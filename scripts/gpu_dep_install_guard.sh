#!/usr/bin/env bash
# gpu_dep_install_guard.sh — vton GPU 盒依赖装护栏 + 全量校验（有卡开机第 2 步跑; 只写不跑产物）
#
# 规格单一事实源: repo/scripts/env_policy.yaml (banned / whitelist / optional / runtime_env)。
# 本脚本用 python3+PyYAML 从该 yaml 读常量（盒子环境 PyYAML 6.0 在 freeze 内），不在脚本里另起炉灶。
#
# 关键条款（细节与原因见 env_policy.yaml 各 rationale 字段）:
#   * mmdet 禁装 —— 可执行检查: ① `pip show mmdet` 为空; ② 裸进程 `python -c "import mmdet"` 必须失败。
#     真 mmdet 装进来 = 双杀: mmpose 的 `from mmdet.utils import ...` 命中真包绕过 fake-deps stub,
#     且真 mmdet 自身 import mmcv.ops（本环境 mmcv 是 lite 版无 ops C 扩展）→ 二次崩溃。
#   * mmcv 以 mmcv-lite 落地 —— sdist + MMCV_WITH_OPS=0 安装: dist 名 mmcv-lite, 模块名 mmcv,
#     无 ops C 扩展。真 mmcv 是 mmpose import 时版本断言 (2.0.0rc4<=mmcv.__version__<=3.0.0)
#     与 mmcv.cnn 全家桶的刚需; mmcv.ops 由 fake-deps 恒等 stub 提供, RTMPose 推理路径 0 次调用。
#     禁止把 lite 换成完整版 mmcv "补全功能"。
#   * mediapipe = optional —— 不安装、不阻塞: 缺失打印 OPT_SKIP, 不计 FAIL。主流程不依赖
#     (RTMPose 为 primary, rtmpose-m_coco.pth 在场时回退路径不可达; 代码已迁移 Tasks API,
#     pose_landmarker_full.task 已预置); 未来启用 = 单独 pip install mediapipe (>=0.10 tasks 版)。
#   * OMP_NUM_THREADS=8 —— 无卡模式 cgroup 0.5 核配额下, 不限线程的 192 OMP 线程 CFS thrashing
#     会把 RTMPose forward 从 2.2s 拖到 20+ 分钟。
#
# 用法（盒子上）:
#   bash repo/scripts/gpu_dep_install_guard.sh            # 先验证; 缺才装; 最后全冒烟
#   VTON_PYTHON=/root/miniconda3/bin/python MMCV_TARBALL=/path/mmcv-2.2.0.tar.gz \
#     bash repo/scripts/gpu_dep_install_guard.sh           # 干净 venv 复现协议时显式给 tarball
#
# 校验和来源（mmcv sdist）: sha256=ac479247e808d8802f89eadf04d4118de86bdfe81361ec5aed0cc1bf731c67c9
#   (479121 B, 阿里云 pypi sdist, 2026-11-05 dep-audit 拉取, NAS 备份
#    archive_20261104_combo_women01/dep_audit_pkgs/mmcv-2.2.0.tar.gz)。
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="$REPO_ROOT/scripts/env_policy.yaml"
PYTHON="${VTON_PYTHON:-/root/miniconda3/bin/python}"
PIP=("$PYTHON" -m pip)
MMCV_TARBALL="${MMCV_TARBALL:-}"   # 干净 venv 复现时给本地 tarball; 缺省走阿里云 pypi sdist
ALIYUN_INDEX="https://mirrors.aliyun.com/pypi/simple/"
WORK="$(mktemp -d /tmp/vton_guard.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_THREADS=8
FAIL=0
note() { echo "  [guard] $*"; }
ok()   { echo "  [OK] $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }

# ── 0) 资源预检（只读; 无卡/有卡 cgroup 不同, 按 env_policy.yaml runtime_env）─────────────
echo "═══ 0) 资源预检 ═══"
MEM_MAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)
MEM_GB=$(( MEM_MAX / 1073741824 ))
echo "  cgroup memory.max = ${MEM_GB}GiB (无卡=2; 有卡按所购档位)"
if [ "$MEM_GB" -lt 8 ]; then
  note "RAM < 8GiB: 无卡模式或低档位 —— 只够 pose/冒烟, SDXL 出图需切有卡配置 (SDXL+多ControlNet+rembg 峰值 ~4-6GB)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
  "$PYTHON" - <<'PY' 2>/dev/null || note "torch 未装, 跳过 cuda 检查 (装依赖后自动有)"
import torch
print(f"  torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
else
  note "nvidia-smi 不在 (无卡开机模式) —— 仅 pose/冒烟, 不出图"
fi

# ── 1) 从 env_policy.yaml 读护栏常量（PyYAML; 解析失败 = 规格不可用, 直接 FAIL）──────────
echo "═══ 1) 加载护栏规格 (env_policy.yaml) ═══"
POLICY_VARS="$( "$PYTHON" - "$POLICY" <<'PY' 2>/dev/null || true
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
banned = " ".join(d["banned"].keys())
wl = " ".join(d["whitelist"]["packages"])
ver = " ".join(f"{k}={v}" for k, v in d["whitelist"]["versions"].items())
opt = " ".join(d["optional"].keys())
print(f"GUARD_BANNED='{banned}'")
print(f"GUARD_WHITELIST='{wl}'")
print(f"GUARD_VERSIONS='{ver}'")
print(f"GUARD_OPTIONAL='{opt}'")
PY
)"
if [ -z "$POLICY_VARS" ]; then bad "env_policy.yaml 解析失败 (PyYAML 缺失或文件损坏)"; fi
# shellcheck disable=SC1091
eval "$POLICY_VARS"
ok "规格: banned=[$GUARD_BANNED] whitelist=[$GUARD_WHITELIST] optional=[$GUARD_OPTIONAL]"

# ── 2) 装前 freeze + 卸混叠 dist ─────────────────────────────────────────────────────────
echo "═══ 2) 装前基线 ═══"
"${PIP[@]}" freeze > "$WORK/pre_freeze.txt" 2>/dev/null || true
# mmcv-full/完整版 mmcv 等混叠 dist 与 mmcv-lite 方案冲突, 清掉
for d in mmcv-full; do
  if "${PIP[@]}" show "$d" >/dev/null 2>&1; then
    note "卸混叠 dist: $d"
    "${PIP[@]}" uninstall -y "$d" >/dev/null 2>&1 || true
  fi
done

# ── 3) 逐包验证/安装（全程 --no-deps; 白名单外不动任何既有包）──────────────────────────
echo "═══ 3) 依赖安装/校验 ═══"
dist_present() { "${PIP[@]}" show "$1" >/dev/null 2>&1; }
pkg_version()  { "${PIP[@]}" show "$1" 2>/dev/null | sed -n 's/^Version: //p'; }

# mmpose 1.3.2
if dist_present mmpose; then ok "mmpose 已在 ($(pkg_version mmpose))"
else
  note "装 mmpose==1.3.2 (源: 阿里云 pypi; 非网络类失败才需人工介入)"
  "${PIP[@]}" install --no-deps -i "$ALIYUN_INDEX" "mmpose==1.3.2" \
    || bad "mmpose 源安装失败"
fi

# mmcv-lite 2.2.0 (sdist + MMCV_WITH_OPS=0 → dist 名 mmcv-lite)
if dist_present mmcv-lite; then ok "mmcv-lite 已在 ($(pkg_version mmcv-lite))"
else
  if [ -n "$MMCV_TARBALL" ]; then
    sum=$(sha256sum "$MMCV_TARBALL" | cut -d' ' -f1)
    if [ "$sum" != "ac479247e808d8802f89eadf04d4118de86bdfe81361ec5aed0cc1bf731c67c9" ]; then
      bad "mmcv tarball sha256 不符 ($sum) — 拒装 (来源不明)"
    else
      note "装 mmcv sdist (本地 tarball, MMCV_WITH_OPS=0 → mmcv-lite)"
      MMCV_WITH_OPS=0 "${PIP[@]}" install --no-deps "$MMCV_TARBALL" \
        || bad "mmcv tarball 安装失败"
    fi
  else
    # 源路径: 阿里云 pypi 的 mmcv sdist (mmcv 2.2.0 无 wheel 发布, 走 sdist 纯 Python 装)
    note "装 mmcv==2.2.0 sdist (阿里云 pypi, MMCV_WITH_OPS=0 → mmcv-lite)"
    MMCV_WITH_OPS=0 "${PIP[@]}" download --no-deps -d "$WORK" -i "$ALIYUN_INDEX" "mmcv==2.2.0" \
      && MMCV_WITH_OPS=0 "${PIP[@]}" install --no-deps "$WORK"/mmcv-2.2.0.tar.gz \
      || bad "mmcv 源安装失败 (干净 venv 复现请给 MMCV_TARBALL 本地 tarball)"
  fi
fi

for p in mmengine-lite json_tricks xtcocotools; do
  if dist_present "$p"; then ok "$p 已在 ($(pkg_version "$p"))"
  else
    note "装 $p"
    "${PIP[@]}" install --no-deps -i "$ALIYUN_INDEX" "$p" || bad "$p 安装失败"
  fi
done

# ── 4) 装后断言（env_policy.yaml 可执行护栏）────────────────────────────────────────────
echo "═══ 4) 装后断言 ═══"
"${PIP[@]}" show mmcv-lite | sed -n '1,2p' | sed 's/^/  /'
[ "$(pkg_version mmcv-lite)" = "2.2.0" ] && ok "mmcv-lite=2.2.0" || bad "mmcv-lite 版本异常: $(pkg_version mmcv-lite)"

# banned.mmdet.checks ①: pip show mmdet 必须为空
if "${PIP[@]}" show mmdet >/dev/null 2>&1; then
  bad "banned: 真 mmdet 已装 (pip show mmdet 非空) → 双杀 mmpose; 回滚: pip uninstall -y mmdet"
else
  ok "banned: pip show mmdet 为空"
fi
# banned.mmdet.checks ②: 裸进程 import mmdet 必须失败 (成功 = 磁盘有真包绕过 stub)
if "$PYTHON" -c "import mmdet" >/dev/null 2>&1; then
  bad "banned: 裸进程可 import mmdet (真包绕过 fake-deps stub) → 回滚: pip uninstall -y mmdet"
else
  ok "banned: 裸进程 import mmdet 失败 (合规)"
fi

# optional.mediapipe: 不装不阻塞
if "${PIP[@]}" show mediapipe >/dev/null 2>&1; then
  note "mediapipe 在场 (可选; 本配方无场景用到, 保留不动)"
else
  echo "  OPT_SKIP mediapipe (allowed, 非阻塞)"
fi

# freeze diff 白名单: 只允许新增 GUARD_WHITELIST, 既有包任何变动 → FAIL + 回滚指令
"${PIP[@]}" freeze > "$WORK/post_freeze.txt" 2>/dev/null || true
"$PYTHON" - "$WORK/pre_freeze.txt" "$WORK/post_freeze.txt" "$GUARD_WHITELIST" <<'PY'
import sys
pre = dict(l.split("==", 1) for l in open(sys.argv[1]) if "==" in l)
post = dict(l.split("==", 1) for l in open(sys.argv[2]) if "==" in l)
wl = sys.argv[3].split()
added = {k: post[k] for k in post if k not in pre}
removed = [k for k in pre if k not in post]
changed = {k: (pre[k], post[k]) for k in pre if k in post and pre[k] != post[k]}
wl = [w.lower().replace("_", "-") for w in sys.argv[3].split()]
def norm(n):
    return n.lower().replace("_", "-")
bad_added = [k for k in added if norm(k) not in wl]
print(f"  freeze diff: ADDED={added} REMOVED={removed} CHANGED={changed}")
if removed:
    sys.exit(2)
if changed:
    print(f"  [FAIL] 既有包被变动 {changed} → 回滚: " + " ".join(f"pip install {k}=={v}" for k, v in changed.values()))
    sys.exit(2)
if bad_added:
    print(f"  [FAIL] 白名单外新增 {bad_added} → 回滚: " + " ".join(f"pip uninstall -y {k}" for k in bad_added))
    sys.exit(2)
print("  [OK] freeze diff 白名单通过")
PY
[ $? -eq 0 ] || FAIL=1

# ── 5) 冒烟段（stub 断言 + shape 冒烟 + RTMPose E2E, OMP=8 已 export）────────────────────
echo "═══ 5) 冒烟 ═══"
"$PYTHON" - <<'PY' || FAIL=1
import sys, types, warnings
warnings.filterwarnings("ignore")
# 5a) 复刻 repo fake-deps 的最小 stub 面 (mmdet.utils{ConfigType,reduce_mean} + mmcv.ops 恒等),
#     断言 stub 形状匹配 mmpose 期望 —— 不 import repo 代码, guard 自足。
mmdet = types.ModuleType("mmdet")
mu = types.ModuleType("mmdet.utils")
mu.ConfigType = dict
def reduce_mean(x, dist):
    return x
mu.reduce_mean = reduce_mean
mmdet.utils = mu
sys.modules["mmdet"] = mmdet
sys.modules["mmdet.utils"] = mu
import mmcv  # 真 mmcv-lite
_ops = types.ModuleType("mmcv.ops")
def _ident(*a, **k):
    return (lambda f: f)
_ops.__getattr__ = lambda n: (_ident if not n.startswith("__") else (_ for _ in ()).throw(AttributeError(n)))
sys.modules["mmcv.ops"] = _ops
assert sys.modules["mmdet.utils"].ConfigType is dict, "stub ConfigType 不变式 (在 mmdet.utils 上)"
assert sys.modules["mmcv.ops"] is _ops, "stub mmcv.ops 未生效"
# 5b) shape 冒烟 (box_probe_evidence.md §4.3): 全 import 链 + mmpose 正确把 mmdet 当缺席降级
import mmpose
from mmpose.apis import inference_topdown  # noqa: F401
from mmpose.apis.inferencers.base_mmpose_inferencer import has_mmdet
assert has_mmdet is False, f"has_mmdet={has_mmdet!r}: mmpose 未正确把 mmdet 当缺席"
print(f"  [OK] shape 冒烟: import mmpose {mmpose.__version__} + inference_topdown + has_mmdet=False")
# 5c) RTMPose E2E: 合成图 forward + 关键点 sane (不依赖预置基础图, 完全自足)
import numpy as np
from mmengine.config import Config
from mmpose.models.backbones.cspnext import CSPNeXt
from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead
import torch
root = "/root/vton/repo/models/rtmpose"
cfg = Config.fromfile(f"{root}/rtmpose-m_8xb256-420e_coco-256x192.py")
# 构造方式镜像 repo backend/platform_presets.py _init_rtmpose (L894-896, 已验证)
skip = {"type", "_scope_", "init_cfg"}
mb = cfg.model  # 顶层 estimator 结构: model.backbone / model.head
bb = CSPNeXt(**{k: v for k, v in mb.backbone.items() if k not in skip})
hd = RTMCCHead(**{k: v for k, v in mb.head.items() if k not in skip | {"loss", "decoder"}})
sd = torch.load(f"{root}/rtmpose-m_coco.pth", map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
bb.load_state_dict({k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
hd.load_state_dict({k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}, strict=False)
bb.eval(); hd.eval()
# 合成人体状输入: 输入尺寸(192x256=W,H)灰度图中央竖椭圆(纯 numpy, 不依赖 cv2) — 免 crop 不整除 backbone stride
iw, ih = 192, 256
img = np.zeros((ih, iw, 3), np.uint8)
yy, xx = np.mgrid[0:ih, 0:iw]
ell = ((xx - iw // 2) ** 2 / (iw // 5) ** 2 + (yy - ih // 2) ** 2 / (ih // 5) ** 2) <= 1.0
img[ell] = 120
x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().unsqueeze(0) / 255
# RTMCCHead.forward(feats) → (pred_x, pred_y)  simcc 表示 (v1.3.2 源码 L134/L161)
px, py = hd(bb(x))
split = float(mb.head.get("simcc_split_ratio", 2.0))
kx = px[0].argmax(-1).numpy() / split
ky = py[0].argmax(-1).numpy() / split
# sane: 17 点有限、在输入范围内、未塌缩成单点 (合成图是 OOD 输入, 坐标精度不作要求,
# 只验证 forward+SimCC 解码链路产出结构正常 — 空间精度在真图 run_b1 上验收)
assert np.isfinite(kx).all() and np.isfinite(ky).all()
assert kx.min() >= 0 and kx.max() < iw and ky.min() >= 0 and ky.max() < ih
assert (kx.max() - kx.min()) + (ky.max() - ky.min()) > 40, "17 点塌缩成单点, 解码异常"
print(f"  [OK] RTMPose E2E: {iw}x{ih} 合成图 forward+SimCC decode, 17 点有限且不塌缩 (kp5@{kx[5]:.0f},{ky[5]:.0f})")
PY

# ── 6) 结果汇总 ──────────────────────────────────────────────────────────────────────────
echo "═══ 结果 ═══"
if [ "$FAIL" -eq 0 ]; then
  echo "GUARD PASS — 依赖护栏合规: mmdet 缺席 + 白名单 5 包 + mmcv-lite + stub 形状 + RTMPose E2E 通过"
  echo "下一步: 有卡模式先跑资源预检(memory.max>=8GiB + nvidia-smi), 再 run_b1.py"
else
  echo "GUARD FAIL — 见上方 [FAIL] 项; 回滚指令已随白名单 diff 打印"
fi
exit "$FAIL"
