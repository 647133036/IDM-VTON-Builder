#!/usr/bin/env bash
# vton 换机体检 —— 30 秒确认"能不能直接跑", 别上来就排查几小时。
# 用法:  bash repo/scripts/preflight.sh [vton根目录]      (默认 /mnt/workspace/vton)
# 全绿 = 直接按 PIPELINE §6/§3 起服务; 有 ✗ = 先按提示补那一项。
set -u
ROOT="${1:-/mnt/workspace/vton}"
REPO="$ROOT/repo"
BG=$'\033[1;32m'; R=$'\033[1;31m'; Y=$'\033[1;33m'; N=$'\033[0m'
pass(){ echo -e "${BG}  ✓${N} $1"; }
fail(){ echo -e "${R}  ✗${N} $1"; }
warn(){ echo -e "${Y}  ?${N} $1"; }

echo "════════ vton 换机体检 ════════"
echo "根目录: $ROOT"

# ── 1) 代码 + 权重 (文件系统层面) ──
echo "─ 代码 / 模型权重"
[ -d "$REPO/backend" ] \
  && pass "代码在: $REPO/backend" \
  || fail "代码缺失 → 先 'cp -a /mnt/workspace/vton/repo …' 或 clone(但 45G 权重 git 里没有, 必须连 NAS 一起拷/挂)"
[ -d "$REPO/models/sdxl/sdxl_base/unet" ] \
  && pass "SDXL 权重在 (repo/models, ~45G, 靠 NAS 文件系统)" \
  || warn "repo/models/sdxl 缺失 → 没挂同一个 NAS 或没拷; 起服务会因权重缺失报错"
[ -f "$REPO/models/seg/u2net.onnx" ] \
  && pass "u2net.onnx 在 (v0.1.2 主体掩膜; 缺了会静默回退纯压白, 不致命)" \
  || warn "u2net 缺失 (可选): 走 §4 gh-proxy 加速镜像补, 别直连 GitHub"

# ── 2) Python 依赖 + GPU/CUDA (进程层面) ──
echo "─ Python 依赖 / GPU"
( cd "$REPO/backend" 2>/dev/null || cd "$REPO" )
python3 - <<'PY'
import importlib.util as u, sys
def has(m): 
    try: return u.find_spec(m) is not None
    except (ImportError, ValueError): return False
need = ["fastapi","uvicorn","torch","diffusers","transformers","rembg","onnxruntime"]
missing = [m for m in need if not has(m)]
for m in need:
    print(("\033[1;32m  ✓\033[0m " if has(m) else "\033[1;31m  ✗\033[0m ")+m)
if missing:
    print("\033[1;31m  ✗ 缺这些 → 先 'pip install -r %s' (torch 走 cuda130 index)\033[0m" % "/mnt/workspace/vton/requirements.txt")

print("─ GPU / CUDA (torch 2.13.0+cu130 需要驱动≥550 / CUDA 13 runtime)")
try:
    import torch
    print(f"  torch {torch.__version__}  |  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  |  cuda runtime: {torch.version.cuda}")
        print("\033[1;32m  ✓ GPU 可用\033[0m")
    else:
        print("\033[1;33m  ? 无可用 GPU (或驱动/CUDA 与 torch cu130 不匹配)\033[0m")
        print("    下面 nvidia-smi 给你驱动版本对号: 驱动<550 → torch cu130 起不来, 会卡几小时。")
except Exception as e:
    print(f"\033[1;31m  ✗ torch 加载失败: {e}\033[0m")
PY
echo "─ nvidia-smi (驱动, 人工对号)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /' || echo "  nvidia-smi 无输出"
else
  warn "nvidia-smi 不存在 (非 GPU 机 / 未装驱动)"
fi

# ── 3) 起服务前的最后一道 ──
echo "─ 提示"
echo "  全绿后: cd $REPO/backend && LICENSE_MODE=commercial WORKERS=1 python3 -m uvicorn main:app --host 127.0.0.1 --port 8000"
echo "  详见 PIPELINE-0.1.1.md §3/§6。  体检完就这, 别开始排查。"
echo "══════════════════════════════════"
