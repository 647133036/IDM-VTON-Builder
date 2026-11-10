"""印花保真管线 GPU 冒烟: 有卡开机后一条命令验证全链路。

用法(GPU 服务器, 项目在 /root/vton/repo):
    python3 scripts/gpu/smoke_print_pipeline.py                # 默认样图
    python3 scripts/gpu/smoke_print_pipeline.py <base> <garment> [outdir]

验证内容:
    1. engine_status: FASHN 权重在位 + refit_engine 生效
    2. FASHN 路径出图(VTON_REFIT_ENGINE=tps + 反选保护 + 印花 ROI inpaint)
    3. print_fidelity / delivery_tier / lock_identity 元数据
    4. 多 seed(VTON_MULTI_SEED=3)按印花保真分择优
通过标准: print_fidelity >= 0.9 → auto 档; 肉眼看字母/图案不漂、
手臂/胸口/脸/发丝不变白。
"""

import json
import os
import sys
import time
from pathlib import Path

# 冒烟环境: 商用模式 + 贴印引擎 + 反选保护 + 印花 ROI 局部 inpaint
os.environ.setdefault("LICENSE_MODE", "commercial")
os.environ.setdefault("VTON_REFIT_ENGINE", "tps")
os.environ.setdefault("VTON_PROTECT_FEATHER", "4")
os.environ.setdefault("VTON_PRINT_INPAINT", "1")
os.environ.setdefault("VTON_INPAINT_REPAIR", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))

DEFAULT_BASE = Path("/root/autodl-tmp/vton_combo/bases/women_01.png")
DEFAULT_GARMENT = Path("/root/autodl-tmp/vton_combo/garments/01_khaite_goya-silk-top_None_main.jpg")
DEFAULT_OUT = Path("/root/autodl-tmp/vton_combo/out_smoke")


def main() -> int:
    args = sys.argv[1:]
    base = Path(args[0]) if len(args) > 0 else DEFAULT_BASE
    garment = Path(args[1]) if len(args) > 1 else DEFAULT_GARMENT
    outdir = Path(args[2]) if len(args) > 2 else DEFAULT_OUT
    for p, label in ((base, "base"), (garment, "garment")):
        if not p.exists():
            print(f"[FAIL] {label} 不存在: {p}")
            return 1
    outdir.mkdir(parents=True, exist_ok=True)

    import engine

    status = engine.engine_status()
    print("== engine_status ==")
    for key in (
        "engine",
        "fashn_available",
        "fashn_reason",
        "sdxl_available",
        "sdxl_inpaint",
        "gpu_available",
        "refit_engine",
        "license_mode",
        "commercial_compliant",
    ):
        print(f"  {key}: {status.get(key)}")
    if not status.get("fashn_available"):
        print(f"[FAIL] FASHN 不可用: {status.get('fashn_reason')}")
        return 1

    from worker import process_job  # noqa: F401  (确认 worker 链可导入)

    # ---- 单 seed: 基础出图 + 元数据 ----
    print("\n== 单 seed 出图 (VTON_SEED=42) ==")
    os.environ["VTON_SEED"] = "42"
    t0 = time.time()
    img = engine.compose_tryon(base, garment, "women", "auto", 2048, "white", "east_asian")
    dt = time.time() - t0
    out1 = outdir / "smoke_tps_single.png"
    img.convert("RGB").save(out1, "PNG")
    print(f"  耗时 {dt:.1f}s → {out1}")
    info = {k: img.info.get(k) for k in ("engine", "print_fidelity", "delivery_tier", "lock_identity", "detail_applied", "engine_degraded", "degraded_reason")}
    print(f"  result.info: {json.dumps(info, ensure_ascii=False)}")
    pf = img.info.get("print_fidelity")
    if pf is None:
        print("[FAIL] print_fidelity 未写入 result.info")
        return 1
    print(f"  印花保真分 {pf:.3f} → 档位 {img.info.get('delivery_tier')}")

    # ---- 多 seed: 择优 ----
    print("\n== 多 seed 出图 (VTON_MULTI_SEED=3) ==")
    os.environ["VTON_MULTI_SEED"] = "3"
    t0 = time.time()
    img_ms = engine.compose_tryon(base, garment, "women", "auto", 2048, "white", "east_asian")
    dt = time.time() - t0
    out2 = outdir / "smoke_tps_multiseed.png"
    img_ms.convert("RGB").save(out2, "PNG")
    ms = img_ms.info.get("multiseed") or {}
    print(f"  耗时 {dt:.1f}s → {out2}")
    print(f"  multiseed: {json.dumps(ms, ensure_ascii=False)}")

    ok = pf >= 0.9 and img.info.get("delivery_tier") == "auto"
    print("\n== 结论 ==")
    print(f"  print_fidelity={pf:.3f} {'>= 0.9 auto 档, PASS' if ok else '< 0.9, 肉眼复核后定档'}")
    print("  肉眼检查项: 字母/图案不漂移; 手臂/胸口不变白; 脸/发丝不变")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
