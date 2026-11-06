#!/usr/bin/env python3
"""Batch VTON runner: real garment + real person -> try-on image.

Calls backend.engine_sdxl.compose_tryon_sdxl (SDXL + ControlNet
[pose+depth|pose+canny] + IP-Adapter + trained VTON LoRA) so weights/paths/
prompts exactly match the production engine. Garment category is auto-mapped
from the source folder's manifest.csv when present.

Usage (GPU machine, production):
  python scripts/run_tryon_batch.py \
    --garments /mnt/workspace/vton/market_analysis/vton_flatlays \
    --persons  /mnt/workspace/vton/market_analysis/vton_models \
    --out      /mnt/workspace/vton/market_analysis/vton_runs \
    --pairs 50 --steps 30 --push-to nas:/mnt/.../vton_runs

CPU preflight (no UNet; verifies pose/canny/lora/ip-adapter assets):
  python scripts/run_tryon_batch.py --preflight --persons DIR

安全护栏(对应 pipeline audit P0/P1):
- 配方固化: 默认 os.environ.setdefault 到「已验证交付配方 C」(WARP=0 DETAIL=0
  LOCK_ID=1 SEED=42); 跑前 assert_recipe() 打印生效配方, 若落到已知会复现
  「整图竖条纹碎裂」的配方 B(WARP=1+DETAIL=1) 直接拒绝(除非 --allow-unverified-recipe)。
- 逐图 try/except + 增量写 report: 一张 OOM 不再报废整批, 已成功的 manifest 不丢。
- 出图即回传: 每张 img.save 后立即 scp 回 NAS(--push-to / VTON_PUSH_TARGET),
  失败重试, 最终仍有失败 → 非零退出(易失数据盘不再静默丢图)。
- glob 过滤 _preflight_* 探测图: preflight 写的探测图不会被当人物跑换衣。
- 降级阈值: 全降级批次非零退出(不再「看似成功」)。
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend.engine_sdxl as eng

# ── 配方固化(pipeline audit P0-1 / TSV 行2) ─────────────────────────────────
# 全量批处理默认走「已验证交付配方 C」。用 setdefault: 操作员仍可用真实 env 覆盖
# (如 depth 实验), 但跑前 assert_recipe() 会核出生效配方并在危险组合上拒绝。
RECIPE_C = {"VTON_WARP": "0", "VTON_DETAIL": "0", "VTON_LOCK_ID": "1", "VTON_SEED": "42"}
for _k, _v in RECIPE_C.items():
    os.environ.setdefault(_k, _v)

RECIPE_KEYS = ("VTON_WARP", "VTON_DETAIL", "VTON_LOCK_ID", "VTON_SEED", "VTON_PATTERN_LOCK")

# manifest subcategory -> engine garment_type ("upper"/"dress"/"lower"/"onesie")
SUBCAT_MAP = {
    "TOPS": "upper",
    "TEES": "upper",
    "BLOUSES": "upper",
    "SHIRTS": "upper",
    "JACKETS": "upper",
    "COATS": "upper",
    "KNITWEAR": "upper",
    "DRESSES": "dress",
    "SKIRTS": "dress",
    "PANTS": "lower",
    "SHORTS": "lower",
    "JEANS": "lower",
    "SWIMWEAR": "dress",
    "ONESIES": "onesie",
    "OTHER": "dress",
}

# preflight 探测图前缀: 绝不能被当人物/衣物吃进批处理(TSV 行18)。
_PREFLIGHT_PREFIX = "_preflight_"


def _list_images(d: Path):
    """列出目录下图片, 过滤 preflight 探测图与隐藏文件。"""
    files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    return [f for f in files if not f.name.startswith(_PREFLIGHT_PREFIX) and not f.name.startswith(".")]


def garment_types_from_manifest(garment_dir: Path, garment_files):
    m = garment_dir / "manifest.csv"
    if not m.exists():
        return {f.name: "upper" for f in garment_files}
    rows = {}
    with open(m) as fh:
        for r in csv.DictReader(fh):
            rows[r.get("file", "")] = SUBCAT_MAP.get((r.get("subcategory") or "OTHER").upper(), "dress")
    return {f.name: rows.get(f.name, "dress") for f in garment_files}


def effective_recipe() -> dict:
    return {k: os.environ.get(k, "") for k in RECIPE_KEYS}


def assert_recipe(allow_unverified: bool = False) -> None:
    """跑前断言生效配方(P0-1)。已知坏组合直接拒绝, 避免烧 GPU 出废图。"""
    r = effective_recipe()
    print(
        "[recipe] effective: WARP={VTON_WARP} DETAIL={VTON_DETAIL} "
        "LOCK_ID={VTON_LOCK_ID} SEED={VTON_SEED} PATTERN_LOCK={VTON_PATTERN_LOCK}".format(**r)
    )
    if allow_unverified:
        print("[recipe] WARN --allow-unverified-recipe: 跳过配方校验(结果大概率不可交付)")
        return
    if r["VTON_WARP"] == "1" and r["VTON_DETAIL"] == "1":
        raise SystemExit(
            "[recipe] REFUSING to run: WARP=1 + DETAIL=1 是未验证的极大路径, "
            "已知复现「整图竖条纹碎裂」(REPORT §2/§4, pipeline audit P0-1)。\n"
            "  全量批处理默认「已验证交付配方 C」= WARP=0 DETAIL=0 LOCK_ID=1 SEED=42。\n"
            "  若确要跑该配方做实验, 加 --allow-unverified-recipe 显式确认。"
        )


def preflight(person_dir: Path) -> int:
    """CPU-only checks: OpenPose + Canny on a real photo, LoRA + IP-Adapter file validation.

    探测图写到 person_dir 下的 .preflight/ 隐藏子目录(而非 person_dir 本身),
    并在文件名带 _preflight_ 前缀 —— 双保险, 保证后续 glob 不会把它当人物。
    """
    from PIL import Image

    people = _list_images(person_dir)
    ok = True
    print(f"[preflight] persons: {len(people)} in {person_dir}")
    probe_dir = person_dir / ".preflight"
    probe_dir.mkdir(parents=True, exist_ok=True)
    if people:
        img = Image.open(people[0]).convert("RGB").resize((1024, 1024))
        t0 = time.time()
        pose = eng._get_openpose_control(img)
        print(f"[preflight] OpenPose OK ({time.time() - t0:.1f}s, out {pose.size})")
        pose.save(probe_dir / f"{_PREFLIGHT_PREFIX}pose.jpg")
        t0 = time.time()
        canny = eng._get_canny_control(img)
        print(f"[preflight] Canny OK ({time.time() - t0:.1f}s)")
        canny.save(probe_dir / f"{_PREFLIGHT_PREFIX}canny.jpg")
    else:
        print("[preflight] WARN no person images yet; detectors still verified on a blank")
        img = Image.new("RGB", (1024, 1024), "white")
        eng._get_openpose_control(img)
        eng._get_canny_control(img)
        print("[preflight] OpenPose + Canny detectors OK")
    try:
        import safetensors

        with safetensors.safe_open(eng.SDXL_DIR / "lora/adapter_model.safetensors", framework="pt") as hdr:
            n = len(list(hdr.keys()))
        print(f"[preflight] LoRA safetensors OK ({n} tensors)")
    except Exception as e:  # noqa: BLE001 - 预检失败是降级为告警(后续 exit5), 不是崩溃
        ok = False
        print(f"[preflight] LoRA check FAILED: {e}")
    print(f"[preflight] IP-Adapter available: {eng.ip_adapter_available()}")
    # depth P0 回归确认(资产层, 无需 GPU): controlnet_depth 模型 + dpt midas 都在才算 ready。
    try:
        st = eng.engine_status()
        print(
            f"[preflight] depth assets: controlnet_depth={st.get('controlnet_depth')} "
            f"midas_ckpt={st.get('depth_midas_ckpt')} -> depth_ready={st.get('depth_ready')}"
        )
        if not st.get("controlnet_depth"):
            print(
                "[preflight] WARN depth ControlNet 模型缺失 → 双 ControlNet 缩水为 pose+canny。"
                "补: bash scripts/download_weights.sh ONLY=depth"
            )
        elif not st.get("depth_midas_ckpt"):
            print(
                "[preflight] WARN dpt_hybrid-midas ckpt 缺失 → depth 会退化灰图@scale0(零贡献)。"
                "补: bash scripts/download_weights.sh ONLY=annotators"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] WARN engine_status() failed: {e}")
    print("[preflight] FULL UNet inference requires GPU (fp32 SDXL set ~39GB > this host's RAM)")
    return 0 if ok else 1


def _push_file(src: Path, target: str, retries: int = 3, timeout: int = 120) -> bool:
    """把单张出图立刻 scp 回 NAS。失败重试, 最终失败返回 False(调用方记非零退出)。

    target 形如 'user@host:/abs/dest/dir'(scp 语义)。BatchMode=yes 避免卡在
    密码交互; StrictHostKeyChecking=accept-new 首连自动记录 host key。
    """
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                [
                    "scp",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    "ConnectTimeout=15",
                    str(src),
                    target,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                if attempt > 1:
                    print(f"[push] {src.name} OK on retry {attempt}")
                return True
            print(
                f"[push] {src.name} attempt {attempt}/{retries} FAILED rc={proc.returncode}: "
                f"{(proc.stderr or '').strip()[:200]}"
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            print(f"[push] {src.name} attempt {attempt}/{retries} ERROR: {e}")
        if attempt < retries:
            time.sleep(min(2**attempt, 10))
    return False


def _write_report_incremental(out: Path, report: list) -> None:
    """每张图后重写 report.json(原子替换), 崩溃也不丢已成功的 manifest。"""
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "report.json.tmp"
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    tmp.replace(out / "report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garments", type=Path, default=None)
    ap.add_argument("--persons", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pairs", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--audience", default="women")
    ap.add_argument("--ethnicity", default="east_asian")
    ap.add_argument("--background", default="white")
    ap.add_argument(
        "--fit",
        default="regular",
        choices=["fitted", "regular", "loose", "oversized"],
        help="garment fit on the model's body: how the fabric should drape/contour",
    )
    ap.add_argument("--preflight", action="store_true")
    # 出图回传钩子(P0-2): 每张图 save 后立即 scp 回 NAS。不设则只落易失数据盘(会告警)。
    ap.add_argument(
        "--push-to",
        default=os.environ.get("VTON_PUSH_TARGET"),
        help="scp 目标, 如 nas:/mnt/workspace/vton/runs (每张出图即回传; 失败非零退出)",
    )
    ap.add_argument("--push-retries", type=int, default=3)
    ap.add_argument(
        "--allow-unverified-recipe", action="store_true", help="允许在已知会碎裂的配方(WARP=1+DETAIL=1)上强行运行"
    )
    ap.add_argument(
        "--max-degraded-frac",
        type=float,
        default=1.0,
        help="降级图占比超过该阈值则非零退出(默认 1.0=全降级才失败; 调低更严)",
    )
    args = ap.parse_args()

    if args.preflight:
        sys.exit(preflight(args.persons))

    # 配方校验在 GPU 检查之前: 配方错就该早停, 别等加载完 39GB 才发现。
    assert_recipe(args.allow_unverified_recipe)

    import torch

    if not torch.cuda.is_available():
        print("ERROR: full inference needs a GPU host (see --preflight for CPU checks)")
        sys.exit(2)

    gdir = args.garments or Path("/mnt/workspace/vton/market_analysis/vton_flatlays")
    gfiles = _list_images(gdir)
    pfiles = _list_images(args.persons)
    n = args.pairs or min(len(gfiles), len(pfiles))
    out = args.out or Path("/mnt/workspace/vton/market_analysis/vton_runs")
    gtypes = garment_types_from_manifest(gdir, gfiles)

    print(f"[tryon] device=cuda pairs={n} steps={args.steps} garments={len(gfiles)} persons={len(pfiles)}")
    print(f"[tryon] out={out} push_to={args.push_to or '(NONE — 结果只落易失数据盘!)'}")
    if not args.push_to:
        print(
            "[tryon] WARN 未配置 --push-to/VTON_PUSH_TARGET: 出图只写本机易失数据盘, "
            "关机/release 即丢(REPORT §2.3 已丢过 3/5 张)。强烈建议配置回传。"
        )
    if not gfiles:
        print(f"ERROR: no garment images in {gdir}")
        sys.exit(2)
    if not pfiles:
        print(f"ERROR: no person images in {args.persons}")
        sys.exit(2)
    gfiles += gfiles * ((n // max(len(gfiles), 1)) + 1)
    pfiles += pfiles * ((n // max(len(pfiles), 1)) + 1)

    out.mkdir(parents=True, exist_ok=True)
    report: list = []
    n_ok = 0
    n_failed = 0
    n_degraded = 0
    n_push_failed = 0
    push_warned = False

    for i in range(n):
        g, p = gfiles[i % len(gfiles)], pfiles[i % len(pfiles)]
        t0 = time.time()
        dest = out / f"tryon_{i:02d}_{p.stem}_x_{g.stem}.jpg"
        r = {
            "pair": i,
            "garment": g.name,
            "person": p.name,
            "out": dest.name,
            "sec": round(time.time() - t0, 1),
            "status": "ok",
            "degraded": False,
            "degraded_reason": "",
        }
        try:
            img = eng.compose_tryon_sdxl(
                base_path=p,
                garment_path=g,
                audience=args.audience,
                garment_type=gtypes.get(g.name, "dress"),
                background=args.background,
                ethnicity=args.ethnicity,
                num_inference_steps=args.steps,
                fit=args.fit,
            )
            img.save(dest, quality=92)
            r["sec"] = round(time.time() - t0, 1)
            r["degraded"] = bool(img.info.get("engine_degraded", False))
            r["degraded_reason"] = img.info.get("degraded_reason", "")
            r["detail_applied"] = bool(img.info.get("detail_applied", False))
            r["lock_identity"] = bool(img.info.get("lock_identity", False))
            if "prewhiten" in img.info:
                r["prewhiten"] = img.info["prewhiten"]
            n_ok += 1
            if r["degraded"]:
                n_degraded += 1
            print(
                f"[tryon] {i + 1}/{n} -> {dest.name} ({r['sec']}s) degraded={r['degraded']}"
                + (f" reason={r['degraded_reason']}" if r["degraded"] else "")
            )
        except Exception as e:  # noqa: BLE001  逐图隔离(P0-2/P1-19): 一张 OOM 不报废整批。
            n_failed += 1
            r["status"] = "error"
            r["error"] = f"{type(e).__name__}: {e}"
            r["sec"] = round(time.time() - t0, 1)
            print(f"[tryon] {i + 1}/{n} FAILED ({r['sec']}s): {r['error']}", file=sys.stderr)

        # 出图即回传(P0-2): save 成功后立刻 scp, 失败重试 + 记非零退出。
        if r["status"] == "ok":
            if args.push_to:
                if not _push_file(dest, args.push_to, retries=args.push_retries):
                    n_push_failed += 1
                    r["push"] = "failed"
                    print(f"[push] {dest.name} 最终回传失败(已重试 {args.push_retries} 次)", file=sys.stderr)
                else:
                    r["push"] = "ok"
            elif not push_warned:
                push_warned = True

        report.append(r)
        _write_report_incremental(out, report)  # 增量写(P1-19): 崩溃不丢 manifest。

    # 汇总 + 非零退出判定(P1-19): 全失败 / 全降级 / 回传失败 都不能「看似成功」。
    produced = n_ok
    degraded_frac = (n_degraded / produced) if produced else 1.0
    print(
        f"[tryon] done: ok={n_ok} failed={n_failed} degraded={n_degraded} "
        f"push_failed={n_push_failed} degraded_frac={degraded_frac:.2f}"
    )
    print(f"[tryon] report -> {out}/report.json")

    if produced == 0:
        print(f"[tryon] FAIL: 0/{n} 张出图成功", file=sys.stderr)
        sys.exit(3)
    if n_push_failed:
        print(f"[tryon] FAIL: {n_push_failed} 张出图回传 NAS 失败(数据盘易失, 需人工抢救)", file=sys.stderr)
        sys.exit(4)
    if degraded_frac > args.max_degraded_frac:
        print(
            f"[tryon] FAIL: 降级占比 {degraded_frac:.2f} > 阈值 {args.max_degraded_frac}(全废图批次不应视为成功)",
            file=sys.stderr,
        )
        sys.exit(5)


if __name__ == "__main__":
    main()
