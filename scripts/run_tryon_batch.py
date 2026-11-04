#!/usr/bin/env python3
"""Batch VTON runner: real garment + real person -> try-on image.

Calls backend.engine_sdxl.compose_tryon_sdxl (SDXL + dual ControlNet
[openpose|canny] + IP-Adapter + trained VTON LoRA) so weights/paths/prompts
exactly match the production engine. Garment category is auto-mapped from
the source folder's manifest.csv when present.

Usage (GPU machine, production):
  python scripts/run_tryon_batch.py \
    --garments /mnt/workspace/vton/market_analysis/vton_flatlays \
    --persons  /mnt/workspace/vton/market_analysis/vton_models \
    --out      /mnt/workspace/vton/market_analysis/vton_runs \
    --pairs 50 --steps 30

CPU preflight (no UNet; verifies pose/canny/lora/ip-adapter assets):
  python scripts/run_tryon_batch.py --preflight --persons DIR
"""
import argparse, csv, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend.engine_sdxl as eng

# manifest subcategory -> engine garment_type ("upper"/"dress"/"lower"/"onesie")
SUBCAT_MAP = {
    "TOPS": "upper", "TEES": "upper", "BLOUSES": "upper", "SHIRTS": "upper",
    "JACKETS": "upper", "COATS": "upper", "KNITWEAR": "upper",
    "DRESSES": "dress", "SKIRTS": "dress",
    "PANTS": "lower", "SHORTS": "lower", "JEANS": "lower",
    "SWIMWEAR": "dress", "ONESIES": "onesie", "OTHER": "dress",
}


def garment_types_from_manifest(garment_dir: Path, garment_files):
    m = garment_dir / "manifest.csv"
    if not m.exists():
        return {f.name: "upper" for f in garment_files}
    rows = {}
    with open(m) as fh:
        for r in csv.DictReader(fh):
            rows[r.get("file", "")] = SUBCAT_MAP.get((r.get("subcategory") or "OTHER").upper(), "dress")
    return {f.name: rows.get(f.name, "dress") for f in garment_files}


def preflight(person_dir: Path) -> int:
    """CPU-only checks: OpenPose + Canny on a real photo, LoRA + IP-Adapter file validation."""
    import torch
    from PIL import Image
    people = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
    ok = True
    print(f"[preflight] persons: {len(people)} in {person_dir}")
    if people:
        img = Image.open(people[0]).convert("RGB").resize((1024, 1024))
        t0 = time.time()
        pose = eng._get_openpose_control(img)
        print(f"[preflight] OpenPose OK ({time.time()-t0:.1f}s, out {pose.size})")
        pose.save(person_dir / "_preflight_pose.jpg")
        t0 = time.time()
        canny = eng._get_canny_control(img)
        print(f"[preflight] Canny OK ({time.time()-t0:.1f}s)")
        canny.save(person_dir / "_preflight_canny.jpg")
    else:
        print("[preflight] WARN no person images yet; detectors still verified on a blank")
        img = Image.new("RGB", (1024, 1024), "white")
        eng._get_openpose_control(img); eng._get_canny_control(img)
        print("[preflight] OpenPose + Canny detectors OK")
    try:
        import safetensors
        with safetensors.safe_open(eng.SDXL_DIR / "lora/adapter_model.safetensors", framework="pt") as hdr:
            n = len(list(hdr.keys()))
        print(f"[preflight] LoRA safetensors OK ({n} tensors)")
    except Exception as e:
        ok = False
        print(f"[preflight] LoRA check FAILED: {e}")
    print(f"[preflight] IP-Adapter available: {eng.ip_adapter_available()}")
    print("[preflight] FULL UNet inference requires GPU (fp32 SDXL set ~39GB > this host's RAM)")
    return 0 if ok else 1


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
    ap.add_argument("--fit", default="regular", choices=["fitted", "regular", "loose", "oversized"],
                    help="garment fit on the model's body: how the fabric should drape/contour")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    if args.preflight:
        sys.exit(preflight(args.persons))

    import torch
    if not torch.cuda.is_available():
        print("ERROR: full inference needs a GPU host (see --preflight for CPU checks)")
        sys.exit(2)

    gdir = args.garments or Path("/mnt/workspace/vton/market_analysis/vton_flatlays")
    gfiles = sorted(gdir.glob("*.jpg")) + sorted(gdir.glob("*.png"))
    pfiles = sorted(args.persons.glob("*.jpg")) + sorted(args.persons.glob("*.png"))
    n = args.pairs or min(len(gfiles), len(pfiles))
    out = args.out or Path("/mnt/workspace/vton/market_analysis/vton_runs")
    gtypes = garment_types_from_manifest(gdir, gfiles)

    print(f"[tryon] device=cuda pairs={n} steps={args.steps} garments={len(gfiles)} persons={len(pfiles)}")
    gfiles += gfiles * ((n // max(len(gfiles), 1)) + 1)
    pfiles += pfiles * ((n // max(len(pfiles), 1)) + 1)

    report = []
    for i in range(n):
        g, p = gfiles[i % len(gfiles)], pfiles[i % len(pfiles)]
        t0 = time.time()
        img = eng.compose_tryon_sdxl(
            base_path=p, garment_path=g, audience=args.audience,
            garment_type=gtypes.get(g.name, "dress"), background=args.background,
            ethnicity=args.ethnicity, num_inference_steps=args.steps, fit=args.fit,
        )
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"tryon_{i:02d}_{p.stem}_x_{g.stem}.jpg"
        img.save(dest, quality=92)
        r = {"pair": i, "garment": g.name, "person": p.name, "out": dest.name,
             "sec": round(time.time() - t0, 1),
             "degraded": bool(img.info.get("engine_degraded", False)),
             "degraded_reason": img.info.get("degraded_reason", "")}
        report.append(r)
        print(f"[tryon] {i+1}/{n} -> {dest.name} ({r['sec']}s) degraded={r['degraded']}")

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[tryon] done -> {out}/report.json")


if __name__ == "__main__":
    main()
