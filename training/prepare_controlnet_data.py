from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image


def _canny_condition(image: Image.Image) -> Image.Image:
    import cv2
    import numpy as np

    arr = np.array(image.convert("L"))
    edges = cv2.Canny(arr, 100, 200)
    return Image.fromarray(edges).convert("RGB")


def _openpose_condition(image: Image.Image) -> Image.Image:
    from controlnet_aux import OpenposeDetector

    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return detector(image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a ControlNet conditioning dataset from example assets")
    parser.add_argument("--source", type=Path, default=Path("public/examples/human"))
    parser.add_argument("--out", type=Path, default=Path("training/cn_dataset"))
    parser.add_argument("--condition", choices=("openpose", "canny"), default="openpose")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--caption", default="a full-body photo of a person wearing casual clothes, studio lighting")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"source not found: {args.source}")
        return 1

    images_dir = args.out / "images"
    conditioning_dir = args.out / "conditioning"
    images_dir.mkdir(parents=True, exist_ok=True)
    conditioning_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(p for p in args.source.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not items:
        print(f"no images in {args.source}")
        return 1

    for i, src in enumerate(items):
        name = f"{i:04d}"
        img = Image.open(src).convert("RGB")

        img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
        cond = _canny_condition(img) if args.condition == "canny" else _openpose_condition(img)
        cond = cond.resize((args.resolution, args.resolution), Image.LANCZOS)

        dst_img = images_dir / f"{name}.png"
        dst_cond = conditioning_dir / f"{name}.png"
        if args.force or not dst_img.exists():
            img.save(dst_img)
        if args.force or not dst_cond.exists():
            cond.save(dst_cond)

    meta = args.out / "captions.csv"
    if args.force or not meta.exists():
        with meta.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "conditioning", "caption"])
            for i in range(len(items)):
                writer.writerow([f"images/{i:04d}.png", f"conditioning/{i:04d}.png", args.caption])

    print(f"{args.condition} dataset ready: {len(items)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
