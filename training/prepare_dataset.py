from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _default_caption(kind: str) -> str:
    if kind == "cloth":
        return "a flat-lay photo of a garment, studio lighting, e-commerce product photo"
    return "a full-body photo of a person wearing a garment, studio lighting, professional fashion photography"


def _collect(source: Path, kind: str, out: Path, force: bool) -> list[Path]:
    items = sorted(p for p in source.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not items:
        return []

    dst_kind = out / kind
    dst_kind.mkdir(parents=True, exist_ok=True)

    captions = {}
    for i, src in enumerate(items):
        dst = dst_kind / f"{i:04d}.png"

        if not dst.exists() or force:
            img = Image.open(src)
            img = img.convert("RGB")
            img.save(dst, "PNG")

        txt = dst.with_suffix(".txt")
        if not txt.exists() or force:
            txt.write_text(_default_caption(kind), encoding="utf-8")
        captions[dst.name] = txt.read_text(encoding="utf-8").strip()

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a tiny LoRA training set from example assets")
    parser.add_argument("--cloth", type=Path, default=Path("public/examples/cloth"))
    parser.add_argument("--human", type=Path, default=Path("public/examples/human"))
    parser.add_argument("--out", type=Path, default=Path("training/dataset"))
    parser.add_argument(
        "--groups", nargs="*", default=("cloth", "human"), help="subsets to include, in order: cloth human"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing normalized images and captions")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for source, kind in [(args.cloth, "cloth"), (args.human, "human")]:
        if kind not in args.groups:
            continue
        if not source.exists():
            print(f"skip {kind}: {source} not found")
            continue
        items = _collect(source, kind, args.out, args.force)
        print(f"{kind}: {len(items)} images -> {args.out / kind}")

    manifest = args.out / "manifest.txt"
    manifest.write_text(
        "\n".join(f"{relative}" for relative in sorted(p.relative_to(args.out) for p in args.out.rglob("*.txt")))
        + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
