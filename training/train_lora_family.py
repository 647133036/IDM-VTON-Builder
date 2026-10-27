from __future__ import annotations

import argparse
import sys
from pathlib import Path

import train_lora_sdxl

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one LoRA per dataset subgroup")
    parser.add_argument("--dataset_dir", type=Path, default=Path("training/dataset"))
    parser.add_argument("--groups", nargs="*", default=("cloth", "human"))
    parser.add_argument("--output_root", type=Path, default=Path("models/sdxl/lora"))
    parser.add_argument("--base_model", default="AI-ModelScope/stable-diffusion-xl-base-1.0")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        print(f"dataset dir missing: {args.dataset_dir}; run prepare_dataset.py first")
        return 1

    failed = []
    for kind in args.groups:
        group_dir = args.dataset_dir / kind
        if not group_dir.exists() or not list(group_dir.glob("*.png")):
            print(f"skip {kind}: empty at {group_dir}")
            continue

        out_dir = args.output_root / kind
        print(f"=== training LoRA: {kind} -> {out_dir} ===", flush=True)
        import argparse as _argparse

        train_args = _argparse.Namespace(
            dataset_dir=group_dir,
            base_model=args.base_model,
            output_dir=out_dir,
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            max_train_steps=args.max_train_steps,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            log_interval=50,
        )
        if train_lora_sdxl.train(train_args) != 0:
            failed.append(kind)

    if failed:
        print(f"failed groups: {failed}")
        return 1
    print("all LoRA adapters saved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
