#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_DIR="${DATASET_DIR:-training/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-models/sdxl/lora}"
BASE_MODEL="${BASE_MODEL:-AI-ModelScope/stable-diffusion-xl-base-1.0}"
STEPS="${STEPS:-800}"

# Prepare the training set from example assets
python training/prepare_dataset.py --out "$DATASET_DIR"

# Train the LoRA
python training/train_lora_sdxl.py \
  --dataset_dir "$DATASET_DIR" \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --max_train_steps "$STEPS"

# Generate verification samples
python training/infer_lora.py \
  --base_model "$BASE_MODEL" \
  --lora_dir "$OUTPUT_DIR" \
  --output_dir training/output

echo "pipeline finished: LoRA at $OUTPUT_DIR, samples at training/output"