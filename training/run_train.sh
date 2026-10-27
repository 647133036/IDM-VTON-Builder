#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${BASE_MODEL:-AI-ModelScope/stable-diffusion-xl-base-1.0}"
MODEL="${MODEL:-lora}"

case "$MODEL" in
  lora)
    DATASET_DIR="${DATASET_DIR:-training/dataset}"
    OUTPUT_DIR="${OUTPUT_DIR:-models/sdxl/lora}"
    STEPS="${STEPS:-800}"

    python3 training/prepare_dataset.py --out "$DATASET_DIR"
    python3 training/train_lora_sdxl.py \
      --dataset_dir "$DATASET_DIR" \
      --base_model "$BASE_MODEL" \
      --output_dir "$OUTPUT_DIR" \
      --max_train_steps "$STEPS"
    python3 training/infer_lora.py \
      --base_model "$BASE_MODEL" \
      --lora_dir "$OUTPUT_DIR" \
      --output_dir training/output
    echo "pipeline finished: LoRA at $OUTPUT_DIR, samples at training/output"
    ;;

  lora_family)
    DATASET_DIR="${DATASET_DIR:-training/dataset}"
    GROUPS="${GROUPS:-cloth human}"
    STEPS="${STEPS:-600}"

    python3 training/prepare_dataset.py --out "$DATASET_DIR"
    python3 training/train_lora_family.py \
      --dataset_dir "$DATASET_DIR" \
      --groups $GROUPS \
      --base_model "$BASE_MODEL" \
      --max_train_steps "$STEPS"
    echo "pipeline finished: per-group LoRA adapters under models/sdxl/lora/"
    ;;

  controlnet)
    DATASET_DIR="${DATASET_DIR:-training/cn_dataset}"
    CONDITION="${CONDITION:-openpose}"
    # 默认输出到引擎实际扫描的目录(models/sdxl/controlnet_<condition>)。
    OUTPUT_DIR="${OUTPUT_DIR:-models/sdxl/controlnet_${CONDITION}}"
    STEPS="${STEPS:-800}"

    python3 training/prepare_controlnet_data.py \
      --source public/examples/human \
      --condition "$CONDITION" \
      --out "$DATASET_DIR"
    python3 training/train_controlnet_sdxl.py \
      --dataset_dir "$DATASET_DIR" \
      --base_model "$BASE_MODEL" \
      --output_dir "$OUTPUT_DIR" \
      --max_train_steps "$STEPS"
    echo "pipeline finished: ControlNet at $OUTPUT_DIR"
    ;;

  *)
    echo "unknown MODEL=$MODEL (lora|lora_family|controlnet)"
    exit 1
    ;;
esac