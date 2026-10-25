#!/usr/bin/env bash
# SDXL + ControlNet + IP-Adapter 权重下载脚本（商用合规）
# 需要: huggingface-cli (pip install huggingface_hub)
# 用法: bash scripts/download_sdxl_weights.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/models/sdxl"

echo "==> SDXL 商用合规权重下载"
echo "    目标目录: ${MODEL_DIR}"
echo ""

if ! command -v huggingface-cli &>/dev/null; then
    echo "未找到 huggingface-cli，正在安装..."
    pip install -q huggingface_hub
fi

mkdir -p "${MODEL_DIR}"

echo "==> 下载 Stable Diffusion XL 基础模型..."
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --include "unet/*" "vae/*" "text_encoder/*" "text_encoder_2/*" "tokenizer/*" "tokenizer_2/*" "scheduler/*" \
    --local-dir "${MODEL_DIR}/sdxl_base"

echo ""
echo "==> 下载 ControlNet Canny (SDXL)..."
huggingface-cli download xinsir/controlnet-canny-sdxl-1.0 \
    --local-dir "${MODEL_DIR}/controlnet_canny"

echo ""
echo "==> 下载 ControlNet OpenPose (SDXL)..."
huggingface-cli download xinsir/controlnet-openpose-sdxl-1.0 \
    --local-dir "${MODEL_DIR}/controlnet_openpose"

echo ""
echo "==> 下载 OpenPose/Canny 预处理器权重 (Annotators)..."
huggingface-cli download lllyasviel/Annotators \
    --local-dir "${MODEL_DIR}/annotators"

echo ""
echo "==> 下载 IP-Adapter image_encoder..."
huggingface-cli download h94/IP-Adapter \
    --include "models/image_encoder/*" \
    --local-dir "${MODEL_DIR}/ip_adapter"

echo ""
echo "==> 下载 IP-Adapter 模型权重..."
huggingface-cli download h94/IP-Adapter \
    --include "ip-adapter-plus_sdxl_vit-h.bin" \
    --local-dir "${MODEL_DIR}/ip_adapter"

echo ""
echo "==> 安装 Python 依赖..."
pip install -q torch>=2.1.0 diffusers>=0.27.0 transformers>=4.40.0 accelerate>=0.30.0 controlnet-aux>=0.0.7 rembg>=2.0.50

echo ""
echo "==> 完成!"
echo "    权重目录: ${MODEL_DIR}"
echo "    验证: LICENSE_MODE=commercial python3 -c \"from backend.engine import engine_status; print(engine_status())\""