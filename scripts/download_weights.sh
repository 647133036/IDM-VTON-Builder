#!/usr/bin/env bash
# IDM-VTON 权重下载脚本
# 需要: huggingface-cli (pip install huggingface_hub)
# 用法: bash scripts/download_weights.sh
# 权重约 20-30GB，需要充足磁盘空间

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/IDM-VTON/ckpt"
HF_REPO="yisol/IDM-VTON"

echo "==> IDM-VTON 权重下载"
echo "    目标目录: ${CKPT_DIR}"
echo "    来源: ${HF_REPO}"
echo ""

if ! command -v huggingface-cli &>/dev/null; then
    echo "未找到 huggingface-cli，正在安装..."
    pip install -q huggingface_hub
fi

mkdir -p "${CKPT_DIR}"

echo "==> 下载主模型权重..."
huggingface-cli download "${HF_REPO}" \
    --include "unet/*" "vae/*" "text_encoder/*" "text_encoder_2/*" "tokenizer/*" "tokenizer_2/*" "scheduler/*" \
    --local-dir "${CKPT_DIR}"

echo ""
echo "==> 下载 OpenPose 模型..."
huggingface-cli download yisol/IDM-VTON \
    --include "ckpt/openpose/*" \
    --local-dir "${REPO_ROOT}/IDM-VTON"

echo ""
echo "==> 下载 DensePose 模型..."
huggingface-cli download yisol/IDM-VTON \
    --include "ckpt/densepose/*" \
    --local-dir "${REPO_ROOT}/IDM-VTON"

echo ""
echo "==> 下载 humanparsing 模型..."
huggingface-cli download yisol/IDM-VTON \
    --include "ckpt/humanparsing/*" \
    --local-dir "${REPO_ROOT}/IDM-VTON"

echo ""
echo "==> 下载 IP-Adapter 模型..."
huggingface-cli download TencentARC/IP-Adapter \
    --include "ip-adapter-plus_sdxl_vit-h.bin" \
    --local-dir "${REPO_ROOT}/IDM-VTON"

echo ""
echo "==> 下载 image_encoder..."
huggingface-cli download TencentARC/IP-Adapter \
    --include "models/image_encoder/*" \
    --local-dir "${REPO_ROOT}/IDM-VTON/ckpt"

echo ""
echo "==> 安装 Python 依赖..."
pip install -q torch>=2.1.0 diffusers>=0.27.0 transformers>=4.40.0 accelerate>=0.30.0

echo ""
echo "==> 完成!"
echo "    权重目录: ${CKPT_DIR}"
echo "    验证: 运行 python3 -c \"from backend.engine import engine_status; print(engine_status())\""
