#!/usr/bin/env bash
# SDXL 换衣引擎权重下载脚本(v0.1.9-pre,pipeline audit P0-4 重写)
#
# 重写原因(旧脚本已作废):旧版下载的是 legacy IDM-VTON/ckpt,而 SDXL 引擎
# 读的是 models/sdxl/*(engine_sdxl.py:SDXL_DIR)。数据盘 release 后按旧脚
# 本重灌会下到错目录,SDXL 仍加载不了,最贵的 ~24G 重下白花。
#
# 目标布局(与 engine_sdxl.py SDXL_DIR + platform_presets.py _*_ROOT 对齐):
#   models/sdxl/sdxl_base/                Stable Diffusion XL base 1.0
#   models/sdxl/controlnet_openpose/      thibaud pose ControlNet (SDXL)
#   models/sdxl/controlnet_canny/         thibaud canny ControlNet (SDXL)
#   models/sdxl/controlnet_depth/         thibaud depth ControlNet (SDXL)  ← P0-5 新增
#   models/sdxl/ip_adapter/               IP-Adapter (base/plus/plus-face, vit-h)
#   models/sdxl/ip_adapter/image_encoder/ OpenCLIP ViT-H/14 图像编码器
#   models/sdxl/lora/                     训练好的 VTON LoRA(adapter_model.safetensors)
#   models/sdxl/annotators/               openpose body/hand/facenet + dpt midas   ← P0-6 补 midas
#   models/mediapipe/                     pose_landmarker_full.task
#   models/rtmpose/                       rtmpose-m ckpt + cfg
#   models/seg/                           u2net.onnx (rembg 主体掩膜)
#   models/sdxl/sdxl_inpaint/             SDXL Inpaint 权重(可选兜底)
#   models/fashn/                         FASHN VTON v1.5 + DWPose(商用主引擎)
#
# 用法:
#   bash scripts/download_weights.sh                    # 全量
#   ONLY=depth bash scripts/download_weights.sh         # 只补 depth 相关(controlnet_depth + dpt midas)
# 环境变量:
#   ONLY=name[,name]  只下指定组(sdxl_base|openpose|canny|depth|ip_adapter|image_encoder|lora|annotators|mediapipe|rtmpose|seg|sdxl_inpaint|fashn)
#   HF_ENDPOINT       默认 https://hf-mirror.com(大陆可达);海外机可清空走官方
#   DRY_RUN=1         只打印将要下载什么,不实际下载
#
# 注意:本脚本**不装/不升任何 Python 依赖**(避免拖动既有依赖版本导致起不来)。
# 依赖安装走 scripts/gpu_dep_install_guard.sh(带 --no-deps + freeze diff 白名单护栏)。
# 验证用 engine_sdxl.engine_status(不是 legacy engine)。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
SDXL_DIR="${MODELS_DIR}/sdxl"
FASHN_DIR="${MODELS_DIR}/fashn"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "==> SDXL 换衣引擎权重下载"
echo "    目标目录: ${MODELS_DIR}"
echo "    HF_ENDPOINT: ${HF_ENDPOINT}"
echo ""

if command -v hf &>/dev/null; then
    HF_BIN="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_BIN="huggingface-cli"
else
    echo "未找到 hf / huggingface-cli。**不自动 pip install**(不拖动依赖);请先单独安装:"
    echo "  pip install -i https://mirrors.aliyun.com/pypi/simple/ \"huggingface_hub\""
    exit 1
fi
echo "    下载器: ${HF_BIN}"

# ── 工具函数 ────────────────────────────────────────────────────────────────
want() {  # want <组名>  —— 配合 ONLY 过滤
    if [ -z "${ONLY:-}" ]; then return 0; fi
    case ",${ONLY}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}
hf_dl() {  # hf_dl <repo> <include-glob> <dest-dir>
    local repo="$1" inc="$2" dest="$3"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "    [DRY_RUN] ${HF_BIN} download ${repo} --include \"${inc}\" --local-dir ${dest}"
        return 0
    fi
    mkdir -p "${dest}"
    "${HF_BIN}" download "${repo}" --include "${inc}" --local-dir "${dest}"
}
# 权重已存在则跳过(幂等, 断点续跑友好)。
have() { [ -e "$1" ]; }

# ── SDXL base(最大, ~6.9G fp16) ───────────────────────────────────────────
if want sdxl_base; then
    echo "==> sdxl_base (stabilityai/stable-diffusion-xl-base-1.0)"
    if have "${SDXL_DIR}/sdxl_base/unet"; then
        echo "    已存在, 跳过"
    else
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "*.json" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "scheduler/*" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "unet/*" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "vae/*" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "text_encoder/*" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "text_encoder_2/*" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "tokenizer/*" "${SDXL_DIR}/sdxl_base"
        hf_dl "stabilityai/stable-diffusion-xl-base-1.0" \
            "tokenizer_2/*" "${SDXL_DIR}/sdxl_base"
    fi
fi

# ── ControlNet(pose / canny / depth),thibaud 单文件 ckpt ───────────────────
if want openpose; then
    echo "==> controlnet_openpose (thibaud/controlnet-openpose-sdxl-1.0)"
    hf_dl "thibaud/controlnet-openpose-sdxl-1.0" \
        "OpenPoseXL2.safetensors" "${SDXL_DIR}/controlnet_openpose"
    hf_dl "thibaud/controlnet-openpose-sdxl-1.0" \
        "config.json" "${SDXL_DIR}/controlnet_openpose"
fi

if want canny; then
    echo "==> controlnet_canny (thibaud/controlnet-canny-sdxl-1.0)"
    hf_dl "thibaud/controlnet-canny-sdxl-1.0" \
        "diffusion_pytorch_model.safetensors" "${SDXL_DIR}/controlnet_canny"
    hf_dl "thibaud/controlnet-canny-sdxl-1.0" \
        "config.json" "${SDXL_DIR}/controlnet_canny"
fi

# P0-5: depth ControlNet 模型本体。**没有它 depth_loaded 恒为 False,
# 宣称的「双 ControlNet」缩水为 pose+canny**(engine_sdxl.py:583-602)。
if want depth; then
    echo "==> controlnet_depth (thibaud/controlnet-depth-sdxl-1.0)  ← P0-5"
    hf_dl "thibaud/controlnet-depth-sdxl-1.0" \
        "diffusion_pytorch_model.safetensors" "${SDXL_DIR}/controlnet_depth"
    hf_dl "thibaud/controlnet-depth-sdxl-1.0" \
        "config.json" "${SDXL_DIR}/controlnet_depth"
fi

# ── IP-Adapter(SDXL vit-h) + 图像编码器 ──────────────────────────────────
if want ip_adapter; then
    echo "==> ip_adapter (h94/IP-Adapter)"
    hf_dl "h94/IP-Adapter" \
        "sdxl/ip-adapter_sdxl_vit-h.safetensors" "${SDXL_DIR}/ip_adapter"
    hf_dl "h94/IP-Adapter" \
        "sdxl/ip-adapter-plus_sdxl_vit-h.safetensors" "${SDXL_DIR}/ip_adapter"
    hf_dl "h94/IP-Adapter" \
        "sdxl/ip-adapter-plus-face_sdxl_vit-h.safetensors" "${SDXL_DIR}/ip_adapter"
fi

if want image_encoder; then
    echo "==> image_encoder (OpenCLIP ViT-H/14, h94/IP-Adapter)"
    hf_dl "h94/IP-Adapter" \
        "sdxl/models/image_encoder/config.json" "${SDXL_DIR}/ip_adapter"
    hf_dl "h94/IP-Adapter" \
        "sdxl/models/image_encoder/model.safetensors" "${SDXL_DIR}/ip_adapter"
fi

# ── LoRA:训练产物不在 HF,提示来源(不联网拉) ─────────────────────────────
if want lora; then
    echo "==> lora (训练好的 VTON LoRA)"
    if have "${SDXL_DIR}/lora/adapter_model.safetensors"; then
        echo "    已存在, 跳过"
    else
        echo "    WARN: ${SDXL_DIR}/lora/adapter_model.safetensors 不存在。"
        echo "      LoRA 是本项目自训练产物, 不在公共 HF。请从 NAS/训练目录拷入:"
        echo "        scp training/output/<run>/pytorch_lora_weights.safetensors ${SDXL_DIR}/lora/adapter_model.safetensors"
        echo "      缺失时引擎仍可跑(无 LoRA), 但成衣风格不匹配。"
    fi
fi

# ── Annotators:openpose body/hand/facenet + **dpt midas**(P0-6)────────────
# dpt_hybrid-midas-501f0c75.pt 是 MidasDetector 的精确文件名。缺它时即便
# controlnet_depth 模型在位, depth 也会退化为纯灰图 @ scale 0.0(零贡献)。
if want annotators; then
    echo "==> annotators (lllyasviel/Annotators)  openpose + dpt-midas"
    hf_dl "lllyasviel/Annotators" \
        "body_pose_model.pth" "${SDXL_DIR}/annotators"
    hf_dl "lllyasviel/Annotators" \
        "hand_pose_model.pth" "${SDXL_DIR}/annotators"
    hf_dl "lllyasviel/Annotators" \
        "facenet.pth" "${SDXL_DIR}/annotators"
    hf_dl "lllyasviel/Annotators" \
        "dpt_hybrid-midas-501f0c75.pt" "${SDXL_DIR}/annotators"   # ← P0-6 新增
fi

# ── mediapipe pose_landmarker(MediaPipe 回退路) ────────────────────────────
if want mediapipe; then
    echo "==> mediapipe (pose_landmarker_full.task, ~9MB)"
    hf_dl "funasr/mediapipe-pose-landmarker" \
        "pose_landmarker_full.task" "${MODELS_DIR}/mediapipe" || \
    echo "    WARN: mediapipe .task 下载失败(回退路可选)。手动下:"
    echo "      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
fi

# ── RTMPose(腰线精度, D1 齐腰切依赖) ──────────────────────────────────────
if want rtmpose; then
    echo "==> rtmpose (rwightman/rtmlib ckpt + cfg)"
    hf_dl "rushi23/rtmpose-m_coco" \
        "rtmpose-m_coco.pth" "${MODELS_DIR}/rtmpose" || \
    echo "    WARN: rtmpose ckpt 下载失败。从 NAS 拷: models/rtmpose/{rtmpose-m_coco.pth, rtmpose-m_8xb256-420e_coco-256x192.py}"
    hf_dl "rushi23/rtmpose-m_coco" \
        "rtmpose-m_8xb256-420e_coco-256x192.py" "${MODELS_DIR}/rtmpose" || true
fi

# ── u2net(rembg 主体掩膜;缺了 force_white 退纯颜色压白) ────────────────────
if want seg; then
    echo "==> seg (u2net.onnx, rembg 主体掩膜)"
    hf_dl "skytnt/anime-seg" "u2net.onnx" "${MODELS_DIR}/seg" || \
    echo "    WARN: u2net.onnx 下载失败(可选, 缺了走纯颜色压白)。"
fi

# ── FASHN VTON v1.5(商用主引擎, Apache-2.0) ────────────────────────────────
# 只拉主模型 + DWPose/YOLOX 权重。fashn-human-parser(NVIDIA SegFormer)
# 含商用限制, 不下载; 商用路径由 backend/engine_fashn.py 注入零张量 stub。
if want fashn; then
    echo "==> fashn (FASHN VTON v1.5 + DWPose, Apache-2.0)"
    if have "${FASHN_DIR}/model.safetensors"; then
        echo "    已存在, 跳过"
    else
        hf_dl "fashn-ai/fashn-vton-1.5" "model.safetensors" "${FASHN_DIR}"
        hf_dl "fashn-ai/DWPose" "yolox_l.onnx" "${FASHN_DIR}/dwpose"
        hf_dl "fashn-ai/DWPose" "dw-ll_ucoco_384.onnx" "${FASHN_DIR}/dwpose"
    fi
fi

# ── SDXL Inpaint(可选兜底;缺时全图 ControlNet + degraded 标记) ────────────
# 注意: 官方 stabilityai/stable-diffusion-xl-inpainting 仓库不存在,
# 用 diffusers 社区微调版 stable-diffusion-xl-1.0-inpainting-0.1,
# 只拉 fp16 变体(约 6G, 引擎 _sdxl_inpaint_repair 以 variant=fp16 加载)。
if want sdxl_inpaint; then
    echo "==> sdxl_inpaint (diffusers/stable-diffusion-xl-1.0-inpainting-0.1, fp16)"
    if have "${SDXL_DIR}/sdxl_inpaint/model_index.json"; then
        echo "    已存在, 跳过"
    else
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "model_index.json" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "scheduler/*" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "unet/config.json" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "unet/diffusion_pytorch_model.fp16.safetensors" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "vae/config.json" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "vae/diffusion_pytorch_model.fp16.safetensors" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "text_encoder/config.json" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "text_encoder/model.fp16.safetensors" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "text_encoder_2/config.json" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "text_encoder_2/model.fp16.safetensors" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "tokenizer/*" "${SDXL_DIR}/sdxl_inpaint"
        hf_dl "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" \
            "tokenizer_2/*" "${SDXL_DIR}/sdxl_inpaint"
    fi
fi

# ── 完成校验(验 SDXL 引擎, 不是 legacy engine) ────────────────────────────
echo ""
echo "==> 完成!权重目录: ${MODELS_DIR}"
echo "    关键资产核对:"
for f in \
    "${SDXL_DIR}/sdxl_base/unet" \
    "${SDXL_DIR}/controlnet_openpose/config.json" \
    "${SDXL_DIR}/controlnet_canny/config.json" \
    "${SDXL_DIR}/controlnet_depth/config.json" \
    "${SDXL_DIR}/ip_adapter/ip-adapter_sdxl_vit-h.safetensors" \
    "${SDXL_DIR}/lora/adapter_model.safetensors" \
    "${SDXL_DIR}/annotators/body_pose_model.pth" \
    "${SDXL_DIR}/annotators/dpt_hybrid-midas-501f0c75.pt" \
    "${MODELS_DIR}/rtmpose/rtmpose-m_coco.pth" \
    "${MODELS_DIR}/seg/u2net.onnx" \
    "${FASHN_DIR}/model.safetensors" \
    "${FASHN_DIR}/dwpose/yolox_l.onnx" \
    "${FASHN_DIR}/dwpose/dw-ll_ucoco_384.onnx" ; do
    if have "$f"; then echo "      ✓ $f"; else echo "      ✗ 缺 $f"; fi
done
echo ""
echo "    验证(用 engine_sdxl, 不是 legacy engine):"
echo "      cd ${REPO_ROOT}/backend && python3 -c \"import engine_sdxl as e; print(e.engine_status())\""
echo "    depth 相关 P0 回归确认: engine_status 应显示 controlnet_depth 在位,"
echo "    且日志不再出现 \"Depth control failed\"(见 RUNBOOK_boot_g4.md 全量批处理阶段)。"
