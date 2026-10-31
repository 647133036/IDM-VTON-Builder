from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from pathlib import Path

from PIL import Image
from platform_presets import (
    blend_tryon,
    clothing_mask,
    detect_garment_extent,
    force_white_background,
    reapply_garment_detail,
)

logger = logging.getLogger("atelier.engine_sdxl")

_GPU_LOCK = threading.Lock()
_PIPE_LOCK = threading.Lock()
_sdxl_state: dict = {
    "pipe": None,
    "ip_adapter_loaded": False,
    "ip_adapter_reason": "",
    "depth_loaded": False,
    "lora_loaded": False,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
SDXL_DIR = _REPO_ROOT / "models" / "sdxl"
_IDM_DIR = _REPO_ROOT / "IDM-VTON"
_TRAINING_DIR = _REPO_ROOT / "training"

AUDIENCE_PROMPTS = {
    "women": "a fashion model wearing this garment, full body front view, professional fashion photography",
    "men": "a male fashion model wearing this garment, full body front view, professional fashion photography",
    "girls": "a child model wearing this garment, correct proportions, realistic body, full body front view",
    "boys": "a boy child model wearing this garment, correct proportions, realistic body, full body front view",
    "baby": "a baby model wearing this garment, correct proportions, realistic body, full body front view",
}

AUDIENCE_NEGATIVE = {
    "women": "worst quality, low quality, monochrome, bad anatomy, extra limbs, missing limbs, adult body on child",
    "men": "worst quality, low quality, monochrome, bad anatomy, extra limbs, missing limbs, female body",
    "girls": "worst quality, low quality, adult body, distorted proportions, big head, small body, extra limbs",
    "boys": "worst quality, low quality, adult body, distorted proportions, big head, small body, extra limbs",
    "baby": "worst quality, low quality, adult body, extra limbs, missing limbs",
}
# 通用反向(全 audience 生效):电商主图禁项——塑料感/空气刷、有色背景、地面阴影、水印文字。
COMMON_NEGATIVE = (
    "plastic, airbrushed, deformed, distorted body, bad anatomy, extra limbs, "
    "colored background, beige background, studio shadow, floor shadow, "
    "watermark, text, logo, signature"
)
# 摄影质感正向(防"假人感"):电商主图需要的写实体感与面料细节。
PHOTO_QUALITY = "photorealistic, 8k uhd, dslr, soft studio lighting, detailed fabric texture, natural fabric drape"

ETHNICITY_PROMPTS = {
    "western_europe": "a model with a slender build, long legs, relatively narrow hips, European body proportions",
    "eastern_europe": "a model with a taller body, slightly broader shoulders, Eastern European body proportions",
    "west_asia": "a model with a medium height, curvy hips, Middle Eastern body proportions",
    "central_asia": "a model with a compact build, shorter legs relative to torso, Central Asian body proportions",
    "east_asia": "a model with a slim petite frame, narrower shoulders, longer torso "
    "relative to legs, East Asian body proportions",
    "southeast_asia": "a model with a smaller and slimmer body, narrower shoulders, Southeast Asian body proportions",
    "africa": "a model with long arms and legs, athletic build, African body proportions",
    "north_america": "a model with a larger and fuller body, wider waist, North American body proportions",
    "south_america": "a model with an hourglass figure, prominent waist-hip contrast, South American body proportions",
    "oceania": "a model with slightly longer legs, fuller body, Australian and Pacific body proportions",
}

GARMENT_PROMPT_SUFFIX = {
    "upper": "shirt, t-shirt, blouse",
    "dress": "dress, gown, skirt",
    "lower": "pants, trousers, shorts, jeans",
    "onesie": "onesie, romper, baby body",
}

# 穿着贴合层:让衣服"穿在身上"而非"贴在身上"——面料随身材垂坠、起褶皱。
FIT_PROMPTS = {
    "fitted": (
        "the garment is worn on the model's body in a slim fit, fabric hugging the figure with "
        "tight, natural wrinkles and folds that follow the body's silhouette"
    ),
    "regular": (
        "the garment is worn on the model's body in a regular fit, fabric draping naturally over "
        "the figure with realistic wrinkles and folds that conform to the body shape"
    ),
    "loose": (
        "the garment is worn on the model's body in a relaxed loose fit, fabric hanging softly "
        "over the figure with gentle natural folds"
    ),
    "oversized": (
        "the garment is worn on the model's body in an oversized fit, fabric draping wide and "
        "loose over the figure with deep natural folds"
    ),
}

FIT_NEGATIVE = (
    "flat paper-doll garment overlay, garment floating off the body, no fabric drape, stiff flat cloth pasted on skin"
)


def sdxl_available() -> tuple[bool, str]:
    # 目录存在还不够:校验关键模型文件,避免"看似可用实则加载即炸"。
    required = [
        ("sdxl_base", "model_index.json"),
        ("controlnet_openpose", "config.json"),
        ("controlnet_canny", "config.json"),
    ]
    for d, f in required:
        if not (SDXL_DIR / d).exists():
            return False, f"Missing weights: {d}"
        if not (SDXL_DIR / d / f).exists():
            return False, f"Missing {f} in {d}"
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "No GPU available"
    return True, "ok"


_IMAGE_PROJ_BASE_KEY = "image_proj.proj.weight"
_IMAGE_PROJ_PLUS_KEY = "image_proj.layers.0.0.norm1.weight"


def _read_image_encoder_dim(ip_adapter_dir: Path) -> int | None:
    """读取本地 image_encoder 的输出维度(projection_dim,缺失时用 hidden_size)。"""
    config_path = Path(ip_adapter_dir) / "image_encoder" / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for key in ("projection_dim", "hidden_size"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _infer_ip_adapter_input_dim(weight_path: Path) -> int | None:
    """从 IP-Adapter 权重的 image_proj 推断其期望的图像 embedding 维度。

    base 变体的 image_proj.proj.weight 形状为 [out, in],取 in;
    plus/plus-face 变体的 image_proj.layers.0.0.norm1.weight 为 1-D,长度即维度。
    """
    try:
        if weight_path.suffix.lower() == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(weight_path), framework="numpy", device="cpu") as handle:
                keys = set(handle.keys())
                if _IMAGE_PROJ_PLUS_KEY in keys:
                    shape = tuple(handle.get_slice(_IMAGE_PROJ_PLUS_KEY).get_shape())
                    if len(shape) == 1:
                        return int(shape[0])
                if _IMAGE_PROJ_BASE_KEY in keys:
                    shape = tuple(handle.get_slice(_IMAGE_PROJ_BASE_KEY).get_shape())
                    if len(shape) == 2:
                        return int(shape[1])
            return None

        if weight_path.suffix.lower() == ".bin":
            import torch

            state = torch.load(str(weight_path), map_location="cpu")
            if not isinstance(state, dict):
                return None
            # 兼容把权重包在 state_dict/model 下的 .bin。
            for wrapper in ("state_dict", "model"):
                nested = state.get(wrapper)
                if isinstance(nested, dict):
                    state = nested
                    break
            for key, is_1d in ((_IMAGE_PROJ_PLUS_KEY, True), (_IMAGE_PROJ_BASE_KEY, False)):
                tensor = state.get(key)
                shape = getattr(tensor, "shape", None)
                if shape is None:
                    continue
                if is_1d and len(shape) == 1:
                    return int(shape[0])
                if not is_1d and len(shape) == 2:
                    return int(shape[1])
    except Exception:
        # 损坏/格式不符的权重文件跳过,不能让它阻断其它候选。
        logger.debug("Could not inspect IP-Adapter weight %s", weight_path, exc_info=True)
    return None


def _ip_adapter_variant_rank(weight_path: Path) -> int:
    """plus 变体优先于 plus-face,两者都优先于 base。"""
    name = weight_path.name.lower()
    if "plus-face" in name:
        return 1
    if "plus" in name:
        return 0
    return 2


def _select_ip_adapter_weight(ip_adapter_dir: Path) -> tuple[Path | None, str]:
    """选择与本地 image_encoder 输出维度配套的 IP-Adapter 权重。

    返回 (权重路径, 降级原因)。找不到任何权重时原因为空字符串;
    有候选但维度不配套时返回 None 和可直接写入 ip_adapter_reason 的说明。
    """
    ip_adapter_dir = Path(ip_adapter_dir)
    # .safetensors 优先;只有在没有匹配项时才回退到 .bin。
    groups = (
        sorted(ip_adapter_dir.glob("*.safetensors")),
        sorted(ip_adapter_dir.glob("*.bin")),
    )
    # 没有安装任何权重时不读取 encoder 配置,保持「未安装 ≠ 降级」的语义。
    if not any(groups):
        return None, ""

    encoder_dim = _read_image_encoder_dim(ip_adapter_dir)
    if encoder_dim is None:
        config_path = ip_adapter_dir / "image_encoder" / "config.json"
        return None, f"Cannot read image_encoder output dim from {config_path}; skipping IP-Adapter"

    inspected: list[tuple[Path, int]] = []
    for paths in groups:
        group_inspected = [(path, dim) for path in paths if (dim := _infer_ip_adapter_input_dim(path)) is not None]
        inspected.extend(group_inspected)
        matches = [path for path, dim in group_inspected if dim == encoder_dim]
        if matches:
            matches.sort(key=lambda path: (_ip_adapter_variant_rank(path), path.name.lower()))
            return matches[0], ""

    if not inspected:
        return None, f"IP-Adapter weights in {ip_adapter_dir} expose no readable image_proj dims; skipping IP-Adapter"

    expected_dims = sorted({dim for _, dim in inspected})
    if len(expected_dims) == 1:
        reason = (
            f"IP-Adapter weight image_proj expects {expected_dims[0]}-dim embeddings "
            f"but local image_encoder outputs {encoder_dim}"
        )
    else:
        dims = ", ".join(str(dim) for dim in expected_dims)
        reason = f"IP-Adapter weights expect one of {dims}-dim embeddings but local image_encoder outputs {encoder_dim}"
    return None, reason


def _pick_ip_adapter_weight(ip_adapter_dir: Path) -> Path | None:
    """纯函数:返回与 image_encoder 配套的权重路径,无匹配时返回 None。"""
    return _select_ip_adapter_weight(ip_adapter_dir)[0]


def ip_adapter_available() -> bool:
    """IP-Adapter 能否真正加载:匹配的权重文件与 image_encoder(CLIP vision)均须存在。"""
    weight_ok = _pick_ip_adapter_weight(SDXL_DIR / "ip_adapter") is not None
    image_encoder_dir = SDXL_DIR / "ip_adapter" / "image_encoder"
    enc_ok = (image_encoder_dir / "config.json").is_file() and (
        (image_encoder_dir / "model.safetensors").is_file() or (image_encoder_dir / "pytorch_model.bin").is_file()
    )
    return weight_ok and enc_ok


_pose_detector = None
_canny_detector = None
_depth_detector = None


def _get_openpose_control(image: Image.Image) -> Image.Image:
    global _pose_detector
    from controlnet_aux import OpenposeDetector

    if _pose_detector is None:
        # 懒加载加锁(NEW-H3 残留):并发首载会重复构建检测器,用 _PIPE_LOCK 单飞。
        with _PIPE_LOCK:
            if _pose_detector is None:
                annotator_dir = SDXL_DIR / "annotators"
                if annotator_dir.exists():
                    _pose_detector = OpenposeDetector.from_pretrained(str(annotator_dir))
                else:
                    _pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _pose_detector(image)


def _get_canny_control(image: Image.Image) -> Image.Image:
    global _canny_detector
    from controlnet_aux import CannyDetector

    if _canny_detector is None:
        # 懒加载加锁(NEW-H3 残留):并发首载会重复构建检测器,用 _PIPE_LOCK 单飞。
        with _PIPE_LOCK:
            if _canny_detector is None:
                _canny_detector = CannyDetector()
    return _canny_detector(image, low_threshold=100, high_threshold=200, detect_resolution=1024, image_resolution=1024)


def _get_depth_control(image: Image.Image) -> Image.Image:
    """人体深度图:换衣贴合体型、产生布料褶皱与立体感的关键条件(贴合改进)。"""
    global _depth_detector
    from controlnet_aux import MidasDetector

    if _depth_detector is None:
        # 懒加载加锁:并发首载会重复构建检测器,用 _PIPE_LOCK 单飞。
        with _PIPE_LOCK:
            if _depth_detector is None:
                annotator_dir = SDXL_DIR / "annotators"
                if annotator_dir.exists():
                    _depth_detector = MidasDetector.from_pretrained(str(annotator_dir))
                else:
                    _depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return _depth_detector(image)


def _load_trained_lora(pipe, lora_dir: Path) -> str | None:
    import sys

    if str(_TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(_TRAINING_DIR))
    from lora_io import load_trained_lora, lora_weight_path

    if lora_weight_path(lora_dir) is None:
        return None
    return load_trained_lora(pipe, lora_dir)


def _remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove

        return remove(image)
    except ImportError:
        return image


def fit_square(im: Image.Image, side: int, pad=(255, 255, 255)) -> Image.Image:
    """保宽高比缩进 square,短边补白。VTON_AR=0 时回退硬拉伸。"""
    if os.environ.get("VTON_AR", "1") == "0":
        return im.resize((side, side), Image.Resampling.LANCZOS)
    w, h = im.size
    scale = side / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = Image.new("RGB", (side, side), pad)
    canvas.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), ((side - nw) // 2, (side - nh) // 2))
    return canvas


def _apply_lora_scale(pipe) -> None:
    """小数据 r16 LoRA 默认 0.7,避免 1.0 过拟合模板句。VTON_LORA 可覆盖。"""
    raw = os.environ.get("VTON_LORA", "0.7").strip()
    try:
        scale = float(raw)
    except ValueError:
        logger.warning("Invalid VTON_LORA=%r; keeping adapter default", raw)
        return
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(["default"], adapter_weights=[scale])
            logger.info("LoRA scale -> %s (env VTON_LORA)", scale)
            return
        except Exception:
            logger.debug("pipe.set_adapters failed; trying fuse-less scale", exc_info=True)
    if hasattr(pipe, "set_lora_scale"):
        pipe.set_lora_scale(scale)
        logger.info("LoRA scale -> %s via set_lora_scale", scale)


def align_garment_to_pose(
    garment: Image.Image,
    audience: str,
    ethnicity: str,
    garment_type: str,
    canvas: int = 1024,
) -> Image.Image:
    """把平铺衣"掰"到人体姿态:按目标人群肩宽/轮廓缩放并做水平网格变形,
    得到一张"正穿在人身上"的服装参考图,供 SDXL IP-Adapter 使用。

    这是 VITON 几何对齐步骤的纯 CPU 简化版——把衣服先贴到目标体型,
    再让生成器在已对齐的参考上重绘纹理,显著提升衣服在身上的位置/比例
    准确性,尤其解决"衣服飘在身外/袖长不合/比例错位"。

    流程:crop 内容 -> fit_garment_to_body(按肩宽等比缩放) ->
    _warp_garment(沿躯干轮廓收窄/展宽) -> 白底 RGB。

    VTON_WARP=0 时关闭(回退原平铺图);VTON_WARP_STRENGTH 可覆盖默认强度。
    """
    if os.environ.get("VTON_WARP", "1") == "0":
        return garment
    try:
        from engine import (
            GARMENT_WARP_STRENGTH,
            _warp_garment,
            fit_garment_to_body,
            garment_alpha,
            torso_box,
        )
    except ImportError:
        logger.debug("align_garment_to_pose: engine funcs unavailable; passthrough")
        return garment

    box = torso_box(audience, ethnicity)
    base_strength = GARMENT_WARP_STRENGTH.get(garment_type, 0.42)
    raw = os.environ.get("VTON_WARP_STRENGTH", "").strip()
    try:
        strength = float(raw) if raw else base_strength
    except ValueError:
        strength = base_strength
        logger.warning("Invalid VTON_WARP_STRENGTH=%r; using default", raw)

    # fit_garment_to_body 期望能取到衣料 alpha(RGBA)或自动估计(RGB)。
    g = garment.convert("RGBA")
    if int(g.split()[-1].getextrema()[0]) >= 250:
        # 不透明图:用 garment_alpha 补一层 alpha 供 bbox/warp 使用。
        g.putalpha(garment_alpha(garment))

    fitted = fit_garment_to_body(g, box, canvas, canvas)
    if fitted.mode != "RGBA":
        fitted = fitted.convert("RGBA")
    warped = _warp_garment(fitted, box, strength)
    if warped.mode != "RGBA":
        warped = warped.convert("RGBA")

    out = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    out.paste(warped, (0, 0), warped)
    return out


def crop_garment_to_content(
    image: Image.Image,
    pad_frac: float = 0.03,
    min_content_frac: float = 0.04,
) -> Image.Image:
    """把服装平铺图裁到内容边界,去掉大片白底/透明边,让 Canny 与 IP-Adapter
    的参考聚焦在服装本体上(颜色/花纹保真度更高)。

    - RGBA 输入按 alpha 判内容;RGB 输入把四角 24x24 中值当背景估计,
      与背景 L1 距离 > 40 判为内容像素。
    - 内容占比 < min_content_frac(检测不到清晰主体,如纯色图/全身浅色且无白边)
      或找不到连通区时,原样返回(安全 no-op,不会比现状更差)。
    """
    import numpy as np

    arr = np.array(image.convert("RGBA"))
    alpha = arr[..., 3]
    # 只有"真"透明图(PNG 平铺)才按 alpha 判内容;不透明 RGB 升成 RGBA 后
    # alpha 恒 255,必须走背景差分,否则会把整图误判成"全是内容"不裁。
    has_transp = image.mode in ("RGBA", "LA") and int(alpha.min()) < 200
    if has_transp:
        mask = alpha > 20
    else:
        rgb = arr[..., :3].astype(np.int16)
        c = 24
        corners = np.concatenate(
            [
                rgb[:c, :c].reshape(-1, 3),
                rgb[:c, -c:].reshape(-1, 3),
                rgb[-c:, :c].reshape(-1, 3),
                rgb[-c:, -c:].reshape(-1, 3),
            ]
        )
        bg = np.median(corners, axis=0)
        mask = np.abs(rgb - bg).sum(axis=2) > 40

    if float(mask.mean()) < min_content_frac:
        return image  # 检测不到清晰主体 → 原样返回(no-op)

    h, w = mask.shape
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return image
    r0 = int(np.where(rows)[0][0])
    r1 = int(np.where(rows)[0][-1])
    c0 = int(np.where(cols)[0][0])
    c1 = int(np.where(cols)[0][-1])

    p = int(min(h, w) * pad_frac)
    r0 = max(0, r0 - p)
    c0 = max(0, c0 - p)
    r1 = min(h - 1, r1 + p)
    c1 = min(w - 1, c1 + p)

    out = arr[r0 : r1 + 1, c0 : c1 + 1]
    if has_transp:
        return Image.fromarray(out, "RGBA")
    return Image.fromarray(out[..., :3], "RGB")


_FP16_WEIGHT_FILES = ("diffusion_pytorch_model.fp16.safetensors", "model.fp16.safetensors")
_PIPELINE_COMPONENTS = {
    "unet": "diffusion_pytorch_model",
    "vae": "diffusion_pytorch_model",
    "text_encoder": "model",
    "text_encoder_2": "model",
}


def _fp16_variant_kwargs(model_dir: Path) -> dict[str, str]:
    """仅当本地存在 fp16 权重变体时传 variant,避免硬编码 fp16 导致 OSError。

    diffusers 会把 variant 拼进权重文件名:ControlNet/UNet/VAE 为
    diffusion_pytorch_model.fp16.safetensors,CLIP text encoder 为
    model.fp16.safetensors。本地目录只有非 variant 文件时必须省略 variant。
    """
    if any((model_dir / name).is_file() for name in _FP16_WEIGHT_FILES):
        return {"variant": "fp16"}
    return {}


def _pipeline_fp16_variant_kwargs(pipeline_dir: Path) -> dict[str, str]:
    """pipeline 只有在所有带权重组件的 fp16 变体都存在时才启用 variant。

    若任一组件缺失 fp16 变体,diffusers 仍会为该组件拼出 .fp16.safetensors
    路径并报 OSError;因此宁可整条 pipeline 回退到非 variant 权重。
    """
    has_variant = False
    for subfolder, stem in _PIPELINE_COMPONENTS.items():
        component_dir = pipeline_dir / subfolder
        if not component_dir.is_dir():
            continue
        variant_file = component_dir / f"{stem}.fp16.safetensors"
        if not variant_file.is_file():
            # 该组件没有 fp16 变体 -> diffusers 会拼出不存在的路径,整条 pipeline 都不能传 variant。
            return {}
        has_variant = True
    return {"variant": "fp16"} if has_variant else {}


def _get_sdxl_pipe(device, dtype):
    # 首次加载必须单飞(NEW-H3):并发请求同时见 pipe=None 会重复构建管线撑爆显存。
    with _PIPE_LOCK:
        if _sdxl_state["pipe"] is not None:
            return _sdxl_state["pipe"], _sdxl_state["ip_adapter_loaded"], _sdxl_state["ip_adapter_reason"]

        import torch
        from diffusers import (
            ControlNetModel,
            DDPMScheduler,
            StableDiffusionXLControlNetPipeline,
        )

        scheduler = DDPMScheduler.from_pretrained(
            str(SDXL_DIR / "sdxl_base"), subfolder="scheduler", local_files_only=True
        )

        # thibaud/controlnet-openpose-sdxl-1.0 ships the weights as a single-file
        # checkpoint (OpenPoseXL2.safetensors) rather than a diffusers subfolder layout.
        openpose_ckpt = SDXL_DIR / "controlnet_openpose"
        openpose_single = openpose_ckpt / "OpenPoseXL2.safetensors"
        if openpose_single.exists():
            controlnet_pose = ControlNetModel.from_single_file(str(openpose_single), torch_dtype=dtype)
        else:
            controlnet_pose = ControlNetModel.from_pretrained(
                str(openpose_ckpt), torch_dtype=dtype, use_safetensors=True, local_files_only=True
            )

        # 本地 canny 目录可能只有非 variant 权重;有 fp16 变体才传 variant。
        def _load_canny():
            return ControlNetModel.from_pretrained(
                str(SDXL_DIR / "controlnet_canny"),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
                **_fp16_variant_kwargs(SDXL_DIR / "controlnet_canny"),
            )

        # 贴合改进:默认只加载 pose+depth(2 个 controlnet)。3 个 controlnet 常驻
        # 会超出 24GB 显存,故不默认全加载;depth 缺失或失败时退回 pose+canny。
        depth_dir = SDXL_DIR / "controlnet_depth"
        depth_loaded = False
        if (depth_dir / "config.json").exists():
            try:
                controlnets = [
                    controlnet_pose,
                    ControlNetModel.from_pretrained(
                        str(depth_dir),
                        torch_dtype=dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        **_fp16_variant_kwargs(depth_dir),
                    ),
                ]
                depth_loaded = True
            except Exception:
                logger.exception("Failed to load depth ControlNet; falling back to canny")
                controlnets = [controlnet_pose, _load_canny()]
        else:
            controlnets = [controlnet_pose, _load_canny()]

        pipe = None
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                str(SDXL_DIR / "sdxl_base"),
                controlnet=controlnets,
                scheduler=scheduler,
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
                **_pipeline_fp16_variant_kwargs(SDXL_DIR / "sdxl_base"),
            )
            # SDXL base VAE 需以 fp32 解码,否则 fp16 下解码溢出产生 NaN/黑图。
            # 此刻组件在 CPU 上,直接改 dtype 即可。
            pipe.vae = pipe.vae.to(torch.float32)

            lora_dir = SDXL_DIR / "lora"
            lora_loaded = False
            try:
                lora_fmt = _load_trained_lora(pipe, lora_dir)
                if lora_fmt:
                    lora_loaded = True
                    logger.info("Loaded trained LoRA (%s) from %s", lora_fmt, lora_dir)
            except Exception:
                logger.exception("Failed to load trained LoRA; continuing without it")

            # 权重必须与本地 image_encoder 的输出维度配套:base(1024)在 1280 编码器上
            # 会 shape mismatch,plus/plus-face(1280)才兼容。按实际 image_proj 维度选择。
            ip_adapter_weight, ip_adapter_reason = _select_ip_adapter_weight(SDXL_DIR / "ip_adapter")
            ip_adapter_loaded = False
            if ip_adapter_weight is not None:
                try:
                    # 以本地目录为 pretrained_model_name_or_path 并显式 subfolder="",
                    # 避免回连 HF,也避免缺省 subfolder 触发 TypeError(H-05)。
                    pipe.load_ip_adapter(
                        str(ip_adapter_weight.parent),
                        subfolder="",
                        weight_name=str(ip_adapter_weight.name),
                        download_kwargs={"local_files_only": True},
                    )
                    pipe.set_ip_adapter_scale(0.9)
                    ip_adapter_loaded = True
                except Exception as exc:
                    # 权重存在但加载失败:标记降级(N4-L6),与「未安装权重」区分。
                    logger.exception("Failed to load IP-Adapter (%s); continuing without it", exc)
                    ip_adapter_reason = (
                        f"IP-Adapter weight present ({ip_adapter_weight.name}) but failed to load: {exc}"
                    )
            elif ip_adapter_reason:
                logger.warning("%s", ip_adapter_reason)
            else:
                logger.warning(
                    "No IP-Adapter weight file found in %s; SDXL runs without image prompt", SDXL_DIR / "ip_adapter"
                )

            # IP-Adapter 激活时不启用 attention slicing:SlicedAttnProcessor 会
            # 覆盖 IPAdapter 处理器,而后者需要 unet 传入的 (text, image) 元组。
            # 无 IP-Adapter 时 slicing 仍是纯内存优化,可安全启用。
            if not ip_adapter_loaded:
                pipe.enable_attention_slicing()

            # 多 controlnet(pose+canny+depth)+IP-Adapter 超出 24GB 显存:
            # offload text_encoder/VAE/IP-Adapter image_encoder 到 CPU,只留
            # unet+controlnet 在 GPU,显著降低峰值显存(贴合改进引入 depth 后)。
            pipe.enable_model_cpu_offload(device=device)
            # VAE tiling/slicing 进一步压低解码峰值显存(多 controlnet 场景);
            # 该管线在部分 diffusers 版本无此方法,缺失时静默跳过。
            for _m in ("enable_vae_slicing", "enable_vae_tiling"):
                if hasattr(pipe, _m):
                    getattr(pipe, _m)()

            _sdxl_state["pipe"] = pipe
            _sdxl_state["ip_adapter_loaded"] = ip_adapter_loaded
            _sdxl_state["ip_adapter_reason"] = ip_adapter_reason
            _sdxl_state["depth_loaded"] = depth_loaded
            _sdxl_state["lora_loaded"] = lora_loaded
            return pipe, ip_adapter_loaded, ip_adapter_reason
        except Exception:
            # 加载失败必须释放已占显存,否则残留半成品会持续占用 VRAM(NEW-H3)。
            logger.exception("Failed to build SDXL pipeline; releasing GPU memory")
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.to("cpu")
                _sdxl_state["pipe"] = None
                _sdxl_state["ip_adapter_loaded"] = False
                _sdxl_state["ip_adapter_reason"] = ""
                _sdxl_state["depth_loaded"] = False
                _sdxl_state["lora_loaded"] = False
            try:
                import gc

                del controlnets
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to release GPU memory after pipeline error")
            raise


def compose_tryon_sdxl(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
    ethnicity: str = "western_europe",
    num_inference_steps: int = 30,
    fit: str = "regular",
    crop_garment: bool = True,
) -> Image.Image:
    import torch
    from PIL import Image as PILImage

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_img = Image.open(base_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")
    if crop_garment:
        garment_img = crop_garment_to_content(garment_img)

    target_w, target_h = 1024, 1024
    human_img = fit_square(base_img, target_w)
    cloth_img = fit_square(garment_img, target_h)

    # 几何对齐:把平铺衣掰到目标人群姿态,作为 IP-Adapter 参考图。
    # VTON_WARP=1(默认)时,生成器看到的是"已穿在身上的衣服",而非平铺贴图,
    # 显著改善衣服在身上的位置/比例/袖长;失败或关闭时回退原平铺图。
    warp_enabled = os.environ.get("VTON_WARP", "1") != "0"
    cloth_ref = cloth_img
    if warp_enabled:
        try:
            cloth_ref = align_garment_to_pose(garment_img, audience, ethnicity, garment_type, target_w)
        except Exception:
            logger.exception("align_garment_to_pose failed; using flat cloth reference")

    prompt = AUDIENCE_PROMPTS.get(audience, AUDIENCE_PROMPTS["women"])
    eth_prompt = ETHNICITY_PROMPTS.get(ethnicity, ETHNICITY_PROMPTS["western_europe"])
    prompt = f"{prompt}, {eth_prompt}"
    garment_suffix = GARMENT_PROMPT_SUFFIX.get(garment_type, "")
    if garment_suffix:
        prompt = f"{prompt}, wearing {garment_suffix}"
    # 穿着贴合:衣服必须"穿"在模特身上——面料垂坠、褶皱随身材(不是平铺贴图)。
    prompt = f"{prompt}, {FIT_PROMPTS.get(fit, FIT_PROMPTS['regular'])}"
    # 摄影质感(防假人感):写实 dslr 质感 + 面料细节。
    prompt = f"{prompt}, {PHOTO_QUALITY}"
    # 额外 prompt 钩子(如"长袖须随手臂到手腕"):仅当设置 VTON_PROMPT_HINT 时追加,默认不变。
    _hint = os.environ.get("VTON_PROMPT_HINT", "").strip()
    if _hint:
        prompt = f"{prompt}, {_hint}"
    if background == "white":
        # 电商主图:背景要求纯白无缝影棚、无地面/阴影(后处理还会强制压白)。
        prompt = f"{prompt}, isolated on a pure white seamless studio backdrop, no floor, no shadow"
    negative_prompt = (
        f"{AUDIENCE_NEGATIVE.get(audience, AUDIENCE_NEGATIVE['women'])}, {FIT_NEGATIVE}, {COMMON_NEGATIVE}"
    )

    pipe, ip_adapter_loaded, ip_adapter_reason = _get_sdxl_pipe(device, dtype)

    _ipa = os.environ.get("VTON_IPA")
    if ip_adapter_loaded and _ipa:
        pipe.set_ip_adapter_scale(float(_ipa))
        logger.info("IP-Adapter scale overridden -> %s (env VTON_IPA)", float(_ipa))

    if _sdxl_state.get("lora_loaded"):
        _apply_lora_scale(pipe)

    control_pose = _get_openpose_control(human_img)

    # 贴合改进:depth 存在时用 pose+depth(深度图约束衣服包裹体型、产生立体
    # 褶皱);depth 不可用(depth_loaded=False)时退回 pose+canny。两者都保证
    # images 数量与已加载的 controlnet 数量一致。
    control_images = [control_pose]
    control_scales = [0.8]
    if _sdxl_state.get("depth_loaded"):
        try:
            control_images.append(_get_depth_control(human_img))
            control_scales.append(0.95)
        except Exception:
            logger.exception("Depth control failed; continuing without it")
            # 保底:中性灰占位图,保证与 controlnet 数量一致,避免 check_inputs 报错。
            control_images.append(PILImage.new("RGB", (target_w, target_h), (128, 128, 128)))
            control_scales.append(0.0)
    else:
        control_images.append(_get_canny_control(cloth_ref))
        # 按品类自动调节 Canny 强度:实心面料(upper/dress/onesie)用 0.55 锁图案;
        # 薄纱/网眼类用 0.4 防鬼影。VTON_CANNY 显式覆盖优先级最高。
        _canny_defaults = {"upper": 0.55, "dress": 0.55, "onesie": 0.55, "lower": 0.45}
        _canny_default = _canny_defaults.get(garment_type, 0.55)
        control_scales.append(float(os.environ.get("VTON_CANNY", str(_canny_default))))

    import os as _os

    generator = torch.Generator(device=device).manual_seed(int(_os.environ.get("VTON_SEED", "42")))

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": control_images,
        "controlnet_conditioning_scale": control_scales,
        "width": target_w,
        "height": target_h,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(os.environ.get("VTON_GUIDANCE", "7.5")),
        "generator": generator,
        "output_type": "pil",
    }

    if ip_adapter_loaded:
        pipe_kwargs["ip_adapter_image"] = cloth_ref

    # 采样(含 denoise)在 autocast 下跑;VAE 解码移出 autocast 并以 fp32 执行,
    # 否则 SDXL fp16 VAE 解码溢出产生 NaN/黑图。
    with _GPU_LOCK, torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        latents = pipe(**{**pipe_kwargs, "output_type": "latent"}).images
    with torch.no_grad():
        decoded = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    result = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    # 服装区遮罩:细节回贴(衣服纹理/颜色)与身份锁(人脸/背景)共用同一张
    # clothing_mask, 只算一次(含 rembg u2net 推理,是重操作)。
    # mask 255=衣服区(reapply 在此贴回原衣纹理颜色),0=人脸/手/背景
    # (blend_tryon 在此贴回原图像素保身份)。两者互补:reapply 管衣服保真,
    # 身份锁管人脸不走形。任一开关开启才算 mask,都关时跳过省一次推理。
    need_mask = os.environ.get("VTON_DETAIL", "1") != "0" or os.environ.get("VTON_LOCK_ID", "1") != "0"
    garment_extent = None
    if need_mask:
        try:
            garment_extent = detect_garment_extent(garment_img)
            logger.info("garment extent: %s", garment_extent)
        except Exception:  # noqa: BLE001
            logger.debug("detect_garment_extent failed; using default mask")
    shared_mask = (
        clothing_mask(human_img, garment_type=garment_type, garment_extent=garment_extent) if need_mask else None
    )

    # 商品细节回贴:AI 出骨架光影,代码保原衣纹理颜色。
    # SDXL 生成的衣服颜色/花纹常与原平铺图有偏差(串色/丢花纹);把原衣的
    # 高频纹理(柔光贴回)+ 颜色分布(Reinhard LAB 迁移)注入生成图衣服区。
    # VTON_DETAIL=1(默认)开启;关闭时跳过。
    detail_applied = False
    if os.environ.get("VTON_DETAIL", "1") != "0" and shared_mask is not None:
        try:
            result = reapply_garment_detail(result, cloth_ref, shared_mask)
            detail_applied = True
            logger.info("reapply_garment_detail: texture+color re-injected on garment region")
        except Exception:
            logger.exception("reapply_garment_detail failed; using raw SDXL output")

    # 身份锁:mask 外(人脸/手/背景)贴回原图 human_img,防止 SDXL 人脸走形。
    # mask 内保留 reapply 修正后的衣服(不被覆盖)。VTON_LOCK_ID=1(默认)开启。
    lock_identity = False
    if os.environ.get("VTON_LOCK_ID", "1") != "0" and shared_mask is not None:
        result = blend_tryon(human_img, result, shared_mask)
        lock_identity = True

    if background == "transparent":
        result = _remove_background(result)
    elif background == "white":
        # 电商主图硬性要求:背景必须纯白 RGB(255,255,255)。扩散模型生成的
        # 背景常偏暖米色/灰,不能直接采用——把边缘连通的近白背景压成纯白,
        # 只改背景区域,模特/衣服原样保留(边缘连通检查避免误伤身上的白色衣物)。
        if result.mode == "RGBA":
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[-1])
            result = bg
        result = force_white_background(result)

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), PILImage.Resampling.LANCZOS)

    # IP-Adapter 权重存在但加载失败时标记降级(N4-L6);「未安装」不视为降级。
    result.info["engine_degraded"] = bool(ip_adapter_reason)
    result.info["degraded_reason"] = ip_adapter_reason
    result.info["lock_identity"] = lock_identity
    result.info["detail_applied"] = detail_applied
    # 出图后清空 CUDA 缓存:VAE 解码的 fp32 中间张量会留下大块显存碎片,
    # 不清会在下一任务/下一解码时放大峰值,触发 cgroup OOM。
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return result
