"""ModelScope 创空间版换衣引擎(与 backend/engine_sdxl.py 同源,去除训练 LoRA)。

权重目录由环境变量 MODELS_DIR 指定(默认 /mnt/workspace/models/sdxl),
权重须在启动时通过 dl_weights.py 预下载到该目录(创建空间环境无外网依赖)。
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from pathlib import Path

from PIL import Image

logger = logging.getLogger("ms_space.engine_sdxl")

_GPU_LOCK = threading.Lock()
_PIPE_LOCK = threading.Lock()
_sdxl_state: dict = {"pipe": None, "ip_adapter_loaded": False, "ip_adapter_reason": "", "depth_loaded": False}

SDXL_DIR = Path(os.environ.get("MODELS_DIR", "/mnt/workspace/models/sdxl"))

# 质量调优参数(环境变量可覆盖,便于创空间配置与离线实验):
#   MS_CLOTH_CANNY=1   加衣服边缘(第3 controlnet)约束衣服结构,避免严重变形;
#   MS_IP_MODEL_REF=1  将模特图纳入 IP-Adapter 参考,保留模特身份(脸/肤色/体型);
#   MS_IP_SCALE / MS_CFG / MS_STEPS 控制 IP-Adapter 强度、引导系数与步数。
#   默认关:实测(魔塔 xGPU)仅衣服参考 + pose/depth 双 controlnet 已自然贴合,
#   模特参考与衣服 canny 反而引入僵硬/混淆,故默认 0。
CLOTH_CANNY = os.environ.get("MS_CLOTH_CANNY", "0") == "1"
IP_MODEL_REF = os.environ.get("MS_IP_MODEL_REF", "0") == "1"
IP_SCALE = float(os.environ.get("MS_IP_SCALE", "0.8"))
CFG = float(os.environ.get("MS_CFG", "7.5"))
STEPS = int(os.environ.get("MS_STEPS", "32"))

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
    "baby": "worst quality, low quality, adult body, distorted proportions, big head, extra limbs, missing limbs",
}

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

# 换衣贴合度:强调合身、布料褶皱与立体垂坠,避免"平面贴纸"感。
FIT_PROMPT = (
    "the garment fits the model's body perfectly, snug and form-fitting, "
    "realistic 3D fabric folds and creases, cloth draping naturally over the "
    "body contour with realistic volume and soft shadows, tailored fit"
)


def sdxl_available() -> tuple[bool, str]:
    required = [
        ("sdxl_base", "model_index.json"),
        ("controlnet_openpose", "config.json"),
        ("controlnet_depth", "config.json"),
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


def ip_adapter_available() -> bool:
    weight_ok = any(
        (SDXL_DIR / "ip_adapter" / name).is_file()
        for name in ("ip-adapter_sdxl_vit-h.safetensors", "ip-adapter_sdxl_vit-h.bin")
    )
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
        with _PIPE_LOCK:
            if _canny_detector is None:
                _canny_detector = CannyDetector()
    return _canny_detector(image, low_threshold=100, high_threshold=200, detect_resolution=1024, image_resolution=1024)


def _get_depth_control(image: Image.Image) -> Image.Image:
    """人体深度图:换衣贴合体型、产生布料褶皱与立体感的关键条件。"""
    global _depth_detector
    from controlnet_aux import MidasDetector

    if _depth_detector is None:
        with _PIPE_LOCK:
            if _depth_detector is None:
                annotator_dir = SDXL_DIR / "annotators"
                if annotator_dir.exists():
                    _depth_detector = MidasDetector.from_pretrained(str(annotator_dir))
                else:
                    _depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return _depth_detector(image)


def _remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove

        return remove(image)
    except ImportError:
        return image


def _get_sdxl_pipe(device, dtype):
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

        openpose_ckpt = SDXL_DIR / "controlnet_openpose"
        openpose_single = openpose_ckpt / "OpenPoseXL2.safetensors"
        if openpose_single.exists():
            controlnet_pose = ControlNetModel.from_single_file(str(openpose_single), torch_dtype=dtype)
        else:
            controlnet_pose = ControlNetModel.from_pretrained(
                str(openpose_ckpt), torch_dtype=dtype, use_safetensors=True, local_files_only=True
            )

        def _load_canny():
            return ControlNetModel.from_pretrained(
                str(SDXL_DIR / "controlnet_canny"),
                torch_dtype=dtype,
                variant="fp16",
                use_safetensors=True,
                local_files_only=True,
            )

        # 默认 pose+depth(2 个 controlnet)。3 个 controlnet 常驻会超出 24GB 显存,
        # 故不默认全加载;depth 缺失或失败时退回 pose+canny。
        depth_dir = SDXL_DIR / "controlnet_depth"
        depth_loaded = False
        controlnets = [controlnet_pose]
        if (depth_dir / "config.json").exists():
            try:
                controlnets.append(
                    ControlNetModel.from_pretrained(
                        str(depth_dir),
                        torch_dtype=dtype,
                        variant="fp16",
                        use_safetensors=True,
                        local_files_only=True,
                    )
                )
                depth_loaded = True
            except Exception:
                logger.exception("Failed to load depth ControlNet; falling back to canny")
                controlnets.append(_load_canny())
        else:
            controlnets.append(_load_canny())
        # 衣服边缘结构(第3 controlnet):约束衣服形状,避免衣服严重变形。
        if CLOTH_CANNY:
            controlnets.append(_load_canny())

        pipe = None
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                str(SDXL_DIR / "sdxl_base"),
                controlnet=controlnets,
                scheduler=scheduler,
                torch_dtype=dtype,
                variant="fp16",
                use_safetensors=True,
                local_files_only=True,
            )
            # SDXL base VAE 需以 fp32 解码,否则 fp16 下解码溢出产生 NaN/黑图。
            pipe.vae = pipe.vae.to(torch.float32)

            # 兼容 .safetensors / .bin 两种常见权重文件。
            ip_adapter_candidates = [
                SDXL_DIR / "ip_adapter" / "ip-adapter_sdxl_vit-h.safetensors",
                SDXL_DIR / "ip_adapter" / "ip-adapter_sdxl_vit-h.bin",
            ]
            ip_adapter_bin = next((p for p in ip_adapter_candidates if p.exists()), None)
            ip_adapter_loaded = False
            ip_adapter_reason = ""
            if ip_adapter_bin is not None:
                try:
                    pipe.load_ip_adapter(
                        str(ip_adapter_bin.parent),
                        subfolder="",
                        weight_name=str(ip_adapter_bin.name),
                        download_kwargs={"local_files_only": True},
                    )
                    pipe.set_ip_adapter_scale(IP_SCALE)
                    ip_adapter_loaded = True
                except Exception as exc:
                    logger.exception("Failed to load IP-Adapter (%s); continuing without it", exc)
                    ip_adapter_reason = f"IP-Adapter weight present ({ip_adapter_bin.name}) but failed to load: {exc}"
            else:
                logger.warning(
                    "No IP-Adapter weight file found in %s; SDXL runs without image prompt", SDXL_DIR / "ip_adapter"
                )

            # IP-Adapter 激活时不启用 attention slicing:SlicedAttnProcessor 会
            # 覆盖 IPAdapter 处理器,而后者需要 unet 传入的 (text, image) 元组。
            if not ip_adapter_loaded:
                pipe.enable_attention_slicing()

            # 多 controlnet+IP-Adapter 超出 24GB 显存:offload 到 CPU 降峰值显存。
            pipe.enable_model_cpu_offload(device=device)
            for _m in ("enable_vae_slicing", "enable_vae_tiling"):
                if hasattr(pipe, _m):
                    getattr(pipe, _m)()

            _sdxl_state["pipe"] = pipe
            _sdxl_state["ip_adapter_loaded"] = ip_adapter_loaded
            _sdxl_state["ip_adapter_reason"] = ip_adapter_reason
            _sdxl_state["depth_loaded"] = depth_loaded
            return pipe, ip_adapter_loaded, ip_adapter_reason
        except Exception:
            logger.exception("Failed to build SDXL pipeline; releasing GPU memory")
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.to("cpu")
                _sdxl_state["pipe"] = None
                _sdxl_state["ip_adapter_loaded"] = False
                _sdxl_state["ip_adapter_reason"] = ""
                _sdxl_state["depth_loaded"] = False
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
    long_side: int = 1024,
    background: str = "white",
    ethnicity: str = "western_europe",
) -> Image.Image:
    import torch
    from PIL import Image as PILImage

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_img = Image.open(base_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")

    target_w, target_h = 1024, 1024

    def _fit_square(im, side, pad=(255, 255, 255)):
        if os.environ.get("VTON_AR", "1") == "0":
            return im.resize((side, side), PILImage.Resampling.LANCZOS)
        w, h = im.size
        scale = side / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        canvas = PILImage.new("RGB", (side, side), pad)
        canvas.paste(im.resize((nw, nh), PILImage.Resampling.LANCZOS), ((side - nw) // 2, (side - nh) // 2))
        return canvas

    human_img = _fit_square(base_img, target_w)
    cloth_img = _fit_square(garment_img, target_h)

    prompt = AUDIENCE_PROMPTS.get(audience, AUDIENCE_PROMPTS["women"])
    eth_prompt = ETHNICITY_PROMPTS.get(ethnicity, ETHNICITY_PROMPTS["western_europe"])
    prompt = f"{prompt}, {eth_prompt}"
    garment_suffix = GARMENT_PROMPT_SUFFIX.get(garment_type, "")
    if garment_suffix:
        prompt = f"{prompt}, wearing {garment_suffix}"
    prompt = f"{prompt}, {FIT_PROMPT}"
    negative_prompt = AUDIENCE_NEGATIVE.get(audience, AUDIENCE_NEGATIVE["women"])

    pipe, ip_adapter_loaded, ip_adapter_reason = _get_sdxl_pipe(device, dtype)

    control_pose = _get_openpose_control(human_img)

    control_images = [control_pose]
    control_scales = [0.8]
    if _sdxl_state.get("depth_loaded"):
        try:
            control_images.append(_get_depth_control(human_img))
            control_scales.append(0.95)
        except Exception:
            logger.exception("Depth control failed; continuing without it")
            control_images.append(PILImage.new("RGB", (target_w, target_h), (128, 128, 128)))
            control_scales.append(0.0)
    else:
        control_images.append(_get_canny_control(cloth_img))
        control_scales.append(0.55)
    # 衣服边缘结构:与第3 controlnet 对应,约束衣服形状避免严重变形。
    if CLOTH_CANNY:
        control_images.append(_get_canny_control(cloth_img))
        control_scales.append(0.55)

    generator = torch.Generator(device=device).manual_seed(42)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": control_images,
        "controlnet_conditioning_scale": control_scales,
        "width": target_w,
        "height": target_h,
        "num_inference_steps": STEPS,
        "guidance_scale": CFG,
        "generator": generator,
        "output_type": "pil",
    }

    if ip_adapter_loaded:
        # 保留模特身份(可选):把原模特图一并作为 IP-Adapter 参考(脸/肤色/发型/体型)。
        # 实测多图 list 在 diffusers 0.42 的 h94 非 plus 路径可运行(魔塔验证),故用 list;
        # 默认关,仅衣服参考。
        pipe_kwargs["ip_adapter_image"] = [human_img, cloth_img] if IP_MODEL_REF else cloth_img

    with _GPU_LOCK, torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        latents = pipe(**{**pipe_kwargs, "output_type": "latent"}).images
    with torch.no_grad():
        decoded = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    result = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    if background == "transparent":
        result = _remove_background(result)
    elif background == "white":
        if result.mode == "RGBA":
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[-1])
            result = bg
        else:
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result.convert("RGB"), (0, 0))
            result = bg

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), PILImage.Resampling.LANCZOS)

    result.info["engine_degraded"] = bool(ip_adapter_reason)
    result.info["degraded_reason"] = ip_adapter_reason
    return result
