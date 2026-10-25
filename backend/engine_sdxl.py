from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
SDXL_DIR = _REPO_ROOT / "models" / "sdxl"
_IDM_DIR = _REPO_ROOT / "IDM-VTON"

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

GARMENT_PROMPT_SUFFIX = {
    "upper": "shirt, t-shirt, blouse",
    "dress": "dress, gown, skirt",
    "lower": "pants, trousers, shorts, jeans",
    "onesie": "onesie, romper, baby body",
}


def sdxl_available() -> tuple[bool, str]:
    required = ["sdxl_base", "controlnet"]
    missing = [d for d in required if not (SDXL_DIR / d).exists()]
    if missing:
        return False, f"缺少权重: {', '.join(missing)}"
    try:
        import torch
    except ImportError:
        return False, "未安装 torch"
    if not torch.cuda.is_available():
        return False, "无可用 GPU"
    return True, "ok"


def _get_openpose_control(image: Image.Image) -> Image.Image:
    from controlnet_aux import OpenPoseDetector
    detector = OpenPoseDetector.from_pretrained("lllyasviel/ControlNet-v1-1")
    control = detector(image)
    return control


def _get_canny_control(image: Image.Image) -> Image.Image:
    from controlnet_aux import CannyDetector
    detector = CannyDetector.from_pretrained("lllyasviel/ControlNet-v1-1")
    control = detector(image, resolution=1024)
    return control


def _remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        return remove(image)
    except ImportError:
        return image


def compose_tryon_sdxl(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
) -> Image.Image:
    import torch
    from PIL import Image as PILImage

    try:
        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            DDPMScheduler,
            ControlNetModel,
        )
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    except ImportError as e:
        raise RuntimeError(f"缺少依赖: {e}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_img = Image.open(base_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")

    target_w, target_h = 1024, 1024
    human_img = base_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
    cloth_img = garment_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)

    prompt = AUDIENCE_PROMPTS.get(audience, AUDIENCE_PROMPTS["women"])
    garment_suffix = GARMENT_PROMPT_SUFFIX.get(garment_type, "")
    if garment_suffix:
        prompt = f"{prompt}, wearing {garment_suffix}"
    negative_prompt = AUDIENCE_NEGATIVE.get(audience, AUDIENCE_NEGATIVE["women"])

    scheduler = DDPMScheduler.from_pretrained(str(SDXL_DIR / "sdxl_base"), subfolder="scheduler")

    controlnet_pose = ControlNetModel.from_pretrained(
        str(SDXL_DIR / "controlnet"), torch_dtype=dtype
    )
    controlnet_canny = ControlNetModel.from_pretrained(
        str(SDXL_DIR / "controlnet"), torch_dtype=dtype
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(SDXL_DIR / "sdxl_base"),
        controlnet=[controlnet_pose, controlnet_canny],
        scheduler=scheduler,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to(device)
    pipe.enable_attention_slicing()

    ip_adapter_path = str(SDXL_DIR / "ip_adapter" / "ip-adapter-plus_sdxl_vit-h.bin")
    if (SDXL_DIR / "ip_adapter" / "ip-adapter-plus_sdxl_vit-h.bin").exists():
        try:
            pipe.load_image_embedding(ip_adapter_path)
        except Exception:
            pass

    control_pose = _get_openpose_control(human_img)
    control_canny = _get_canny_control(cloth_img)

    generator = torch.Generator(device=device).manual_seed(42)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "control_image": [control_pose, control_canny],
        "controlnet_conditioning_scale": [0.8, 0.6],
        "width": target_w,
        "height": target_h,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "generator": generator,
        "output_type": "pil",
    }

    if (SDXL_DIR / "ip_adapter" / "ip-adapter-plus_sdxl_vit-h.bin").exists():
        pipe_kwargs["image"] = cloth_img

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            output = pipe(**pipe_kwargs)

    result = output.images[0]

    if background == "transparent":
        result = _remove_background(result)
    elif background == "white":
        if result.mode == "RGBA":
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[-1])
            result = bg
        else:
            result = PILImage.new("RGB", result.size, (255, 255, 255))
            result.paste(result.convert("RGB"))

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), PILImage.Resampling.LANCZOS)

    del pipe
    del controlnet_pose
    del controlnet_canny
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result
