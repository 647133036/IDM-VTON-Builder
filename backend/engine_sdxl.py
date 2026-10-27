from __future__ import annotations

import logging
import threading
from pathlib import Path

from PIL import Image

logger = logging.getLogger("atelier.engine_sdxl")

_GPU_LOCK = threading.Lock()
_sdxl_state: dict = {"pipe": None, "ip_adapter_loaded": False}

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


def sdxl_available() -> tuple[bool, str]:
    required = ["sdxl_base", "controlnet_openpose", "controlnet_canny"]
    missing = [d for d in required if not (SDXL_DIR / d).exists()]
    if missing:
        return False, f"Missing weights: {', '.join(missing)}"
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "No GPU available"
    return True, "ok"


_pose_detector = None
_canny_detector = None


def _get_openpose_control(image: Image.Image) -> Image.Image:
    global _pose_detector
    from controlnet_aux import OpenposeDetector

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
        _canny_detector = CannyDetector()
    return _canny_detector(image, low_threshold=100, high_threshold=200, detect_resolution=1024, image_resolution=1024)


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


def _get_sdxl_pipe(device, dtype):
    if _sdxl_state["pipe"] is not None:
        return _sdxl_state["pipe"], _sdxl_state["ip_adapter_loaded"]

    from diffusers import (
        ControlNetModel,
        DDPMScheduler,
        StableDiffusionXLControlNetPipeline,
    )

    scheduler = DDPMScheduler.from_pretrained(str(SDXL_DIR / "sdxl_base"), subfolder="scheduler")

    controlnet_pose = ControlNetModel.from_pretrained(str(SDXL_DIR / "controlnet_openpose"), torch_dtype=dtype)
    controlnet_canny = ControlNetModel.from_pretrained(str(SDXL_DIR / "controlnet_canny"), torch_dtype=dtype)

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

    lora_dir = SDXL_DIR / "lora"
    try:
        _load_trained_lora(pipe, lora_dir)
    except Exception:
        logger.exception("Failed to load trained LoRA; continuing without it")

    ip_adapter_bin = SDXL_DIR / "ip_adapter" / "ip-adapter-plus_sdxl_vit-h.bin"
    ip_adapter_loaded = False
    if ip_adapter_bin.exists():
        try:
            # 直接以本地目录作为 pretrained_model_name_or_path,避免回连 HF(H-05)。
            pipe.load_ip_adapter(
                str(ip_adapter_bin.parent),
                weight_name=str(ip_adapter_bin.name),
                download_kwargs={"local_files_only": True},
            )
            pipe.set_ip_adapter_scale(0.8)
            ip_adapter_loaded = True
        except Exception:
            logger.exception("Failed to load IP-Adapter; continuing without it")

    _sdxl_state["pipe"] = pipe
    _sdxl_state["ip_adapter_loaded"] = ip_adapter_loaded
    return pipe, ip_adapter_loaded


def compose_tryon_sdxl(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
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
    human_img = base_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
    cloth_img = garment_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)

    prompt = AUDIENCE_PROMPTS.get(audience, AUDIENCE_PROMPTS["women"])
    eth_prompt = ETHNICITY_PROMPTS.get(ethnicity, ETHNICITY_PROMPTS["western_europe"])
    prompt = f"{prompt}, {eth_prompt}"
    garment_suffix = GARMENT_PROMPT_SUFFIX.get(garment_type, "")
    if garment_suffix:
        prompt = f"{prompt}, wearing {garment_suffix}"
    negative_prompt = AUDIENCE_NEGATIVE.get(audience, AUDIENCE_NEGATIVE["women"])

    pipe, ip_adapter_loaded = _get_sdxl_pipe(device, dtype)

    control_pose = _get_openpose_control(human_img)
    control_canny = _get_canny_control(cloth_img)

    generator = torch.Generator(device=device).manual_seed(42)

    pipe_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": [control_pose, control_canny],
        "controlnet_conditioning_scale": [0.8, 0.6],
        "width": target_w,
        "height": target_h,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "generator": generator,
        "output_type": "pil",
    }

    if ip_adapter_loaded:
        pipe_kwargs["ip_adapter_image"] = cloth_img

    with _GPU_LOCK, torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
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
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result.convert("RGB"), (0, 0))
            result = bg

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), PILImage.Resampling.LANCZOS)

    return result
