from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
SDXL_DIR = _REPO_ROOT / "models" / "sdxl"
_IDM_DIR = _REPO_ROOT / "IDM-VTON"


def sdxl_available() -> tuple[bool, str]:
    required = ["sdxl_base", "controlnet", "ip_adapter"]
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


def compose_tryon_sdxl(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
) -> Image.Image:
    import torch

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
    human_img = base_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    cloth_img = garment_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    prompt_map = {
        "upper": "a fashion model wearing this shirt, full body front view",
        "dress": "a fashion model wearing this dress, full body front view",
        "lower": "a fashion model wearing these pants, full body front view",
        "onesie": "a baby wearing this onesie, full body front view",
    }
    prompt = prompt_map.get(garment_type, "a fashion model wearing this garment")
    negative_prompt = "worst quality, low quality, monochrome, bad anatomy, extra limbs, missing limbs"

    scheduler = DDPMScheduler.from_pretrained(str(SDXL_DIR / "sdxl_base"), subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(
        str(SDXL_DIR / "controlnet"), torch_dtype=dtype
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(SDXL_DIR / "sdxl_base"),
        controlnet=controlnet,
        scheduler=scheduler,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to(device)
    pipe.enable_attention_slicing()

    try:
        from IPAdapter import IPAdapter
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(SDXL_DIR / "ip_adapter" / "image_encoder"), torch_dtype=dtype
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            str(SDXL_DIR / "ip_adapter" / "image_encoder")
        )
        ip_adapter = IPAdapter(
            pipe,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            adapter_model_state_dict=torch.load(
                SDXL_DIR / "ip_adapter" / "ip-adapter-plus_sdxl_vit-h.bin"
            ),
        )
        ip_adapter.set_weight(0.8)
    except Exception:
        pass

    control_image = human_img.copy()

    generator = torch.Generator(device=device).manual_seed(42)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=human_img,
                control_image=control_image,
                width=target_w,
                height=target_h,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                output_type="pil",
            )

    result = output.images[0]

    if background == "white":
        bg = Image.new("RGB", result.size, (255, 255, 255))
        bg.paste(result, mask=result.split()[-1] if result.mode == "RGBA" else None)
        result = bg

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), Image.Resampling.LANCZOS)

    del pipe
    del controlnet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result
