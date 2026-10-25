from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

TORSOS = {
    "women": dict(x=0.21, y=0.178, w=0.58, h=0.39, shoulder=0.96, hem=0.62, neck=0.22, sleeve=0.22),
    "men": dict(x=0.18, y=0.155, w=0.64, h=0.42, shoulder=0.98, hem=0.64, neck=0.20, sleeve=0.20),
    "girls": dict(x=0.24, y=0.21, w=0.52, h=0.36, shoulder=0.92, hem=0.70, neck=0.24, sleeve=0.16),
    "boys": dict(x=0.22, y=0.20, w=0.56, h=0.38, shoulder=0.94, hem=0.72, neck=0.22, sleeve=0.16),
    "baby": dict(x=0.24, y=0.24, w=0.52, h=0.46, shoulder=0.90, hem=0.78, neck=0.28, sleeve=0.14),
}

GARMENT_BOX = {
    "upper": dict(y=0.00, h=1.00),
    "dress": dict(y=0.00, h=1.55),
    "lower": dict(y=0.92, h=0.95),
    "onesie": dict(y=0.00, h=1.35),
}

GARMENT_PROMPTS = {
    "upper": "a model wearing this shirt",
    "dress": "a model wearing this dress",
    "lower": "a model wearing these pants",
    "onesie": "a model wearing this onesie",
}

WEIGHTS_DIR = Path(os.getenv("IDM_WEIGHTS_DIR", str(Path(__file__).resolve().parent.parent / "IDM-VTON" / "ckpt")))
_REPO_ROOT = Path(__file__).resolve().parent.parent
_IDM_DIR = _REPO_ROOT / "IDM-VTON"
_LICENSE_MODE = os.getenv("LICENSE_MODE", "").strip().lower()


def _license_allows_idm() -> tuple[bool, str]:
    if _LICENSE_MODE == "commercial":
        return False, "LICENSE_MODE=commercial 禁用 IDM-VTON（CC BY-NC-SA 禁止商用）"
    if _LICENSE_MODE == "research":
        return True, ""
    if _LICENSE_MODE == "":
        return False, "未设置 LICENSE_MODE 环境变量，IDM-VTON 默认禁用"
    return False, f"未知 LICENSE_MODE={_LICENSE_MODE}，仅支持 research 或 commercial"


def _idm_vton_available() -> tuple[bool, str]:
    allowed, reason = _license_allows_idm()
    if not allowed:
        return False, reason

    ckpt = _IDM_DIR / "ckpt"
    required = ["unet", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
    missing = [d for d in required if not (ckpt / d).exists()]
    if missing:
        return False, f"缺少权重: {', '.join(missing)}"
    if not (ckpt / "unet" / "diffusion_pytorch_model.safetensors").exists() and not (ckpt / "unet" / "diffusion_pytorch_model.bin").exists():
        return False, "unet 权重文件不存在"
    try:
        import torch
    except ImportError:
        return False, "未安装 torch"
    if not torch.cuda.is_available():
        return False, "无可用 GPU"
    return True, "ok"


def detect_garment_crop(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    h, w, _ = arr.shape
    corners = np.concatenate(
        [
            arr[:8, :8].reshape(-1, 3),
            arr[:8, -8:].reshape(-1, 3),
            arr[-8:, :8].reshape(-1, 3),
            arr[-8:, -8:].reshape(-1, 3),
            arr[:8, w // 2 - 4 : w // 2 + 4].reshape(-1, 3),
        ]
    )
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > 22
    ys, xs = np.where(mask)
    if len(xs) < 80:
        return rgb
    pad = int(min(w, h) * 0.03)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, w)
    y1 = min(int(ys.max()) + pad, h)
    return rgb.crop((x0, y0, x1, y1))


def garment_alpha(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    bg = np.median(np.concatenate([arr[0, :].reshape(-1, 3), arr[:, 0].reshape(-1, 3)]), axis=0)
    dist = np.linalg.norm(arr - bg, axis=2)
    alpha = np.clip((dist - 10) / 32.0, 0, 1)
    a = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
    a = a.filter(ImageFilter.GaussianBlur(1.2))
    a = ImageOps.autocontrast(a)
    return a.point(lambda p: 0 if p < 24 else p)


def shirt_mask(width: int, height: int, box: dict) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    sw = box["shoulder"] * width
    hw = box["hem"] * width
    nw = box["neck"] * width
    sleeve = box["sleeve"] * height
    cx = width / 2
    neck_y = height * 0.04
    shoulder_y = height * 0.10
    pit_y = shoulder_y + sleeve
    hem_y = height * 0.97
    pts = [
        (cx - nw / 2, neck_y),
        (cx + nw / 2, neck_y),
        (cx + sw / 2, shoulder_y),
        (cx + sw / 2 + width * 0.02, pit_y),
        (cx + hw / 2, hem_y),
        (cx - hw / 2, hem_y),
        (cx - sw / 2 - width * 0.02, pit_y),
        (cx - sw / 2, shoulder_y),
    ]
    draw.polygon(pts, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(2, width // 80)))


def fit_garment(garment: Image.Image, target: tuple[int, int]) -> Image.Image:
    tw, th = target
    gw, gh = garment.size
    scale = max(tw / gw, th / gh)
    nw, nh = max(1, int(gw * scale)), max(1, int(gh * scale))
    resized = garment.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = max(0, int((nh - th) * 0.12))
    return resized.crop((left, top, left + tw, top + th))


def match_color(garment: Image.Image) -> Image.Image:
    g = ImageEnhance.Color(garment).enhance(1.06)
    g = ImageEnhance.Contrast(g).enhance(1.05)
    g = ImageEnhance.Sharpness(g).enhance(1.12)
    return g


def _composite_engine(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int,
    background: str,
) -> Image.Image:
    base = Image.open(base_path).convert("RGBA")
    w, h = base.size
    garment = detect_garment_crop(Image.open(garment_path))
    garment = match_color(garment)
    alpha = garment_alpha(garment)
    garment = garment.convert("RGBA")
    garment.putalpha(alpha)

    box = dict(TORSOS.get(audience, TORSOS["women"]))
    adj = GARMENT_BOX.get(garment_type, GARMENT_BOX["upper"])
    box["y"] = box["y"] + adj["y"] * box["h"]
    box["h"] = box["h"] * adj["h"]

    tx = int(box["x"] * w)
    ty = int(box["y"] * h)
    tw = int(box["w"] * w)
    th = int(box["h"] * h)
    tw = max(tw, 32)
    th = max(th, 32)

    fitted = fit_garment(garment, (tw, th))
    body = shirt_mask(tw, th, box)
    g_a = fitted.split()[-1]
    merged_a = ImageEnhance.Brightness(g_a).enhance(1.0)
    merged_a = Image.fromarray(
        (np.asarray(merged_a).astype(np.float32) * (np.asarray(body).astype(np.float32) / 255.0)).astype(np.uint8),
        mode="L",
    )
    fitted.putalpha(merged_a)

    shadow = merged_a.filter(ImageFilter.GaussianBlur(8))
    shadow_layer = Image.new("RGBA", base.size, (25, 20, 16, 0))
    shadow_img = Image.new("L", base.size, 0)
    shadow_img.paste(ImageEnhance.Brightness(shadow).enhance(0.35), (tx + 2, ty + 6))
    shadow_layer.putalpha(shadow_img)

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(fitted, (tx, ty), fitted)

    canvas = Image.new("RGBA", base.size, (255, 255, 255, 255) if background == "white" else (0, 0, 0, 0))
    canvas.alpha_composite(base)
    canvas.alpha_composite(shadow_layer)
    canvas.alpha_composite(layer)

    if background == "white":
        out = Image.new("RGB", canvas.size, (255, 255, 255))
        out.paste(canvas, mask=canvas.split()[-1])
        result = out
    else:
        result = canvas

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), Image.Resampling.LANCZOS)
    return result


_engine_state: dict = {"pipe": None, "available": None, "reason": ""}


def _load_idm_vton_pipeline():
    if _engine_state["available"] is False:
        return None
    if _engine_state["pipe"] is not None:
        return _engine_state["pipe"]

    avail, reason = _idm_vton_available()
    if not avail:
        _engine_state["available"] = False
        _engine_state["reason"] = reason
        return None

    _engine_state["available"] = True
    _engine_state["reason"] = ""

    sys_path = str(_IDM_DIR)
    if sys_path not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = sys_path + os.pathsep + os.environ.get("PYTHONPATH", "")

    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_tryon import UNet2DConditionModel
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

    ckpt_path = str(_IDM_DIR / "ckpt")
    device = torch.device("cuda:0")

    unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet", torch_dtype=torch.float16)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(ckpt_path, subfolder="unet_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae", torch_dtype=torch.float16)
    scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")

    text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(ckpt_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
    tokenizer_2 = AutoTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer_2")

    pipe = TryonPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        unet_encoder=unet_encoder,
        scheduler=scheduler,
    )
    pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    _engine_state["pipe"] = pipe
    return pipe


def _idm_vton_engine(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int,
    background: str,
) -> Image.Image:
    import torch
    from PIL import Image as PILImage

    pipe = _load_idm_vton_pipeline()
    if pipe is None:
        return _composite_engine(base_path, garment_path, audience, garment_type, long_side, background)

    device = pipe.device if hasattr(pipe, "device") else torch.device("cuda:0")

    base_img = PILImage.open(base_path).convert("RGB")
    garment_img = PILImage.open(garment_path).convert("RGB")

    target_w, target_h = 768, 1024
    human_img = base_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
    cloth_img = garment_img.resize((target_w, target_h), PILImage.Resampling.LANCZOS)

    prompt = GARMENT_PROMPTS.get(garment_type, "a model wearing this garment")
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    mask_img = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_img)
    box = dict(TORSOS.get(audience, TORSOS["women"]))
    mx = int(box["x"] * target_w)
    my = int(box["y"] * target_h)
    mw = int(box["w"] * target_w)
    mh = int(box["h"] * target_h)
    mask_draw.rectangle([mx, my, mx + mw, my + mh], fill=(255, 255, 255))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(16))

    generator = torch.Generator(device=device).manual_seed(42)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda"):
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=human_img,
                mask_image=mask_img,
                cloth=cloth_img,
                width=target_w,
                height=target_h,
                num_inference_steps=30,
                guidance_scale=2.0,
                generator=generator,
                output_type="pil",
            )

    result = output.images[0] if hasattr(output, "images") else output

    if background == "white":
        bg = PILImage.new("RGB", result.size, (255, 255, 255))
        bg.paste(result, mask=result.split()[-1] if result.mode == "RGBA" else None)
        result = bg

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), PILImage.Resampling.LANCZOS)
    return result


def compose_tryon(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
) -> Image.Image:
    avail, reason = _idm_vton_available()
    if avail:
        try:
            return _idm_vton_engine(base_path, garment_path, audience, garment_type, long_side, background)
        except Exception:
            pass
    return _composite_engine(base_path, garment_path, audience, garment_type, long_side, background)


def engine_status() -> dict:
    avail, reason = _idm_vton_available()
    license_allowed, license_reason = _license_allows_idm()
    return {
        "idm_vton_available": avail,
        "reason": reason if not avail else "",
        "engine": "idm-vton" if avail else "idm-vton-composite-fallback",
        "license_mode": _LICENSE_MODE or "unset",
        "license_allows_idm": license_allowed,
        "license_reason": license_reason if not license_allowed else "",
        "commercial_compliant": _LICENSE_MODE == "commercial",
        "weights_dir": str(WEIGHTS_DIR),
        "idm_vton_dir": str(_IDM_DIR),
    }


def guess_audience(filename: str, fallback: str = "women") -> str:
    name = filename.lower()
    mapping = [
        ("baby", "baby"),
        ("infant", "baby"),
        ("toddler", "baby"),
        ("boy", "boys"),
        ("girl", "girls"),
        ("kids", "girls"),
        ("child", "girls"),
        ("men", "men"),
        ("male", "men"),
        ("women", "women"),
        ("female", "women"),
        ("lady", "women"),
    ]
    for key, value in mapping:
        if key in name:
            return value
    return fallback


def guess_garment_type(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ("dress", "gown", "skirt")):
        return "dress"
    if any(k in name for k in ("pant", "trouser", "short", "jean")):
        return "lower"
    if any(k in name for k in ("onesie", "romper", "body")):
        return "onesie"
    return "upper"
