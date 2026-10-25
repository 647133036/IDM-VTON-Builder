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


def compose_tryon(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
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
