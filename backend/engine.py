from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("atelier.engine")

_GPU_LOCK = threading.Lock()
_PIPE_LOCK = threading.Lock()

TORSOS = {
    "women": dict(x=0.21, y=0.178, w=0.58, h=0.39, shoulder=0.96, hem=0.62, neck=0.22, sleeve=0.22),
    "men": dict(x=0.18, y=0.155, w=0.64, h=0.42, shoulder=0.98, hem=0.64, neck=0.20, sleeve=0.20),
    "girls": dict(x=0.24, y=0.21, w=0.52, h=0.36, shoulder=0.92, hem=0.70, neck=0.24, sleeve=0.16),
    "boys": dict(x=0.22, y=0.20, w=0.56, h=0.38, shoulder=0.94, hem=0.72, neck=0.22, sleeve=0.16),
    "baby": dict(x=0.24, y=0.24, w=0.52, h=0.46, shoulder=0.90, hem=0.78, neck=0.28, sleeve=0.14),
}

ETHNICITY_FIT = {
    "western_europe": dict(w=1.00, h=1.00, shoulder=1.00),
    "eastern_europe": dict(w=1.02, h=1.02, shoulder=1.02),
    "west_asia": dict(w=1.01, h=1.00, shoulder=1.01),
    "central_asia": dict(w=0.98, h=1.00, shoulder=0.98),
    "east_asia": dict(w=0.95, h=1.02, shoulder=0.95),
    "southeast_asia": dict(w=0.94, h=0.99, shoulder=0.94),
    "africa": dict(w=1.00, h=0.96, shoulder=1.00),
    "north_america": dict(w=1.06, h=1.02, shoulder=1.05),
    "south_america": dict(w=1.03, h=1.00, shoulder=1.02),
    "oceania": dict(w=1.03, h=0.98, shoulder=1.02),
}


def torso_box(audience: str, ethnicity: str) -> dict:
    box = dict(TORSOS.get(audience, TORSOS["women"]))
    fit = ETHNICITY_FIT.get(ethnicity, ETHNICITY_FIT["western_europe"])
    ow = box["w"]
    box["w"] *= fit["w"]
    box["h"] *= fit["h"]
    box["shoulder"] *= fit["shoulder"]
    box["x"] += (ow - box["w"]) / 2
    return box


GARMENT_BOX = {
    "upper": dict(y=0.00, h=1.00, w="shoulder"),
    "tunic": dict(y=0.00, h=1.30, w="shoulder"),
    "dress": dict(y=0.00, h=1.55, w="shoulder"),
    "lower": dict(y=0.92, h=0.95, w="hem"),
    "onesie": dict(y=0.00, h=1.35, w="shoulder"),
}

# 简易 2D 变形的强度(0~1):越大越贴合模特轮廓。纯 CPU,不依赖 GPU。
GARMENT_WARP_STRENGTH = {
    "upper": 0.42,
    "tunic": 0.45,
    "dress": 0.48,
    "onesie": 0.42,
    "lower": 0.28,
}

GARMENT_PROMPTS = {
    "upper": "a model wearing this shirt",
    "tunic": "a model wearing this tunic top",
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
        return False, "LICENSE_MODE=commercial disables IDM-VTON (CC BY-NC-SA forbids commercial use)"
    if _LICENSE_MODE == "research":
        return True, ""
    if _LICENSE_MODE == "":
        return False, "LICENSE_MODE not set; IDM-VTON disabled by default"
    return False, f"Unknown LICENSE_MODE={_LICENSE_MODE}; only research or commercial supported"


def _idm_vton_available() -> tuple[bool, str]:
    allowed, reason = _license_allows_idm()
    if not allowed:
        return False, reason

    ckpt = WEIGHTS_DIR
    # 必须校验 unet_encoder(cross-attention encoder)与 scheduler,
    # 否则目录齐了也会在推理时报 KeyError(NEW-M4)。
    required = [
        "unet",
        "unet_encoder",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
    ]
    missing = [d for d in required if not (ckpt / d).exists()]
    if missing:
        return False, f"Missing weights: {', '.join(missing)}"
    unet_weight = ckpt / "unet"
    missing_unet = not (unet_weight / "diffusion_pytorch_model.safetensors").exists()
    if missing_unet and not (unet_weight / "diffusion_pytorch_model.bin").exists():
        return False, "unet weight file missing"
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "No GPU available"
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
    # 背景 = 四角 24x24 patch 中值: 平铺图常把衣料铺到上/左边缘, 旧"顶行+左列
    # 中值"会把衣料本身当背景 → alpha 全图塌软(pattern-lock 贴回只剩两三成,
    # 胸口被 SDXL 灰白洗掉; 袖条顶部半透明渐变, 袖像"从中段才开始")。
    # 四角是背景的概率高得多; 四角也是衣料(内容占满画布)时退回旧边线中值。
    k = 24
    h, w = arr.shape[:2]
    k = max(4, min(k, h, w))
    corners = np.concatenate(
        [
            arr[:k, :k].reshape(-1, 3),
            arr[:k, -k:].reshape(-1, 3),
            arr[-k:, :k].reshape(-1, 3),
            arr[-k:, -k:].reshape(-1, 3),
        ]
    )
    bg = np.median(corners, axis=0)
    if bg.mean() < 110:
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


def _silhouette_scale(ys: np.ndarray, box: dict, strength: float) -> np.ndarray:
    """每行的水平缩放系数,让衣料跟随躯干轮廓(肩宽、腰部收窄、下摆展开)。

    依据 shirt_mask 的轮廓关键点插值;强度 0 表示不缩放(原样贴图)。
    """
    if strength <= 0:
        return np.ones_like(ys)
    shoulder = box["shoulder"]
    hem = box["hem"]
    neck = box["neck"]
    sleeve = box["sleeve"]
    pit = 0.10 + sleeve
    # (y, 半宽占 box 宽度比例) 关键点:颈/肩/腋下/下摆。
    keys = np.array(
        [
            [0.00, neck / 2],
            [0.10, shoulder / 2],
            [pit, (shoulder + 0.02) / 2],
            [0.97, hem / 2],
            [1.00, hem / 2],
        ]
    )
    ratio = np.interp(ys, keys[:, 0], keys[:, 1])
    ratio = np.clip(ratio, 0.35, 0.55)  # 限制极端收窄
    full = 0.5  # 未变形时衣料近似铺满 box 宽度
    scale = 1.0 - strength * (1.0 - ratio / full)
    return np.clip(scale, 0.72, 1.06)


def _warp_garment(garment: Image.Image, box: dict, strength: float) -> Image.Image:
    """2D 网格变形:沿躯干轮廓水平收窄/展宽 + 垂直锚点对齐(领口→颈部、下摆→腰部)。

    纯 CPU(cv2.remap 双线性采样),无 GPU 依赖;cv2 缺失时原样返回。

    垂直对齐受 VTON_WARP_VERTICAL(默认 1.0)控制:
      - 1.0:衣料内容(从 alpha 检测的领口→下摆)映射到躯干框(颈部→腰部),
            修正 fit_garment_to_body 仅按肩宽等比缩放导致的垂直偏位。
      - 0.0:仅水平变形(与 v0.1.6 行为一致)。
    """
    if strength <= 0:
        return garment
    try:
        import cv2
    except ImportError:
        return garment
    tw, th = garment.size
    if tw < 4 or th < 4:
        return garment
    arr = np.asarray(garment.convert("RGBA")).astype(np.float32)
    ys = np.linspace(0.0, 1.0, th)
    scale = _silhouette_scale(ys, box, strength)
    cx = (tw - 1) / 2.0
    xg, yg = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
    # 目标像素 -> 源像素:沿中心线按该行缩放系数反算。
    map_x = (cx + (xg - cx) / scale[:, None]).astype(np.float32)

    # 垂直锚点对齐:衣料内容(领口→下摆)映射到躯干框(颈部→腰部)。
    v_strength = float(os.environ.get("VTON_WARP_VERTICAL", "1.0"))
    map_y = yg.astype(np.float32)
    _remap_bottom = -1.0
    if v_strength > 0:
        alpha = arr[..., 3]
        # 垂直锚点取中央带(0.32-0.68 宽): 平铺图的侧垂袖尖/裹身衣垂带常垂过
        # 衣身下摆(04469 实测全 bbox 939 vs 中央带 811), 全 bbox 锚会把
        # "袖尖"当下摆 → 衣身被整体上提, hem 截在肚脐上、下摆结压到胸口。
        # 中央带空(开襟/两片)时退回全 bbox 行。
        _c0, _c1 = int(tw * 0.32), max(int(tw * 0.32) + 4, int(tw * 0.68))
        content_rows = np.where(alpha[:, _c0:_c1].sum(axis=1) > 20)[0]
        if len(content_rows) <= 10:
            content_rows = np.where(alpha.sum(axis=1) > 20)[0]
        if len(content_rows) > 10:
            content_top = float(content_rows[0])
            content_bottom = float(content_rows[-1])
            target_top = float(np.clip(box["y"] * th, 0.0, th - 1.0))
            target_bottom = float(np.clip((box["y"] + box["h"]) * th, 1.0, float(th)))
            if target_bottom > target_top + 1 and content_bottom > content_top + 1:
                aligned = np.interp(
                    np.arange(th, dtype=np.float32),
                    [0.0, target_top, target_bottom, float(th)],
                    [0.0, content_top, content_bottom, float(th)],
                ).astype(np.float32)
                map_y = (yg * (1.0 - v_strength) + aligned[:, None] * v_strength).astype(np.float32)
                _remap_bottom = target_bottom

    out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    if v_strength > 0 and _remap_bottom > 0:
        # 垂直 remap 尾段 [target_bottom, th] <- [content_bottom, th] 在
        # 衣料贴到 fit 画布底(content_bottom≈th)时退化为常数采样: 下摆
        # 边缘行被竖直拖影到画布底(09263 实测 alpha 内容到 1023)。mask
        # 截线以下虽被裁, 但 alpha 拖影会进 pattern_lock 的衣料统计与
        # IP-Adapter 参考。下摆锚行以下直接清零。
        _cut = int(min(float(th), _remap_bottom + 4.0))
        out[_cut:, :, 3] = 0
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def remove_watermark(img: Image.Image) -> Image.Image:
    """去除角部半透明白/灰水印(logo/文字)。

    启发式:角部区域内,低饱和、亮薄特征(top-hat)判定为水印,
    再用 cv2.inpaint(Telea) 填补。无法可靠检出时原样返回。
    纯 CPU;cv2 缺失时无操作。
    """
    try:
        import cv2
    except ImportError:
        return img
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    h, w, _ = arr.shape
    if h < 60 or w < 60:
        return img
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # 亮 + 低饱和 = 中性水印候选。
    cand = ((top_hat > 26) & (sat < 60)).astype(np.uint8) * 255
    # 顶部/底部整条带(可覆盖横贯的水印文字),避免误伤衣料主体中段。
    band = np.zeros((h, w), np.uint8)
    bh = int(h * 0.18)
    band[:bh, :] = 255
    band[h - bh :, :] = 255
    cand = cv2.bitwise_and(cand, band)
    n = int(cand.sum() // 255)
    total = h * w
    if n < 30 or n > total * 0.06:
        return img
    cand = cv2.dilate(cand, np.ones((5, 5), np.uint8), iterations=1)
    inpainted = cv2.inpaint(bgr, cand, 4, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))


def _garment_bbox(img: Image.Image, alpha_thresh: int = 35) -> tuple[int, int, int, int] | None:
    """衣料真实外接框:由 alpha 通道阈值得到,忽略透明背景。"""
    a = np.asarray(img.split()[-1]) if img.mode == "RGBA" else np.asarray(garment_alpha(img))
    ys, xs = np.where(a > alpha_thresh)
    if len(xs) < 16:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_garment_to_body(garment: Image.Image, box: dict, tw: int, th: int) -> Image.Image:
    """按衣料实测宽度等比缩放到目标体宽,保留衣料真实长宽比。

    解决「衣服宽、模特窄」:不再强行铺满躯干框,而是把衣料自身宽度
    映射到目标体宽,使其完美落在模特身上;窄衣料也不会被拉宽。

    目标体宽由 box["width_key"] 决定(默认 shoulder):上衣/连衣裙用肩宽,
    下装用 hem(髋宽),避免裤子被缩放到肩宽过宽。
    """
    bbox = _garment_bbox(garment)
    if bbox is None:
        return fit_garment(garment, (tw, th))
    bx0, by0, bx1, by1 = bbox
    gw = bx1 - bx0 + 1
    gh = by1 - by0 + 1
    if gw < 4 or gh < 4:
        return fit_garment(garment, (tw, th))
    width_key = box.get("width_key", "shoulder")
    width_ratio = box.get(width_key, box["shoulder"])
    target_w = max(1.0, width_ratio * tw)
    scale = target_w / gw
    min_scale = (th * 0.55) / gh
    scale = max(min_scale, scale)
    nw = max(1, round(gw * scale))
    nh = max(1, round(gh * scale))
    crop = garment.crop((bx0, by0, bx1 + 1, by1 + 1))
    resized = crop.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas_h = max(th, nh + int(th * 0.04))
    canvas = Image.new("RGBA", (tw, canvas_h), (0, 0, 0, 0))
    x = (tw - nw) // 2
    y = int(th * 0.04)
    canvas.paste(resized, (x, y), resized)
    return canvas


def _arm_column_bands(human_img: Image.Image) -> tuple[int, int, int, int] | None:
    """从白底人物图自适应左右手臂列带(画布坐标)。

    上半身行带(肩下到肘上)取非白剪影, 按列连通段切分: 最左段=左臂,
    最右段=右臂; 臂与躯干粘连(剪影单块)时按主体宽 0.18 截取两端臂带
    (本底图实测: 剪影宽 331 → 左臂 354-410 ≈ 0.17x331); 剪影铺满画布
    (背景非白)时返回 None, 调用方退回硬编码画布比例。
    前提: 画布白底(compose 的 prewhiten + background=white 保证)。
    """
    arr = np.asarray(human_img.convert("RGB"))
    h, w = arr.shape[:2]
    y0, y1 = int(h * 0.28), int(h * 0.60)
    if y1 <= y0 + 4:
        return None
    body = np.abs(arr[y0:y1].astype(np.int16) - 255).sum(axis=2) > 90
    occ = body.mean(axis=0)
    cols = np.where(occ > 0.15)[0]
    if len(cols) < 8:
        return None
    runs = []
    start = prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c - prev > 8:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    if len(runs) >= 2:
        l0, l1 = runs[0]
        r0, r1 = runs[-1]
        cap = int(w * 0.12)
        l1 = min(l1, l0 + cap)
        r0 = max(r0, r1 - cap)
    elif runs and (runs[0][1] - runs[0][0]) < int(w * 0.75):
        # 臂贴躯干(剪影单块): 臂在主体两缘, 按剪影宽 0.18 截臂带
        # (0.12c 截宽会外溢到侧腹, 袖料压到髋部)。
        l0, r1 = runs[0]
        bw = int(np.clip((r1 - l0) * 0.18, w * 0.03, w * 0.10))
        l1 = l0 + bw
        r0 = r1 - bw
    else:
        return None
    if r0 <= l1 or (l1 - l0) < w * 0.02 or (r1 - r0) < w * 0.02:
        return None
    return int(l0), int(l1), int(r0), int(r1)


def _arm_polylines(pose: dict | None) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
    """左右臂 (肘, 腕) 归一化坐标对, 按图像 x 升序(小 x ↔ 图像左臂带)。

    COCO/MediaPipe 的左右是人物视角(面镜头时人物左臂在图像右), 引擎按
    图像 x 排序配对, 不依赖左右命名。肘/腕配不齐(缺关键点)返回 None,
    调用方退回竖直条。
    """
    if not pose:
        return None

    def _pt(k):
        if k and len(k) >= 3 and k[2] >= 0.3:
            return (float(k[0]), float(k[1]))
        return None

    pairs = []
    for e_key, w_key in (("elbow_l", "wrist_l"), ("elbow_r", "wrist_r")):
        e, w = _pt(pose.get(e_key)), _pt(pose.get(w_key))
        if e and w:
            pairs.append((e, w))
    if len(pairs) != 2:
        return None
    pairs.sort(key=lambda p: p[0][0])
    return pairs


def _arm_row_anchors(ch: int, pose: dict | None, sleeve_len: str) -> tuple[int, int, int] | None:
    """臂列竖直锚点: 返回 (shoulder_y, strip_h_l, strip_h_r), 算不出返回 None。

    pose 肩/腕 y(缺省全身比例 0.20/0.62), 夹紧肩 0.12-0.32c、腕=肩+
    (0.18-0.50)c, 避免腕点噪声把袖条拉到画布底(09263 L 带到 y1.0)。
    左右臂腕点常不等高: 优先 per-side wrist_l/wrist_r(c9: 旧 averaged
    wrist 实为肘线——RTMPose 索引 7/8 是肘, 袖条只到肘=长袖读作中袖),
    缺一侧用 averaged wrist, 再缺用比例。strip_h: 长袖=各臂到腕,
    短袖=肩下 0.40 臂长(肘上)。
    """
    if pose:
        sh = pose.get("shoulder")
        shoulder_y = int(sh[1] * ch) if sh and sh[2] >= 0.3 else int(ch * 0.20)

        def _wy(k):
            return int(k[1] * ch) if k and k[2] >= 0.3 else None

        wa = _wy(pose.get("wrist"))
        yl = _wy(pose.get("wrist_l"))
        yr = _wy(pose.get("wrist_r"))
        if yl is None:
            yl = wa if wa is not None else int(ch * 0.62)
        if yr is None:
            yr = wa if wa is not None else int(ch * 0.62)
    else:
        shoulder_y = int(ch * 0.20)
        yl = yr = int(ch * 0.62)
    shoulder_y = int(np.clip(shoulder_y, ch * 0.12, ch * 0.32))
    lo, hi = shoulder_y + int(ch * 0.18), shoulder_y + int(ch * 0.50)
    yl = int(np.clip(yl, lo, hi))
    yr = int(np.clip(yr, lo, hi))
    if min(yl, yr) - shoulder_y < ch * 0.08:
        return None
    if sleeve_len == "long":
        hl, hr = yl - shoulder_y, yr - shoulder_y
    else:
        hl = max(int(ch * 0.05), int((yl - shoulder_y) * 0.40))
        hr = max(int(ch * 0.05), int((yr - shoulder_y) * 0.40))
    return shoulder_y, hl, hr


def _build_sleeve_strips(
    garment: Image.Image,
    cw: int,
    ch: int,
    pose: dict | None = None,
    sleeve_len: str = "long",
    arm_cols: tuple[int, int, int, int] | None = None,
) -> Image.Image | None:
    """构建全画布袖条 RGBA 覆盖层(左右袖从平铺图裁出, 竖直贴到臂列)。

    refit_sleeves_to_arms 把它贴进 warp 参考图(IP-Adapter/pattern-lock 用);
    compose 生成后又把同一覆盖层 alpha 合成到结果图手臂上 —— 走廊/前臂
    常被"∩生成主体"裁掉(SDXL 不把前臂画满 → 走廊白底被剔, c7 实测
    cut_row 以下 in-apply=0, 袖条只能贴到肘上), 生成后合成不再依赖
    apply_mask, 袖料确定盖到腕。

    袖子摆放按裁片横宽比自适应:
    - 横向摊开(宽>1.25x高): 旋转 90° 袖口朝下(左袖 CCW, 右袖 CW);
    - 向下折放(高≥宽, 肩端在上袖口在下): 不转, 直接贴。
    外侧袖带若混入不连肩的低位散段(A 字衣摆侧缘), 按"取最靠上(连肩线)的
    连续行段"只保留袖段。RGB 平铺图先补 garment_alpha 再裁, 白底像素
    alpha=0 贴合时为 no-op(修"袖子贴成白块, 模型看不见袖料")。右袖裁片
    坐标从切片相对坐标换回全图坐标需 +tc1。

    竖直范围用 pose 的肩/腕归一化 y(无 pose 时用全身比例 0.20/0.62), 并夹紧到
    肩 0.12-0.32c、腕=肩+(0.18-0.50)c, 避免腕点噪声把袖条拉到画布底。横向优先
    用人物剪影手臂列(arm_cols, 由 _arm_column_bands 从 human_img 算出): 硬编码
    画布比例按"全身正面基模"标定, 换底图(手臂贴身/letterbox 画布)时袖条会贴到
    白背景上, SDXL 与 pattern-lock 都救不回(c6 实测: 左袖条 297-363 vs 实际左臂
    354-410, 仅 19px 压臂)。arm_cols=None 时退回画布比例(袖宽 0.065c)。短袖只
    贴肩下 0.40 臂长(肘上)。任何一步算不出合法手臂列返回 None。sleeve_len
    为 "long"/"short"。
    """
    if sleeve_len not in ("long", "short"):
        return None
    overlay = Image.new("RGBA", (int(cw), int(ch)), (0, 0, 0, 0))
    cx = cw / 2.0

    # ---- 竖直锚点: pose 肩/各臂腕 y, 无 pose 用全身比例 ----
    anchors = _arm_row_anchors(ch, pose, sleeve_len)
    if anchors is None:
        return None
    shoulder_y, h_l, h_r = anchors
    # 顶补 deltoid 帽: 肩关节点在臂顶(三角肌)下方 ~0.025c, 袖条从关节起
    # 会露出一段"无袖帽"的肩头(c9-r2 实测臂带顶部 30 行仍是 SDXL 肤)。
    top_y = max(0, shoulder_y - int(ch * 0.025))

    # ---- 横向: 优先人物剪影手臂列, 拿不到退回全身正面基模标定比例 ----
    # 基模(全身正面 1024)躯干半宽≈0.145c, 手臂悬在躯干外侧, 手臂宽≈0.065c:
    # 左手臂列=[cx-0.21c, cx-0.145c], 右手臂列=[cx+0.145c, cx+0.21c]。
    if arm_cols is not None:
        al0, al1, ar0, ar1 = arm_cols
        arm_w = int(np.clip(al1 - al0, 10, 0.10 * cw))
        px_l, px_r = int(al0), int(ar0)
        logger.info("sleeve strips at silhouette arm cols L %d-%d R %d-%d (w=%d)", al0, al1, ar0, ar1, arm_w)
    else:
        arm_w = int(max(10, 0.065 * cw))
        px_l = int(cx - 0.21 * cw)
        px_r = int(cx + 0.145 * cw)
    if px_l < 0:
        px_l = 0
    if px_r + arm_w > cw:
        px_r = cw - arm_w
    if px_l + arm_w > px_r:  # 两臂列重叠, 放不下
        return None

    # ---- 从原平铺图裁左右袖 ----
    g = garment.convert("RGBA")
    ga = np.asarray(g.getchannel("A"))
    if int(ga.min()) >= 240:
        # 不透明(RGB/JPEG)输入: 补估计 alpha, 让裁片自带真实前景
        # (白底 alpha≈0 → 贴合时背景 no-op, 不再整块贴白)。
        g.putalpha(garment_alpha(garment))
        ga = np.asarray(g.getchannel("A"))
    content = ga > 40
    xs = np.where(content)[1]
    if len(xs) < 64:
        return None
    gx0, gx1 = int(xs.min()), int(xs.max())
    gcw = max(1, gx1 - gx0 + 1)
    tc0 = gx0 + int(gcw * 0.22)
    tc1 = gx0 + int(gcw * 0.78)

    def _sleeve_strip(side: str) -> Image.Image | None:
        band_c = content[:, :tc0] if side == "L" else content[:, tc1:]
        rows = np.where(band_c.any(axis=1))[0]
        if len(rows) == 0:
            return None
        # 连续行段(行间空隙>30px 断开); 取起始行最小(连着肩线)的一段=袖,
        # 丢弃衣摆侧缘/散开下摆等不连肩的低位段。
        segs = []
        start = prev = int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r - prev > 30:
                segs.append((start, prev))
                start = r
            prev = r
        segs.append((start, prev))
        sy0, sy1 = min(segs)
        _yy, xx = np.where(band_c[sy0 : sy1 + 1])
        if len(xx) < max(32, gcw * ch // 400):
            return None
        sx0 = int(xx.min())
        sx1 = int(xx.max()) + 1
        if side == "R":
            # np.where 返回切片内相对坐标, 换回全图坐标要 +tc1。
            sx0 += tc1
            sx1 += tc1
        crop = g.crop((sx0, sy0, sx1, sy1 + 1))
        if sx1 - sx0 > (sy1 + 1 - sy0) * 1.25:
            # 横向摊开: 左袖袖口在裁片左缘 → 90° CCW 袖口朝下; 右袖镜像 → 90° CW。
            if side == "L":
                crop = crop.transpose(Image.Transpose.ROTATE_90)
            else:
                crop = crop.transpose(Image.Transpose.ROTATE_270)
        # 向下折放: 长度轴已竖直(肩端在上袖口在下), 不转。
        return crop

    from PIL import ImageFilter

    # 缎带折线: 肩(臂带中心)→肘→腕。前臂常斜出臂带外(c9 实测腕 x=306/707
    # 在带 345-404/617-676 外), 竖条在肘以下会盖到躯干/空隙而漏掉真臂。
    polys = _arm_polylines(pose)
    band_cx = {"L": (px_l + px_l + arm_w) / 2.0, "R": (px_r + px_r + arm_w) / 2.0}
    lay = np.zeros((int(ch), int(cw), 4), dtype=np.uint8)

    for idx, (side, _px, h_side) in enumerate((("L", px_l, h_l), ("R", px_r, h_r))):
        sl = _sleeve_strip(side)
        if sl is None:
            continue
        end_y = min(int(ch) - 1, shoulder_y + h_side)
        total_h = end_y - top_y
        if total_h < 8:
            continue
        sl = sl.resize((arm_w, total_h), Image.Resampling.LANCZOS)
        # 袖条必须实心: 裁片只是袖子的包围盒(横摊袖 90° 转过来), 袖形斜在
        # 盒内、肩端/袖口收尖 → 直接拉伸后 alpha 顶部渐入、底部仅 10% 列
        # 有料(c9 实测 09263 row491 alpha=24/255), 贴出来读作"中袖/碎袖"。
        # 主色实心打底(裁片实心像素中值, 不足退回全衣中值), 真实袖纹(二值
        # 化 alpha+羽化)叠上面 —— 袖料确定盖满肩-腕/肩-肘整段。
        arr = np.array(sl)
        solid = arr[:, :, 3] >= 100
        if int(solid.sum()) >= max(32, arm_w // 2):
            fill = np.median(arr[:, :, :3][solid].astype(np.float32), axis=0)
        else:
            fill = np.median(np.asarray(g)[..., :3][content].astype(np.float32), axis=0)
        fill = np.clip(fill, 0, 255).astype(np.uint8)
        # 裁片透明处的主色填充 + 实心处真实袖纹, 一起逐行铺进缎带。
        tex_rgb = np.where(solid[:, :, None], arr[:, :, :3], fill.reshape(1, 1, 3)).astype(np.uint8)
        tex_a = np.where(solid, np.uint8(255), np.uint8(0))
        tex_img = Image.fromarray(np.dstack([tex_rgb, tex_a]), "RGBA")
        tex_img.putalpha(tex_img.getchannel("A").filter(ImageFilter.GaussianBlur(2)))
        tex = np.asarray(tex_img)  # (total_h, arm_w, 4)

        # 每行缎带中心 x: 上臂在臂带内竖直, 肘前 25% 开始向外弯, 肘→腕沿前臂。
        ex = ey = wx = wy = -1.0
        if polys is not None:
            (ex_n, ey_n), (wx_n, wy_n) = polys[idx]
            ex, ey, wx, wy = ex_n * cw, ey_n * ch, wx_n * cw, wy_n * ch
        bend_start = shoulder_y + (ey - shoulder_y) * 0.75 if 0 < shoulder_y < ey < wy else -1
        # 袖口位置: 长袖=腕(带罗纹收口), 短袖=0.40 臂长。
        cuff_v = 1.0 - max(0.03, min(0.06, 26.0 / max(1, total_h)))
        for k in range(total_h):
            r = top_y + k
            if bend_start < 0:
                xc, w_row = band_cx[side], arm_w
            else:
                if r < bend_start:
                    xc, w_row = band_cx[side], arm_w
                elif r <= ey:
                    t = (r - bend_start) / max(1, ey - bend_start)
                    xc = band_cx[side] + (ex - band_cx[side]) * t
                    w_row = arm_w
                else:
                    t = (r - ey) / max(1, wy - ey)
                    xc = ex + (wx - ex) * t
                    w_row = int(arm_w * (1.0 - 0.35 * min(1.0, (r - shoulder_y) / max(1, wy - shoulder_y))))
            c0 = int(np.clip(xc - w_row / 2.0, 0, cw - 2))
            c1 = int(np.clip(c0 + w_row, c0 + 2, cw))
            xs = np.linspace(0, arm_w - 1, c1 - c0).astype(int)
            seg = c1 - c0
            u = (np.arange(seg, dtype=np.float32) + 0.5) / max(1, seg)
            v = k / max(1, total_h - 1)
            # 写实明暗(纯确定性, 无随机): 平铺衣纹是影棚平光, 直贴读作"纯色
            # 填充/铁上去"。四层: 柱面(边暗中亮) + 上亮下暗垂坠 + 低频竖褶
            # (两相位正弦, 幅度 3.5%) + 袖口罗纹(末端 4-6% 压暗 + 缝线行)。
            mult = (0.87 + 0.13 * np.sin(np.pi * np.clip(u, 0.0, 1.0))) * (1.0 - 0.07 * v)
            mult *= 1.0 + 0.035 * np.sin(2.4 * np.pi * u + idx * 2.1) * np.sin(np.pi * min(1.0, v * 1.6))
            # 腋下阴影: 靠躯干侧的内缘, 上臂段(30%)再压 6% —— 袖子"长在"躯干上。
            if v < 0.30:
                _inner = u > 0.85 if side == "L" else u < 0.15
                mult = np.where(_inner, mult * 0.94, mult)
            if v >= cuff_v:
                mult = np.where(np.abs(v - (cuff_v + 0.008)) < 0.006, mult * 0.82, mult * 0.92)
            lay[r, c0:c1, :3] = np.clip(tex[k, xs, :3] * mult[:, None], 0, 255)
            # 缎带整段实心: 纹理洞(裁片透明处)已用主色填充, alpha 统一 255,
            # 边缘羽化交给最后的整体 blur —— 旧"直接拉伸"会留半透明渐变。
            lay[r, c0:c1, 3] = 255
    if lay[:, :, 3].any():
        # 边缘羽化 2px: 1px 羽化混出的 1px 中间色环被用户读作"袖子边上
        # 别的颜色", 2px 柔和过渡 + 贴回带先垫原臂肤, 混色=料x肤更自然。
        lay[:, :, 3] = np.asarray(Image.fromarray(lay[:, :, 3], "L").filter(ImageFilter.GaussianBlur(2)))
        overlay = Image.fromarray(lay, "RGBA")
    return overlay


def refit_sleeves_to_arms(
    warped: Image.Image,
    garment: Image.Image,
    canvas: int,
    pose: dict | None = None,
    sleeve_len: str = "long",
    arm_cols: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """袖几何对齐: 把平铺衣的左右袖(_build_sleeve_strips)合成到 warp 参考图。

    2D warp 只把衣身"充气"包到躯干框, 折不下袖子 → 参考图手臂位置没有袖料
    → 生成模型只能画裸臂。本函数在 warp 参考图上贴袖条, 让 IP-Adapter /
    pattern-lock 在手臂位置看到真实袖纹。袖不出合法覆盖层时原样返回
    (RGBA copy)。纯 CPU。
    """
    out = warped.convert("RGBA").copy()
    ov = _build_sleeve_strips(garment, out.size[0], out.size[1], pose, sleeve_len, arm_cols)
    if ov is None:
        return out
    out.paste(ov, (0, 0), ov)
    return out


def sleeve_strip_overlay(
    garment: Image.Image,
    canvas: int,
    pose: dict | None = None,
    sleeve_len: str = "long",
    arm_cols: tuple[int, int, int, int] | None = None,
) -> Image.Image | None:
    """生成后确定性袖条覆盖层(compose 在贴回/肤色校正之后合成)。

    与 refit_sleeves_to_arms 同一份袖条(同 pose/列带), 保证参考图与结果
    图的袖子位置一致; 返回 None 表示算不出袖条(调用方跳过)。
    """
    return _build_sleeve_strips(garment, canvas, canvas, pose, sleeve_len, arm_cols)


def arm_refit_overlay(
    garment: Image.Image,
    human_img: Image.Image,
    canvas: int,
    pose: dict | None = None,
    sleeve_len: str = "long",
    arm_cols: tuple[int, int, int, int] | None = None,
) -> Image.Image | None:
    """生成后臂区覆盖层: 袖条(布料) + 袖端以下原图手臂/手(原肤色)。

    SDXL 重画的手臂常发白发灰, 短袖时还常自作主张把袖画到肘下; 走廊把
    前臂圈进重画区后, 袖端以下的生成肤色/多画袖料都留在出图里(c8 用户
    目测: 三件全部"长/短袖读作中袖、手臂颜色变了")。本覆盖层在臂列带内:
    (1) 袖端以下贴回原图像素(袖端到画布底, 仅主体, 羽化), 恢复原肤色并
    抹掉 SDXL 多画的袖; (2) 与 refit 同源的袖条盖在上面。返回 None 表示
    袖型不支持/臂列算不出(调用方跳过)。纯 CPU。
    """
    if sleeve_len not in ("long", "short"):
        return None
    cw = int(canvas)
    if arm_cols is None:
        arm_cols = _arm_column_bands(human_img)
    if arm_cols is None:
        return None
    anchors = _arm_row_anchors(cw, pose, sleeve_len)
    if anchors is None:
        return None
    shoulder_y, h_l, h_r = anchors
    # 贴回带起点 = 各臂袖端行(左右腕不等高, 各自接缝); 袖条顶补不算(在袖内)。
    y0_l = min(shoulder_y + h_l, cw - 1)
    y0_r = min(shoulder_y + h_r, cw - 1)

    base_rgb = human_img.convert("RGB")
    if base_rgb.size != (cw, cw):
        base_rgb = base_rgb.resize((cw, cw), Image.Resampling.LANCZOS)
    base = np.asarray(base_rgb)
    subj = np.abs(base.astype(np.int16) - 255).sum(axis=2) > 90

    ov = np.zeros((cw, cw, 4), dtype=np.uint8)
    for c0, c1, y0 in (
        (arm_cols[0], arm_cols[1], y0_l),
        (arm_cols[2], arm_cols[3], y0_r),
    ):
        c0, c1 = int(c0), int(c1) + 1
        ov[y0:, c0:c1, :3] = base[y0:, c0:c1]
        ov[y0:, c0:c1, 3] = subj[y0:, c0:c1].astype(np.uint8) * 255
    # 前臂/手缎带贴回: 前臂在肘下斜出臂带外(带内贴回盖不到), 沿肘→腕
    # 折线(腕以下按前臂斜率延到指尖)贴回原图, 修带外 SDXL 灰白前臂。
    polys = _arm_polylines(pose)
    if polys is not None:
        arm_w = int(np.clip(max(arm_cols[1] - arm_cols[0], arm_cols[3] - arm_cols[2]), 10, cw * 0.10))
        for idx, (y0, band_cx) in enumerate(
            ((y0_l, (arm_cols[0] + arm_cols[1]) / 2.0), (y0_r, (arm_cols[2] + arm_cols[3]) / 2.0))
        ):
            (ex_n, ey_n), (wx_n, wy_n) = polys[idx]
            ex, ey, wx, wy = ex_n * cw, ey_n * cw, wx_n * cw, wy_n * cw
            if not (0 < shoulder_y < ey < wy):
                continue
            slope = (wx - ex) / max(1.0, wy - ey)
            hand_end = min(cw - 1, int(wy + cw * 0.06))
            for r in range(int(y0), hand_end + 1):
                if r <= wy:
                    t = min(1.0, max(0.0, (r - shoulder_y) / max(1, wy - shoulder_y)))
                    if r >= ey:
                        xc = ex + slope * (r - ey)
                    else:
                        te = min(1.0, max(0.0, (r - shoulder_y) / max(1, ey - shoulder_y)))
                        xc = band_cx + (ex - band_cx) * te
                    w_row = int(arm_w * (1.0 - 0.35 * t))
                else:
                    xc = wx + slope * (r - wy)
                    w_row = max(24, int(arm_w * 0.55))
                c0 = int(np.clip(xc - w_row / 2.0, 0, cw - 2))
                c1 = int(np.clip(c0 + w_row, c0 + 2, cw))
                ov[r, c0:c1, :3] = base[r, c0:c1]
                ov[r, c0:c1, 3] = subj[r, c0:c1].astype(np.uint8) * 255
    overlay = Image.fromarray(ov, "RGBA")
    # 3px 羽化出自然边界, 再按主体收边(羽化不越出原图人物)。
    a = np.asarray(overlay.getchannel("A").filter(ImageFilter.GaussianBlur(3)))
    overlay.putalpha(Image.fromarray(np.minimum(a, subj.astype(np.uint8) * 255), "L"))

    strips = _build_sleeve_strips(garment, cw, cw, pose, sleeve_len, arm_cols)
    if strips is not None:
        overlay = Image.alpha_composite(overlay, strips)
    return overlay


def _composite_engine(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int,
    background: str,
    ethnicity: str = "western_europe",
) -> Image.Image:
    base = Image.open(base_path).convert("RGBA")
    w, h = base.size
    garment_src = remove_watermark(Image.open(garment_path))
    garment = detect_garment_crop(garment_src)
    garment = match_color(garment)
    alpha = garment_alpha(garment)
    garment = garment.convert("RGBA")
    garment.putalpha(alpha)

    box = torso_box(audience, ethnicity)
    adj = GARMENT_BOX.get(garment_type, GARMENT_BOX["upper"])
    box["y"] = box["y"] + adj["y"] * box["h"]
    box["h"] = box["h"] * adj["h"]
    box["width_key"] = adj.get("w", "shoulder")

    tx = int(box["x"] * w)
    ty = int(box["y"] * h)
    tw = int(box["w"] * w)
    th = int(box["h"] * h)
    tw = max(tw, 32)
    th = max(th, 32)

    fitted = fit_garment_to_body(garment, box, tw, th)
    fitted = _warp_garment(fitted, box, GARMENT_WARP_STRENGTH.get(garment_type, 0.4))
    fw, fh = fitted.size
    body = shirt_mask(fw, fh, box)
    g_a = fitted.split()[-1]
    merged_a = ImageEnhance.Brightness(g_a).enhance(1.0)
    merged_a = Image.fromarray(
        (np.asarray(merged_a).astype(np.float32) * (np.asarray(body).astype(np.float32) / 255.0)).astype(np.uint8),
        mode="L",
    )
    fitted.putalpha(merged_a)

    # 接触阴影:模糊半径随衣料尺寸缩放,略偏移并加深,模拟身体投影。
    blur_r = max(4, int(max(tw, th) * 0.035))
    shadow = merged_a.filter(ImageFilter.GaussianBlur(blur_r))
    shadow_layer = Image.new("RGBA", base.size, (25, 20, 16, 0))
    shadow_img = Image.new("L", base.size, 0)
    shadow_img.paste(ImageEnhance.Brightness(shadow).enhance(0.42), (tx + 2, ty + int(th * 0.02)))
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


def _get_compliance_advice(engine: str, license_mode: str, sdxl_avail: bool, fashn_avail: bool = False) -> str:
    if license_mode == "commercial" and engine == "fashn-vton":
        return (
            "Commercial mode: using FASHN VTON v1.5 (Apache-2.0, pixel-space, maskless); "
            "human-parser is stubbed (NVIDIA SegFormer terms). Letters/large prints still "
            "use deterministic pattern_lock."
        )
    if license_mode == "commercial" and engine == "sdxl-commercial":
        extra = " FASHN weights missing; " if not fashn_avail else " "
        return (
            f"Commercial mode:{extra}using SDXL commercial engine; register on the "
            "Stability AI model page and complete attribution."
        )
    if license_mode == "commercial" and engine == "composite-fallback":
        return (
            "Commercial mode: FASHN and SDXL weights missing, using the in-repo composite engine "
            "(original PIL code, no third-party weights); compliant but limited in fidelity."
        )
    if license_mode == "research" and engine == "idm-vton":
        return "Research mode: using IDM-VTON; non-commercial use only (CC BY-NC-SA 4.0)."
    if license_mode == "research" and engine == "composite-fallback":
        return "Research mode: IDM-VTON unavailable, using composite engine (limited results)."
    if license_mode == "":
        return "LICENSE_MODE not set; using composite engine. Set research or commercial."
    return f"Current engine: {engine}; please verify compliance."


def _load_idm_vton_pipeline():
    if _engine_state["available"] is False:
        return None
    if _engine_state["pipe"] is not None:
        return _engine_state["pipe"]

    # 首载必须单飞(NEW-H3 残留):并发请求同时见 pipe=None 会重复构建管线撑爆显存。
    with _PIPE_LOCK:
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
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.unet_hacked_tryon import UNet2DConditionModel
        from transformers import (
            AutoTokenizer,
            CLIPTextModel,
            CLIPTextModelWithProjection,
        )

        ckpt_path = str(WEIGHTS_DIR)
        device = torch.device("cuda:0")

        unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet", torch_dtype=torch.float16)
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            ckpt_path, subfolder="unet_encoder", torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae", torch_dtype=torch.float16)
        scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")

        text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            ckpt_path, subfolder="text_encoder_2", torch_dtype=torch.float16
        )
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
    ethnicity: str = "western_europe",
) -> Image.Image:
    import torch
    from PIL import Image as PILImage

    pipe = _load_idm_vton_pipeline()
    if pipe is None:
        return _composite_engine(base_path, garment_path, audience, garment_type, long_side, background, ethnicity)

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
    box = torso_box(audience, ethnicity)
    mx = int(box["x"] * target_w)
    my = int(box["y"] * target_h)
    mw = int(box["w"] * target_w)
    mh = int(box["h"] * target_h)
    mask_draw.rectangle([mx, my, mx + mw, my + mh], fill=(255, 255, 255))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(16))

    generator = torch.Generator(device=device).manual_seed(42)

    with _GPU_LOCK, torch.no_grad(), torch.amp.autocast(device_type="cuda"):
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
        # 电商主图硬性要求:背景必须纯白 RGB(255,255,255),边缘连通近白区压白。
        from platform_presets import force_white_background, foreground_mask

        if result.mode == "RGBA":
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[-1])
            result = bg
        else:
            result = result.convert("RGB")
        # pipeline audit 行36 + 远端复核#3: legacy IDM-VTON 路径原来直接 force_white_background(result)
        # 不传 subject_mask → 胸口白块 union 修复被绕过(生产默认走 SDXL, legacy 仍可被选)。
        # 保护掩膜 = 基模主体 并 生成图主体(注释承诺的并集, 此前漏了基模一侧):
        # 基模主体保住"原本是人体的区域", 生成图主体保住"生成后仍是衣物的区域",
        # 两者并集外(纯背景)才压白。任一侧取不到掩膜则用另一侧; 都取不到 = 原行为。
        _legacy_subj = None
        _base_subj = foreground_mask(human_img)
        _res_subj = foreground_mask(result)
        if _base_subj is not None and _res_subj is not None:
            _legacy_subj = _base_subj | _res_subj
        elif _base_subj is not None:
            _legacy_subj = _base_subj
        elif _res_subj is not None:
            _legacy_subj = _res_subj
        result = force_white_background(result, subject_mask=_legacy_subj)

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
    ethnicity: str = "western_europe",
) -> Image.Image:
    # 实际采用的引擎与降级原因挂在 result.info 上,由 worker 落库(NEW-H4)。
    engine_name = "composite-fallback"
    degraded = False
    degraded_reason = ""
    license_allowed, _ = _license_allows_idm()
    if license_allowed:
        avail, _ = _idm_vton_available()
        if avail:
            try:
                img = _idm_vton_engine(
                    base_path, garment_path, audience, garment_type, long_side, background, ethnicity
                )
                engine_name = "idm-vton"
                img.info["engine"] = engine_name
                img.info["engine_degraded"] = False
                img.info["degraded_reason"] = ""
                return img
            except Exception:
                logger.exception("IDM-VTON engine failed; falling back to composite engine")
                degraded = True
                degraded_reason = "IDM-VTON engine failed at runtime; fell back to composite"

    if _LICENSE_MODE == "commercial":
        fashn_reason = ""
        try:
            from engine_fashn import compose_tryon_fashn, fashn_available

            fashn_avail, fashn_reason = fashn_available()
            if fashn_avail:
                try:
                    img = compose_tryon_fashn(
                        base_path, garment_path, audience, garment_type, long_side, background, ethnicity
                    )
                    engine_name = "fashn-vton"
                    img.info["engine"] = engine_name
                    img.info.setdefault("engine_degraded", False)
                    img.info.setdefault("degraded_reason", "")
                    return img
                except Exception:
                    logger.exception("FASHN engine failed at runtime; falling back to SDXL")
                    degraded = True
                    degraded_reason = "FASHN engine failed at runtime; fell back to SDXL"
        except ImportError:
            fashn_reason = "fashn_vton not installed"

        sdxl_reason = ""
        try:
            from engine_sdxl import compose_tryon_sdxl, sdxl_available

            sdxl_avail, sdxl_reason = sdxl_available()
            if sdxl_avail:
                try:
                    img = compose_tryon_sdxl(
                        base_path, garment_path, audience, garment_type, long_side, background, ethnicity
                    )
                    engine_name = "sdxl-commercial"
                    img.info["engine"] = engine_name
                    img.info.setdefault("engine_degraded", False)
                    img.info.setdefault("degraded_reason", "")
                    if degraded:
                        prev = img.info.get("degraded_reason") or ""
                        img.info["engine_degraded"] = True
                        img.info["degraded_reason"] = "; ".join(x for x in (degraded_reason, prev) if x)
                    return img
                except Exception:
                    logger.exception("SDXL engine failed at runtime; falling back to composite engine")
                    degraded = True
                    degraded_reason = "SDXL engine failed at runtime; fell back to composite"
        except ImportError:
            sdxl_reason = "diffusers not installed"
        if (fashn_reason or sdxl_reason) and not degraded:
            logger.warning(
                "commercial FASHN/SDXL unavailable (fashn=%s sdxl=%s); using in-repo composite engine",
                fashn_reason,
                sdxl_reason,
            )

    img = _composite_engine(base_path, garment_path, audience, garment_type, long_side, background, ethnicity)
    img.info["engine"] = engine_name
    img.info["engine_degraded"] = degraded
    img.info["degraded_reason"] = degraded_reason
    return img


def engine_status() -> dict:
    avail, reason = _idm_vton_available()
    license_allowed, license_reason = _license_allows_idm()

    # 贴印引擎: grid(默认)/tps/optical_flow, 由 VTON_REFIT_ENGINE 控制
    import os as _os_eng

    refit_engine = _os_eng.environ.get("VTON_REFIT_ENGINE", "grid").lower()
    if refit_engine not in ("grid", "tps", "optical_flow"):
        refit_engine = "grid"

    sdxl_avail = False
    sdxl_reason = ""
    ip_adapter_avail = False
    sdxl_inpaint = False
    try:
        from engine_sdxl import engine_status as sdxl_engine_status
        from engine_sdxl import ip_adapter_available, sdxl_available

        sdxl_avail, sdxl_reason = sdxl_available()
        ip_adapter_avail = ip_adapter_available()
        sdxl_inpaint = bool(sdxl_engine_status().get("sdxl_inpaint"))
    except Exception as e:  # noqa: BLE001
        sdxl_reason = str(e)

    fashn_avail = False
    fashn_reason = ""
    try:
        from engine_fashn import fashn_available

        fashn_avail, fashn_reason = fashn_available()
    except Exception as e:  # noqa: BLE001
        fashn_reason = str(e)

    gpu_available = False
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_available = False

    engine = "composite-fallback"
    commercial_compliant = False

    if license_allowed and avail:
        engine = "idm-vton"
        commercial_compliant = False
    elif _LICENSE_MODE == "commercial" and fashn_avail:
        engine = "fashn-vton"
        commercial_compliant = True
    elif _LICENSE_MODE == "commercial" and sdxl_avail:
        engine = "sdxl-commercial"
        commercial_compliant = True
    elif _LICENSE_MODE == "commercial":
        engine = "composite-fallback"
        commercial_compliant = True

    return {
        "idm_vton_available": avail,
        "idm_vton_reason": reason if not avail else "",
        "fashn_available": fashn_avail,
        "fashn_reason": fashn_reason if not fashn_avail else "",
        "sdxl_available": sdxl_avail,
        "sdxl_reason": sdxl_reason if not sdxl_avail else "",
        "sdxl_inpaint": sdxl_inpaint,
        "ip_adapter_available": ip_adapter_avail,
        "gpu_available": gpu_available,
        "refit_engine": refit_engine,
        "engine": engine,
        "license_mode": _LICENSE_MODE or "unset",
        "license_allows_idm": license_allowed,
        "license_reason": license_reason if not license_allowed else "",
        "commercial_compliant": commercial_compliant,
        "compliance_advice": _get_compliance_advice(engine, _LICENSE_MODE, sdxl_avail, fashn_avail),
        "weights_dir": str(WEIGHTS_DIR),
        "idm_vton_dir": str(_IDM_DIR),
        "sdxl_dir": str(_REPO_ROOT / "models" / "sdxl"),
        "fashn_dir": str(_REPO_ROOT / "models" / "fashn"),
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
    if any(k in name for k in ("tunic", "longline")):
        return "tunic"
    if any(k in name for k in ("pant", "trouser", "short", "jean")):
        return "lower"
    if any(k in name for k in ("onesie", "romper", "body")):
        return "onesie"
    return "upper"
