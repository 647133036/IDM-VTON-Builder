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
    if v_strength > 0:
        alpha = arr[..., 3]
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

    out = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
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


def refit_sleeves_to_arms(
    warped: Image.Image,
    garment: Image.Image,
    canvas: int,
    pose: dict | None = None,
    sleeve_len: str = "long",
) -> Image.Image:
    """袖几何对齐: 把平铺衣的左右袖合成到目标画布的手臂列。

    2D warp 只把衣身"充气"包到躯干框, 折不下袖子 → 参考图手臂位置没有袖料
    (alpha=0) → 生成模型只能画裸臂 → 出图长袖变短袖/无袖。本函数从原平铺图
    裁出左右袖, 转成竖直(袖口朝下), 缩放到手臂宽/长, 合成到画布肩→腕的外侧
    手臂列, 让 IP-Adapter/pattern-lock 在手臂上有真实袖纹。纯 CPU。

    袖子摆放按裁片横宽比自适应:
    - 横向摊开(宽>1.25x高): 旋转 90° 袖口朝下(左袖 CCW, 右袖 CW);
    - 向下折放(高≥宽, 肩端在上袖口在下): 不转, 直接贴。
    外侧袖带若混入不连肩的低位散段(A 字衣摆侧缘), 按"取最靠上(连肩线)的
    连续行段"只保留袖段。RGB 平铺图先补 garment_alpha 再裁, 白底像素
    alpha=0 贴合时为 no-op(修"袖子贴成白块, 模型看不见袖料")。右袖裁片
    坐标从切片相对坐标换回全图坐标需 +tc1。

    竖直范围用 pose 的肩/腕归一化 y(无 pose 时用全身比例 0.20/0.62); 横向按
    全身正面基模标定。短袖只贴肩下 0.40 臂长(肘上)。任何一步算不出合法手臂列
    则原样返回(退回旧行为, 不破坏已有结果)。sleeve_len 为 "long"/"short"。
    """
    if sleeve_len not in ("long", "short"):
        return warped
    out = warped.convert("RGBA").copy()
    cw, ch = out.size
    cx = cw / 2.0

    # ---- 竖直锚点: pose 肩/腕 y, 无 pose 用全身比例 ----
    if pose:
        sh = pose.get("shoulder")
        wr = pose.get("wrist")
        shoulder_y = int(sh[1] * ch) if sh and sh[2] >= 0.3 else int(ch * 0.20)
        wrist_y = int(wr[1] * ch) if wr and wr[2] >= 0.3 else int(ch * 0.62)
    else:
        shoulder_y, wrist_y = int(ch * 0.20), int(ch * 0.62)
    if wrist_y - shoulder_y < ch * 0.08:
        return out
    arm_len = wrist_y - shoulder_y
    strip_h = arm_len if sleeve_len == "long" else max(int(ch * 0.05), int(arm_len * 0.40))

    # ---- 横向: 全身正面基模标定的画布比例(手臂列在躯干外侧) ----
    # 基模(全身正面 1024)躯干半宽≈0.145c, 手臂悬在躯干外侧, 手臂宽≈0.05c:
    # 左手臂列=[cx-0.20c, cx-0.15c], 右手臂列=[cx+0.15c, cx+0.20c]。
    arm_w = int(max(10, 0.05 * cw))
    px_l = int(cx - 0.20 * cw)  # 左臂外缘(袖外侧)
    px_r = int(cx + 0.15 * cw)  # 右臂内缘(向外出到 0.20c)
    if px_l < 0:
        px_l = 0
    if px_r + arm_w > cw:
        px_r = cw - arm_w
    if px_l + arm_w > px_r:  # 两臂列重叠, 放不下
        return out

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
        return out
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

    for side, px in (("L", px_l), ("R", px_r)):
        sl = _sleeve_strip(side)
        if sl is None:
            continue
        sl = sl.resize((arm_w, strip_h), Image.Resampling.LANCZOS)
        # 去白底: alpha 太低的像素(背景/抠图残边)置 0, 避免贴进白晕。
        arr = np.array(sl)  # copy: np.asarray 返回只读视图, 无法写回
        al = arr[:, :, 3].astype(np.uint8)
        al[al < 60] = 0
        arr[:, :, 3] = al
        sl = Image.fromarray(arr, "RGBA")
        # 羽化: 只模糊 alpha 通道(不动 RGB), 消掉硬矩形边缘。
        sl.putalpha(sl.getchannel("A").filter(ImageFilter.GaussianBlur(2)))
        out.paste(sl, (px, shoulder_y), sl)
    return out


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


def _get_compliance_advice(engine: str, license_mode: str, sdxl_avail: bool) -> str:
    if license_mode == "commercial" and engine == "sdxl-commercial":
        return (
            "Commercial mode: using SDXL commercial engine; register on the "
            "Stability AI model page and complete attribution."
        )
    if license_mode == "commercial" and engine == "composite-fallback":
        return (
            "Commercial mode: SDXL weights missing, using the in-repo composite engine "
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
                    # SDXL 内部若已标记降级(如 IP-Adapter 加载失败 N4-L6),保留其信息。
                    img.info.setdefault("engine_degraded", False)
                    img.info.setdefault("degraded_reason", "")
                    return img
                except Exception:
                    logger.exception("SDXL engine failed at runtime; falling back to composite engine")
                    degraded = True
                    degraded_reason = "SDXL engine failed at runtime; fell back to composite"
        except ImportError:
            sdxl_reason = "diffusers not installed"
        if sdxl_reason and not degraded:
            logger.warning("commercial SDXL weights unavailable (%s); using in-repo composite engine", sdxl_reason)

    img = _composite_engine(base_path, garment_path, audience, garment_type, long_side, background, ethnicity)
    img.info["engine"] = engine_name
    img.info["engine_degraded"] = degraded
    img.info["degraded_reason"] = degraded_reason
    return img


def engine_status() -> dict:
    avail, reason = _idm_vton_available()
    license_allowed, license_reason = _license_allows_idm()

    sdxl_avail = False
    sdxl_reason = ""
    ip_adapter_avail = False
    try:
        from engine_sdxl import ip_adapter_available, sdxl_available

        sdxl_avail, sdxl_reason = sdxl_available()
        ip_adapter_avail = ip_adapter_available()
    except Exception as e:  # noqa: BLE001
        sdxl_reason = str(e)

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
    elif _LICENSE_MODE == "commercial" and sdxl_avail:
        engine = "sdxl-commercial"
        commercial_compliant = True
    elif _LICENSE_MODE == "commercial" and not sdxl_avail:
        engine = "composite-fallback"
        commercial_compliant = True

    return {
        "idm_vton_available": avail,
        "idm_vton_reason": reason if not avail else "",
        "sdxl_available": sdxl_avail,
        "sdxl_reason": sdxl_reason if not sdxl_avail else "",
        "ip_adapter_available": ip_adapter_avail,
        "gpu_available": gpu_available,
        "engine": engine,
        "license_mode": _LICENSE_MODE or "unset",
        "license_allows_idm": license_allowed,
        "license_reason": license_reason if not license_allowed else "",
        "commercial_compliant": commercial_compliant,
        "compliance_advice": _get_compliance_advice(engine, _LICENSE_MODE, sdxl_avail),
        "weights_dir": str(WEIGHTS_DIR),
        "idm_vton_dir": str(_IDM_DIR),
        "sdxl_dir": str(_REPO_ROOT / "models" / "sdxl"),
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
