"""电商平台店规适配层:不同平台的预处理模板。

平台硬性规则(来自电商后台要求):
- 背景必须纯白 RGB(255,255,255)、无阴影;
- 主体居中、1:1 比例;
- 无文字/水印(生成链路不加水印,模板层不添加);
- 各平台尺寸与文件大小上限不同。

模板定义:
- shopify:    2048x2048, JPEG 控制在 5MB 以内(上限 20MB,5MB 内加载体验更好);
- amazon:     1600x1600(开启缩放需最长边 >=1000), 背景强制纯白;
- aliexpress: 1000x1000(主图 >=800x800), JPEG 控制在 4.5MB 以内(上限 5MB)。
"""

from __future__ import annotations

import atexit
import io
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.platform_presets")

# u2net(主体掩膜)模型放 NAS 持久目录, 换机不丢、免重下; 缺失才让 rembg 自动下载。
_REPO_ROOT = Path(__file__).resolve().parent.parent
_U2NET_ROOT = _REPO_ROOT / "models" / "seg"
_POSE_ROOT = _REPO_ROOT / "models" / "mediapipe"
_RTMPOSE_ROOT = _REPO_ROOT / "models" / "rtmpose"

Platform = Literal["shopify", "amazon", "aliexpress"]

PLATFORM_PRESETS: dict[str, dict] = {
    # size: 输出正方形边长(px);max_bytes: 成品 JPEG 目标体积上限;force_white: 强制纯白底
    "shopify": {"size": 2048, "max_bytes": 5 * 1024 * 1024, "force_white": True},
    "amazon": {"size": 1600, "max_bytes": 20 * 1024 * 1024, "force_white": True},
    "aliexpress": {"size": 1000, "max_bytes": int(4.5 * 1024 * 1024), "force_white": True},
}
PLATFORM_SET = {"none", *PLATFORM_PRESETS}


_HAAR_FACE_CASCADE = "haarcascade_frontalface_default.xml"


def _detect_face_mask(
    arr: np.ndarray,
    margin_frac: float = 0.5,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    down_margin_frac: float | None = None,
) -> np.ndarray:
    """Haar 检脸,返回人脸 bbox 外扩掩膜(uint8 0/1)。

    margin_frac 控制上/左/右外扩;down_margin_frac(默认=margin_frac)控制向下外扩。
    服装遮罩场景传较小 down_margin_frac,避免脸区吞掉脖子/胸口。
    无 cv2 / 级联缺失 / 未检到脸时返回全 0。
    """
    dm = margin_frac if down_margin_frac is None else down_margin_frac
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        import cv2

        gray = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + _HAAR_FACE_CASCADE)
        if cascade.empty():
            return mask
        min_side = max(32, int(0.05 * min(h, w)))
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(min_side, min_side),
        )
    except Exception:  # noqa: BLE001
        return mask
    if len(faces):
        for x, y, fw, fh in faces:
            x, y, fw, fh = int(x), int(y), int(fw), int(fh)
            mx = int(max(fw, fh) * margin_frac)
            md = int(max(fw, fh) * dm)
            x0, y0 = max(0, x - mx), max(0, y - mx)
            x1 = min(w, x + fw + mx)
            y1 = min(h, y + fh + md)
            mask[y0:y1, x0:x1] = 1
    return mask


def _heuristic_face_mask(
    arr: np.ndarray,
    subject_mask: np.ndarray | None,
    frac: float = 0.18,
) -> np.ndarray:
    """cv2 5.0 兜底:用主体 bbox 顶部 frac 比例估算脸区。返回 uint8 0/1。

    cv2 5.0 移除 CascadeClassifier/CASCADE_SCALE_IMAGE, _detect_face_mask 全走
    except 返回全零。此函数用 subject_mask(rembg 前景)的 bbox 顶部估算脸区,
    让 clothing_mask 和 force_white_background 在 cv2 5.0 下仍能保护人脸。
    """
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if subject_mask is None:
        return mask
    sm = np.asarray(subject_mask, dtype=np.uint8) > 0
    if not sm.any():
        return mask
    ys = np.where(sm.any(axis=1))[0]
    if len(ys) == 0:
        return mask
    y_top = int(ys.min())
    subj_h = max(1, int(ys.max()) - y_top)
    face_band = min(h, y_top + int(subj_h * frac))
    mask[:face_band] = sm[:face_band].astype(np.uint8)
    return mask


def _detect_skin(rgb: np.ndarray) -> np.ndarray:
    """YCbCr 肤色检测。返回 uint8 0/255,255=肤色。

    标准 YCbCr 肤色范围:Cb 77-127, Cr 133-173。
    用于 reapply_garment_detail 排除手臂皮肤(防服装纹理贴到皮肤上发白)
    和 _correct_skin_tone 校正 SDXL 生成肤色。

    形态学开运算去散点噪声,闭运算填小孔,保证肤色区连通完整。
    """
    try:
        import cv2

        ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        # OpenCV YCrCb 通道顺序: [Y, Cr, Cb]
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]
        skin = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
        return skin
    except Exception:  # noqa: BLE001
        return np.zeros(rgb.shape[:2], dtype=np.uint8)


def _correct_skin_tone(
    original: Image.Image,
    result: Image.Image,
    cloth_mask: np.ndarray | None = None,
) -> Image.Image:
    """把 result 中肤色像素的色调校正到 original 的肤色色调。

    SDXL 重画手臂皮肤时常丢失原始肤色(发白发灰)。此函数用 YCbCr 肤色检测
    找到两图的皮肤区,计算 original 肤色均值,把 result 肤色向其偏移。
    肤色掩码经 GaussianBlur 羽化,避免校正边界硬切。

    cloth_mask(可选,生成图上新衣遮罩): 长袖会盖住原图手臂位置——原图那里
    是肤色,"原图肤色位置"兜底会把袖子像素强拉成肤色(袖子发黄)。给了遮罩
    就只在非衣服区统计与校正:袖子区域一概排除。

    脸区排除:SDXL 通常正确生成脸部肤色,全局平移会让脸色不自然。
    用前景 bbox 顶部 12% 估算脸区,从校正统计和校正应用两端都排除。
    12% ≈ 头顶到下巴(头高约占全身 ~1/8):脖子(~13-18%)纳入校正 ——
    SDXL 重画的脖子常发白, 必须拉回原图肤色。
    """
    try:
        import cv2
    except ImportError:
        return result

    orig = np.array(original.convert("RGB"), dtype=np.uint8)
    res = np.array(result.convert("RGB"), dtype=np.uint8)
    if orig.shape[:2] != res.shape[:2]:
        res = np.array(
            Image.fromarray(res).resize((orig.shape[1], orig.shape[0]), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

    orig_skin = _detect_skin(orig)
    res_skin = _detect_skin(res)
    if not orig_skin.any():
        return result

    # 排除脸区:用原图前景 bbox 顶部 12% 估算(头顶~下巴)。
    # 发白结果图常整片 >240, 用 result 估脸会得到空脸罩, 把脸也拉偏。
    h, w = res.shape[:2]
    non_white = orig.mean(axis=2) < 240
    face_mask = np.zeros((h, w), dtype=np.uint8)
    if non_white.any():
        ys = np.where(non_white.any(axis=1))[0]
        face_h = max(1, int((ys[-1] - ys[0]) * 0.12))
        face_mask[ys[0] : ys[0] + face_h, :] = 1
    orig_skin = orig_skin * (1 - face_mask)
    res_skin = res_skin * (1 - face_mask)
    if not orig_skin.any():
        return result
    # 新衣遮罩: 校正贴回区内的生成肤色(胸口/手臂两个肤色根因),
    # 衣料本身(非肤色)不拉, 避免袖子发黄。原图肤色位置兜底只用于
    # 非衣料区(发白手臂掉出 YCbCr 窗时仍拉回)。
    if cloth_mask is not None:
        cm = np.asarray(cloth_mask)
        if cm.shape[:2] != (h, w):
            cm = np.array(Image.fromarray(cm.astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR))
        in_cloth = cm > 127
        fallback = orig_skin * (~in_cloth)
        res_skin = np.maximum(res_skin * in_cloth, fallback)
        if not res_skin.any():
            return result
    else:
        res_skin = np.maximum(res_skin, orig_skin)
    if not orig_skin.any():
        return result

    orig_mean = orig[orig_skin > 0].mean(axis=0).astype(np.float32)
    res_mean = res[res_skin > 0].mean(axis=0).astype(np.float32)
    shift = orig_mean - res_mean
    if np.abs(shift).max() < 2:
        return result

    res_f = res.astype(np.float32)
    skin_mask = (res_skin > 0).astype(np.float32)
    skin_mask = cv2.GaussianBlur(skin_mask, (0, 0), 3.0)
    res_f += shift[None, None, :] * skin_mask[..., None]
    return Image.fromarray(np.clip(res_f, 0, 255).astype(np.uint8), "RGB")


def _seam_repair(
    image: Image.Image,
    mask_u8: np.ndarray,
    grow_px: int = 8,
    radius: int = 5,
) -> Image.Image:
    """用 cv2.inpaint (Telea) 平滑 mask 边界接缝。

    blend_tryon 后,mask 边界处原图像素与生成像素的接缝可能有色阶断层
    (胸口白块/领口硬边)。此函数提取 mask 边界带(grow_px 宽),用 Telea
    算法以周边真实像素填充接缝,实现平滑过渡。

    Telea 算法原理:从遮罩边界向内,按到边界距离的逆序填充,每个像素
    取周围已知像素的加权平均。效果类似"用周围肤色涂抹接缝"。
    """
    try:
        import cv2

        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        mask = np.asarray(mask_u8)
        if mask.shape[:2] != arr.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize((arr.shape[1], arr.shape[0]), Image.Resampling.BILINEAR))

        # 边界带: dilate - erode = boundary band (width ≈ 2*grow_px)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px, grow_px))
        dilated = cv2.dilate(mask, k)
        eroded = cv2.erode(mask, k)
        seam = ((dilated > 20) & (eroded < 20)).astype(np.uint8) * 255

        if not seam.any():
            return image

        repaired = cv2.inpaint(arr, seam, radius, cv2.INPAINT_TELEA)
        return Image.fromarray(repaired, "RGB")
    except Exception:  # noqa: BLE001
        return image


_INPAINT_PIPE = None
_INPAINT_LOCK = threading.Lock()


def _build_inpaint_mask(
    image: Image.Image,
    mask_u8: np.ndarray,
    original: Image.Image | None = None,
    grow_px: int = 8,
    anomaly_thresh: int = 30,
    print_roi: np.ndarray | None = None,
) -> np.ndarray:
    """构建聚焦 inpaint 遮罩:接缝带 + 肤色异常区 + 印花 ROI 边界带。

    不再用整张 shared_mask(会重画整个衣服区),改为只针对真正有问题的区域:
    1. 接缝带:mask 边界 dilate-erode 带(同 _seam_repair)
    2. 肤色异常区:result 中皮肤像素与 original 对应像素 L1 距离 > thresh,
       排除脸区(前景顶部 18%)
    3. 印花 ROI 边界带(VTON_PRINT_INPAINT=1 时由 detect_print_roi 提供):
       确定性贴印区(字母/图案内部)不进 inpaint——内部像素由 SKU 平铺图
       像素构成,重绘会把字母再画漂;只有贴印区与生成衣料的边界带进
       局部 inpaint 做混合。
    """
    import cv2

    mask = np.asarray(mask_u8)
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    if mask.shape[:2] != arr.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize((arr.shape[1], arr.shape[0]), Image.Resampling.BILINEAR))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px, grow_px))
    dilated = cv2.dilate(mask, k)
    eroded = cv2.erode(mask, k)
    combined = ((dilated > 20) & (eroded < 20)).astype(np.uint8) * 255

    if print_roi is not None:
        pr = np.asarray(print_roi)
        if pr.ndim == 3:
            pr = pr[..., 0]
        if pr.shape[:2] != combined.shape[:2]:
            pr = np.array(
                Image.fromarray(pr.astype(np.uint8)).resize(
                    (combined.shape[1], combined.shape[0]), Image.Resampling.BILINEAR
                )
            )
        combined = cv2.bitwise_or(combined, (pr > 20).astype(np.uint8) * 255)

    if original is not None:
        orig = np.array(original.convert("RGB"), dtype=np.uint8)
        if orig.shape[:2] == arr.shape[:2]:
            res_skin = _detect_skin(arr)
            if res_skin.any():
                h, w = arr.shape[:2]
                non_white = arr.mean(axis=2) < 240
                face_mask = np.zeros((h, w), dtype=np.uint8)
                if non_white.any():
                    ys = np.where(non_white.any(axis=1))[0]
                    face_h = max(1, int((ys[-1] - ys[0]) * 0.18))
                    face_mask[ys[0] : ys[0] + face_h, :] = 1
                res_skin = res_skin * (1 - face_mask)
                if res_skin.any():
                    diff = np.linalg.norm(arr.astype(np.float32) - orig.astype(np.float32), axis=2)
                    anomaly = (res_skin > 0) & (diff > anomaly_thresh)
                    combined = np.maximum(combined, anomaly.astype(np.uint8) * 255)

    return combined


def _sdxl_inpaint_repair(
    image: Image.Image,
    mask_u8: np.ndarray,
    prompt: str = "natural skin texture, smooth transition, seamless blend, matching surrounding skin tone",
    strength: float = 0.35,
    device=None,
    original: Image.Image | None = None,
    print_roi: np.ndarray | None = None,
) -> Image.Image:
    """SDXL Inpainting 二次修复:低降噪强度局部重绘接缝/肤色异常/印花边界区。

    需要 SDXL Inpainting 模型 (stable-diffusion-xl-1.0-inpainting-0.1)。
    用 StableDiffusionXLInpaintPipeline 对聚焦 mask 区域做低强度(0.35)重绘,
    模型主要基于周边已有像素做微调,而非彻底重画。

    v0.1.7 改进:
    - 遮罩从整张 shared_mask 改为接缝带 + 肤色异常区(_build_inpaint_mask),
      避免重画整个衣服区。
    - 管线单例(_INPAint_PIPE + _INPAint_LOCK),不再每次调用 from_pretrained + del。
    - original 参数传入原图,用于检测肤色异常区。

    印花 ROI(VTON_PRINT_INPAINT=1):
    - print_roi 为 detect_print_roi 输出的印花边界带, 只混合贴印区与生成
      衣料的过渡带, 字母/图案内部像素保持确定性贴印, 不重绘。

    启用条件:
      1. VTON_INPAINT_REPAIR=1 或 VTON_PRINT_INPAINT=1 环境变量开启
      2. SDXL Inpainting 模型已下载到 models/sdxl/sdxl_inpaint/
      3. torch + diffusers 可用

    未满足时原样返回(安全回退,不回归)。
    """
    global _INPAINT_PIPE

    _repair_on = os.environ.get("VTON_INPAINT_REPAIR", "0") == "1"
    _print_on = os.environ.get("VTON_PRINT_INPAINT", "0") == "1"
    if not (_repair_on or _print_on):
        return image

    try:
        import cv2

        focused = _build_inpaint_mask(image, mask_u8, original=original, print_roi=print_roi)
        if not focused.any():
            return image

        inpaint_dir = Path(__file__).resolve().parent.parent / "models" / "sdxl" / "sdxl_inpaint"
        if not (inpaint_dir / "model_index.json").exists():
            logger.warning("VTON_INPAINT_REPAIR=1 but model not found at %s; skipping", inpaint_dir)
            return image

        with _INPAINT_LOCK:
            if _INPAINT_PIPE is None:
                import torch
                from diffusers import StableDiffusionXLInpaintPipeline
                from engine_sdxl import _pipeline_fp16_variant_kwargs

                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                _INPAINT_PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(
                    str(inpaint_dir),
                    torch_dtype=dtype,
                    local_files_only=True,
                    **_pipeline_fp16_variant_kwargs(inpaint_dir),
                )
                if torch.cuda.is_available():
                    _INPAINT_PIPE.enable_model_cpu_offload()
                    _INPAINT_PIPE.vae = _INPAINT_PIPE.vae.to(torch.float32)
            pipe = _INPAINT_PIPE

        mask_grown = cv2.dilate(focused, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))
        mask_blurred = cv2.GaussianBlur(mask_grown, (0, 0), 4.0)
        mask_pil = Image.fromarray(mask_blurred.astype(np.uint8), "L")

        with _INPAINT_LOCK:
            result = pipe(
                prompt=prompt,
                image=image.convert("RGB"),
                mask_image=mask_pil,
                strength=strength,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

        return result
    except Exception:
        logger.debug("_sdxl_inpaint_repair failed; returning unmodified image", exc_info=True)
        return image


def force_white_background(
    img: Image.Image,
    border_band: int = 8,
    grow_dist: int = 250,
    face_guard: bool = True,
    face_margin: float = 0.5,
    subject_mask: np.ndarray | None = None,
) -> Image.Image:
    """把影棚背景压成纯白 (255,255,255),边缘锚定区域生长 + 深色内容保护。

    扩散模型输出的"白底"常是暖米色/中灰影棚,且主体周围有软光箱
    反光形成的**亮光渐变环**(亮度 180~230,比锚点亮 80~100)。
    四个关键坑(实测踩出):
    1) 锚点污染:黑发丝贴画面边缘时,全带中值被发色拉暗,背景判反
       ——黑发反落入候选区被压白。对策:锚点只在**亮像素**(亮度
       >= 85,影棚背景必为浅色)上取中值;亮像素不足退化为纯白锚点。
    2) 光渐变环残留:主体轮廓外一圈背景比锚点亮 ~200(L1 距离),
       90 的阈值盖不住 → 轮廓周围一圈灰色光晕。grow_dist=250
       盖住渐变环;安全靠连通性:模特身上的衣物/皮肤即使入候选,
       也被主体包围、不与画面边缘连通,不会被压白。
    3) 深色内容误伤:候选区连通时可能裹住贴边发丝。对策:亮度
       < 85 的深色像素(头发/深色衣物)及 2px 抗锯齿边保护不动。
    4) 宽亮色通路:发质 1-6px 细缝可被腐蚀切断,但"脸→脖子→浅色衣
       →背景"形成的**宽**通路(整条都在 L1<250 门限内)腐蚀切不断,
       脸部随之被压白(实测面部区纯白占比 0.826→0.754,正常 ~0.53)。
       对策:face_guard 时先 Haar 检脸,把人脸 bbox 外扩 face_margin
       直接豁免压白(保留原肤色);检不到脸回退现状行为,不误伤无脸图。
    5) 浅色主体被啃:深色像素保护(lum<85)救不了浅色发丝/雪纺袖口/
       手指缝隙/头顶高光——它们本就浅,又常与背景同 L1 门限内连通,
       被"边界生长"误压白,身体/头发轮廓缺角(实测 3 张 khaite 换白底后)。
       对策:subject_mask(rembg/u2net 前景)时,主体区 + 2px 抗锯齿边
       无条件保留、只白背景;mask 为 None 时行为与现状完全一致(不回归)。
    """
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    b = min(border_band, h // 4, w // 4)
    band = np.concatenate(
        [
            arr[:b].reshape(-1, 3),
            arr[-b:].reshape(-1, 3),
            arr[:, :b].reshape(-1, 3),
            arr[:, -b:].reshape(-1, 3),
        ]
    )
    lum_band = band.mean(axis=1)
    light = band[lum_band >= 85]
    # 锚点:亮像素中值;边缘全暗(异常)时退化为纯白
    anchor = np.median(light, axis=0) if len(light) > 0 else np.array([255.0, 255.0, 255.0])
    dist = np.abs(arr - anchor).sum(axis=2)
    mask = (dist < grow_dist).astype(np.uint8)
    lum = arr.mean(axis=2)
    dark = (lum < 85).astype(np.uint8)
    # 宽亮色通路保护:检到脸则把外扩框记下来,各出口处对该区豁免压白。
    # face_guard=False(或检不到脸/无 cv2)时全 0,行为与现状完全一致。
    face_mask = _detect_face_mask(arr, margin_frac=face_margin) if face_guard else np.zeros((h, w), dtype=np.uint8)
    # cv2 5.0 fallback: CascadeClassifier 被移除,_detect_face_mask 返回全零。
    # 用 subject_mask bbox 顶部 18% 估算脸区,防止胸口/脖子被压白。
    if face_guard and not face_mask.any() and subject_mask is not None:
        face_mask = _heuristic_face_mask(arr, subject_mask)
    try:
        import cv2

        k3 = np.ones((3, 3), np.uint8)
        # 切断 1-6px 亮缝(发质内部断丝/抗锯齿亮缝):不腐蚀时,亮色"脸"区
        # 会经细缝与边缘连通背景打通 → 整张脸落入选区被压白(实测回归)。
        # 腐蚀 3 次后,只有"宽"背景通路保留连通性;细缝被切断,被深色内容
        # 包围的亮区(脸/亮色衣物)孤立,不再被压白。
        m_er = cv2.erode(mask, k3, iterations=3)
        _num, labels = cv2.connectedComponents(m_er)
        border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        border = [int(v) for v in border if v > 0]
        if border:
            keep = np.isin(labels, border).astype(np.uint8)
            # 映射回原候选像素:腐蚀每边削 3px,dilate 3 次还原宽度,再与原
            # mask 相与防止越界扩张
            keep = cv2.dilate(keep, k3, iterations=3) & mask
            # 画面外圈 4px 例外:候选像素无条件压白。主体与画面边缘之间的
            # 细亮缝(<7px)腐蚀后连通域会死,没有这条外圈兜底会在缝里
            # 留下色偏残留
            ring = np.zeros_like(mask)
            ring[:4, :] = 1
            ring[-4:, :] = 1
            ring[:, :4] = 1
            ring[:, -4:] = 1
            keep = keep | (ring & mask)
            keep = cv2.dilate(keep, k3, iterations=1)
            if dark.any():
                # 深色内容(头发/深色衣物)+ 2px 抗锯齿边保护:不动原色
                protect = cv2.dilate(dark, k3, iterations=2)
                keep = keep & (1 - protect)
            if subject_mask is not None:
                # 主体掩膜(rembg/u2net 前景):主体区 + 2px 抗锯齿边绝不压白,
                # 治浅色发丝/袖口/手指缝隙/头顶高光被当背景啃掉(坑 5)。
                sm = np.asarray(subject_mask, dtype=np.uint8) > 0
                if sm.any():
                    keep = keep & (~cv2.dilate(sm.astype(np.uint8), k3, iterations=2)).astype(np.uint8)
            out = np.where(keep[..., None] > 0, 255, arr)
            if face_mask.any():
                # 人脸区豁免:无条件保留原色(脸/脖子不随背景压白)
                out = np.where(face_mask[..., None] > 0, arr, out)
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    except ImportError:  # 无 cv2 时退化为阈值压白(不做连通/保护)
        keep2 = (mask & (1 - dark)) if dark.any() else mask
        if subject_mask is not None:
            keep2 = keep2 & (~(np.asarray(subject_mask, dtype=np.uint8) > 0)).astype(np.uint8)
        out = np.where(keep2[..., None] > 0, 255, arr)
        if face_mask.any():
            out = np.where(face_mask[..., None] > 0, arr, out)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if subject_mask is not None:
        keep_all = mask & (~(np.asarray(subject_mask, dtype=np.uint8) > 0)).astype(np.uint8)
        out = np.where(keep_all[..., None] > 0, 255, arr)
    else:
        out = np.where(mask[..., None] > 0, 255, arr)
    if face_mask.any():
        out = np.where(face_mask[..., None] > 0, arr, out)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


_FG_SESSION = None
_FG_SESSION_WARNED = False
# rembg 失败短路缓存(镜像 _RTMPOSE_LOAD_ERROR, pipeline audit 行34):
# 坏掉时 _FG_SESSION 恒 None, 不短路则每张图都重进失败块重试 new_session(批量拖慢)
# 且 warning 仅首遍。置位后直接返回 None, 不再重试。
_FG_SESSION_ERROR: str | None = None

_POSE_SESSION = None
_POSE_SESSION_WARNED = False


def _u2net_onnx_path():
    """找本地完整 u2net.onnx(NAS 模型目录 > U2NET_HOME 两种布局 > rembg 默认缓存)。

    找不到就**不**触发 rembg 的 GitHub 下载(本环境很慢、可能卡好几分钟), 直接回退
    纯颜色压白。要启用主体掩膜: 把 u2net.onnx(~176MB) 放到 repo/models/seg/ 即生效。
    只认 >10MB 的文件, 避免误用残缺的临时下载。
    """
    import os as _os

    u2home = Path(_os.environ.get("U2NET_HOME", "~/.u2net")).expanduser()
    candidates = [
        _U2NET_ROOT / "u2net.onnx",  # NAS: repo/models/seg/
        u2home / "u2net.onnx",  # U2NET_HOME 扁平(legacy)
        u2home / "models" / "u2net" / "u2net.onnx",  # U2NET_HOME 新版布局
    ]
    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size > 10_000_000:
                return c
        except OSError:
            continue
    return None


def foreground_mask(
    img: Image.Image,
    min_foreground_frac: float = 0.05,
) -> np.ndarray | None:
    """主体(前景)掩膜:(h,w) uint8,1=主体(保留),0=背景(可压白)。

    用 rembg(u2net)推理前景。任一环节不可用(未装 rembg / 模型缺失 /
    前景占比过低像无主体 / 推理异常)都返回 None,调用方据此回退到纯颜色
    压白(行为不回归)。模型会话全局缓存,避免每张重新加载。

    这就是治"换白底后身体/头发被啃"的开关:把本函数返回的 mask 传给
    force_white_background(subject_mask=...),主体区就再不会被压白。
    """
    global _FG_SESSION, _FG_SESSION_WARNED, _FG_SESSION_ERROR
    # 失败短路(行34):上次 session 初始化已失败, 不再每张重试。
    if _FG_SESSION_ERROR is not None:
        return None
    try:
        from rembg import new_session, remove
    except Exception:  # noqa: BLE001
        return None
    model_path = _u2net_onnx_path()
    if model_path is None:
        # 本地无完整 u2net.onnx → 不触发(可能很慢的)下载, 直接回退纯颜色压白(快)。
        if not _FG_SESSION_WARNED:
            logger.warning("未找到 u2net.onnx(放 repo/models/seg/ 或预下载);force_white 回退纯颜色压白")
            _FG_SESSION_WARNED = True
        return None
    if _FG_SESSION is None:
        import os as _os

        # rembg 靠 U2NET_HOME 找本地 onnx(没有 download_root 形参)。把模型所在目录指过去,
        # 就不再碰 GitHub。多核机器上 u2net 这种小模型若不限线程, onnxruntime 自动开满
        # 线程反而拖死 → 显式限 intra=8/inter=1。
        _os.environ.setdefault("U2NET_HOME", str(model_path.parent))
        try:
            so = None
            try:
                import onnxruntime as _ort

                so = _ort.SessionOptions()
                so.intra_op_num_threads = 8
                so.inter_op_num_threads = 1
            except Exception:  # noqa: BLE001
                so = None
            # 定向 TypeError 重试(行17/35): 旧报告所称的修复其实不存在——原代码是单次
            # 带 sess_opts 调用 + 宽 except 直接放弃整个 rembg。此处仅对 sess_opts 不被
            # 接受(TypeError)重试一次不带 kwarg 的调用; 宽 except 仍留作最后兜底。
            if so is not None:
                try:
                    _FG_SESSION = new_session("u2net", sess_opts=so)
                except TypeError:
                    logger.warning("rembg new_session 不接受 sess_opts;重试不带该参数")
                    _FG_SESSION = new_session("u2net")
            else:
                _FG_SESSION = new_session("u2net")
        except Exception as e:  # noqa: BLE001
            _FG_SESSION_ERROR = str(e)
            if not _FG_SESSION_WARNED:
                logger.warning(
                    "rembg u2net session 初始化失败(已短路缓存, 不再逐图重试);force_white 回退纯颜色压白: %s", e
                )
                _FG_SESSION_WARNED = True
            return None
    try:
        cut = remove(img, session=_FG_SESSION)
        alpha = np.array(cut.convert("RGBA"))[..., 3]
        m = (alpha > 127).astype(np.uint8)
        if m.mean() < min_foreground_frac:
            return None  # 几乎全前景/异常,不信任
        return m
    except Exception as e:  # noqa: BLE001
        # remove() 推理失败(行17): 原代码无日志完全静默。补 warning(仍回退纯颜色压白)。
        logger.warning("rembg remove() 推理失败;本张回退纯颜色压白: %s", e)
        return None


def detect_garment_extent(garment: Image.Image) -> dict:
    """从平铺衣料图分析袖长和衣长。

    返回:
        sleeve: "long" | "short" | "sleeveless"
        hem: "waist" | "mid" | "long"
    用于动态 clothing_mask:长袖→mask 含手臂,短袖→只含大臂,无袖→不含手臂。

    袖翼判定(逐行连通段, 含 bbox 水平中点的一段是躯干):
    - 侧垂袖: 断开在身侧、与躯干留缝的衣料段; 段宽须 >= 12% 衣宽
      (裹身衣的绑带/腰带只有 ~10%, 会被宽度过滤掉), 且袖尖不垂过
      衣身下摆(躯干底行 + 容差) —— 长过下摆的段外内容是绑带/腰带。
    - 连肩袖: 行宽超躯干宽 25% 且在衣身上半 35%(肩周加宽)。
      A 字裙摆与躯干连成一段且在下半区, 不会被误判成袖;
      高瘦直筒无侧翼内容 → sleeveless(旧 aspect>1.35 规则的误判源,已删)。
    袖长基准(不能用整衣长的固定比例): 衣长从齐腰到及踝变化很大,
    "垂过衣长 38% 即长袖"在长款上会把短袖帽误判成长袖。改按衣长分级
    估"肩→腰"基准距(人体比例近似): waist=1.0ch, mid=0.55ch, long=0.40ch,
    袖尖超过基准 70% 才判长袖。
    衣长用躯干中段连通段宽(不含垂袖)算高宽比, 宽袖不压衣长。
    """
    arr = np.array(garment.convert("RGBA"))
    _h, _w = arr.shape[:2]
    alpha = arr[..., 3]
    has_transp = int(alpha.min()) < 200

    if has_transp:
        content = alpha > 30
    else:
        rgb = arr[..., :3].astype(np.float32)
        corners = np.median(rgb[:24, :24].reshape(-1, 3), axis=0)
        content = np.abs(rgb - corners).sum(axis=2) > 40

    ys, xs = np.where(content)
    if len(ys) < 64:
        return {"sleeve": "short", "hem": "waist", "neckline": "closed"}

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    ch = max(1, y1 - y0)
    cw = max(1, x1 - x0)
    aspect = ch / cw

    # 逐行连通段:含 bbox 水平中点的一段是躯干,段外内容是垂袖;
    # A 字裙摆与躯干连成一段,不会被误判成袖。高瘦直筒整行是一段 → 无袖翼。
    cx = (x0 + x1) // 2 - x0
    n_rows = ch + 1
    row_span = np.zeros(n_rows, dtype=np.float32)
    central_w = np.zeros(n_rows, dtype=np.float32)
    gap_wing = np.zeros(n_rows, dtype=bool)
    wing_w_min = max(3.0, cw * 0.12)
    for i in range(n_rows):
        xs_row = np.where(content[y0 + i, x0 : x1 + 1])[0]
        if xs_row.size == 0:
            continue
        row_span[i] = xs_row[-1] - xs_row[0] + 1
        breaks = np.where(np.diff(xs_row) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [xs_row.size - 1]))
        run_lo = xs_row[starts]
        run_hi = xs_row[ends]
        widths = run_hi - run_lo + 1
        contains = (run_lo <= cx) & (cx <= run_hi)
        if contains.any():
            j = int(np.argmax(np.where(contains, widths, -1)))
        else:
            j = int(np.argmin(np.abs((run_lo + run_hi) * 0.5 - cx)))
        cl = int(run_lo[j])
        cr = int(run_hi[j])
        central_w[i] = cr - cl + 1
        outside = int((xs_row < cl).sum() + (xs_row > cr).sum())
        # 段外连通段须够宽(>=12% 衣宽): 绑带/腰带只有 ~10%, 与袖筒区分。
        outer_w = widths[(run_hi < cl) | (run_lo > cr)]
        gap_wing[i] = outside >= 3 and (float(outer_w.max()) if outer_w.size else 0.0) >= wing_w_min

    # 袖尖不垂过衣身下摆: 超过躯干底行的段外内容是绑带/腰带, 一律不算袖。
    has_central = central_w > 0
    hem_row = int(np.where(has_central)[0][-1]) if has_central.any() else n_rows - 1
    row_idx = np.arange(n_rows)
    gap_wing &= row_idx <= hem_row + max(3, ch // 50)

    mid_lo = ch // 3
    mid_hi = min(2 * ch // 3 + 1, n_rows)
    seg = central_w[mid_lo:mid_hi]
    torso_w = float(seg.max()) if seg.size else float(cw)
    if torso_w < 1:
        torso_w = float(cw)

    # 衣身高用"衣身本体"行数: 中心段宽 >= 45% 躯干宽的最后一行 + 1。
    # 整 bbox 高会被垂到底的细长内容撑大 —— 裹身衣垂带/长腰带挂在 bbox
    # 中心附近时, 逐行中心段一直存在(04469 实测: 绑带把 ch 撑到躯干宽的
    # 2.1 倍 → 中长款误判); 宽袖/裙摆不受影响(那些段本身够宽)。
    # 一行都不达阈值时按整 bbox 高兜底(高瘦直筒整行一段, 等价旧行为)。
    _body_wide = central_w >= torso_w * 0.45
    body_rows = int(np.where(_body_wide)[0][-1]) + 1 if _body_wide.any() else n_rows
    body_ratio = body_rows / float(n_rows)

    # 衣身高 / 躯干宽:宽袖不再把整体 aspect 拉扁。
    # 短款 ≈1.0-1.4, 中长款(髋/大腿)≈1.6-2.6, 裙级≈3.0+。
    # long 旁路同改 body_rows(垂带把 aspect 撑过 2 的假"长款"不再触发)。
    cratio = body_rows / max(torso_w, 1.0)
    if cratio > 3.0 or body_rows > 2.0 * max(cw, 1):
        hem = "long"
    elif cratio >= 1.55:
        hem = "mid"
    else:
        hem = "waist"

    # 袖:侧垂袖(任何高度,已滤绑带)或上半区行宽超躯干 25%(连肩短袖/落肩)。
    width_wing = (row_span > torso_w * 1.25) & (row_idx < n_rows * 0.35)
    wing_rows = gap_wing | width_wing
    if not wing_rows.any():
        sleeve = "sleeveless"
        drop_px = 0.0
        sleeve_ratio = 0.0
    else:
        drop_px = float(np.where(wing_rows)[0][-1])
        # 袖长基准:肩→腰距离按衣长分级估计,袖尖过基准 70% 才算长袖。
        ref_px = ch * {"waist": 1.0, "mid": 0.55, "long": 0.40}[hem]
        sleeve_ratio = drop_px / max(ref_px, 1.0)
        sleeve = "long" if sleeve_ratio >= 0.70 else "short"

    # 领口开口度: 平铺图拍法(背部片垫在领口后)使 V/开口领的中央上部仍被
    # 衣料填实, 像素级判不开口。默认 closed; 要开孔需显式传 neckline="open"。
    neckline = "closed"

    return {
        "sleeve": sleeve,
        "hem": hem,
        "aspect": round(aspect, 2),
        "neckline": neckline,
        "central_ratio": round(cratio, 2),
        "sleeve_drop": round(drop_px / max(n_rows - 1, 1), 2),
        "sleeve_ratio": round(sleeve_ratio, 2),
        # 衣身本体行占 bbox 高比例: auto 类型推断用它区分"真中长款"(衣身
        # 宽到底)和"垂带撑高的短款"(04469: 0.84)。
        "body_ratio": round(body_ratio, 2),
    }


_RTMPOSE_BACKBONE = None
_RTMPOSE_HEAD = None
_RTMPOSE_INPUT_WH = (192, 256)  # (W, H)
_RTMPOSE_SPLIT_RATIO = 2.0
_RTMPOSE_LOAD_ERROR: str | None = None
_WAIST_SOURCE_COUNTS: dict[str, int] = {"rtmpose": 0, "mediapipe": 0, "contour": 0, "default": 0}

# fake deps 注入: RTMPOSE_FAKE_DEPS=0 时跳过 (真实 mmcv/mmdet 已装时不需要).
# 注入时机: 首次调用 _init_rtmpose 时, 在 mmpose import 之前.
# 升级风险: mmpose 版本变动可能新增 import 路径, 此 fake 可能失效.
_RTMPOSE_FAKE_DEPS = os.environ.get("RTMPOSE_FAKE_DEPS", "1") == "1"
_FAKE_DEPS_INJECTED = False

# MediaPipe PoseLandmarker (tasks API) 单例缓存 + 加载错误短路.
_POSE_LANDMARKER = None
_POSE_LANDMARKER_ERROR: str | None = None


def _module_importable(name: str) -> bool:
    """逐模块在场判断(pipeline audit 行27/28): 真的能 import 就不注 stub.

    用 importlib.util.find_spec 判断, 不实际执行模块(避免真 mmdet/mmcv.ops 的
    重依赖被拉起来). name 缺失或 import 失败 → False → 才注入 stub。
    """
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _inject_fake_deps() -> None:
    """注入假 xtcocotools/mmdet(.utils)/mmcv.ops(/.transforms) 模块, 绕过 mmpose import 链.

    边界(2026-11-05 盒子实测 + mmpose 1.3.2 源码 grep 定案, 详见 box_probe_evidence.md):
    这些 stub **只为 mmpose 的 import 链挡枪 —— 管线没有任何真实检测任务, 没有 mmdet
    检测器**。RTMPose 是纯姿态估计(输入 bbox), 推理路径不碰 mmdet 的功能代码; mmdet
    只在 mmpose 的模块加载时被 import(下面列出的那一处), stub 挡的是 import 时元数据/
    模块存在性, 不是运行时调用。因此:
      * 禁止把真 mmdet 装回来"用"——本环境 mmdet 永远卸载(装真 mmdet 会因要求 mmcv<2.2.0
        与 mmcv-lite 2.2.0 冲突, 且真 mmdet 的模块会绕过 stub 导致 mmpose import 链连锁断)。
      * 真 mmcv 必须装(mmcv-lite): mmpose/__init__.py 在 import 时硬断言
        2.0.0rc4 <= mmcv.__version__ <= 3.0.0 并 import mmcv.cnn 全家桶, stub 面太大不可取。
      * 仅 mmcv.ops(C 扩展, lite 版没编)做恒等 stub: RTMPose-m 推理路径(cspnext+rtmcc)
        grep 确认 0 次引用 mmcv.ops, 它只在 dekr/edpose/posewarper 等其它 head 里出现。

    mmpose 1.3.2 源码 grep: 包内 Python 代码对 mmdet 的全部引用仅 3 处——
      1. models/heads/hybrid_heads/rtmo_head.py:9  `from mmdet.utils import ConfigType, reduce_mean`
         ← **唯一模块顶层硬 import**, 加载 mmpose.models 即触发, 需 stub mmdet.utils。
      2. apis/inferencers/base_mmpose_inferencer.py `try: from mmdet.apis.det_inferencer
         import DetInferencer except: has_mmdet=False` ← 守卫降级, mmdet 缺席时 has_mmdet=False。
      3. utils/config_utils.py:16 `from mmdet.datasets import transforms` ← 函数内 lazy(源码注释
         明写 "use lazy import to avoid hard dependence on mmdet"), RTMPose 路径不触发。
    无模块加载时读取 mmdet.__version__ / mmdet.apis 存在性检查。故 stub 面 = mmdet.utils
    {ConfigType, reduce_mean} 已是最小充分; 冒烟 `import mmpose; from mmpose.apis import
    inference_topdown` 在 stub 下走通且 base_mmpose_inferencer.has_mmdet==False(stub 形状匹配)。

    行27/28 修法: **逐模块独立判断**, 只在真实 import 不可用时才注入 stub,
    真实 xtcocotools 在场不再短路其余 stub 的注入(旧代码检测到它就直接 return,
    缺 mmdet 的机器上 RTMPose 静默死)。RTMPOSE_FAKE_DEPS=0 关闭注入。

    2026-11-06 盒上 g4 实测修正(根因): mmcv-lite 随附**纯 Python 的 mmcv.ops 包** ——
    find_spec("mmcv.ops") 能命中, 但其真实 __init__ → active_rotated_filter →
    ext_loader.load_ext("_ext") → `import mmcv._ext`(C 扩展, lite 版未编译) →
    ModuleNotFoundError。逐模块 find_spec 判断会把这种"盘上存在、import 即崩"的
    mmcv.ops 误放行 → stub 被跳过 → mmpose.models 的 import 链(EDPoseHead 顶层
    硬 import mmcv.ops)整体崩 → _init_rtmpose 失败 → pose 退轮廓法 → 四缺陷模式
    (胸口压白/手臂发灰/下摆硬切/图案断裂)同时回归。
    **定案: RTMPOSE_FAKE_DEPS=1 时对 mmdet(.utils)/mmcv.ops/mmcv._ext 三条路径
    无条件注册 stub, 不再逐模块 find_spec; 开关关闭时保持原行为(不注入)。**
    边界: 面已被真实模块 import 进 sys.modules 的不夺取(先到先得, 防混合态)。
    mmcv.ops/_ext 的 stub 带 PEP 562 __getattr__ 占位, `import X` 与
    `from X import Y` 两种引用形式同时命中(load_ext 的 hasattr 断言在占位下也过)。
    xtcocotools / mmcv.transforms(.utils) 保持逐模块 find_spec: lite 版随附的
    纯 Python transforms 可 import、无 _ext 地雷(2026-11-06 g4 实测 import 链
    未在 transforms 面崩)。

    注入后断言确认 fake 生效。
    """
    global _FAKE_DEPS_INJECTED
    if _FAKE_DEPS_INJECTED or not _RTMPOSE_FAKE_DEPS:
        return
    import types as _t

    class _F:
        def __init__(self, *a, **kw):
            pass

        def __call__(self, *a, **kw):
            return _F()

        def __getattr__(self, n):
            return _F()

    # --- xtcocotools: 真包在场(含编译扩展)就不注, 否则注入空壳(行28) ---
    if not _module_importable("xtcocotools") and "xtcocotools" not in sys.modules:
        fake = _t.ModuleType("xtcocotools")
        fake.__path__ = []
        fake.coco = type("M", (), {"COCO": _F})
        fake.mask = type("M", (), {"mask": lambda *a, **kw: None})
        fake.cocoeval = type("M", (), {"COCOeval": _F})
        sys.modules["xtcocotools"] = fake
        sys.modules["xtcocotools.coco"] = fake.coco
        sys.modules["xtcocotools.mask"] = fake.mask
        sys.modules["xtcocotools.cocoeval"] = fake.cocoeval

    # --- mmdet / mmdet.utils: 开关打开 → 无条件 stub, 不查 find_spec(2026-11-06 定案) ---
    # 真 mmdet 已被 import 进 sys.modules 的不夺取(先到先得); 未被 import 时即使盘上有
    # 真包也注 stub —— mmpose import 链只用 mmdet.utils{ConfigType, reduce_mean} 占位面,
    # 管线无 mmdet 检测任务, 真/假在此面等价(box_probe_evidence.md §1)。
    if "mmdet.utils" not in sys.modules:
        md = sys.modules.get("mmdet")
        if md is None:
            md = _t.ModuleType("mmdet")
            md.__path__ = []
            # 测试不变式(backend/tests/test_platform_presets.py:889): ConfigType is dict
            md.ConfigType = dict
            md.reduce_mean = lambda *a, **kw: None
            sys.modules["mmdet"] = md
        elif not hasattr(md, "ConfigType"):
            md.ConfigType = dict
            md.reduce_mean = lambda *a, **kw: None
        mu = _t.ModuleType("mmdet.utils")
        mu.ConfigType = dict
        mu.reduce_mean = lambda *a, **kw: None
        sys.modules["mmdet.utils"] = mu

    # --- mmcv.ops: 开关打开 → 无条件 stub(2026-11-06 定案, 见 docstring 根因) ---
    # 不再用 find_spec: mmcv-lite 的纯 Python mmcv.ops 盘上存在但 import 即崩(缺 _ext)。
    # MSDA 是 mmpose import 链唯一模块顶层需要的名字(edpose_head L14 /
    # deformable_detr_layers L7); DeformConv2d 等只在函数级惰性 import, RTMPose 路径
    # 0 次调用。占位类在实例化/调用时抛 NotImplementedError(带调用栈): 若有真实计算
    # 路径走到 stub, 立刻显式失败, 而不是拿到静默假数据(2026-11-06 定案)。
    if "mmcv.ops" not in sys.modules:
        ops = _t.ModuleType("mmcv.ops")
        ops.__path__ = []

        def _ops_placeholder(name: str, base=object):
            def _raise(self, *a, **kw):
                raise NotImplementedError(
                    f"mmcv.ops.{name} is a no-op stub (RTMPOSE_FAKE_DEPS=1, 无 C 扩展); "
                    f"real op was called — pipeline should not reach here"
                )

            return type(name, (base,), {"__init__": _raise, "__call__": _raise})

        ops.MultiScaleDeformableAttention = _ops_placeholder("MSDA")

        def _ops_getattr(n):
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)  # dunder 不走占位
            # PEP 562: `from mmcv.ops import X` import 时命中占位类; 真被实例化才抛。
            return _ops_placeholder(n)

        ops.__getattr__ = _ops_getattr
        sys.modules["mmcv.ops"] = ops

    # --- mmcv._ext: load_ext 的 import 目标; lite 版未编译 → 无条件注册裸 stub ---
    # 显式属性名单定案: load_ext('_ext', [funcs...]) 的 hasattr 断言只在真实 mmcv.ops
    # 子模块被 import 时执行 —— 而 mmcv.ops 已被上面的 stub 门控, 永不执行 → 链内需要
    # 的显式名单为空。故不做 catch-all __getattr__(会吞拼写错误), 属性访问走默认
    # AttributeError → 任何真实使用立刻显式失败。`import mmcv._ext` /
    # `importlib.import_module("mmcv._ext")` 走 sys.modules 查找, 命中即返回。
    if "mmcv._ext" not in sys.modules:
        ext = _t.ModuleType("mmcv._ext")
        ext.__path__ = []  # 与其余 stub 一致: _is_injected_stub 识别面
        ext.__doc__ = (
            "no-op stub registered by platform_presets._inject_fake_deps "
            "(RTMPOSE_FAKE_DEPS=1): mmcv-lite ships no _ext C extension."
        )
        sys.modules["mmcv._ext"] = ext

    # mmcv.transforms 是 mmcv 2.x 才有的子包;老环境(mmcv 1.7)缺它而 mmpose 的
    # RTMPose config 链会 import 它。lite 版随附的纯 Python transforms 可 import
    # (无 _ext 地雷, 2026-11-06 实测), 保持逐模块 find_spec 判断(行27/28)。
    if not _module_importable("mmcv.transforms") and "mmcv.transforms" not in sys.modules:
        tfs = _t.ModuleType("mmcv.transforms")
        tfs.__path__ = []

        def _tfs_getattr(n):
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)  # dunder 不走 stub, 否则 inspect/getattr 拿到 _F
            return _F  # PEP 562: BaseTransform/LoadImageFromFile 等

        tfs.__getattr__ = _tfs_getattr
        sys.modules["mmcv.transforms"] = tfs
        # utils: cache_randomness/avoid_cache_randomness 是装饰器, 给恒等实现
        tfs_utils = _t.ModuleType("mmcv.transforms.utils")
        tfs_utils.__path__ = []

        def _ident(*a, **kw):
            return lambda f: f

        tfs_utils.cache_randomness = _ident
        tfs_utils.avoid_cache_randomness = _ident
        sys.modules["mmcv.transforms.utils"] = tfs_utils

    _FAKE_DEPS_INJECTED = True
    # 断言 fake 生效(2026-11-06 三条路径无条件条款): mmdet(.utils)/mmcv.ops/mmcv._ext
    # 注入后必在 sys.modules(本轮注册的 stub, 或先导入的真实模块 —— 都算合规, 缺席即 FAIL)。
    for _face in ("mmdet", "mmdet.utils", "mmcv.ops", "mmcv._ext"):
        assert _face in sys.modules, f"fake deps injection failed: {_face}"
    # 保持逐模块 find_spec 的面(xtcocotools/mmcv.transforms 家族): 真实 import 不可用时
    # 才断言必须落进 sys.modules —— 本机(真 mmcv/真 xtcocotools)与盒子(mmcv-lite)
    # 两类环境都成立, 且缺面的机器上仍会 fail-loud。
    for _face in ("xtcocotools", "mmcv.transforms", "mmcv.transforms.utils"):
        if not _module_importable(_face):
            assert _face in sys.modules, f"fake deps injection failed: {_face}"


def _locate_rtmpose_config() -> Path:
    """定位 RTMPose-m 的 .mim config(pipeline audit 行13).

    旧代码把回退路径硬编码成 /usr/local/lib/python3.11/dist-packages/..., GPU 机
    (py3.10/conda env/venv)不存在 → 回退失效 → RTMPose 加载失败 → 腰线退轮廓法。
    现按优先级动态定位:
      1. repo: models/rtmpose/<cfg>.py(NAS 持久目录, 首选)
      2. mmpose.__file__ 推出的 <pkg>/.mim/configs/...(pip 装的 mmpose 1.3.2 带全 configs)
      3. importlib.util.find_spec("mmpose") 拿 submodule_search_locations 兜底
      4. sys.prefix / site-packages 下的 glob 兜底(任意 py 版本)
    找不到返回 repo 路径(让 Config.fromfile 抛错, 走 _RTMPOSE_LOAD_ERROR 短路)。
    """
    rel = Path("body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py")
    repo_cfg = Path(_RTMPOSE_ROOT) / "rtmpose-m_8xb256-420e_coco-256x192.py"
    if repo_cfg.exists():
        return repo_cfg

    candidates: list[Path] = []
    try:
        import mmpose  # 已由 _inject_fake_deps 保证可 import

        pkg_dir = Path(mmpose.__file__).resolve().parent
        candidates.append(pkg_dir / ".mim" / "configs" / rel)
    except Exception:  # noqa: BLE001
        pass
    try:
        import importlib.util

        spec = importlib.util.find_spec("mmpose")
        if spec is not None and spec.submodule_search_locations:
            for loc in list(spec.submodule_search_locations):
                candidates.append(Path(loc) / ".mim" / "configs" / rel)
    except Exception:  # noqa: BLE001
        pass

    for c in candidates:
        try:
            if c.is_file():
                return c
        except OSError:
            continue

    # 最后兜底: 在已知 site 目录里 glob(不限 py3.11)。
    roots = [Path(sys.prefix)]
    try:
        import site

        roots.extend(Path(p) for p in site.getsitepackages())
        roots.append(Path(site.getusersitepackages()))
    except Exception:  # noqa: BLE001
        pass
    for root in roots:
        try:
            hits = list(root.glob(f"mmpose*/.mim/configs/{rel}"))
        except OSError:
            hits = []
        if hits:
            return hits[0]

    logger.warning("RTMPose .mim config not found in %s; RTMPose will fail to load", [str(c) for c in candidates])
    return repo_cfg


def _init_rtmpose():
    """惰性加载 RTMPose-m backbone + head. 成功返回 (backbone, head), 否则 None."""
    global _RTMPOSE_BACKBONE, _RTMPOSE_HEAD, _RTMPOSE_SPLIT_RATIO, _RTMPOSE_LOAD_ERROR
    if _RTMPOSE_BACKBONE is not None:
        return _RTMPOSE_BACKBONE, _RTMPOSE_HEAD
    if _RTMPOSE_LOAD_ERROR is not None:
        return None
    ckpt_path = Path(_RTMPOSE_ROOT) / "rtmpose-m_coco.pth"
    if not ckpt_path.exists():
        return None
    try:
        _inject_fake_deps()
        if _RTMPOSE_FAKE_DEPS and not _FAKE_DEPS_INJECTED:
            raise ImportError("RTMPOSE_FAKE_DEPS=1 but injection failed")
        from mmengine.config import Config
        from mmpose.models.backbones.cspnext import CSPNeXt
        from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead

        cfg_path = _locate_rtmpose_config()
        cfg = Config.fromfile(str(cfg_path))
        codec = cfg.get("codec", {}) if hasattr(cfg, "get") else {}
        split = codec.get("simcc_split_ratio") if isinstance(codec, dict) else None
        if split is None:
            split = cfg.model.head.get("simcc_split_ratio", _RTMPOSE_SPLIT_RATIO)
        _RTMPOSE_SPLIT_RATIO = float(split)
        _skip = {"type", "_scope_", "init_cfg"}
        backbone = CSPNeXt(**{k: v for k, v in cfg.model.backbone.items() if k not in _skip})
        head = RTMCCHead(**{k: v for k, v in cfg.model.head.items() if k not in _skip | {"loss", "decoder"}})

        import torch as _torch

        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        backbone.load_state_dict(
            {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}, strict=True
        )
        head.load_state_dict({k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}, strict=False)
        backbone.eval()
        head.eval()
        _RTMPOSE_BACKBONE = backbone
        _RTMPOSE_HEAD = head
        logger.info("RTMPose-m loaded (simcc_split_ratio=%.1f)", _RTMPOSE_SPLIT_RATIO)
        return backbone, head
    except Exception as e:  # noqa: BLE001
        _RTMPOSE_LOAD_ERROR = str(e)
        logger.error(
            "RTMPose init failed (checkpoint present); not falling back to MediaPipe: %s",
            e,
        )
        return None


def _detect_pose_keypoints(img: Image.Image) -> dict | None:
    """关键点检测, 返回 hip/shoulder/knee/wrist(+wrist_l/wrist_r) 归一化坐标.

    优先 RTMPose-m (COCO 17 关键点, CPU 可跑, 精度优于 MediaPipe),
    回退 MediaPipe Pose, 再回退 None.

    Returns:
        {"hip": (x, y, vis), "shoulder": ..., "knee": ..., "wrist": (x, y, vis),
         "wrist_l": (x, y, vis), "wrist_r": (x, y, vis)} 或 None.
        坐标归一化到 [0,1], vis 为 visibility 置信度.
        wrist 供长袖手臂走廊用(衣袖区保留到腕部); wrist_l/r 供袖条
        按左右臂各自腕点定袖长(左右臂常不等高); 缺失时走廊逻辑静默跳过。
    """
    ckpt = Path(_RTMPOSE_ROOT) / "rtmpose-m_coco.pth"
    if ckpt.exists():
        result = _detect_pose_rtmpose(img)
        if result is not None:
            result["_source"] = "rtmpose"
            return result
        # checkpoint 在但加载/推理失败: 不混用 MediaPipe, 让腰线走轮廓法
        return None
    result = _detect_pose_mediapipe(img)
    if result is not None:
        result["_source"] = "mediapipe"
    return result


def _simcc_decode(
    sx: np.ndarray,
    sy: np.ndarray,
    bbox: tuple[int, int, int, int],
    img_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """SimCC argmax 解码 + crop→原图坐标逆变换 + 归一化到原图 [0,1].

    变换链:
      argmax / split_ratio → 模型输入坐标 (0..iw, 0..ih)
      → / iw * crop_w + cx0 → 原图像素坐标
      → / img_w → 归一化 [0,1]

    crop 为正方形 bbox (max(w,h)*1.25) 直接 resize 到 (iw, ih), 非 letterbox,
    逆变换 x/y 独立缩放, 无 padding 需减除. 归一化分母为原图宽高(非 crop),
    下游 waist_r = y_norm / H_subject 依赖这一点.
    """
    iw, ih = _RTMPOSE_INPUT_WH
    cx0, cy0, crop_w, crop_h = bbox
    w, h = img_wh
    kx = sx.argmax(axis=-1).astype(np.float64) / _RTMPOSE_SPLIT_RATIO
    ky = sy.argmax(axis=-1).astype(np.float64) / _RTMPOSE_SPLIT_RATIO
    kx = kx / iw * crop_w + cx0
    ky = ky / ih * crop_h + cy0
    kx = np.clip(kx / w, 0.0, 1.0)
    ky = np.clip(ky / h, 0.0, 1.0)
    return kx, ky


def _rtmpose_keypoints_sane(kp: dict, img_w: int, img_h: int) -> bool:
    """RTMPose 关键点解剖学合理性校验(pipeline audit 行32).

    SimCC 无 vis, 旧代码把 vis 硬编码 1.0 → 下游 vis>=0.3 门对 rtmpose 是空操作。
    这里用几何先验替代 vis:
      - hip/shoulder/knee 必须落在图内(0..1);
      - 肩在下髋之上、髋在膝之上(站立姿态, 容忍小翻转);
      - 肩-髋 与 髋-膝 都要有足够像素跨度(>= 主体高度的 3%), 否则说明检测塌成一团。
    返回 False → 调用方拒收(返回 None), 腰线走轮廓法, 不拿错坐标切衣长。
    """
    min_span = 0.03
    pts = []
    for name in ("shoulder", "hip", "knee"):
        x, y, _ = kp.get(name, (0.0, 0.0, 0.0))
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        pts.append((name, x, y))
    sy = {n: y for n, _, y in pts}
    sx = {n: x for n, x, _ in pts}
    if not (sy["shoulder"] < sy["hip"] < sy["knee"]):
        return False
    px_per_norm_y = max(1, img_h)
    sh_hip_px = abs(sy["hip"] - sy["shoulder"]) * px_per_norm_y
    hip_knee_px = abs(sy["knee"] - sy["hip"]) * px_per_norm_y
    if sh_hip_px < min_span * px_per_norm_y or hip_knee_px < min_span * px_per_norm_y:
        return False
    # 侧面照肩宽可能塌成一条竖线, 但肩点仍须在图内(上面已查)。
    del sx
    return True


def _detect_pose_rtmpose(img: Image.Image) -> dict | None:
    """RTMPose-m 推理: bbox 裁切 → resize 256x192 → backbone → SimCC 解码."""
    global _POSE_SESSION_WARNED
    ret = _init_rtmpose()
    if ret is None:
        return None
    backbone, head = ret

    try:
        import cv2

        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]

        # bbox from gray threshold (no rembg needed)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        fg = (gray < 235).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        ys, xs = np.where(fg > 0)
        if len(ys) < 10:
            return None
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        # expand bbox 1.25
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = int(max(x1 - x0, y1 - y0) * 1.25 / 2)
        cx_i, cy_i = int(cx), int(cy)
        cx0, cy0, cx1, cy1 = cx_i - half, cy_i - half, cx_i + half, cy_i + half
        pl, pt = max(0, -cx0), max(0, -cy0)
        pr, pb = max(0, cx1 - w), max(0, cy1 - h)
        padded = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), mode="constant")
        crop = padded[cy0 + pt : cy1 + pt, cx0 + pl : cx1 + pl]
        iw, ih = _RTMPOSE_INPUT_WH
        resized = cv2.resize(crop, (iw, ih))
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        norm = (resized.astype(np.float32) - mean) / std
        import torch as _torch

        tensor = _torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)

        with _torch.no_grad():
            feats = backbone(tensor)
            sx, sy = head(feats)

        kx, ky = _simcc_decode(
            sx.cpu().numpy()[0],
            sy.cpu().numpy()[0],
            (cx0, cy0, cx1 - cx0, cy1 - cy0),
            (w, h),
        )

        # COCO 17: 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
        # 9=L_wrist, 10=R_wrist, 11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee。
        # c9 勘误: 旧代码把 7/8(肘)当腕 —— "wrist"=avg(7,8) 落在肘线
        # (~0.50c), 袖条/走廊只到肘, 长袖出图读作中袖。真腕在 9/10。
        # SimCC 无 vis, 硬编码 1.0; vis>=0.3 门只对 MediaPipe 回退路径生效。
        def avg(i, j):
            return (float((kx[i] + kx[j]) / 2), float((ky[i] + ky[j]) / 2), 1.0)

        def one(i):
            return (float(kx[i]), float(ky[i]), 1.0)

        kp = {
            "hip": avg(11, 12),
            "shoulder": avg(5, 6),
            "knee": avg(13, 14),
            "wrist": avg(9, 10),
            "wrist_l": one(9),
            "wrist_r": one(10),
            "elbow_l": one(7),
            "elbow_r": one(8),
        }
        # 行32: SimCC 没有 vis, 上面硬编码 1.0 使下游 vis>=0.3 门变空操作 —— 错但落
        # 在 [0,1] 范围内的检测会被当成"rtmpose 源"采信(比 None 更糟)。加独立解剖学
        # 合理性校验: 关键点须在图内, 肩-髋-膝垂直有序且跨距够大; 不合理则拒收(None),
        # 让腰线走轮廓法而非拿一个错坐标去切。
        if not _rtmpose_keypoints_sane(kp, w, h):
            logger.warning(
                "RTMPose keypoints anatomically implausible (hip=%.3f,%.3f shoulder=%.3f,%.3f "
                "knee=%.3f,%.3f); rejecting to avoid a wrong waist cut",
                *kp["hip"][:2],
                *kp["shoulder"][:2],
                *kp["knee"][:2],
            )
            return None
        return kp
    except Exception:  # noqa: BLE001
        if not _POSE_SESSION_WARNED:
            logger.warning("RTMPose inference failed; falling back")
            _POSE_SESSION_WARNED = True
        return None


def _pose_landmarker_task_path():
    """找本地 pose_landmarker_full.task(NAS 模型目录 > MEDIAPIPE_HOME)。

    找不到返回 None → MediaPipe 回退静默降级为 None(腰线回退 contour/default)。
    模型约 9MB, 只认 >1MB 的文件避免误用残缺下载。下载地址:
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
    """
    import os as _os

    home = Path(_os.environ.get("MEDIAPIPE_HOME", "~/.mediapipe")).expanduser()
    candidates = [
        _POSE_ROOT / "pose_landmarker_full.task",  # NAS: repo/models/mediapipe/
        home / "pose_landmarker_full.task",
    ]
    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size > 1_000_000:
                return c
        except OSError:
            continue
    return None


def _init_pose_landmarker():
    """惰性创建并缓存 MediaPipe PoseLandmarker(mediapipe.tasks 新版 API)。

    旧版 `mp.solutions.pose.Pose()` 在本机 mediapipe(无 solutions 模块)已被移除,
    必须用 `mediapipe.tasks.python.vision.PoseLandmarker` + 下载的 .task 模型文件。
    """
    global _POSE_LANDMARKER, _POSE_LANDMARKER_ERROR
    if _POSE_LANDMARKER is not None:
        return _POSE_LANDMARKER
    if _POSE_LANDMARKER_ERROR is not None:
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    except Exception as e:  # noqa: BLE001
        _POSE_LANDMARKER_ERROR = f"mediapipe.tasks 不可用: {e}"
        logger.warning("MediaPipe 回退不可用: %s", _POSE_LANDMARKER_ERROR)
        return None
    model_path = _pose_landmarker_task_path()
    if model_path is None:
        _POSE_LANDMARKER_ERROR = (
            "未找到 pose_landmarker_full.task(放 repo/models/mediapipe/ 或 MEDIAPIPE_HOME; "
            "下载见 _pose_landmarker_task_path docstring)"
        )
        logger.warning("MediaPipe 回退不可用: %s", _POSE_LANDMARKER_ERROR)
        return None
    try:
        opts = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _POSE_LANDMARKER = vision.PoseLandmarker.create_from_options(opts)
        return _POSE_LANDMARKER
    except Exception as e:  # noqa: BLE001
        _POSE_LANDMARKER_ERROR = str(e)
        logger.warning("MediaPipe PoseLandmarker 初始化失败: %s", e)
        return None


def _detect_pose_mediapipe(img: Image.Image) -> dict | None:
    """MediaPipe Pose 回退(新版 tasks API: PoseLandmarker)。

    依赖已下载的 pose_landmarker_full.task(见 _pose_landmarker_task_path)。
    检测失败或无模型时静默返回 None, 腰线回退 contour/default。
    """
    global _POSE_SESSION_WARNED
    lk = _init_pose_landmarker()
    if lk is None:
        return None

    arr = np.array(img.convert("RGB"))
    try:
        import mediapipe as mp

        res = lk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]  # 取第一人; 每个元素是 33 个 landmark 的 list
    except Exception:  # noqa: BLE001
        if not _POSE_SESSION_WARNED:
            logger.warning("MediaPipe inference failed; falling back")
            _POSE_SESSION_WARNED = True
        return None

    def avg_lm(i, j):
        lm_l, lm_r = lm[i], lm[j]
        return (
            (lm_l.x + lm_r.x) / 2,
            (lm_l.y + lm_r.y) / 2,
            ((getattr(lm_l, "visibility", 0) or 0) + (getattr(lm_r, "visibility", 0) or 0)) / 2,
        )

    hip = avg_lm(23, 24)
    shoulder = avg_lm(11, 12)
    knee = avg_lm(25, 26)
    wrist = avg_lm(15, 16)  # 15=L_wrist, 16=R_wrist (长袖手臂走廊用)
    if hip[2] < 0.3:
        return None
    return {
        "hip": hip,
        "shoulder": shoulder,
        "knee": knee,
        "wrist": wrist,
        "wrist_l": (lm[15].x, lm[15].y, getattr(lm[15], "visibility", 0) or 0),
        "wrist_r": (lm[16].x, lm[16].y, getattr(lm[16], "visibility", 0) or 0),
        "elbow_l": (lm[13].x, lm[13].y, getattr(lm[13], "visibility", 0) or 0),
        "elbow_r": (lm[14].x, lm[14].y, getattr(lm[14], "visibility", 0) or 0),
    }


def _estimate_waist_ratio(
    subj: np.ndarray,
    y0: int,
    span: int,
    img: Image.Image | None = None,
    pose_keypoints: dict | None = None,
) -> float:
    """估算腰线占身高的比例。

    三层降级:
    1. Pose 关键点 shoulder→hip 插值(hip/shoulder 置信度均 >= 0.3)
    2. 轮廓法行宽最窄点(躯干区 30%-75%, convolve 平滑)
    3. 0.62 固定默认值

    轮廓法两层保护:
    - 中位宽度过近(min/median > 0.95,即 < 5% 差异)→ 无明显收腰
    - 最窄点左右不对称(一侧被裁切/遮挡)→ 不可信
    """
    h = subj.shape[0]
    # y0/span 由调用方从同一份 subject mask 算出后传入, pose 插值与轮廓法共用,
    # 禁止在本函数内重算. clothing_mask 的 knee_r 也用同一对 y0/span.

    # 1. Pose: 几何门控短路, 通过才插值并立即返回 (按最终来源计数).
    kp = pose_keypoints
    if kp is None and img is not None:
        kp = _detect_pose_keypoints(img)
    if kp and kp["hip"][2] >= 0.3 and kp["shoulder"][2] >= 0.3:
        hip_r = (kp["hip"][1] * h - y0) / max(1, span)
        sho_r = (kp["shoulder"][1] * h - y0) / max(1, span)
        if hip_r > sho_r + 0.05:
            knee_ok = True
            if "knee" in kp and kp["knee"][2] >= 0.3:
                knee_r = (kp["knee"][1] * h - y0) / max(1, span)
                knee_ok = knee_r > hip_r
            if knee_ok:
                ratio = sho_r + (hip_r - sho_r) * 0.55
                if 0.45 <= ratio <= 0.65:
                    src = kp.get("_source", "rtmpose")
                    if src not in ("rtmpose", "mediapipe"):
                        src = "rtmpose"
                    _record_waist_source(src)
                    return float(ratio)

    # 2. Contour: 同一 y0/span. 无明显收腰或不对称 → 不记 contour, 归 default.
    torso_top = y0 + int(span * 0.30)
    torso_bottom = min(y0 + int(span * 0.75), h)
    if torso_bottom > torso_top + 5:
        widths = np.array([subj[r].sum() for r in range(torso_top, torso_bottom)], dtype=np.float32)
        if len(widths) >= 5:
            kernel = np.ones(5, dtype=np.float32) / 5
            smooth = np.convolve(widths, kernel, mode="valid")
            min_idx = int(np.argmin(smooth)) + 2
            min_w = float(smooth.min())
            median_w = float(np.median(smooth))
            if not (median_w > 0 and min_w / median_w > 0.95):
                waist_row = torso_top + min_idx
                all_cols = np.where(subj.any(axis=0))[0]
                symmetric = True
                if len(all_cols) > 0 and 0 <= waist_row < h:
                    body_center = (float(all_cols[0]) + float(all_cols[-1])) / 2
                    waist_cols = np.where(subj[waist_row] > 0)[0]
                    if len(waist_cols) > 4:
                        left = body_center - float(waist_cols[0])
                        right = float(waist_cols[-1]) - body_center
                        if max(left, right) > 0 and abs(left - right) / max(left, right) > 0.30:
                            symmetric = False
                if symmetric:
                    _record_waist_source("contour")
                    return float(np.clip((torso_top + min_idx - y0) / max(1, span), 0.45, 0.75))
    _record_waist_source("default")
    return 0.62


def _record_waist_source(src: str) -> None:
    _WAIST_SOURCE_COUNTS[src] = _WAIST_SOURCE_COUNTS.get(src, 0) + 1
    total = sum(_WAIST_SOURCE_COUNTS.values())
    if total == 1 or total % 50 == 0:
        logger.info("waist sources: %s", waist_source_summary())


def waist_source_summary() -> str:
    c = _WAIST_SOURCE_COUNTS
    return f"RTMPose {c['rtmpose']} / MediaPipe {c['mediapipe']} / contour {c['contour']} / default {c['default']}"


def _log_waist_source_summary() -> None:
    total = sum(_WAIST_SOURCE_COUNTS.values())
    if total:
        logger.info("waist sources final: %s", waist_source_summary())


atexit.register(_log_waist_source_summary)


def _debug_dump_mask_stage(mask_u8: np.ndarray, img: Image.Image, label: str) -> None:
    """VTON_DEBUG_MASK=1 时, 把 clothing_mask 的阶段遮罩落盘(绿=mask 区)。

    阶段序列: hemcut(截线后) → corridor(长袖手臂走廊后) → carve(领口/胸口开孔后)
    → final(erode+blur 后)。哪一步"把皮肤圈进去 / 把衣料挖掉", 看对应阶段图即可,
    不用猜。tag 取 VTON_DEBUG_TAG(衣物名, engine 传入), 目录取 VTON_DEBUG_DIR(默认 .)。
    关开关时本函数零成本(一个 env 读 + 返回)。
    """
    if os.environ.get("VTON_DEBUG_MASK", "0") != "1":
        return
    try:
        from pathlib import Path as _P

        import cv2

        m = np.asarray(mask_u8)
        vis = np.array(img.convert("RGB"))
        if m.shape[:2] != vis.shape[:2]:
            m = cv2.resize(
                m.astype(np.float32),
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        overlay = vis.copy()
        mz = m > 127
        overlay[mz] = (overlay[mz].astype(np.float32) * 0.55 + np.array([0, 190, 0], dtype=np.float32) * 0.45).astype(
            np.uint8
        )
        tag = os.environ.get("VTON_DEBUG_TAG", "img")
        ddir = _P(os.environ.get("VTON_DEBUG_DIR", "."))
        ddir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(ddir / f"maskstage_{tag}_{label}.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
    except Exception:
        logger.debug("mask stage dump failed (%s)", label, exc_info=True)


def clothing_mask(
    img: Image.Image,
    garment_type: str = "upper",
    subject_mask: np.ndarray | None = None,
    face_margin: float = 0.35,
    garment_extent: dict | None = None,
) -> np.ndarray | None:
    """服装替换区:主体减脸(及按品类+袖长动态裁切),返回 uint8 0-255;构不出则 None。

    这是「只换衣、锁身份」的遮罩:255=贴生成衣物,0=保留原图像素(脸/发/手/背景)。
    garment_extent(from detect_garment_extent)控制:
      sleeve: long → 手臂走廊保留到腕(衣袖区); short → 上臂含袖、肘下前臂挖出;
              sleeveless → 不含手臂
      hem: waist → 截线=腰; mid(中长款) → 腰+0.22(髋/大腿, tunic 级);
           long → tunic 腰+0.30 / upper 腰+0.18
      neckline: open(V/scoop) → 胸口露肤区开孔保留原肤色; closed(高领) → 胸口保持衣料
    截线优先级写死: top 品类(upper/auto/tunic)只用腰线系, 不与 knee 截断混用
    (knee_r 只服务 dress/onesie)。
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    subj = subject_mask if subject_mask is not None else foreground_mask(img)
    if subj is None:
        return None
    subj = (np.asarray(subj) > 0).astype(np.uint8)
    if subj.shape[:2] != (h, w) or float(subj.mean()) < 0.04:
        return None

    # Pose keypoints — call once, reuse for waist + knee
    pose_kp = _detect_pose_keypoints(img)

    face = _detect_face_mask(arr, margin_frac=face_margin, down_margin_frac=0.08)
    cloth = subj.copy()
    if face is not None and face.any():
        cloth[face > 0] = 0
    else:
        heur = _heuristic_face_mask(arr, subj)
        if heur.any():
            cloth[heur > 0] = 0
    ys = np.where(subj > 0)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(1, y1 - y0)

    # dress/onesie 需额外排除脖子: 脸区(顶部 ~18%)以下到 22% 是颈区,
    # 不应被服装遮罩覆盖。上衣/下装不需此处理(领口在衣物范围内)。
    if garment_type in ("dress", "onesie"):
        neck_end = min(h, y0 + int(span * 0.22))
        cloth[:neck_end] = 0

    sleeve = "short" if garment_extent is None else garment_extent.get("sleeve")
    sleeve_eff = sleeve or "short"
    hem = garment_extent.get("hem", "waist") if garment_extent else "waist"
    _dbg_tag = os.environ.get("VTON_DEBUG_TAG", "img")

    waist_r = _estimate_waist_ratio(subj, y0, span, pose_keypoints=pose_kp)

    # 膝关键点(若有)用于 dress/onesie 下摆精确截断.
    # knee_r 是相对 subject span 的归一化比例 (0-1), 不是像素坐标;
    # 后续 cut = knee_r - 0.02 中的 0.02 ≈ 身高 2% 安全边距.
    knee_r: float | None = None
    if pose_kp and pose_kp["knee"][2] >= 0.3:
        knee_y_px = pose_kp["knee"][1] * h
        knee_r = (knee_y_px - y0) / max(1, span)

    # 截线优先级(写死, 有测试): top 品类(upper/auto/tunic) 一律腰线系, 不与 knee
    # 截断混用 —— knee_r 只服务 dress/onesie; hem=mid(中长款) 即使品类被显式
    # 定成 upper 也走 tunic 截线。
    # 上衣截线兜底盖过裤头: 与 align_garment_to_pose 的 warp 框底同口径(pose
    # 髋点 + 0.03h)。hem 分类为 waist 的上衣常实穿过腰(c10 实测 09263/09290
    # 分类 waist → 截线在肚脐上 → 生成区漏肚脐, "腰间只盖裤头两边")。
    _hip_cut_floor = -1
    if garment_type in ("upper", "auto"):
        _kp_hip = pose_kp.get("hip") if pose_kp else None
        if _kp_hip and len(_kp_hip) >= 3 and _kp_hip[2] >= 0.3:
            _hip_cut_floor = int(float(_kp_hip[1]) * h + 0.03 * h)
    if garment_type in ("upper", "auto", "tunic"):
        if garment_type == "tunic":
            _off = 0.30 if hem == "long" else 0.22
        elif hem == "mid":
            _off = 0.22
        elif hem == "long":
            _off = 0.18
        elif sleeve_eff == "sleeveless":
            _off = -0.07
        elif sleeve_eff == "long":
            _off = 0.10
        else:
            _off = 0.0
        cut_row = y0 + int(span * (waist_r + _off))
        if _hip_cut_floor > cut_row:
            logger.info(
                "[clothing_mask:%s] hemcut floor: %d -> %d (cover waistband, pose hip)",
                _dbg_tag,
                cut_row,
                _hip_cut_floor,
            )
            cut_row = _hip_cut_floor
        cloth[cut_row:] = 0
        _debug_dump_mask_stage(cloth * 255, img, "hemcut")
        logger.info(
            "[clothing_mask:%s] hemcut: type=%s hem=%s sleeve=%s waist_r=%.3f off=%+.2f "
            "cut_row=%d (span=%d, knee_r=%s)",
            _dbg_tag,
            garment_type,
            hem,
            sleeve_eff,
            waist_r,
            _off,
            cut_row,
            span,
            (f"{knee_r:.3f}" if knee_r is not None else "absent"),
        )
    elif garment_type == "dress":
        if knee_r is not None and 0.75 <= knee_r <= 0.95:
            # 有膝关键点: normal 截到膝, long 截到膝下 5%
            cut = knee_r + 0.05 if hem == "long" else knee_r - 0.02
            cloth[y0 + int(span * cut) :] = 0
        elif hem == "long":
            cloth[y0 + int(span * (waist_r + 0.35)) :] = 0
        else:
            cloth[y0 + int(span * (waist_r + 0.28)) :] = 0
    elif garment_type == "lower":
        cloth[: y0 + int(span * (waist_r - 0.14))] = 0
    elif garment_type == "onesie":
        if knee_r is not None and 0.75 <= knee_r <= 0.95:
            cut = knee_r + 0.05 if hem == "long" else knee_r - 0.02
            cloth[y0 + int(span * cut) :] = 0
        elif sleeve_eff == "long":
            cloth[y0 + int(span * (waist_r + 0.33)) :] = 0
        else:
            cloth[y0 + int(span * (waist_r + 0.28)) :] = 0
    if float(cloth.mean()) < 0.02:
        return None

    # 长袖手臂走廊(top 品类): 下摆截线是通宽直线, 但长袖衣服的袖子要盖住整条
    # 手臂到腕。不恢复截线到腕之间的两侧手臂列, 截线以下会被 blend_tryon
    # 贴回基模裸臂 → 袖子齐腰被截、手臂变白。
    # 走廊 = 主体两侧外 25% 列, 只限主体内部。腕点缺失时按身高 85% 估腕,
    # 避免 RTMPose/MediaPipe 不可用时走廊整段跳过。
    def _restore_arm_corridor(to_row: int) -> bool:
        xs_all = np.where(subj.any(axis=0))[0]
        if not len(xs_all):
            return False
        x0s, x1s = int(xs_all.min()), int(xs_all.max())
        arm_w = max(8, int(max(1, x1s - x0s) * 0.25))
        w_row = min(h, int(to_row))
        if w_row <= cut_row + 4:
            return False
        for cs, ce in ((0, x0s + arm_w), (x1s - arm_w, w)):
            cloth[cut_row:w_row, cs:ce] = (subj[cut_row:w_row, cs:ce] > 0).astype(np.uint8)
        _debug_dump_mask_stage(cloth * 255, img, "corridor")
        logger.info(
            "[clothing_mask:%s] corridor ON (sleeve=%s, to_row=%d)",
            _dbg_tag,
            sleeve,
            w_row,
        )
        return True

    _want_corridor = sleeve == "long" or (sleeve is None and garment_type in ("upper", "auto", "tunic"))
    if garment_type in ("upper", "auto", "tunic") and _want_corridor:
        restored = False
        if pose_kp:
            wrist = pose_kp.get("wrist")
            if wrist is not None and wrist[2] >= 0.3:
                wr = (wrist[1] * h - y0) / max(1, span)
                sr = (pose_kp["shoulder"][1] * h - y0) / max(1, span)
                if wr > sr + 0.05:
                    restored = _restore_arm_corridor(y0 + int(span * wr))
        if not restored:
            _restore_arm_corridor(y0 + int(span * 0.85))
    else:
        logger.info(
            "[clothing_mask:%s] corridor OFF: sleeve=%s, type=%s (need long or auto-unknown)",
            _dbg_tag,
            sleeve,
            garment_type,
        )

    # 领口/胸口肤色开孔(top 品类): 中央条带(脖子 + V 领露肤)目前在 mask 内被
    # 生成像素贴回 —— SDXL 重画的脖子/胸口常发白发灰, 与基模原肤色不符。把
    # "原图肤色"(dilate 15px 兜发灰边界)+ 颈区"浅色皮"(原图亮且低饱和, 兜住
    # 皮肤检测器漏掉的近白皮)挖出 → blend_tryon 贴回原像素, 恢复模特原肤色。
    # 判定只看原图(human_img), 不受衣物自身浅色(如白衬衫领)干扰; 开孔在最终
    # erode 之前做, 本步不膨胀 mask, 开孔不会被形态学补回。
    if garment_type in ("upper", "auto", "tunic"):
        skin_o = _detect_skin(arr)
        try:
            import cv2

            skin_d = cv2.dilate(skin_o, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        except ImportError:
            skin_d = skin_o
        c0, c1 = int(w * 0.30), int(w * 0.70)
        neck_top = y0 + int(span * 0.14)
        neck_bot = y0 + int(span * 0.22)
        chest_bot = y0 + int(span * 0.45)
        # 颈区: 无论领型, 露出的脖子必须是模特原肤色(领口以下才是衣料)。
        if neck_bot > neck_top:
            zo = arr[neck_top:neck_bot, c0:c1]
            lum = zo.astype(np.float32).mean(axis=2)
            sat = zo.max(axis=2).astype(np.int32) - zo.min(axis=2).astype(np.int32)
            hit = (skin_d[neck_top:neck_bot, c0:c1] > 0) | ((lum > 195) & (sat < 45))
            zreg = cloth[neck_top:neck_bot, c0:c1]
            zreg[hit] = 0
        # 胸口区: 仅开口领(领口开口度检测=open, 胸口真露皮)开孔; 高领/立领
        # (closed) 胸口区是衣料, 保持覆盖, 否则会把衣料挖出露出基模打底衫。
        if (
            chest_bot > neck_bot
            and (garment_extent.get("neckline", "closed") if garment_extent else "closed") == "open"
        ):
            zreg = cloth[neck_bot:chest_bot, c0:c1]
            if zreg.any():
                zreg[skin_d[neck_bot:chest_bot, c0:c1] > 0] = 0
        _debug_dump_mask_stage(cloth * 255, img, "carve")

    # 短袖/无袖:排除手臂皮肤。检测原图肤色(YCbCr),从 mask 两侧减去
    # 皮肤像素 → 手臂皮肤标记为 0(保留原图),只有衣物区标记为 255(允许
    # 生成替换)。躯干中心不减 → 胸口 V 领露肤仍可被新衣物覆盖。
    # 长袖不减:新衣物需覆盖整条手臂,手臂皮肤应被生成结果替换。
    # 短袖只挖肘下前臂,上臂留给袖料(整臂挖出 → 袖条无法贴回 → 04469 无袖)。
    # 用主体水平外扩 25% 定位手臂区(而非固定图像宽度比例)。
    if garment_type in ("upper", "auto", "tunic") and sleeve in ("short", "sleeveless"):
        skin = _detect_skin(arr)
        if skin.any():
            xs = np.where(subj.any(axis=0))[0]
            if len(xs):
                x0s, x1s = int(xs.min()), int(xs.max())
                subj_w = max(1, x1s - x0s)
                arm_w = max(8, int(subj_w * 0.25))
                left_end = x0s + arm_w
                right_start = x1s - arm_w
                if sleeve == "short":
                    elbow_row = y0 + int(span * 0.42)
                    cloth[elbow_row:, :left_end][skin[elbow_row:, :left_end] > 0] = 0
                    cloth[elbow_row:, right_start:][skin[elbow_row:, right_start:] > 0] = 0
                else:
                    cloth[:, :left_end][skin[:, :left_end] > 0] = 0
                    cloth[:, right_start:][skin[:, right_start:] > 0] = 0

    try:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cloth = cv2.erode(cloth, k, iterations=1)
        if float(cloth.mean()) < 0.015:
            return None
        out_mask = cv2.GaussianBlur((cloth * 255).astype(np.uint8), (0, 0), 6)
    except Exception:  # noqa: BLE001
        out_mask = (cloth * 255).astype(np.uint8)
    _debug_dump_mask_stage(out_mask, img, "final")
    return out_mask


def limit_mask_to_subject(
    mask: Image.Image | None,
    subject: Image.Image | None,
    dilate_px: int = 3,
) -> Image.Image | None:
    """几何 mask 与生成图主体求交, 只保留"生成器真画出来的内容"区域。

    clothing_mask 是几何外推(长袖走廊 85% 兜底/下摆截线), 生成器实际没画满
    的地方残留生成图白背景 —— blend 直接用几何 mask 会把白背景(被 Reinhard
    染成衣色)半透明盖到原图上(09290 髋部粉块 + 黑矩形根因)。与生成图
    u2net 主体求交后, 走廊内没画出的区域保持原图像素。

    - dilate_px: u2net 边界偏保守, 主体外扩几像素, 避免新衣边缘被切硬边。
    - subject 为 None/尺寸不齐/交集为空 → 原样返回 mask(fail-open:
      u2net 挂掉时退回旧行为, 不让整条后处理链断掉)。
    - mask 的羽化带(0-255 渐变)在主体内原样保留, 只在主体外归零。
    - 双类型兼容: 生产链路里 clothing_mask 返回 uint8 ndarray(0-255),
      foreground_mask 返回 uint8 ndarray(0/1); 单测里常喂 PIL Image。
      输入是什么类型, 输出就是什么类型(下游 blend/reapply/seam 按
      ndarray 消费, 不能换型)。
    """
    if mask is None:
        return None
    if subject is None:
        return mask
    m = np.asarray(mask.convert("L")) if isinstance(mask, Image.Image) else np.asarray(mask)
    s = np.asarray(subject.convert("L")) if isinstance(subject, Image.Image) else np.asarray(subject)
    if s.ndim == 3:
        s = s.mean(axis=2)
    if s.shape[:2] != m.shape[:2]:
        return mask
    s_bin = s > 0
    if dilate_px > 0:
        try:
            import cv2

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            s_bin = cv2.dilate(s_bin.astype("uint8"), k, iterations=1) > 0
        except Exception:  # noqa: BLE001
            pass
    out = np.where(s_bin, m, 0)
    if not (out > 0).any():
        return mask
    if isinstance(mask, Image.Image):
        return Image.fromarray(out.astype("uint8"))
    return out.astype("uint8")


def blend_tryon(
    original: Image.Image,
    generated: Image.Image,
    mask_u8: np.ndarray,
) -> Image.Image:
    """mask 255 取 generated,0 取 original。

    身份区(mask=0,脸/手/脚)用 cv2.seamlessClone 无缝贴回原模特像素,
    避免硬边;seamlessClone 失败或多连通区域时回退羽化 alpha 混合。
    """
    orig = original.convert("RGB")
    gen = generated.convert("RGB")
    if gen.size != orig.size:
        gen = gen.resize(orig.size, Image.Resampling.LANCZOS)
    o = np.array(orig, dtype=np.uint8)
    g = np.array(gen, dtype=np.uint8)
    alpha = np.asarray(mask_u8)
    if alpha.shape[:2] != o.shape[:2]:
        alpha = np.array(Image.fromarray(alpha).resize((o.shape[1], o.shape[0]), Image.Resampling.BILINEAR))

    try:
        import cv2

        identity_mask = (alpha < 20).astype(np.uint8) * 255
        identity_mask = cv2.morphologyEx(
            identity_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(identity_mask, connectivity=8)

        result = g.copy()
        blended_any = False
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 200:
                continue
            comp_mask = (labels == i).astype(np.uint8) * 255
            cx, cy = centroids[i]
            cx, cy = round(cx), round(cy)
            if not (0 <= cx < o.shape[1] and 0 <= cy < o.shape[0]) or comp_mask[cy, cx] == 0:
                ys_c, xs_c = np.where(labels == i)
                if len(ys_c) == 0:
                    continue
                cx, cy = int(xs_c[len(xs_c) // 2]), int(ys_c[len(ys_c) // 2])
            if 0 <= cx < o.shape[1] and 0 <= cy < o.shape[0] and comp_mask[cy, cx] > 0:
                try:
                    result = cv2.seamlessClone(o, result, comp_mask, (cx, cy), cv2.NORMAL_CLONE)
                    blended_any = True
                except Exception:  # noqa: BLE001
                    continue

        if blended_any:
            cloth_a = (alpha.astype(np.float32) / 255.0)[..., None]
            cloth_a = np.clip(cloth_a * 3.0, 0, 1)
            out = (g.astype(np.float32) * cloth_a + result.astype(np.float32) * (1.0 - cloth_a)).astype(np.uint8)
            return Image.fromarray(out, "RGB")
    except Exception:  # noqa: BLE001
        pass

    a = (alpha.astype(np.float32) / 255.0)[..., None]
    out = np.clip(g.astype(np.float32) * a + o.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def center_crop_square(img: Image.Image) -> Image.Image:
    """1:1 居中裁剪(电商主图必须正方形,主体居中)。"""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def apply_platform_preset(
    img: Image.Image,
    platform: str,
    use_subject_mask: bool = True,
) -> tuple[Image.Image, dict | None]:
    """应用平台模板;platform="none" 或未知时原样返回 (img, None)。

    use_subject_mask=True(默认):压白前先用 rembg 求主体掩膜,只白背景、
    绝不啃主体;rembg 不可用时自动回退到纯颜色压白(等价旧行为)。
    """
    preset = PLATFORM_PRESETS.get(platform)
    if preset is None:
        return img, None
    out = center_crop_square(img)
    out = out.resize((preset["size"], preset["size"]), Image.Resampling.LANCZOS)
    if preset.get("force_white"):
        sm = foreground_mask(out) if use_subject_mask else None
        out = force_white_background(out, subject_mask=sm)
    return out, preset


def encode_jpeg_within(
    img: Image.Image,
    max_bytes: int,
    start_quality: int = 94,
    min_quality: int = 60,
) -> bytes:
    """导出 JPEG 并保证体积 <= max_bytes(质量从高往低调,达到上限才降级)。"""
    rgb = img.convert("RGB")
    quality = start_quality
    while True:
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes or quality <= min_quality:
            return data
        quality = max(min_quality, quality - 6)


def save_platform_result(
    img: Image.Image,
    platform: str,
    out_path,
) -> tuple[int, int]:
    """模板裁剪/压白 + JPEG 限体积落盘;返回 (size_px, jpeg_bytes)。"""
    out, preset = apply_platform_preset(img, platform)
    if preset is None:
        out.convert("RGB").save(out_path, "JPEG", quality=94, optimize=True)
        return out.size[0], 0
    data = encode_jpeg_within(out, preset["max_bytes"])
    out_path.write_bytes(data)
    return out.size[0], len(data)


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """统一 env float 解析 + clamp(pipeline audit 行23/24/30).

    旧代码 10 个 VTON_PATTERN_LOCK_* 与若干 VTON_* 直接 float(env) → 坏值(如
    "0,75")在未包裹的 overlay 里抛异常 → 整步退回裸 SDXL;LIGHTING_MIN>LIGHTING_MAX
    时 clip 把所有像素钉在 min, 图像整体偏色。这里坏值 warning + 用 default,
    越界值 warning + clamp 进 [lo, hi]。
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        val = float(default)
    else:
        try:
            val = float(str(raw).strip())
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using default %s", name, raw, default)
            val = float(default)
    if lo is not None and val < lo:
        logger.warning("%s=%s below minimum %s; clamping", name, val, lo)
        val = lo
    if hi is not None and val > hi:
        logger.warning("%s=%s above maximum %s; clamping", name, val, hi)
        val = hi
    return val


def _pattern_lock_overlay(
    gen: np.ndarray,
    ref: np.ndarray,
    mask_f32: np.ndarray,
    skin_u8: np.ndarray,
    light_strength: float,
    ref_alpha: Image.Image | None = None,
) -> Image.Image:
    """Warp 后的衣覆盖衣服区;生成图只提供褶皱光影(PLTON 频率分离)。

    lighting = 空间明暗场: LF(L_gen) 按衣料区中位数归一(不除 LF_ref, 防 ref
    影棚阴影以倒数污染明暗场)
    shade_lf = 1 + (lighting - 1) * light_strength * lf_strength
    shade_hf = 1 + (lighting - 1) * light_strength * hf_strength
    lit = LF_ref * shade_lf + HF_ref * shade_hf
    底色按 lf_strength 调,花纹按 hf_strength 调(默认半强度,保对比度)。
    light_strength=0 时 lit=cloth_ref(纯贴图,快速回退基准)。

    空间明暗场(修"贴上去"贴纸感, C13 三件实证): 旧实现逐像素亮度比钳 ±8%,
    SDXL 的身体侧影/褶皱阴影(20-40% 明暗)被砍平 → 面料平光如贴纸。新实现:
    双方 LF 比值按衣料区中位数归一 —— 全局亮暗差(SDXL 整件画亮/画暗、胸口
    整片高光)被中位数吸收(胸口变白不回归), 只传空间相对明暗; 非对称钳制:
    阴影宽(-38%, "穿上"的关键线索)、高光窄(+12%, 防发白)。

    环境变量:
      VTON_PATTERN_LOCK_LIGHT_STRENGTH  (1.0)  光照总闸:0=精确 ref,1=全效
      VTON_PATTERN_LOCK_LF_STRENGTH     (1.0)  底色光照响应系数(1=全强度)
      VTON_PATTERN_LOCK_HF_STRENGTH     (0.5)  花纹光照响应系数(0=不受光照,1=全强度)
    VTON_PATTERN_LOCK_LF_BLUR_SIGMA   (8.0)  低频提取高斯 sigma
      VTON_PATTERN_LOCK_LIGHTING_MIN    (0.62) 空间明暗场下限(阴影通道, 归一后)
      VTON_PATTERN_LOCK_LIGHTING_MAX    (1.12) 空间明暗场上限(高光通道, 归一后)
      VTON_PATTERN_LOCK_SKIN_WEIGHT     (0.85) 肤色降权
      VTON_PATTERN_LOCK_FEATHER_PX      (5)    mask 边缘羽化像素
      VTON_PATTERN_LOCK_SKIP_WHITE_BG   (1)    跳过画布边缘连通白底
      VTON_PATTERN_LOCK_WHITE_THRESH    (245)  白底 RGB 判定阈值
      VTON_PATTERN_LOCK_FOLD_GAIN       (0.45) gen 褶皱光影叠加强度(0=关)
    """
    import os

    import cv2

    ls_val = _env_float("VTON_PATTERN_LOCK_LIGHT_STRENGTH", light_strength, 0.0, 1.0)
    ls = float(np.clip(ls_val, 0.0, 1.0))
    lf_strength = _env_float("VTON_PATTERN_LOCK_LF_STRENGTH", 1.0, 0.0, 4.0)
    hf_strength = _env_float("VTON_PATTERN_LOCK_HF_STRENGTH", 0.5, 0.0, 4.0)
    lf_blur = _env_float("VTON_PATTERN_LOCK_LF_BLUR_SIGMA", 8.0, 0.0, 64.0)
    light_min = _env_float("VTON_PATTERN_LOCK_LIGHTING_MIN", 0.62, 0.0, 4.0)
    light_max = _env_float("VTON_PATTERN_LOCK_LIGHTING_MAX", 1.12, 0.0, 4.0)
    if light_min > light_max:
        # 行30: min>max 时 np.clip 把所有像素钉在 min(整体偏色)。交换并告警。
        logger.warning(
            "VTON_PATTERN_LOCK_LIGHTING_MIN(%s) > MAX(%s); swapping to keep lighting meaningful",
            light_min,
            light_max,
        )
        light_min, light_max = min(light_min, light_max), max(light_min, light_max)
    skin_w = _env_float("VTON_PATTERN_LOCK_SKIN_WEIGHT", 0.85, 0.0, 1.0)
    feather = _env_float("VTON_PATTERN_LOCK_FEATHER_PX", 5, 1.0, 64.0)
    skip_white = os.environ.get("VTON_PATTERN_LOCK_SKIP_WHITE_BG", "1") == "1"
    white_thresh = _env_float("VTON_PATTERN_LOCK_WHITE_THRESH", 245, 128, 255)

    if skip_white:
        ref_u8 = np.clip(ref, 0, 255).astype(np.uint8)
        near_white = (
            (ref_u8[..., 0] > white_thresh) & (ref_u8[..., 1] > white_thresh) & (ref_u8[..., 2] > white_thresh)
        ).astype(np.uint8)
        # 行26: 只在"非衣物区"找画布白。旧代码直接对 near_white 连通域 + 触边判定,
        # 白色/浅色衣物触到画布边就被整块当"画布白"清零 → 白衣印花锁不住被 SDXL 重画。
        # 先与 ~mask(mask=衣物 255)求交, 白衣像素不再参与画布白判定。
        garment = (mask_f32 > 127).astype(np.uint8)
        near_white = near_white * (1 - garment)
        _n_labels, labels = cv2.connectedComponents(near_white, connectivity=8)
        border_ids = set(labels[0].tolist())
        border_ids.update(labels[-1].tolist())
        border_ids.update(labels[:, 0].tolist())
        border_ids.update(labels[:, -1].tolist())
        border_ids.discard(0)
        canvas_white = np.isin(labels, list(border_ids)) if border_ids else np.zeros_like(near_white, dtype=bool)
        content = (~canvas_white).astype(np.float32)[..., None]
    else:
        content = np.ones((*ref.shape[:2], 1), dtype=np.float32)

    # 衣物 alpha(来自 warp 出的 RGBA 参考): alpha=0 处(画布白底/无衣料区,
    # 如参考图没覆盖到的袖子区)不贴入 —— 否则白底整块贴到模特身上 = 袖子白晕/
    # 脖子灰带根因。白衣 alpha=1 照常贴(比白底启发式更准, 二者相乘双保险)。
    # 必须取 A 通道: convert("L") 对 RGBA 取的是 RGB 亮度而非 alpha —— 深色
    # 衣(藏青/墨绿, luma 65-90)贴回不透明度被压到 0.25-0.35, gen 像素(肚皮
    # 肤色/白晕)透出 65%+, 漏肚脐/图案失真/结淡全由此出。
    if ref_alpha is not None:
        _a = np.asarray(ref_alpha.getchannel("A") if ref_alpha.mode in ("RGBA", "LA") else ref_alpha.convert("L"))
        if _a.shape[:2] != ref.shape[:2]:
            _a = np.array(Image.fromarray(_a).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR))
        # 半透边缘(alpha<150)不贴: mask 膨胀核(kx~56px)会让 mask 越过面料
        # 形边界, 半透 alpha(40-130)按 content 比例混入 = 袖缘浅蓝/杂色 fringe。
        _a = _a.copy()
        _a[_a < 150] = 0
        content = content * (_a.astype(np.float32) / 255.0)[..., None]

    mask_norm = (mask_f32 / 255.0)[..., None]
    edge_weight = mask_norm
    try:
        dist = cv2.distanceTransform((mask_f32 > 20).astype(np.uint8), cv2.DIST_L2, 5)
        if float(dist.max()) > 3:
            # 行29: FEATHER_PX=0 → dist/0 → NaN → uint8 未定义 → 边缘黑噪点。
            # _env_float 已把下限钉在 1.0, 这里再加 nan_to_num 双保险。
            edge_fall = np.nan_to_num(np.clip(dist / max(feather, 1.0), 0, 1), nan=1.0, posinf=1.0, neginf=0.0)
            edge_weight = edge_fall[..., None] * mask_norm
    except Exception:  # noqa: BLE001
        pass

    skin_factor = 1.0
    if skin_u8.any():
        skin_factor = 1.0 - (skin_u8.astype(np.float32) / 255.0)[..., None] * skin_w
        # 参考图有衣料的像素不降权: 袖条盖在生成裸臂上时仍贴袖料
        # (旧逻辑按生成图肤色降权 85% → 长袖参考有袖、出图仍无袖)。
        if ref_alpha is not None:
            _fa = np.asarray(ref_alpha.getchannel("A") if ref_alpha.mode == "RGBA" else ref_alpha.convert("L"))
            if _fa.shape[:2] != ref.shape[:2]:
                _fa = np.array(Image.fromarray(_fa).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR))
            fabric = (_fa > 40)[..., None]
            skin_factor = np.where(fabric, np.ones_like(skin_factor), skin_factor)

    lf_ref = cv2.GaussianBlur(ref, (0, 0), lf_blur)
    hf_ref = ref - lf_ref

    gen_gray = gen[..., 0] * 0.299 + gen[..., 1] * 0.587 + gen[..., 2] * 0.114
    # 身体尺度明暗场: gen 的 LF 亮度按衣料区中位数归一(不除 LF_ref ——
    # C14 实测教训: 平铺 ref 自身带影棚阴影, 藏青照片 LF 暗至 5-20, 除法让
    # ref 的拍摄光影以倒数混入比值场: 34% 像素钉在 +12% 高光帽、整件均值
    # 压暗 10 而空间结构没涨)。ref 出 albedo, gen 出空间明暗, 各管各的;
    # 中位数吸收 SDXL 全局亮暗差(胸口变白不回归), 非对称钳制: 阴影宽
    # (-38%, "穿上"的关键线索)、高光窄(+12%, 防发白)。
    lf_gen_gray = cv2.GaussianBlur(gen_gray, (0, 0), lf_blur)
    fabric_px = (mask_f32 > 127) & (content[..., 0] > 0.5)
    shade_med = float(np.median(lf_gen_gray[fabric_px])) if fabric_px.any() else float(np.median(lf_gen_gray))
    lighting = np.clip(lf_gen_gray / max(shade_med, 0.05), light_min, light_max)
    # 中位数归一只保中位=1: 左偏分布(暗尾)会整体压暗(C14 实测整件 -10)。
    # 单次按"裁剪后均值"再归一, 残余压到 ±3%(病态双峰 ±8%); 不迭代到底
    # —— 迭代收敛会把场的对比度压平(阴影被逐步抬回), 与传身体明暗相悖。
    lit_px = lighting[fabric_px] if fabric_px.any() else lighting
    lighting = np.clip(lighting / max(float(lit_px.mean()), 0.05), light_min, light_max)

    shade_lf = 1.0 + (lighting - 1.0) * ls * lf_strength
    shade_hf = 1.0 + (lighting - 1.0) * ls * hf_strength
    lit = np.clip(lf_ref * shade_lf[..., None] + hf_ref * shade_hf[..., None], 0, 255)

    fold_gain = _env_float("VTON_PATTERN_LOCK_FOLD_GAIN", 0.45, 0.0, 1.5)
    if fold_gain > 0:
        # 贴回面料叠加 gen 的中频褶皱光影(带通, 零均值): 平铺面料获得生成
        # 图的身体起伏/褶皱明暗 → "穿上去"而非"贴上去"(贴纸感根因之一是
        # 全量贴回后面料完全平光)。带通整体零均值不引入提亮(胸口变白不
        # 回归), 花纹仍被 HF 锁定。0=关闭(A/B 用)。
        folds = cv2.GaussianBlur(gen_gray, (0, 0), 10.0) - cv2.GaussianBlur(gen_gray, (0, 0), 35.0)
        lit = np.clip(lit + folds[..., None] * fold_gain, 0, 255)

    alpha = edge_weight * content * skin_factor
    out = np.nan_to_num(lit * alpha + gen * (1.0 - alpha), nan=0.0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def reapply_garment_detail(
    generated: Image.Image,
    garment_ref: Image.Image,
    mask_u8: np.ndarray,
    texture_strength: float = 0.70,
    color_strength: float = 0.85,
    pattern_lock: bool = False,
    light_strength: float = 1.0,
    ref_alpha: Image.Image | None = None,
) -> Image.Image:
    """AI 出骨架光影,代码保商品纹理颜色。

    pattern_lock=True: warp 后的 garment_ref 直接覆盖衣服区(锁死花纹/logo),
    生成图只按亮度比调制褶皱光影;mask 边缘仍衰减,接缝交给生成像素。
    pattern_lock=False: 高频柔光 + Reinhard 颜色迁移(旧路径)。

    ref_alpha(warp 出的 RGBA 参考, 可 None): 贴回与 Reinhard source 统计都按
    alpha 取"衣物内容"——alpha=0 的画布白底/无衣料区不贴入模特(修袖子白晕/
    脖子灰带), 也不进 LAB 颜色统计; alpha=1 的白衣照常贴(保护白衣印花)。
    None 时退回白底连通域启发式(非近白像素=内容, 阈值 245)。

    四步纯 CPU 后处理(SDXL 生成之后、身份锁之前执行):

    1. **高频纹理提取**: garment_ref(已对齐姿态的服装图)减去高斯模糊低频,
       得到花纹/面料质感的高频分量(居中到 128 中性灰)。
    2. **柔光混合贴回**: 把高频以 softlight 模式叠加到 generated 的衣服区,
       加回 SDXL 丢失的面料纹理/花纹细节(128 无影响,>128 提亮,<128 压暗)。
    3. **Reinhard LAB 颜色迁移**: 把生成图衣服区的颜色分布(LAB 均值/方差)
        拉到原衣的颜色统计 —— 等价颜色直方图匹配,强制颜色一致。
        两侧都用衣物区统计:target 取 mask 区,source 取衣料内容区
        (ref_alpha 或白底阈值),画布白不进统计。
    4. **mask 羽化混合**: 只动衣服区(mask 255),人脸/背景/手保持生成图原样;
       texture/color 各有独立强度,mask 边缘平滑过渡无硬边。

    无 cv2 时原样返回 generated(安全回退,不回归)。
    """
    try:
        import cv2
    except ImportError:
        return generated

    gen = np.array(generated.convert("RGB"), dtype=np.float32)
    ref = np.array(garment_ref.convert("RGB"), dtype=np.float32)
    h, w = gen.shape[:2]
    if ref.shape[:2] != (h, w):
        ref = np.array(
            Image.fromarray(ref.astype(np.uint8)).resize((w, h), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    mask = np.asarray(mask_u8, dtype=np.float32)
    if mask.shape[:2] != (h, w):
        mask = np.array(
            Image.fromarray(mask_u8.astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    mask_norm = (mask / 255.0)[..., None]

    skin = _detect_skin(gen.astype(np.uint8))

    if pattern_lock:
        # 行29: overlay 调用无守卫, 坏 env 值崩了 → 整步退回裸 SDXL(静默丢商品纹理)。
        # 上游已 fail-loud log, 这里保底回退到高频柔光路径而非裸生成图。
        try:
            return _pattern_lock_overlay(gen, ref, mask, skin, light_strength, ref_alpha=ref_alpha)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pattern_lock overlay failed (%s); falling back to softlight path", exc)

    edge_weight = mask_norm
    try:
        dist = cv2.distanceTransform((mask > 20).astype(np.uint8), cv2.DIST_L2, 5)
        d_max = float(dist.max())
        if d_max > 3:
            edge_fall = np.clip(dist / (d_max * 0.25), 0, 1)
            edge_weight = (edge_fall[..., None]) * mask_norm
    except Exception:  # noqa: BLE001
        pass

    skin_factor = 1.0
    if skin.any():
        skin_factor = 1.0 - (skin.astype(np.float32) / 255.0)[..., None] * 0.85

    # 平铺参考图的"衣料内容"区:Reinhard source 统计与高频只取衣物本体。
    # 白画布(常多于衣料像素)进 LAB 均值/方差会把整次颜色迁移拉白、失真;
    # 画布与衣料边界的高频梯度贴入模特会出白边晕染。
    if ref_alpha is not None:
        # 同 _pattern_lock_overlay: 必须取 A 通道。convert("L") 对 RGBA 取
        # RGB 亮度, 深色衣 src_content 几乎全 False(→ 全图兜底统计被画布白拉偏)。
        _a = np.asarray(ref_alpha.getchannel("A") if ref_alpha.mode in ("RGBA", "LA") else ref_alpha.convert("L"))
        if _a.shape[:2] != (h, w):
            _a = np.array(Image.fromarray(_a).resize((w, h), Image.Resampling.BILINEAR))
        src_content = _a > 127
    else:
        _ref_u8 = np.clip(ref, 0, 255).astype(np.uint8)
        # 245 同 pattern_lock 白画布阈值;白衣需走 ref_alpha 通路区分。
        src_content = _ref_u8.min(axis=2) < 245
    if float(src_content.mean()) < 0.05:
        src_content = np.ones((h, w), dtype=bool)
    try:
        _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        content_dil = cv2.dilate(src_content.astype(np.uint8), _k) > 0
    except Exception:  # noqa: BLE001
        content_dil = src_content

    low_freq = cv2.GaussianBlur(ref, (0, 0), 4.0)
    high_freq = ref - low_freq + 128.0
    high_freq[~content_dil] = 128.0

    softlight = gen + (high_freq - 128.0) / 128.0 * gen * (255.0 - gen) / 255.0
    softlight = np.clip(softlight, 0, 255)
    tex_a = edge_weight * texture_strength * skin_factor

    textured = softlight * tex_a + gen * (1.0 - tex_a)

    # 步骤3: Reinhard LAB 颜色迁移(等价颜色直方图匹配)。
    # source=ref(原衣颜色基准), target=textured(待校正); 只在 mask 区统计 target。
    target_rgb = np.clip(textured, 0, 255).astype(np.uint8)
    source_rgb = np.clip(ref, 0, 255).astype(np.uint8)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    m_bin = mask > 10
    if m_bin.any():
        t_mean = target_lab[m_bin].mean(axis=0)
        t_std = target_lab[m_bin].std(axis=0) + 1e-6
    else:
        flat = target_lab.reshape(-1, 3)
        t_mean = flat.mean(axis=0)
        t_std = flat.std(axis=0) + 1e-6
    # source 统计只在衣料内容区:整图统计会被画布白拉偏(白底占大半时尤甚)。
    s_flat = source_lab[src_content]
    if s_flat.shape[0] < 16:
        s_flat = source_lab.reshape(-1, 3)
    s_mean = s_flat.mean(axis=0)
    s_std = s_flat.std(axis=0) + 1e-6

    corrected_lab = (target_lab - t_mean) / t_std * s_std + s_mean
    corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    col_a = mask_norm * color_strength * skin_factor
    out = corrected_rgb * col_a + gen * (1.0 - col_a)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
