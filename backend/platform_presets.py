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

import io
import logging
import os
import threading
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.platform_presets")

# u2net(主体掩膜)模型放 NAS 持久目录, 换机不丢、免重下; 缺失才让 rembg 自动下载。
_REPO_ROOT = Path(__file__).resolve().parent.parent
_U2NET_ROOT = _REPO_ROOT / "models" / "seg"

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
) -> Image.Image:
    """把 result 中肤色像素的色调校正到 original 的肤色色调。

    SDXL 重画手臂皮肤时常丢失原始肤色(发白发灰)。此函数用 YCbCr 肤色检测
    找到两图的皮肤区,计算 original 肤色均值,把 result 肤色向其偏移。
    肤色掩码经 GaussianBlur 羽化,避免校正边界硬切。

    脸区排除:SDXL 通常正确生成脸部肤色,全局平移会让脸色不自然。
    用前景 bbox 顶部 18% 估算脸区,从校正统计和校正应用两端都排除。
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
    if not orig_skin.any() or not res_skin.any():
        return result

    # 排除脸区:前景 bbox 顶部 18% 估算脸区,统计和应用两端都排除。
    h, w = res.shape[:2]
    non_white = res.mean(axis=2) < 240
    face_mask = np.zeros((h, w), dtype=np.uint8)
    if non_white.any():
        ys = np.where(non_white.any(axis=1))[0]
        face_h = max(1, int((ys[-1] - ys[0]) * 0.18))
        face_mask[ys[0] : ys[0] + face_h, :] = 1
    orig_skin = orig_skin * (1 - face_mask)
    res_skin = res_skin * (1 - face_mask)
    if not orig_skin.any() or not res_skin.any():
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
) -> np.ndarray:
    """构建聚焦 inpaint 遮罩:接缝带 + 肤色异常区。

    不再用整张 shared_mask(会重画整个衣服区),改为只针对真正有问题的区域:
    1. 接缝带:mask 边界 dilate-erode 带(同 _seam_repair)
    2. 肤色异常区:result 中皮肤像素与 original 对应像素 L1 距离 > thresh,
       排除脸区(前景顶部 18%)
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
) -> Image.Image:
    """SDXL Inpainting 二次修复:低降噪强度局部重绘接缝/肤色异常区。

    需要 SDXL Inpainting 模型 (stable-diffusion-xl-1.0-inpainting-0.1)。
    用 StableDiffusionXLInpaintPipeline 对聚焦 mask 区域做低强度(0.35)重绘,
    模型主要基于周边已有像素做微调,而非彻底重画。

    v0.1.7 改进:
    - 遮罩从整张 shared_mask 改为接缝带 + 肤色异常区(_build_inpaint_mask),
      避免重画整个衣服区。
    - 管线单例(_INPAINT_PIPE + _INPAINT_LOCK),不再每次调用 from_pretrained + del。
    - original 参数传入原图,用于检测肤色异常区。

    启用条件:
      1. VTON_INPAINT_REPAIR=1 环境变量开启
      2. SDXL Inpainting 模型已下载到 models/sdxl/sdxl_inpaint/
      3. torch + diffusers 可用

    未满足时原样返回(安全回退,不回归)。
    """
    global _INPAINT_PIPE

    if os.environ.get("VTON_INPAINT_REPAIR", "0") != "1":
        return image

    try:
        import cv2

        focused = _build_inpaint_mask(image, mask_u8, original=original)
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

                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                _INPAINT_PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(
                    str(inpaint_dir), torch_dtype=dtype, local_files_only=True
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
    global _FG_SESSION, _FG_SESSION_WARNED
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
            _FG_SESSION = new_session("u2net", sess_opts=so) if so is not None else new_session("u2net")
        except Exception:  # noqa: BLE001
            if not _FG_SESSION_WARNED:
                logger.warning("rembg u2net session 初始化失败;force_white 回退纯颜色压白")
                _FG_SESSION_WARNED = True
            return None
    try:
        cut = remove(img, session=_FG_SESSION)
        alpha = np.array(cut.convert("RGBA"))[..., 3]
        m = (alpha > 127).astype(np.uint8)
        if m.mean() < min_foreground_frac:
            return None  # 几乎全前景/异常,不信任
        return m
    except Exception:  # noqa: BLE001
        return None


def detect_garment_extent(garment: Image.Image) -> dict:
    """从平铺衣料图分析袖长和衣长。

    返回:
        sleeve: "long" | "short" | "sleeveless"
        hem: "waist" | "long" (衣长是否过腰)
    用于动态 clothing_mask:长袖→mask 含手臂,短袖→只含大臂,无袖→不含手臂。
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
        return {"sleeve": "short", "hem": "waist"}

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    ch = max(1, y1 - y0)
    cw = max(1, x1 - x0)
    aspect = ch / cw

    top_band = content[y0 : y0 + ch // 4, x0:x1]
    mid_band = content[y0 + ch // 3 : y0 + 2 * ch // 3, x0:x1]
    bot_band = content[y0 + 3 * ch // 4 : y1, x0:x1]
    top_w = float(top_band.mean())
    mid_w = float(mid_band.mean())
    bot_w = float(bot_band.mean())

    shoulder_spread = top_w / max(mid_w, 0.01)

    sleeve = "short"
    if aspect > 1.35 or bot_w > mid_w * 0.6:
        sleeve = "long"
    elif shoulder_spread < 1.15 or top_w < 0.15:
        sleeve = "sleeveless"

    hem = "long" if aspect > 1.6 else "waist"

    return {"sleeve": sleeve, "hem": hem, "aspect": round(aspect, 2)}


def clothing_mask(
    img: Image.Image,
    garment_type: str = "upper",
    subject_mask: np.ndarray | None = None,
    face_margin: float = 0.35,
    garment_extent: dict | None = None,
) -> np.ndarray | None:
    """服装替换区:主体减脸(及按品类+袖长动态裁切),返回 uint8 0-255;构不出则 None。

    这是「只换衣、锁身份」的遮罩:255=贴生成衣物,0=保留原图像素(脸/发/手/背景)。
    garment_extent(from detect_garment_extent)控制手臂区纳入范围:
      long sleeve → 含整条手臂(62%→88%)
      short sleeve → 只含大臂(62%,当前默认)
      sleeveless → 不含手臂(55%)
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    subj = subject_mask if subject_mask is not None else foreground_mask(img)
    if subj is None:
        return None
    subj = (np.asarray(subj) > 0).astype(np.uint8)
    if subj.shape[:2] != (h, w) or float(subj.mean()) < 0.04:
        return None
    face = _detect_face_mask(arr, margin_frac=face_margin, down_margin_frac=0.08)
    cloth = subj.copy()
    if face is not None and face.any():
        cloth[face > 0] = 0
    else:
        # cv2 5.0 fallback: CascadeClassifier 被移除,_detect_face_mask 返回全零。
        # 用主体 bbox 顶部 18% 估算脸区,保护脸/头发不被服装遮罩覆盖。
        heur = _heuristic_face_mask(arr, subj)
        if heur.any():
            cloth[heur > 0] = 0
    ys = np.where(subj > 0)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(1, y1 - y0)

    sleeve = garment_extent.get("sleeve", "short") if garment_extent else "short"
    hem = garment_extent.get("hem", "waist") if garment_extent else "waist"

    if garment_type in ("upper", "auto"):
        # 衣长(hem)决定躯干覆盖范围;袖长(sleeve)微调手臂区。
        # hem=long → 长款上衣/束腰衣,垂过裤腰(解决"沿裤子齐边剪"问题)。
        # sleeve=long → 略下延含手臂区(72%,替代原 88% 过度延伸躯干)。
        if hem == "long":
            cloth[y0 + int(span * 0.80) :] = 0
        elif sleeve == "sleeveless":
            cloth[y0 + int(span * 0.55) :] = 0
        elif sleeve == "long":
            cloth[y0 + int(span * 0.72) :] = 0
        else:
            cloth[y0 + int(span * 0.62) :] = 0
    elif garment_type == "dress":
        if hem == "long":
            cloth[y0 + int(span * 0.96) :] = 0
        else:
            cloth[y0 + int(span * 0.82) :] = 0
    elif garment_type == "lower":
        cloth[: y0 + int(span * 0.48)] = 0
    elif garment_type == "onesie":
        if sleeve == "long":
            cloth[y0 + int(span * 0.95) :] = 0
        else:
            cloth[y0 + int(span * 0.82) :] = 0
    if float(cloth.mean()) < 0.02:
        return None

    # 短袖/无袖:排除手臂皮肤。检测原图肤色(YCbCr),从 mask 两侧减去
    # 皮肤像素 → 手臂皮肤标记为 0(保留原图),只有衣物区标记为 255(允许
    # 生成替换)。躯干中心不减 → 胸口 V 领露肤仍可被新衣物覆盖。
    # 长袖不减:新衣物需覆盖整条手臂,手臂皮肤应被生成结果替换。
    # 用主体水平外扩 25% 定位手臂区(而非固定图像宽度比例)。
    if garment_type in ("upper", "auto") and sleeve in ("short", "sleeveless"):
        skin = _detect_skin(arr)
        if skin.any():
            xs = np.where(subj.any(axis=0))[0]
            if len(xs):
                x0s, x1s = int(xs.min()), int(xs.max())
                subj_w = max(1, x1s - x0s)
                arm_w = max(8, int(subj_w * 0.25))
                left_end = x0s + arm_w
                right_start = x1s - arm_w
                cloth[:, :left_end][skin[:, :left_end] > 0] = 0
                cloth[:, right_start:][skin[:, right_start:] > 0] = 0

    try:
        import cv2

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        cloth = cv2.erode(cloth, k, iterations=1)
        if float(cloth.mean()) < 0.015:
            return None
        return cv2.GaussianBlur((cloth * 255).astype(np.uint8), (0, 0), 6)
    except Exception:  # noqa: BLE001
        return (cloth * 255).astype(np.uint8)


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


def _pattern_lock_overlay(
    gen: np.ndarray,
    ref: np.ndarray,
    mask_f32: np.ndarray,
    skin_u8: np.ndarray,
    light_strength: float,
) -> Image.Image:
    """Warp 后的衣覆盖衣服区;生成图只提供褶皱光影(PLTON 频率分离)。

    lighting = clamp(L_gen / L_ref, lighting_min, lighting_max)
    shade_lf = 1 + (lighting - 1) * light_strength * lf_strength
    shade_hf = 1 + (lighting - 1) * light_strength * hf_strength
    lit = LF_ref * shade_lf + HF_ref * shade_hf
    底色按 lf_strength 调,花纹按 hf_strength 调(默认半强度,保对比度)。
    light_strength=0 时 lit=cloth_ref(纯贴图,快速回退基准)。

    环境变量:
      VTON_PATTERN_LOCK_LIGHT_STRENGTH  (1.0)  光照总闸:0=精确 ref,1=全效
      VTON_PATTERN_LOCK_LF_STRENGTH     (1.0)  底色光照响应系数(1=全强度)
      VTON_PATTERN_LOCK_HF_STRENGTH     (0.5)  花纹光照响应系数(0=不受光照,1=全强度)
      VTON_PATTERN_LOCK_LF_BLUR_SIGMA   (8.0)  低频提取高斯 sigma
      VTON_PATTERN_LOCK_LIGHTING_MIN    (0.75) 亮度比下限(clamp 作用于 lighting)
      VTON_PATTERN_LOCK_LIGHTING_MAX    (1.25) 亮度比上限
      VTON_PATTERN_LOCK_SKIN_WEIGHT     (0.85) 肤色降权
      VTON_PATTERN_LOCK_FEATHER_PX      (5)    mask 边缘羽化像素
      VTON_PATTERN_LOCK_SKIP_WHITE_BG   (1)    跳过画布边缘连通白底
      VTON_PATTERN_LOCK_WHITE_THRESH    (245)  白底 RGB 判定阈值
    """
    import os

    import cv2

    env_ls = os.environ.get("VTON_PATTERN_LOCK_LIGHT_STRENGTH")
    ls_val = float(env_ls) if env_ls else light_strength
    ls = float(np.clip(ls_val, 0.0, 1.0))
    lf_strength = float(os.environ.get("VTON_PATTERN_LOCK_LF_STRENGTH", "1.0"))
    hf_strength = float(os.environ.get("VTON_PATTERN_LOCK_HF_STRENGTH", "0.5"))
    lf_blur = float(os.environ.get("VTON_PATTERN_LOCK_LF_BLUR_SIGMA", "8.0"))
    light_min = float(os.environ.get("VTON_PATTERN_LOCK_LIGHTING_MIN", "0.75"))
    light_max = float(os.environ.get("VTON_PATTERN_LOCK_LIGHTING_MAX", "1.25"))
    skin_w = float(os.environ.get("VTON_PATTERN_LOCK_SKIN_WEIGHT", "0.85"))
    feather = float(os.environ.get("VTON_PATTERN_LOCK_FEATHER_PX", "5"))
    skip_white = os.environ.get("VTON_PATTERN_LOCK_SKIP_WHITE_BG", "1") == "1"
    white_thresh = int(os.environ.get("VTON_PATTERN_LOCK_WHITE_THRESH", "245"))

    if skip_white:
        ref_u8 = np.clip(ref, 0, 255).astype(np.uint8)
        near_white = (
            (ref_u8[..., 0] > white_thresh) & (ref_u8[..., 1] > white_thresh) & (ref_u8[..., 2] > white_thresh)
        ).astype(np.uint8)
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

    mask_norm = (mask_f32 / 255.0)[..., None]
    edge_weight = mask_norm
    try:
        dist = cv2.distanceTransform((mask_f32 > 20).astype(np.uint8), cv2.DIST_L2, 5)
        if float(dist.max()) > 3:
            edge_fall = np.clip(dist / feather, 0, 1)
            edge_weight = edge_fall[..., None] * mask_norm
    except Exception:  # noqa: BLE001
        pass

    skin_factor = 1.0
    if skin_u8.any():
        skin_factor = 1.0 - (skin_u8.astype(np.float32) / 255.0)[..., None] * skin_w

    lf_ref = cv2.GaussianBlur(ref, (0, 0), lf_blur)
    hf_ref = ref - lf_ref

    gen_gray = gen[..., 0] * 0.299 + gen[..., 1] * 0.587 + gen[..., 2] * 0.114
    ref_gray = ref[..., 0] * 0.299 + ref[..., 1] * 0.587 + ref[..., 2] * 0.114
    lighting = np.clip(gen_gray / (ref_gray + 1.0), light_min, light_max)

    shade_lf = 1.0 + (lighting - 1.0) * ls * lf_strength
    shade_hf = 1.0 + (lighting - 1.0) * ls * hf_strength
    lit = np.clip(lf_ref * shade_lf[..., None] + hf_ref * shade_hf[..., None], 0, 255)

    alpha = edge_weight * content * skin_factor
    out = lit * alpha + gen * (1.0 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def reapply_garment_detail(
    generated: Image.Image,
    garment_ref: Image.Image,
    mask_u8: np.ndarray,
    texture_strength: float = 0.70,
    color_strength: float = 0.85,
    pattern_lock: bool = False,
    light_strength: float = 1.0,
) -> Image.Image:
    """AI 出骨架光影,代码保商品纹理颜色。

    pattern_lock=True: warp 后的 garment_ref 直接覆盖衣服区(锁死花纹/logo),
    生成图只按亮度比调制褶皱光影;mask 边缘仍衰减,接缝交给生成像素。
    pattern_lock=False: 高频柔光 + Reinhard 颜色迁移(旧路径)。

    四步纯 CPU 后处理(SDXL 生成之后、身份锁之前执行):

    1. **高频纹理提取**: garment_ref(已对齐姿态的服装图)减去高斯模糊低频,
       得到花纹/面料质感的高频分量(居中到 128 中性灰)。
    2. **柔光混合贴回**: 把高频以 softlight 模式叠加到 generated 的衣服区,
       加回 SDXL 丢失的面料纹理/花纹细节(128 无影响,>128 提亮,<128 压暗)。
    3. **Reinhard LAB 颜色迁移**: 把生成图衣服区的颜色分布(LAB 均值/方差)
       拉到原衣的颜色统计 —— 等价颜色直方图匹配,强制颜色一致。
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
        return _pattern_lock_overlay(gen, ref, mask, skin, light_strength)

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

    low_freq = cv2.GaussianBlur(ref, (0, 0), 4.0)
    high_freq = ref - low_freq + 128.0

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
    s_flat = source_lab.reshape(-1, 3)
    s_mean = s_flat.mean(axis=0)
    s_std = s_flat.std(axis=0) + 1e-6

    corrected_lab = (target_lab - t_mean) / t_std * s_std + s_mean
    corrected_lab = np.clip(corrected_lab, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    col_a = mask_norm * color_strength * skin_factor
    out = corrected_rgb * col_a + gen * (1.0 - col_a)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
