from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.quality_gate")

# 质检分阈值: 低于此分进 review 档
DEFAULT_SSIM_THRESHOLD = 0.70
DEFAULT_OCR_THRESHOLD = 0.80


def ssim_score(ref_roi: Image.Image, out_roi: Image.Image) -> float:
    """结构相似度: 比较贴印参考 ROI 与结果 ROI, 返回 0-1。

    图案类 SKU 用 SSIM 判断印花排布是否保留。
    低于阈值的任务进复核档。

    scikit-image 不可用时回退到灰度归一化互相关 (NCC) 的近似 SSIM,
    保证纯 numpy/PIL 环境也能出分。
    """
    ref = np.array(ref_roi.convert("RGB")).astype(np.float32)
    out = np.array(out_roi.convert("RGB")).astype(np.float32)
    if out.shape != ref.shape:
        out = np.array(
            Image.fromarray(out.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]), Image.Resampling.LANCZOS)
        ).astype(np.float32)
    try:
        from skimage.metrics import structural_similarity as ssim

        score = float(ssim(ref, out, channel_axis=2))
        return max(0.0, min(1.0, score))
    except ImportError:
        return _fallback_ssim(ref, out)
    except Exception:
        logger.exception("ssim_score failed")
        return 0.0


def _fallback_ssim(ref: np.ndarray, out: np.ndarray) -> float:
    """无 skimage 时: 灰度均值/方差 SSIM (Wang et al. 核心公式, C1/C2 默认)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    gray_ref = ref.mean(axis=2)
    gray_out = out.mean(axis=2)
    mu_r = gray_ref.mean()
    mu_o = gray_out.mean()
    sigma_r = gray_ref.std()
    sigma_o = gray_out.std()
    sigma_ro = ((gray_ref - mu_r) * (gray_out - mu_o)).mean()
    num = (2 * mu_r * mu_o + C1) * (2 * sigma_ro + C2)
    den = (mu_r**2 + mu_o**2 + C1) * (sigma_r**2 + sigma_o**2 + C2)
    return max(0.0, min(1.0, float(num / den)))


def ocr_score(ref_roi: Image.Image, out_roi: Image.Image) -> float:
    """OCR 逐字比对: 平铺图文字 vs 结果图文字, 返回 0-1。

    用 Apache 2.0 的 Tesseract 或 PaddleOCR。缺省时返回 0.0 (任务进 review)。
    """
    ref_text = _ocr_engine(ref_roi)
    out_text = _ocr_engine(out_roi)
    if not ref_text or not out_text:
        return 0.0
    # 逐字比对: 用 Levenshtein 距离 / 参考长度
    import difflib

    ratio = difflib.SequenceMatcher(None, ref_text, out_text).ratio()
    return max(0.0, min(1.0, float(ratio)))


def _ocr_engine(img: Image.Image) -> str:
    """调用 OCR 引擎, 缺省时返回空串。"""
    try:
        import pytesseract

        return pytesseract.image_to_string(img).strip()
    except ImportError:
        pass
    try:
        import paddleocr

        ocr = paddleocr.PaddleOCR(use_angle_cls=False, lang="en")
        res = ocr.ocr(np.array(img), cls=False)
        lines = []
        if res and res[0]:
            for item in res[0]:
                if item and len(item) >= 2:
                    lines.append(item[1])
        return " ".join(lines).strip()
    except ImportError:
        logger.warning("no OCR engine available (pytesseract/paddleocr)")
    except Exception:
        logger.exception("OCR engine failed")
    return ""


def delivery_tier(score: float, sku_has_text: bool = False) -> str:
    """按质检分 + SKU 类型分档: auto / review。

    - 纯色 SKU: 无质检分, 直接 auto。
    - 图案类 (SSIM): score >= SSIM_THRESHOLD → auto, 否则 review。
    - 文字类 (OCR): score >= OCR_THRESHOLD → auto, 否则 review。
    """
    if not sku_has_text:
        threshold = float(os.environ.get("VTON_QG_SSIM", str(DEFAULT_SSIM_THRESHOLD)))
    else:
        threshold = float(os.environ.get("VTON_QG_OCR", str(DEFAULT_OCR_THRESHOLD)))
    return "auto" if score >= threshold else "review"


def score_for_sku(ref_roi: Image.Image, out_roi: Image.Image, sku_has_text: bool = False) -> tuple[float, str]:
    """对 SKU 计算质检分 + 档位, 返回 (score, tier)。"""
    score = ocr_score(ref_roi, out_roi) if sku_has_text else ssim_score(ref_roi, out_roi)
    tier = delivery_tier(score, sku_has_text)
    return score, tier
