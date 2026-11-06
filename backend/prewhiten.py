"""输入预白底(pipeline audit P0-3 / TSV 行4 + 守卫 行41)。

灰 cyclorama / 街拍灰背景直进引擎会在头后留下灰块(REPORT §2.4),
khaite 5 张里 1→4 的收益全靠这一步。历史上它只活在归档临时脚本
(run_b1.py / cpu_postfix.py), repo 与批处理都没有 —— 本模块把它
固化进 repo, 作为引擎的**强制输入步骤**。

核心算法 = 四角 flood-fill(tol=20, FLOODFILL_FIXED_RANGE), 与
run_b1.py 同款, 保护浅色衣物(只有与画布角连通的近背景色块被压白)。

关键改进(行41): tol=20 裸用有双向风险 —— 非 uniform/米色背景要么没压净,
要么把 tol 提到 45 反过来吃掉浅色衣物(REPORT:63 踩过)。所以加两道守卫,
越界**告警不静默**:
  1. changed-pixel 比例守卫: 压白像素占比过低(背景本就近白, no-op)或过高
     (flood-fill 把主体也吞了)都拦下 —— 过高时**拒绝本次预白底**, 原图返回。
  2. u2net 主体交集守卫: 压白区若与 u2net 前景掩膜显著重叠, 说明吃了主体,
     同样**拒绝**。u2net 不可用(取不到主体掩膜)或掩膜尺寸不符时 **fail-closed 拒绝**
     (远端复核#4: 拿不到主体 = 无法证明没啃主体, 裸跑 changed_frac 守卫会漏过
     浅色成衣面料被啃 <85% 的情况) —— 返回原图, 不静默降级。

被拒绝时返回原图(安全 no-op, 不会比现状更差)并打 warning, 绝不静默产出
"头后灰块"或"浅色衣被啃"的图。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.prewhiten")

# 四角 flood-fill 种子点(与 run_b1.py / cpu_postfix.py 一致)。
_CORNERS = "corners"


def _flood_fill_white(arr: np.ndarray, tol: int) -> tuple[np.ndarray, np.ndarray]:
    """四角 flood-fill 把近背景色压成纯白。

    返回 (whitened_uint8, changed_bool_mask)。cv2 不可用时抛 ImportError 由调用方处理。
    """
    import cv2

    im = arr.copy()
    h, w = im.shape[:2]
    flags = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flags[sy, sx]:
            continue
        cv2.floodFill(
            im,
            flags,
            (sx, sy),
            (255, 255, 255),
            (tol, tol, tol),
            (tol, tol, tol),
            cv2.FLOODFILL_FIXED_RANGE,
        )
    changed = np.any(im != arr, axis=2)
    return im, changed


def prewhiten(img: Image.Image, tol: int = 20) -> Image.Image:
    """裸四角 flood-fill 预白底(无守卫)。保留给已知干净 studio 图与测试用。"""
    arr = np.array(img.convert("RGB"))
    out, _ = _flood_fill_white(arr, tol)
    return Image.fromarray(out)


def prewhiten_guarded(
    img: Image.Image,
    tol: int = 20,
    min_changed_frac: float = 0.005,
    max_changed_frac: float = 0.85,
    max_subject_overlap: float = 0.05,
    subject_mask: np.ndarray | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """带守卫的预白底。返回 (image, info)。

    info 字段:
      status: applied | already_white | rejected_changed_ratio | rejected_subject_overlap | error
      tol, changed_frac: 本次压白像素占比
      subject_overlap: 压白区与主体掩膜的交集占主体比例(None=守卫跳过)
      reason: 人读原因(告警时非空)
    """
    info: dict[str, Any] = {"status": "applied", "tol": tol, "changed_frac": 0.0, "subject_overlap": None, "reason": ""}
    try:
        arr = np.array(img.convert("RGB"))
        out, changed = _flood_fill_white(arr, tol)
    except ImportError:
        # 无 cv2: 无法 flood-fill, 原图返回(安全 no-op)。
        info["status"] = "error"
        info["reason"] = "cv2 unavailable; prewhiten skipped"
        logger.warning("prewhiten: %s", info["reason"])
        return img, info
    except Exception as e:  # noqa: BLE001
        info["status"] = "error"
        info["reason"] = f"flood-fill failed: {type(e).__name__}: {e}"
        logger.warning("prewhiten: %s", info["reason"])
        return img, info

    changed_frac = float(changed.mean()) if changed.size else 0.0
    info["changed_frac"] = round(changed_frac, 4)

    # 守卫 1: changed-pixel 比例(行41)。
    if changed_frac < min_changed_frac:
        info["status"] = "already_white"
        info["reason"] = f"background already near-white (changed_frac={changed_frac:.4f} < {min_changed_frac})"
        logger.info("prewhiten: %s", info["reason"])
        return img, info
    if changed_frac > max_changed_frac:
        info["status"] = "rejected_changed_ratio"
        info["reason"] = (
            f"flood-fill ate {changed_frac:.2%} of pixels (> {max_changed_frac:.0%}); "
            "refusing to prewhiten — background likely non-uniform or subject-colored. "
            "Returning ORIGINAL image (lower tol or clean the backdrop first)."
        )
        logger.warning("prewhiten GUARD: %s", info["reason"])
        return img, info

    # 守卫 2: u2net 主体交集(行41 + 远端复核#4)。fail-closed: 拿不到主体掩膜 = 无法证明
    # 没啃到主体, 拒绝本次预白底(返回原图), 不裸跑 changed_frac 守卫(浅色成衣面料被啃
    # <85% 会漏过); 尺寸不符同理(掩膜与图错位时交集无意义)。
    if subject_mask is None:
        try:
            from platform_presets import foreground_mask

            subject_mask = foreground_mask(img)
        except Exception:  # noqa: BLE001
            subject_mask = None
    if subject_mask is None:
        info["subject_overlap"] = None
        info["status"] = "rejected_subject_overlap"
        info["reason"] = (
            "subject mask unavailable (u2net/rembg failed); refusing to prewhiten "
            "without the subject guard (fail-closed)"
        )
        logger.warning("prewhiten GUARD: %s", info["reason"])
        return img, info
    subj = np.asarray(subject_mask) > 0
    if subj.shape != changed.shape:
        info["subject_overlap"] = None
        info["status"] = "rejected_subject_overlap"
        info["reason"] = (
            f"subject mask shape {subj.shape} != changed {changed.shape}; "
            "refusing to prewhiten without a valid subject guard (fail-closed)"
        )
        logger.warning("prewhiten GUARD: %s", info["reason"])
        return img, info
    subj_area = int(subj.sum())
    overlap = float((changed & subj).sum()) / max(1, subj_area)
    info["subject_overlap"] = round(overlap, 4)
    if overlap > max_subject_overlap:
        info["status"] = "rejected_subject_overlap"
        info["reason"] = (
            f"flood-fill overlaps {overlap:.2%} of the u2net subject (> {max_subject_overlap:.0%}); "
            "refusing to prewhiten — it would eat light-colored garment/body. "
            "Returning ORIGINAL image."
        )
        logger.warning("prewhiten GUARD: %s", info["reason"])
        return img, info

    return Image.fromarray(out), info


def prewhiten_input(img: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """引擎强制输入步骤入口。VTON_PREWHITEN=0 可关闭(默认开)。

    tol 可用 VTON_PREWHITEN_TOL 覆盖(默认 20, 与已验证交付一致)。
    """
    if os.environ.get("VTON_PREWHITEN", "1") == "0":
        return img, {"status": "disabled", "reason": "VTON_PREWHITEN=0"}
    raw_tol = os.environ.get("VTON_PREWHITEN_TOL", "20").strip()
    try:
        tol = int(raw_tol)
    except ValueError:
        logger.warning("Invalid VTON_PREWHITEN_TOL=%r; using 20", raw_tol)
        tol = 20
    return prewhiten_guarded(img, tol=tol)
