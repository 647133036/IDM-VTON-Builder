from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.preserve_refit")

# 贴印参考图长边下限: 禁止先缩到 1024 再 warp 丢掉印花高频。
REFIT_REF_LONG_SIDE_MIN = 1024


def protect_inverse_refit(
    result: Image.Image,
    source: Image.Image,
    recast_mask: Image.Image,
    feather_px: int = 4,
) -> Image.Image:
    """遮罩外保护贴回: 源图像素贴回结果图, 仅重绘区域内保留生成像素。

    替代 blend_tryon + _seam_repair 两步为一步:
    1. protect_mask = 全图 - (recast_mask 膨胀 feather_px)
    2. result[protect_mask] = source[protect_mask]
    3. 边界带(膨胀带 - 未膨胀带)用 cv2.inpaint TELEA 平滑过渡

    脸、发丝、手、露肤手臂、背景保持源图像素; 遮罩内保留生成像素。
    VTON_LOCK_ID=0 时调用方跳过本函数。

    Args:
        result: 生成图 (RGB)
        source: 源图 (RGB), 与 result 同尺寸
        recast_mask: 重绘区域掩膜 (L, 255=重绘, 0=保护)
        feather_px: 边界带宽度, 0=无羽化直接贴回

    Returns:
        RGB 贴回结果
    """
    res = result.convert("RGB")
    src = source.convert("RGB")
    if res.size != src.size:
        src = src.resize(res.size, Image.Resampling.LANCZOS)
    r_arr = np.array(res, dtype=np.uint8)
    s_arr = np.array(src, dtype=np.uint8)

    mask = np.asarray(recast_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != r_arr.shape[:2]:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((r_arr.shape[1], r_arr.shape[0]), Image.Resampling.BILINEAR)
        )

    # 保护区域 = 全图 - recast_mask(膨胀羽化带)
    if feather_px > 0:
        try:
            import cv2

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
            dilated = cv2.dilate(mask, k)
            boundary = (dilated > 20).astype(np.uint8) * 255
            protect_mask = ((mask <= 20) | (boundary > 20)).astype(np.uint8)
            protect_mask[boundary > 20] = 0  # 边界带不进保护, 留给 Telea
        except ImportError:
            protect_mask = (mask <= 20).astype(np.uint8)
            boundary = None
    else:
        protect_mask = (mask <= 20).astype(np.uint8)
        boundary = None

    # 贴回源图像素到保护区域
    out = r_arr.copy()
    out[protect_mask > 0] = s_arr[protect_mask > 0]

    # 边界带 Telea 接缝修复
    if boundary is not None and boundary.any():
        try:
            import cv2

            out = cv2.inpaint(out, boundary, 5, cv2.INPAINT_TELEA)
        except Exception:  # noqa: BLE001
            pass

    return Image.fromarray(out, "RGB")


def tps_warp(
    ref_img: Image.Image,
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    canvas_size: tuple[int, int],
) -> Image.Image:
    """RBF TPS 几何 warp: 把 SKU 平铺图投到人体表面。

    控制点来自 RTMPose 关键点 + 服装轮廓采样。TPS 比网格 remap 更平滑,
    高频印花不随扩散重解释。

    Args:
        ref_img: SKU 平铺图 (RGBA)
        src_pts: 平铺图锚点 (领口/肩/下摆/袖口), 至少 4 个
        dst_pts: 人体对应位置, 与 src_pts 等长
        canvas_size: 输出画布 (w, h)

    Returns:
        RGBA warp canvas, 透明背景
    """
    if len(src_pts) < 4 or len(src_pts) != len(dst_pts):
        logger.warning("tps_warp: need >= 4 anchor pairs, got %d; returning original", len(src_pts))
        return ref_img.resize(canvas_size, Image.Resampling.LANCZOS)

    try:
        import cv2

        arr = np.array(ref_img, dtype=np.float32)
        h, w = ref_img.size[1], ref_img.size[0]
        canvas_w, canvas_h = canvas_size

        # 源锚点 (x, y) 在平铺图上
        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        # 构建 TPS: 仿射 + RBF 径向基
        n = len(src)
        A = np.zeros((n, 2))
        A[:, 0] = src[:, 0]
        A[:, 1] = src[:, 1]
        B = np.zeros((n, 2))
        B[:, 0] = dst[:, 0]
        B[:, 1] = dst[:, 1]

        # 求解 TPS 系数
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = _rbf_kernel(A[i], A[j])
        P = np.vstack([np.ones((1, n)), A.T])
        M = np.zeros((n + 2, n + 2))
        M[2:, 2:] = K
        M[:2, 2:] = P
        M[2:, :2] = P.T
        rhs = np.zeros((n + 2, 2))
        rhs[2:, :] = B
        sol = np.linalg.solve(M, rhs)
        a0 = sol[0, :]
        a1 = sol[1, :]
        w_coeff = sol[2:, :]

        # 生成目标像素网格
        xg, yg = np.meshgrid(np.arange(canvas_w, dtype=np.float64), np.arange(canvas_h, dtype=np.float64))
        gx = np.stack([xg, yg], axis=-1).reshape(-1, 2)

        # TPS 映射: target -> source
        K_g = np.zeros((len(gx), n))
        for i in range(n):
            K_g[:, i] = _rbf_kernel_batch(gx, A[i])
        src_coords = a0 + a1 @ gx.T
        src_coords += K_g @ w_coeff
        src_coords = src_coords.T

        sx = np.clip(src_coords[:, 0], 0, w - 1).astype(np.float32).reshape(canvas_h, canvas_w)
        sy = np.clip(src_coords[:, 1], 0, h - 1).astype(np.float32).reshape(canvas_h, canvas_w)

        map_x = sx
        map_y = sy
        warped = cv2.remap(
            arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        return Image.fromarray(warped.astype(np.uint8), "RGBA")
    except Exception:
        logger.exception("tps_warp failed; falling back to LANCZOS resize")
        return ref_img.resize(canvas_size, Image.Resampling.LANCZOS)


def _rbf_kernel(p1: np.ndarray, p2: np.ndarray) -> float:
    """Thin-plate-spline RBF kernel: r^2 * log(r)."""
    r = float(np.linalg.norm(p1 - p2))
    if r < 1e-12:
        return 0.0
    return r * r * np.log(r)


def _rbf_kernel_batch(pts: np.ndarray, center: np.ndarray) -> np.ndarray:
    """向量化 RBF kernel: 多个点到单个中心."""
    d = pts - center
    r = np.linalg.norm(d, axis=1)
    out = np.zeros_like(r)
    valid = r > 1e-12
    r2 = r[valid]
    out[valid] = r2 * r2 * np.log(r2)
    return out


def detect_print_roi(
    cloth_ref: Image.Image,
    apply_mask: Image.Image,
    edge_px: int = 4,
    hf_thresh: float = 12.0,
    min_area_frac: float = 0.002,
) -> np.ndarray | None:
    """印花 ROI 边界带: warp 参考图在贴回区内的高频区(logo/字母/大图案)的边缘带。

    高频 = |灰度 - 高斯模糊| 超阈值的区域(印花/文字/缝合线), 收缩到贴回
    mask 内, 形态学去噪后取 dilate-erode 边界带。字母/图案内部像素由
    确定性贴印(pattern_lock/TPS)构成, 不进 inpaint——只有贴印区与生成
    衣料的边界带进局部 inpaint 做混合。

    Args:
        cloth_ref: warp 后的 SKU 平铺参考图 (RGB/RGBA)
        apply_mask: 贴回区域掩膜 (L, 255=重绘)
        edge_px: 边界带半宽(px)
        hf_thresh: 高频能量阈值(0-255 灰度差)
        min_area_frac: ROI 面积占比下限, 低于此返回 None(纯色款无印花)

    Returns:
        uint8 边界带掩膜(255=进 inpaint), 纯色/失败返回 None
    """
    try:
        import cv2

        ref = np.asarray(cloth_ref.convert("L"), dtype=np.float32)
        mask = np.asarray(apply_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )

        blur = cv2.GaussianBlur(ref, (0, 0), 4.0)
        hf = np.abs(ref - blur)
        roi = ((hf > hf_thresh) & (mask > 127)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)

        if float(roi.mean()) / 255.0 < min_area_frac:
            return None

        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px * 2 + 1, edge_px * 2 + 1))
        dil = cv2.dilate(roi, ke)
        ero = cv2.erode(roi, ke)
        band = cv2.subtract(dil, ero)
        return band
    except Exception:
        logger.exception("detect_print_roi failed")
        return None


def optical_flow_refit(
    refit_img: Image.Image,
    gen_img: Image.Image,
    recast_mask: Image.Image,
) -> Image.Image:
    """Farneback 光流微移: 把贴印像素向生成图褶皱方向微移, 局部对齐。

    在重绘区域内计算光流, 微移贴印像素使褶皱与生成图一致, 不引入新像素。

    Args:
        refit_img: TPS 贴印结果 (RGBA)
        gen_img: 扩散生成图 (RGB), 与 refit_img 同尺寸
        recast_mask: 重绘区域掩膜 (L, 255=重绘)

    Returns:
        光流微移后的 RGBA 图
    """
    try:
        import cv2

        ref = np.array(refit_img, dtype=np.uint8)
        gen = np.array(gen_img, dtype=np.uint8)
        if gen.ndim == 2:
            gen = np.stack([gen] * 3, axis=2)
        if ref.ndim == 2:
            ref = np.stack([ref] * 3, axis=2)

        mask = np.asarray(recast_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )

        if not (mask > 20).any():
            return refit_img

        ref_gray = (
            cv2.cvtColor(ref[..., :3], cv2.COLOR_RGB2GRAY) if ref.shape[2] > 3 else ref.mean(axis=2).astype(np.uint8)
        )
        gen_gray = cv2.cvtColor(gen[..., :3], cv2.COLOR_RGB2GRAY) if gen.ndim == 3 else gen

        flow = cv2.calcOpticalFlowFarneback(
            ref_gray,
            gen_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # 微移: 光流 * 0.5 (保守, 避免吃进非褶皱区)
        h, w = ref.shape[:2]
        xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = np.clip(xg + flow[..., 0] * 0.5, 0, w - 1).astype(np.float32)
        map_y = np.clip(yg + flow[..., 1] * 0.5, 0, h - 1).astype(np.float32)

        warped = cv2.remap(ref, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # 仅重绘区域内应用光流结果, 保护区域保持原贴印
        out = ref.copy()
        region = mask > 20
        out[region] = warped[region]
        return Image.fromarray(out, "RGBA")
    except Exception:
        logger.exception("optical_flow_refit failed; returning unmodified refit")
        return refit_img


def run_multiseed(
    task_fn,
    seeds: list[int],
    scorer,
) -> tuple[Image.Image, list[tuple[int, float]], int]:
    """多 seed 出图按质检分排序, 返回最高分结果。

    Args:
        task_fn: 可调 task_fn(seed) -> Image.Image
        seeds: seed 列表
        scorer: 可调 scorer(image, seed) -> float (质检分, 越高越好)

    Returns:
        (best_result, all_scores, chosen_seed)
    """
    results = []
    for seed in seeds:
        img = task_fn(seed)
        score = float(scorer(img, seed))
        results.append((seed, score, img))
    results.sort(key=lambda x: x[1], reverse=True)
    best_seed, best_score, best_img = results[0]
    all_scores = [(s, sc) for s, sc, _ in results]
    logger.info("multiseed: %d seeds, chosen=%d score=%.4f", len(seeds), best_seed, best_score)
    return best_img, all_scores, best_seed


def refit_seeds_default() -> list[int]:
    """默认多 seed 列表: VTON_MULTI_SEED 控制数量, 纯色 =1."""
    n = int(os.environ.get("VTON_MULTI_SEED", "3"))
    n = max(1, min(n, 8))
    return [42 + i * 1000 for i in range(n)]
