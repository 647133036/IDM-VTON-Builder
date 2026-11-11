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
            dilated = cv2.dilate(mask, k)
            # 边界带 = 膨胀带 - 原始遮罩 (薄环)。此前误把整块膨胀区传给
            # cv2.inpaint, TELEA 把整件衣服当待修复区, 从四周白背景重建
            # 出白色糊块 (GPU 实测: apply 区 0.37 画布全部变白)。
            boundary = ((dilated > 20) & (mask <= 20)).astype(np.uint8) * 255
            protect_mask = (mask <= 20).astype(np.uint8)
            protect_mask[boundary > 0] = 0  # 边界带不进保护, 留给 Telea
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

        # 源锚点 (x, y) 在平铺图上, 目标锚点在画布上
        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        # 逆向 TPS: 以目标锚点为节点、源锚点为值, 求 g: 画布坐标 -> 平铺图坐标,
        # remap 时对每个画布像素取 g 求采样位置(逆向映射, 高频印花不走样)。
        # Bookstein 方程组: [[K, P^T], [P, 0]] (n+3)x(n+3), P = [1; x; y] (3, n)。
        #
        # 坐标归一化: 像素尺度下 K 项 ~ r^2 log r (1e6) 与 P 项 (~1e2) 混尺度,
        # cond(M) 实测 ~1e13, 同锚点跨 LAPACK 版本解发散(GPU r5 实测 warp 塌成
        # 斜条, fidelity 0.498)。统一缩到 [0,1]^2 求解再反缩放, 节点精确性不变。
        n = len(src)
        dst_min = dst.min(axis=0)
        dst_rng = np.maximum(dst.max(axis=0) - dst_min, 1e-9)
        src_min = src.min(axis=0)
        src_rng = np.maximum(src.max(axis=0) - src_min, 1e-9)
        dst_n = (dst - dst_min) / dst_rng
        src_n = (src - src_min) / src_rng

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = _rbf_kernel(dst_n[i], dst_n[j])
        P = np.vstack([np.ones((1, n)), dst_n.T])
        M = np.zeros((n + 3, n + 3))
        M[:n, :n] = K
        M[:n, n:] = P.T
        M[n:, :n] = P
        rhs = np.zeros((n + 3, 2))
        rhs[:n, :] = src_n
        sol = np.linalg.solve(M, rhs)
        # 求解守卫: 病态锚点(共线/重复)可能静默解出垃圾, 残差检查兜底,
        # 触发即抛错走 LANCZOS 回退, 不把烂映射交给贴印。
        _resid = float(np.abs(M @ sol - rhs).max())
        if not np.isfinite(_resid) or _resid > 1e-6:
            raise ValueError(f"TPS solve residual too large: {_resid}")
        w_coeff = sol[:n, :]
        aff = sol[n:, :]

        # 生成目标像素网格
        xg, yg = np.meshgrid(np.arange(canvas_w, dtype=np.float64), np.arange(canvas_h, dtype=np.float64))
        gx = np.stack([xg, yg], axis=-1).reshape(-1, 2)
        gx_n = (gx - dst_min) / dst_rng

        # TPS 映射: 目标画布 -> 源平铺图
        K_g = np.zeros((len(gx), n))
        for i in range(n):
            K_g[:, i] = _rbf_kernel_batch(gx_n, dst_n[i])
        G = np.hstack([np.ones((len(gx), 1)), gx_n])
        src_coords = (G @ aff + K_g @ w_coeff) * src_rng + src_min

        # 采样越界裁剪前先记 valid 掩膜: np.clip 会把画布外采样点硬夹到
        # 源图边缘(边缘涂抹), 对 alpha 全开的纯色衣会把整张画布涂满;
        # 越界像素必须显式清零(透明), 不能复用边缘值。
        # 半像素容差: TPS 数值噪声让边界锚点坐标出现 -1e-6 / (w-1)+1e-6,
        # 严格 >=0 会把本该有效的边界像素清零(锚点对齐回归)。
        valid = (
            (src_coords[:, 0] >= -0.5)
            & (src_coords[:, 0] <= w - 0.5)
            & (src_coords[:, 1] >= -0.5)
            & (src_coords[:, 1] <= h - 0.5)
        ).reshape(canvas_h, canvas_w)

        sx = np.clip(src_coords[:, 0], 0, w - 1).astype(np.float32).reshape(canvas_h, canvas_w)
        sy = np.clip(src_coords[:, 1], 0, h - 1).astype(np.float32).reshape(canvas_h, canvas_w)

        map_x = sx
        map_y = sy
        warped = cv2.remap(
            arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        warped[~valid] = 0
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


def print_core_mask(
    cloth_ref: Image.Image,
    keep_mask: np.ndarray | Image.Image,
    hf_thresh: float = 12.0,
    min_area_frac: float = 0.004,
    grow_px: int = 3,
) -> np.ndarray | None:
    """印花核心区(全量掩膜, 非边界带): warp 参考图在保留区内的高频区。

    与 detect_print_roi(dilate-erode 边界带, 供局部 inpaint)互补:
    本函数返回印花本体(logo/字母/大图案/纹理)的实心区, 作为确定性
    贴印区域与印花保真分的测区; grow_px 外扩盖住抗锯齿缝并裁回
    keep_mask, 不越出衣料。纯色款(高频不足)返回 None, 调用方回退
    整件衣料贴回。

    Args:
        cloth_ref: warp 后的 SKU 参考图 (RGB/RGBA)
        keep_mask: 衣料保留区掩膜 (L/ndarray, 255=衣料)
        hf_thresh: 高频能量阈值(0-255 灰度差)
        min_area_frac: 核心区面积占比下限, 低于此返回 None
        grow_px: 核心区外扩像素(边缘覆盖), 0=不扩

    Returns:
        uint8 核心区掩膜(255=贴印区), 纯色/失败返回 None
    """
    try:
        import cv2

        ref = np.asarray(cloth_ref.convert("L"), dtype=np.float32)
        mask = np.asarray(keep_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype("uint8")).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )
        blur = cv2.GaussianBlur(ref, (0, 0), 4.0)
        hf = np.abs(ref - blur)
        roi = ((hf > hf_thresh) & (mask > 127)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)
        if float(roi.mean()) / 255.0 < min_area_frac:
            return None
        if grow_px > 0:
            ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1))
            roi = np.minimum(cv2.dilate(roi, ke), mask.astype(np.uint8))
        return roi
    except Exception:
        logger.exception("print_core_mask failed")
        return None


def clip_mask_to_garment_alpha(
    apply_mask: np.ndarray | Image.Image | None,
    ref_alpha: Image.Image | None,
    grow_px: int = 6,
) -> np.ndarray | Image.Image | None:
    """mask ∩ 衣料 alpha(膨胀余量): 纯裁剪, 不并集不外扩。

    与 intersect_mask_with_garment_alpha 相反方向的守卫: 那个把 mask
    并上实心 alpha 再裁(为 SDXL 模板路径兜住宽松款悬垂面料), 本函数
    只做 min(mask, dilate(alpha)) —— FASHN 路径衣服由生成器画定,
    检测区就是衣服边界, 再向模板 alpha 形并集会把贴印推进空气里
    (GPU 实测 apply 0.161→0.3685, 印花重影)。
    """
    if apply_mask is None or ref_alpha is None:
        return apply_mask
    try:
        import cv2
    except ImportError:
        return apply_mask
    m = np.asarray(apply_mask)
    if m.ndim == 3:
        m = m[..., 0]
    h, w = m.shape
    ra_src = ref_alpha.getchannel("A") if ref_alpha.mode == "RGBA" else ref_alpha.convert("L")
    ra = np.asarray(ra_src, dtype=np.uint8)
    if ra.shape != (h, w):
        ra = np.asarray(ra_src.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, grow_px) * 2 + 1, max(1, grow_px) * 2 + 1))
    ra_dil = cv2.dilate(ra, ke)
    return np.minimum(m, ra_dil)


def refit_cloth_to_drawn_garment(
    garment: Image.Image,
    canvas_size: tuple[int, int],
    drawn_mask: np.ndarray | Image.Image,
) -> tuple[Image.Image, Image.Image] | None:
    """把平铺衣重投到生成器实际画的衣服几何(bbox 八锚点 TPS)。

    身体模板先验(torso_box)与 FASHN 实际画的衣服存在落位/比例差,
    模板 warp 的衣料形直接当贴回区会把印花贴到空气里(重影/气泡感,
    GPU 实测 fidelity 0.847)。这里以生成图检测出的衣服区(drawn_mask)
    bbox 为目标, 四角+四边中点共 8 锚点 TPS 把平铺衣投到同一几何,
    贴印/保真分全部以新几何为准; 生成图检测区即衣服边界, 后续用
    clip_mask_to_garment_alpha 纯裁剪(不并集)。

    Args:
        garment: 平铺衣 RGB (白底商品图)
        canvas_size: 生成图尺寸 (w, h)
        drawn_mask: 生成图检测出的衣服区 (L/ndarray, 255=衣服)

    Returns:
        (RGB 白底参考, RGBA alpha 参考) 同 align_garment_to_pose 口径;
        drawn_mask 无效/退化/TPS 失败时 None, 调用方回退模板路径。
    """
    try:
        from engine import garment_alpha
    except ImportError:
        logger.debug("refit_cloth_to_drawn_garment: engine.garment_alpha unavailable")
        return None
    m = np.asarray(drawn_mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != (canvas_size[1], canvas_size[0]):
        logger.warning("refit_cloth_to_drawn_garment: mask shape %s != canvas %s", m.shape, canvas_size)
        return None
    ys, xs = np.where(m > 127)
    if len(ys) < 16:
        return None
    dx0, dx1 = float(xs.min()), float(xs.max())
    dy0, dy1 = float(ys.min()), float(ys.max())
    cw, ch = float(canvas_size[0]), float(canvas_size[1])
    dw, dh = dx1 - dx0, dy1 - dy0
    if dw < 8 or dh < 8:
        return None
    if not (0.04 <= dw / cw <= 0.95 and 0.05 <= dh / ch <= 0.98):
        logger.info("refit_cloth_to_drawn_garment: drawn bbox degenerate (%.2f, %.2f); fallback", dw / cw, dh / ch)
        return None

    g = garment.convert("RGBA")
    if int(g.split()[-1].getextrema()[0]) >= 250:
        alpha_ch = garment_alpha(garment)
        # 纯色无白底: 色距 alpha 可能全零, 回退实心衣料(整图即内容)
        if np.asarray(alpha_ch).max() <= 10:
            alpha_ch = Image.new("L", g.size, 255)
        g.putalpha(alpha_ch)
    a = np.asarray(g.split()[-1])
    ays, axs = np.where(a > 10)
    if len(ays) == 0:
        axs = np.array([0, g.size[0] - 1])
        ays = np.array([0, g.size[1] - 1])
    sx0, sx1 = float(axs.min()), float(axs.max())
    sy0, sy1 = float(ays.min()), float(ays.max())
    sw, sh = max(1.0, sx1 - sx0), max(1.0, sy1 - sy0)

    fracs = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (1.0, 0.5), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    src_pts = [(sx0 + fx * sw, sy0 + fy * sh) for fx, fy in fracs]
    dst_pts = [(dx0 + fx * dw, dy0 + fy * dh) for fx, fy in fracs]
    try:
        warped = tps_warp(g, src_pts, dst_pts, canvas_size)
    except Exception:
        logger.exception("refit_cloth_to_drawn_garment: tps failed; fallback to template path")
        return None
    if warped.mode != "RGBA":
        warped = warped.convert("RGBA")
    # 覆盖率守卫: warp 塌缩(数值病态/锚点退化)时 alpha 只剩零散斜条,
    # 贴印/保真分会拿到稀疏 sliver(GPU r5 实测 frac 0.011 vs 预期 0.14,
    # fidelity 0.498); 低于 dst bbox 面积 x 源内容填充率的一半即判失败,
    # 回退模板路径。
    wa = np.asarray(warped.split()[-1])
    dst_frac = (dw * dh) / (cw * ch)
    src_fill = len(axs) / max(1.0, sw * sh)
    if float((wa > 10).mean()) < 0.5 * dst_frac * src_fill:
        logger.warning("refit_cloth_to_drawn_garment: warped coverage collapsed; fallback to template path")
        return None
    out = Image.new("RGB", canvas_size, (255, 255, 255))
    out.paste(warped, (0, 0), warped)
    return out, warped


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
