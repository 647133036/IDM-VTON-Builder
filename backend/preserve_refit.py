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
    color_thresh: float = 60.0,
    min_area_frac: float = 0.004,
    grow_px: int = 3,
) -> np.ndarray | None:
    """印花核心区: 与衣料基色色距大的近中性色/异色相实心块。

    v1 高频带把门襟/纽扣/褶皱全当印花(GPU r6 视觉验收: 门襟中线纵向
    拉丝); v2 基色色距过滤挡不住同色相褶皱阴影(真丝暗褶的色度偏移
    同样 >40)。v3 三轴联合, 商用印花语义:
    1) 色距 > color_thresh: 与 keep 区基色(中位 RGB)欧氏距离足够大;
    2) 近中性色(像素 RGB max-min < 40): 白/灰/黑印花, 面料同色相的
       褶皱阴影天然带面料色度, 不满足;
    3) 或色相偏移(像素色度向量与基色色度向量夹角 > 30 度): 彩色印花;
       褶皱阴影沿基色色相方向变暗/变亮, 色相守恒, 被排除。
    形态过滤: 3px 腐蚀存活 + 连通域面积 >= max(150, 8% 最大块) +
    bbox 短边 >= 8px(细缝线/纽扣碎片即便膨胀串联也会先在成分级被杀)。
    回膨胀(恢复腐蚀 + grow_px 外扩盖抗锯齿缝), 裁回 keep_mask。

    与 detect_print_roi(dilate-erode 边界带, 供局部 inpaint)互补,
    本函数返回印花本体实心区, 作为确定性贴印区域与印花保真分测区。
    纯色款/同色相暗褶面料(全中性灰底的深褶有残留风险)返回 None,
    调用方回退整件衣料贴回。

    Args:
        cloth_ref: warp 后的 SKU 参考图 (RGB/RGBA)
        keep_mask: 衣料保留区掩膜 (L/ndarray, 255=衣料)
        color_thresh: 与基色的欧氏色距阈值(0-255 RGB 尺度)
        min_area_frac: 候选/核心区面积占衣料区(keep)比下限, 低于此返回 None
        grow_px: 核心区外扩像素(边缘覆盖), 0=不扩

    Returns:
        uint8 核心区掩膜(255=贴印区), 纯色/失败返回 None
    """
    try:
        import cv2

        ref = np.asarray(cloth_ref.convert("RGB"), dtype=np.float32)
        mask = np.asarray(keep_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype("uint8")).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )
        # keep 内缩 4px + cand 腐蚀 3px 合计清 warp 白边距环: keep 的
        # dilate(alpha,6) 外沿是纯白画布 RGB(近中性+大色距, 会成环状
        # cand, 其凸包=整件衣 → 恒等整件贴回), 环宽=6px; 内缩 4px 留
        # 2px 残环由 cand 腐蚀(3px)清零。不能用衣料 alpha 门槛: 白墨
        # 印花与画布同色, garment_alpha 把笔触判成 alpha=0 透明洞(实测
        # min 0), 满印墨迹网连通触边, 任何 alpha 拓扑内外判定无解;
        # 内缩 8px 会误杀贴边真印花且小画布占比失真。
        ka_img = (mask > 127).astype(np.uint8)
        ka_inner = cv2.erode(ka_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        ka = ka_inner.reshape(-1)
        if ka.sum() < 64:
            return None
        keep_area = float(np.asarray(mask > 127).sum())
        px = ref.reshape(-1, 3)
        base = np.median(px[ka], axis=0)
        dist = np.linalg.norm(px - base, axis=-1)
        neutral = (px.max(-1) - px.min(-1)) < 40.0
        cv_px = px - px.mean(-1, keepdims=True)
        cv_base = base - float(base.mean())
        n_px = np.linalg.norm(cv_px, axis=-1)
        cos = (cv_px @ cv_base) / np.maximum(n_px * float(np.linalg.norm(cv_base)), 1e-6)
        hue_shift = cos < 0.866
        # 灰基色(白/灰衣)的 cv_base≈0, 色相轴失效: 彩色印花改判
        # "像素自身带显著色度"(褶皱在灰衣上是纯亮度变化, 天然无色度)
        base_neutral = float(base.max() - base.min()) < 15.0
        px_chromatic = n_px > 25.0
        # 亮向约束(彩色基色): 近中性印花必须比基色亮。深褶阴影脱饱和后
        # 同样过 dist+neutral 轴(1536 实测腹部 188x193 暗块), 亮度方向是
        # 印花(亮)/阴影(暗)的本质分界; 黑印花款回退整件贴回仍交付。
        bright = np.ones_like(neutral) if base_neutral else px.mean(-1) >= float(base.mean()) + 10.0
        cand = ka & (dist > color_thresh) & ((neutral & bright) | hue_shift | (base_neutral & px_chromatic))
        cand = cand.reshape(ref.shape[:2])
        # 候选/核心门槛按衣料区(keep)占比而非画布占比: FASHN 原生 576x768 画布下
        # warp 把印花面积缩 16 倍, 画布占比门把真印花判成纯色(r8 GPU 实测 core=None
        # 回退整件贴回, 印花全程被 hf/folds 调制融化)。衣料占比跨工作分辨率不变;
        # 绝对下限只拦纯噪声, 实质过滤由腐蚀+成分级 150px 面积门承担。
        if float(cand.sum()) < max(16.0, min_area_frac * keep_area):
            return None
        c8 = cand.astype(np.uint8) * 255
        # cand 腐蚀只清 1px 噪声: 白墨笔触本身仅 3-4px 细, 半径 2 的腐蚀
        # 把笔触网打成碎末(实测 135 个 <160px 碎片, 全过不了面积门 → 误判
        # 纯色); keep 内缩 4px 残下的 2px 白环由成分级 bbox 跨度门拒绝。
        er = cv2.erode(c8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if er.max() == 0:
            return None
        num, lab, stats, _ = cv2.connectedComponentsWithStats((er > 0).astype(np.uint8))
        if num <= 1:
            return None
        # keep(dilated-alpha) 外沿的纯白画布残环: 环宽仅 2px(内缩 4px 后)
        # 但包围整件衣, 凸包=整件 → 恒等整件贴回(灾难)。成分 bbox 跨过
        # keep 尺寸 70% 即拒: 真印花(胸印花 bbox 39%)安全余量充足。
        k_ys, k_xs = np.where(ka_img)
        k_w = float(k_xs.max() - k_xs.min())
        k_h = float(k_ys.max() - k_ys.min())
        # 成分级纹理门(笔触/高光判别): 白色印花笔触与丝绸高光斑在颜色轴上
        # 不可分(同为亮中性+大色距, GPU 实测高光斑全过 v3.1 颜色轴), 只有
        # 纹理可分 —— 笔触=边缘高能, 高光=平滑梯度。测度带 = cand ∩ 成分
        # 支撑(dilate23) 减成分自身: 印花候选的边界笔触/实心 logo 的边界
        # 落在带内(canny 边缘线就在 cand 边界上, erode 成分中心距自身边界
        # 天然 ≥3px, 固定外环带会错过边缘线), 高光斑梯度平缓 canny 不点火
        # (大面积实心 logo 的边缘密度不会被面积稀释)。
        gray = np.asarray(cloth_ref.convert("L"), dtype=np.uint8)
        if gray.shape[:2] != ref.shape[:2]:
            gray = np.asarray(cloth_ref.convert("L").resize((ref.shape[1], ref.shape[0])), dtype=np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90) > 0
        kd_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        cand_b = cand.copy()
        # 面积门用绝对 150px: 0.08x最大成分的相对门在有大型种子时
        # 把中尺寸印花片段(下腹印花腐蚀碎片)整批误杀(实测丢 2/4 印花);
        # 粉尘(≤20px)由绝对门拦, 大结构由 bbox 跨度门+纹理门+亮向轴拦。
        floor = 150.0
        seeds = np.zeros(ref.shape[:2], dtype=np.uint8)
        for i in range(1, num):
            if float(stats[i, cv2.CC_STAT_AREA]) < floor:
                continue
            if min(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])) < 8:
                continue
            if float(stats[i, cv2.CC_STAT_WIDTH]) > 0.7 * k_w or float(stats[i, cv2.CC_STAT_HEIGHT]) > 0.7 * k_h:
                continue
            comp = (lab == i).astype(np.uint8)
            band = cand_b & (cv2.dilate(comp, kd_out) > 0) & (lab != i)
            if band.sum() < 32:
                continue
            # canny 密度单判据: 笔触≥11%/实心logo边界≈49%(GPU+合成实测),
            # 高光斑≤2.1% —— 5% 门两侧各余量一倍; 拉普拉斯能量对边缘
            # 锐度敏感(LANCZOS 软化后 15.8 与高光 14 同档), 不做门槛。
            if float(edges[band].mean()) * 100.0 < 5.0:
                continue
            seeds[lab == i] = 1
        # 种子发芽: 笔触网被 cand 腐蚀(1px)碎成大量 <floor 片段, 但整个
        # 网络在 cand 上连通 —— 种子所在的 cand 成分整块并入, 碎片随
        # 种子复活(实测: 只用腐蚀成分会丢 60% 印花面积, 鬼影残留);
        # 噪声粉尘与种子不相交, 留在门外。凸包按 cand 成分逐个填充:
        # 细笔触印花只贴笔触会留 FASHN 鬼影(NCC -0.4), 凸包=印花场
        # 整场贴参考原样。
        if seeds.max() == 0:
            return None
        _cnum, clab, _, _ = cv2.connectedComponentsWithStats((cand_b > 0).astype(np.uint8))
        sel_ids = {int(v) for v in np.unique(clab[seeds > 0])} - {0}
        hulls = []
        for cid in sel_ids:
            pts = np.column_stack(np.where(clab == cid))
            if len(pts) < 32:
                continue
            hull = cv2.convexHull(pts[:, ::-1].astype(np.int32).reshape(-1, 1, 2))
            hulls.append(hull)
        if not hulls:
            return None
        field = np.zeros(ref.shape[:2], dtype=np.uint8)
        cv2.fillPoly(field, hulls, 255)
        if float((field > 0).sum()) < max(32.0, min_area_frac * keep_area):
            return None
        roi = cv2.dilate(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if grow_px > 0:
            roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1)))
        return np.minimum(roi, mask.astype(np.uint8))
    except Exception:
        logger.exception("print_core_mask failed")
        return None


def ghost_print_mask(
    gen_img: Image.Image,
    cloth_ref: Image.Image,
    keep_mask: np.ndarray | Image.Image,
    exclude_mask: np.ndarray | Image.Image | None = None,
) -> np.ndarray | None:
    """生成图上的印花鬼影检测区(FASHN 自绘印花与 TPS 贴印几何错位时擦除用)。

    FASHN 生成器按自己的理解摆放印花(实测比平铺几何偏上), 贴印核心只盖
    TPS 映射位置 → 生成图的鬼影整套暴露在核心场之外(视觉验收: 上方
    半透明猴头重影)。本函数在生成图上找鬼影笔画: 同色轴(亮中性+大色距,
    基色取参考图衣料区中位数 —— prewhiten 已证两图基色一致)+ bbox
    跨度门(整件环状结构); 无纹理门(生成图丝绸渲染处处带噪, 鬼影 4.3
    与高光 4.5 lapE 同档, 实测色轴零高光假阳性)。调用方用参考内容
    贴回该区(ref 在此处是干净底色, folds/hf 默认保身体光影) 即完成擦除。

    Returns:
        uint8 掩膜(255=鬼影擦除区), 无检出/失败返回 None
    """
    try:
        import cv2

        gen = np.asarray(gen_img.convert("RGB"), dtype=np.float32)
        ref = np.asarray(cloth_ref.convert("RGB"), dtype=np.float32)
        mask = np.asarray(keep_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != gen.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype("uint8")).resize((gen.shape[1], gen.shape[0]), Image.Resampling.BILINEAR)
            )
        ka_img = (mask > 127).astype(np.uint8)
        ka_inner = cv2.erode(ka_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        ka = ka_inner.reshape(-1)
        if ka.sum() < 64:
            return None
        px_ref = ref.reshape(-1, 3)
        base = np.median(px_ref[ka], axis=0)
        px = gen.reshape(-1, 3)
        dist = np.linalg.norm(px - base, axis=-1)
        neutral = (px.max(-1) - px.min(-1)) < 40.0
        bright = px.mean(-1) >= float(base.mean()) + 10.0
        cand = (ka & (dist > 60.0) & (neutral & bright)).reshape(gen.shape[:2])
        if exclude_mask is not None:
            ex = np.asarray(exclude_mask)
            if ex.ndim == 3:
                ex = ex[..., 0]
            if ex.shape[:2] != gen.shape[:2]:
                ex = np.array(
                    Image.fromarray(ex.astype("uint8")).resize((gen.shape[1], gen.shape[0]), Image.Resampling.BILINEAR)
                )
            cand = cand & (
                cv2.dilate((ex > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) == 0
            )
        if float(cand.sum()) < 32:
            return None
        er = cv2.erode(cand.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if er.max() == 0:
            return None
        num, lab, stats, _ = cv2.connectedComponentsWithStats((er > 0).astype(np.uint8))
        if num <= 1:
            return None
        k_ys, k_xs = np.where(ka_img)
        k_w = float(k_xs.max() - k_xs.min())
        k_h = float(k_ys.max() - k_ys.min())
        # 无纹理门: 生成图丝绸渲染处处带噪(lapE 鬼影 4.3-5.2 vs 高光 4.5,
        # canny10/30 0.7-18% vs 7.8% 同档互叠), 纹理无法区分; 色轴已足:
        # 生成图暖调高光(230,215,175 类)不过中性轴, 实测 gen 侧 cand 只有
        # 鬼影+印花残迹, 零高光假阳性。擦除语义兜底: 贴回 ref+LF —— ref
        # 有的内容重画为正确版本, ref 没有的才是擦幻觉。
        hulls = []
        for i in range(1, num):
            if float(stats[i, cv2.CC_STAT_AREA]) < 150:
                continue
            if min(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])) < 8:
                continue
            if float(stats[i, cv2.CC_STAT_WIDTH]) > 0.7 * k_w or float(stats[i, cv2.CC_STAT_HEIGHT]) > 0.7 * k_h:
                continue
            pts = np.column_stack(np.where(lab == i))
            hulls.append(cv2.convexHull(pts[:, ::-1].astype(np.int32).reshape(-1, 1, 2)))
        if not hulls:
            return None
        field = np.zeros(gen.shape[:2], dtype=np.uint8)
        cv2.fillPoly(field, hulls, 255)
        roi = cv2.dilate(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        return np.minimum(roi, mask.astype(np.uint8))
    except Exception:
        logger.exception("ghost_print_mask failed")
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
        else:
            # 白墨印花洞修复: 四角背景色估计把与画布同色的印花笔触判成
            # 透明(真实 SKU alpha 实测 min 0), 贴回 content 门把笔触整
            # 个跳过 → 只贴笔触间基色, 白印花全程留 FASHN 鬼影(r9 GPU
            # 实测 alpha<0.5 区 NCC 0.05, |diff| 102)。
            # 连通性(任何 alpha 阈值)在此失败: 满印 SKU 笔触网触到衣料
            # 轮廓, 画布/羽化/笔触连成一个边框连通块(GPU 实测 holes=0)。
            # 改用 solid 质量(alpha>=150)洪泛: 从边框外洪泛 alpha<150 的
            # 外部区域, 被芥末黄质量包围而未触达的 = 内部洞(印花/白色
            # 部件), 补实心并连带 3px 笔触羽化。
            try:
                import cv2

                a_np = np.asarray(alpha_ch).copy()
                h_, w_ = a_np.shape
                edge = np.zeros((h_, w_), dtype=bool)
                edge[0, :] = a_np[0, :] < 150
                edge[-1, :] = a_np[-1, :] < 150
                edge[:, 0] = a_np[:, 0] < 150
                edge[:, -1] = a_np[:, -1] < 150
                if edge.any():
                    _ys, _xs = np.where(edge)
                    seed_x, seed_y = int(_xs[0]), int(_ys[0])
                    ff = (a_np >= 150).astype(np.uint8)
                    ffmask = np.zeros((h_ + 2, w_ + 2), np.uint8)
                    cv2.floodFill(ff, ffmask, (seed_x, seed_y), 2)
                    holes = (ff == 0) & (a_np < 150)
                    if holes.any():
                        _k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        raise_px = cv2.dilate(holes.astype(np.uint8), _k3) > 0
                        a_np[raise_px & (a_np < 150)] = 255
                        alpha_ch = Image.fromarray(a_np)
            except Exception:  # noqa: BLE001
                logger.debug("refit alpha hole repair skipped")
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
