from __future__ import annotations

import os

import numpy as np
import preserve_refit
import quality_gate
from PIL import Image

# ---------- protect_inverse_refit ----------


def _make_pair(size=(64, 64)):
    """result=全白, source=全黑, mask=左半 255(重绘) 右半 0(保护)."""
    w, h = size
    result = Image.new("RGB", (w, h), (255, 255, 255))
    source = Image.new("RGB", (w, h), (0, 0, 0))
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[:, : w // 2] = 255  # 左半重绘, 右半保护
    mask = Image.fromarray(mask_arr, "L")
    return result, source, mask


def test_protect_inverse_refit_keeps_source_in_protect_region():
    result, source, mask = _make_pair()
    out = preserve_refit.protect_inverse_refit(result, source, mask, feather_px=0)
    arr = np.array(out)
    # 右半(保护)应为源图(全黑), 左半(重绘)应为结果(全白)
    w_half = arr.shape[1] // 2
    assert arr[:, :w_half].mean() > 200
    assert arr[:, w_half:].mean() < 55


def test_protect_inverse_refit_telea_boundary_smooths():
    result, source, mask = _make_pair()
    out = preserve_refit.protect_inverse_refit(result, source, mask, feather_px=3)
    arr = np.array(out)
    h, w = arr.shape[:2]
    mid = w // 2
    # Telea 只修保护侧边界薄环(mask 边缘外 feather_px=3, 即 mid..mid+2):
    # apply 侧保持生成白, 环内产生白→黑过渡, 环外贴源图黑。
    assert (arr[h // 2, mid - 4 : mid, 0] == 255).all(), "apply side of band must keep generated pixels"
    assert arr[h // 2, mid : mid + 3, 0].min() < 255, "ring must be Telea-smoothed transition"
    assert arr[h // 2, mid + 5, 0] == 0, "beyond ring must be pasted source pixels"


def test_protect_inverse_refit_feather_zero_no_telea():
    result, source, mask = _make_pair()
    out = preserve_refit.protect_inverse_refit(result, source, mask, feather_px=0)
    arr = np.array(out)
    h, w = arr.shape[:2]
    mid = w // 2
    # 无羽化时, 保护区像素 = 源图(全黑), 重绘区像素 = 结果(全白)
    # mid-2 在重绘区(左半 mask=255), 应保留结果白; mid+2 在保护区(右半 mask=0), 贴源图黑
    assert arr[h // 2, mid - 2, 0] == 255
    assert arr[h // 2, mid + 2, 0] == 0


def test_protect_inverse_refit_telea_preserves_apply_interior():
    """Telea 只许修边界薄环; apply 区内部必须保留生成像素。

    GPU 回归: 此前把整块膨胀区当 inpaint 掩膜, TELEA 把整件衣服
    从四周白背景重建, apply 区 0.37 画布全部变白(白漏斗)。
    """
    result, source, mask = _make_pair()
    out = preserve_refit.protect_inverse_refit(result, source, mask, feather_px=3)
    arr = np.array(out)
    w = arr.shape[1]
    # apply 区内部(左半, 远离边界带): 保持 result 全白
    assert arr[:, : w // 2 - 8].mean() > 250, "apply interior must keep generated pixels"
    # 保护区内部(右半, 远离边界带): 贴回 source 全黑
    assert arr[:, w // 2 + 8 :].mean() < 5


def test_protect_inverse_refit_size_mismatch_resizes_source():
    result = Image.new("RGB", (32, 32), (255, 255, 255))
    source = Image.new("RGB", (16, 16), (0, 0, 0))
    mask = Image.new("L", (32, 32), 0)
    mask.paste(255, (0, 0, 16, 32))  # 左半重绘
    out = preserve_refit.protect_inverse_refit(result, source, mask, feather_px=0)
    assert out.size == (32, 32)


def test_protect_inverse_refit_all_protect_returns_source():
    result = Image.new("RGB", (16, 16), (255, 0, 0))
    source = Image.new("RGB", (16, 16), (0, 255, 0))
    mask = Image.new("L", (16, 16), 0)  # 全保护
    out = preserve_refit.protect_inverse_refit(result, source, mask, feather_px=0)
    arr = np.array(out)
    assert arr[8, 8, 1] == 255  # 绿色通道


# ---------- tps_warp ----------


def test_tps_warp_anchor_alignment():
    """4 锚点 TPS: 平铺图角点映射到画布对应位置."""
    ref = Image.new("RGBA", (100, 200), (200, 100, 50, 255))
    # 平铺图 4 角 -> 画布 4 角 (warp 到 64x64)
    src = [(0, 0), (100, 0), (100, 200), (0, 200)]
    dst = [(0, 0), (64, 0), (64, 64), (0, 64)]
    out = preserve_refit.tps_warp(ref, src, dst, (64, 64))
    assert out.size == (64, 64)
    arr = np.array(out)
    # 画布四角应有 alpha=255 (来自 ref 的角)
    assert arr[0, 0, 3] > 127
    assert arr[0, -1, 3] > 127
    assert arr[-1, 0, 3] > 127
    assert arr[-1, -1, 3] > 127


def test_tps_warp_fewer_than_4_anchors_falls_back():
    ref = Image.new("RGBA", (100, 200), (200, 100, 50, 255))
    src = [(0, 0), (100, 0), (0, 200)]  # 仅 3 个
    dst = [(0, 0), (64, 0), (0, 64)]
    out = preserve_refit.tps_warp(ref, src, dst, (64, 64))
    assert out.size == (64, 64)  # 回退 LANCZOS resize


def test_tps_warp_mismatched_anchors_falls_back():
    ref = Image.new("RGBA", (100, 200), (200, 100, 50, 255))
    src = [(0, 0), (100, 0), (100, 200), (0, 200)]
    dst = [(0, 0), (64, 0)]  # 数量不等
    out = preserve_refit.tps_warp(ref, src, dst, (64, 64))
    assert out.size == (64, 64)


def test_tps_warp_identity_mapping():
    """src == dst 时, 中心像素颜色应保持不变."""
    ref = Image.new("RGBA", (64, 64), (120, 200, 30, 255))
    pts = [(10, 10), (54, 10), (54, 54), (10, 54)]
    out = preserve_refit.tps_warp(ref, pts, pts, (64, 64))
    arr = np.array(out)
    center = arr[32, 32, :3]
    assert abs(int(center[0]) - 120) < 30
    assert abs(int(center[1]) - 200) < 30


def test_tps_warp_node_exactness_distinguishes_from_lanczos():
    """TPS 在锚点节点处必须精确映射: 节点标记色落在目标锚点位置。

    回归: 旧实现 TPS 系统阵尺寸错((n+2) vs P(3,n)), 100% 抛 ValueError 走
    LANCZOS 兜底, 全部既有测试因纯色图而无法区分。本测试用非均匀锚点,
    LANCZOS 等比缩放会把标记放在比例位置, 与目标锚点位置不同。
    """
    w, h = 100, 200
    ref = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    r = np.zeros((h, w, 4), dtype=np.uint8)
    markers = {
        (0, 0): (255, 0, 0),
        (w - 1, 0): (0, 255, 0),
        (w - 1, h - 1): (0, 0, 255),
        (0, h - 1): (255, 255, 0),
    }
    for (mx, my), color in markers.items():
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = mx + dx, my + dy
                if 0 <= x < w and 0 <= y < h:
                    r[y, x] = (*color, 255)
    ref = Image.fromarray(r, "RGBA")

    src = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    # 非均匀目标锚点: 若退化为 LANCZOS 等比缩放, 标记会落在 (79,119) 而非 (40,118)
    dst = [(2, 2), (78, 2), (40, 118), (2, 118)]
    out = preserve_refit.tps_warp(ref, src, dst, (80, 120))
    arr = np.array(out)
    # 节点精确: 每个目标锚点像素就是对应源节点标记色
    checks = [
        ((2, 2), (255, 0, 0)),
        ((78, 2), (0, 255, 0)),
        ((40, 118), (0, 0, 255)),
        ((2, 118), (255, 255, 0)),
    ]
    for (x, y), color in checks:
        px = arr[y, x, :3]
        assert all(abs(int(px[i]) - color[i]) <= 40 for i in range(3)), f"node ({x},{y}): {tuple(px)} != {color}"
    # 关键区分点: dst 节点 (40,118) 映射蓝色。LANCZOS 等比缩放下该像素采样源图
    # (50, 196) 附近 = 黑背景; TPS 才会把蓝标记放到这里(上面循环已断言)。
    # 反向确认: (40,118) 若为 TPS 节点, 蓝标记不会再占 (79,119) 比例位置 —— 但
    # TPS 外插也可能落到那里, 该点无区分力, 不作断言。


# ---------- optical_flow_refit ----------


def test_optical_flow_refit_returns_same_size():
    refit = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
    gen = Image.new("RGB", (32, 32), (200, 200, 200))
    mask = Image.new("L", (32, 32), 255)  # 全重绘
    out = preserve_refit.optical_flow_refit(refit, gen, mask)
    assert out.size == (32, 32)
    assert out.mode == "RGBA"


def test_optical_flow_refit_protect_region_unchanged():
    refit = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
    gen = Image.new("RGB", (32, 32), (200, 200, 200))
    mask_arr = np.zeros((32, 32), dtype=np.uint8)
    mask_arr[:, :16] = 255  # 左半重绘, 右半保护
    mask = Image.fromarray(mask_arr, "L")
    out = preserve_refit.optical_flow_refit(refit, gen, mask)
    arr = np.array(out)
    # 右半(保护)应保持原 refit 像素
    assert arr[16, 24, 0] == 100
    assert arr[16, 24, 3] == 255


def test_optical_flow_refit_no_mask_returns_original():
    refit = Image.new("RGBA", (16, 16), (50, 50, 50, 255))
    gen = Image.new("RGB", (16, 16), (200, 200, 200))
    mask = Image.new("L", (16, 16), 0)  # 全保护
    out = preserve_refit.optical_flow_refit(refit, gen, mask)
    arr = np.array(out)
    assert arr[8, 8, 0] == 50


# ---------- run_multiseed ----------


def test_run_multiseed_picks_highest_score():
    def task_fn(seed):
        return Image.new("RGB", (4, 4), (seed % 256, 0, 0))

    def scorer(img, seed):
        return float(seed % 10)  # seed 10→0, 20→0, 5→5

    seeds = [10, 20, 5]
    best, scores, chosen = preserve_refit.run_multiseed(task_fn, seeds, scorer)
    assert chosen == 5
    assert best.size == (4, 4)
    assert len(scores) == 3
    # scores 按分降序
    assert scores[0][0] == 5
    assert scores[0][1] == 5.0


def test_run_multiseed_tie_breaks_first():
    calls = []

    def task_fn(seed):
        calls.append(seed)
        return Image.new("RGB", (2, 2))

    def scorer(img, seed):
        return 1.0  # 全同分

    _best, _scores, chosen = preserve_refit.run_multiseed(task_fn, [1, 2, 3], scorer)
    assert chosen == 1  # 同分取第一个
    assert calls == [1, 2, 3]


def test_refit_seeds_default():
    try:
        os.environ["VTON_MULTI_SEED"] = "3"
        seeds = preserve_refit.refit_seeds_default()
        assert len(seeds) == 3
        os.environ["VTON_MULTI_SEED"] = "1"
        seeds1 = preserve_refit.refit_seeds_default()
        assert len(seeds1) == 1
        os.environ["VTON_MULTI_SEED"] = "99"
        seeds_capped = preserve_refit.refit_seeds_default()
        assert len(seeds_capped) == 8  # cap at 8
    finally:
        os.environ.pop("VTON_MULTI_SEED", None)


# ---------- quality_gate ----------


def test_ssim_score_identical_rois():
    img = Image.new("RGB", (16, 16), (128, 64, 32))
    score = quality_gate.ssim_score(img, img)
    assert score > 0.95


def test_ssim_score_different_rois():
    ref = Image.new("RGB", (32, 32), (128, 64, 32))
    # 在 ref 上叠加棋盘格高频图案
    arr = np.array(ref, dtype=np.uint8)
    arr[0::2, 0::2] = (255, 255, 255)
    arr[1::2, 1::2] = (0, 0, 0)
    ref = Image.fromarray(arr, "RGB")
    out = Image.new("RGB", (32, 32), (255, 0, 0))  # 纯色 vs 棋盘
    score = quality_gate.ssim_score(ref, out)
    assert score < 0.8, f"pattern vs solid should differ, got {score}"


def test_ssim_score_size_mismatch_resizes():
    ref = Image.new("RGB", (32, 32), (128, 64, 32))
    out = Image.new("RGB", (16, 16), (128, 64, 32))
    score = quality_gate.ssim_score(ref, out)
    assert 0.0 <= score <= 1.0


def test_delivery_tier_auto_for_high_score():
    assert quality_gate.delivery_tier(0.85, sku_has_text=False) == "auto"
    assert quality_gate.delivery_tier(0.90, sku_has_text=True) == "auto"


def test_delivery_tier_review_for_low_score():
    assert quality_gate.delivery_tier(0.50, sku_has_text=False) == "review"
    assert quality_gate.delivery_tier(0.30, sku_has_text=True) == "review"


def test_delivery_tier_custom_threshold():
    os.environ["VTON_QG_SSIM"] = "0.50"
    try:
        assert quality_gate.delivery_tier(0.60, sku_has_text=False) == "auto"
        assert quality_gate.delivery_tier(0.40, sku_has_text=False) == "review"
    finally:
        os.environ.pop("VTON_QG_SSIM", None)


def test_score_for_sku_pattern():
    ref = Image.new("RGB", (32, 32), (128, 64, 32))
    out = Image.new("RGB", (32, 32), (128, 64, 32))
    score, tier = quality_gate.score_for_sku(ref, out, sku_has_text=False)
    assert score > 0.9
    assert tier == "auto"


# ---------- detect_print_roi ----------


def _print_garment(size=(64, 64)):
    """灰底 + 中央白块印花(24x24)的衣片参考图, mask 盖住中央半区."""
    w, h = size
    arr = np.full((h, w), 128, dtype=np.uint8)
    y0, x0 = h // 2 - 12, w // 2 - 12
    arr[y0 : y0 + 24, x0 : x0 + 24] = 230
    cloth = Image.fromarray(arr, "L").convert("RGB")
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4] = 255
    return cloth, Image.fromarray(mask_arr, "L")


def test_detect_print_roi_returns_edge_band():
    cloth, mask = _print_garment()
    band = preserve_refit.detect_print_roi(cloth, mask)
    assert band is not None
    assert band.dtype == np.uint8
    assert (band == 255).any()
    w, h = cloth.size
    y0 = h // 2 - 12
    # 印花块边缘附近应有边界带
    assert band[y0 + 1, w // 2] == 255 or band[y0 - 1, w // 2] == 255
    # 印花块深部(块中心, 距边界 12px)不进 inpaint
    assert band[h // 2, w // 2] == 0


def test_detect_print_roi_solid_color_returns_none():
    cloth = Image.new("RGB", (64, 64), (128, 128, 128))
    mask = Image.new("L", (64, 64), 255)
    assert preserve_refit.detect_print_roi(cloth, mask) is None


def test_detect_print_roi_outside_mask_ignored():
    cloth, _mask = _print_garment()
    # mask 只盖住左上角(远离中央印花) → ROI 面积不足 → None
    small_mask_arr = np.zeros((64, 64), dtype=np.uint8)
    small_mask_arr[0:16, 0:16] = 255
    small_mask = Image.fromarray(small_mask_arr, "L")
    assert preserve_refit.detect_print_roi(cloth, small_mask) is None


# ---------- print_fidelity_score ----------


def _board_arr(size=(64, 64), cell=4, a=30, b=210):
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w]
    board = np.where(((ys // cell) + (xs // cell)) % 2 == 0, a, b).astype(np.uint8)
    return np.stack([board] * 3, axis=-1)


def test_print_fidelity_score_identical_pattern():
    arr = _board_arr()
    res = Image.fromarray(arr, "RGB")
    mask = Image.new("L", (64, 64), 255)
    assert quality_gate.print_fidelity_score(res, res, mask) > 0.99


def test_print_fidelity_score_solid_pair_returns_one():
    res = Image.new("RGB", (64, 64), (200, 30, 30))
    ref = Image.new("RGB", (64, 64), (40, 40, 180))
    mask = Image.new("L", (64, 64), 255)
    assert quality_gate.print_fidelity_score(res, ref, mask) == 1.0


def test_print_fidelity_score_inverted_pattern_low():
    arr = _board_arr()
    res = Image.fromarray(arr, "RGB")
    ref = Image.fromarray(255 - arr, "RGB")
    mask = Image.new("L", (64, 64), 255)
    assert quality_gate.print_fidelity_score(res, ref, mask) < 0.2


def test_print_fidelity_score_empty_mask_returns_zero():
    res = Image.new("RGB", (64, 64), (200, 30, 30))
    mask = Image.new("L", (64, 64), 0)
    assert quality_gate.print_fidelity_score(res, res, mask) == 0.0


def test_print_fidelity_score_size_mismatch_resizes_ref():
    small = Image.fromarray(_board_arr((64, 64), cell=4), "RGB")
    big = Image.fromarray(_board_arr((128, 128), cell=8), "RGB")
    mask = Image.new("L", (128, 128), 255)
    # 缩放后棋盘相位一致 → 高分
    assert quality_gate.print_fidelity_score(big, small, mask) > 0.9


# ---------- print_core_mask ----------


def test_print_core_mask_detects_logo_within_keep():
    w, h = 96, 128
    ref = Image.new("RGB", (w, h), (235, 233, 231))
    from PIL import ImageDraw

    d = ImageDraw.Draw(ref)
    d.ellipse((36, 48, 60, 72), fill=(200, 30, 30))  # 红色 logo(异色相)
    keep = np.zeros((h, w), dtype=np.uint8)
    keep[16:112, 12:84] = 255
    roi = preserve_refit.print_core_mask(ref, keep)
    assert roi is not None
    ys, xs = np.where(roi > 0)
    # 腐蚀 3px + 回膨胀 4px + grow 3px: 外扩 ≈ 7px 容差
    assert xs.min() >= 29 and xs.max() <= 67
    assert ys.min() >= 41 and ys.max() <= 79
    # 不越出 keep 区
    assert roi[:16, :].max() == 0 and roi[112:, :].max() == 0


def test_print_core_mask_solid_returns_none():
    ref = Image.new("RGB", (64, 64), (200, 200, 200))
    keep = np.full((64, 64), 255, dtype=np.uint8)
    assert preserve_refit.print_core_mask(ref, keep) is None


def test_print_core_mask_excludes_same_hue_folds():
    """褶皱阴影沿基色色相方向变暗 → 必须排除; 白印花(近中性)保留。"""
    w, h = 96, 128
    ref = Image.new("RGB", (w, h), (216, 198, 151))  # 芥末黄基色
    from PIL import ImageDraw

    d = ImageDraw.Draw(ref)
    d.rectangle((20, 40, 34, 100), fill=(140, 120, 75))  # 同色相深褶(长条)
    d.rectangle((50, 50, 80, 84), fill=(250, 249, 246))  # 白色印花
    keep = np.full((h, w), 255, dtype=np.uint8)
    roi = preserve_refit.print_core_mask(ref, keep)
    assert roi is not None
    _ys, xs = np.where(roi > 0)
    # 只有白印花区, 褶皱长条(<=x=34)必须为空
    assert 43 <= xs.min() <= 57 and xs.max() <= 87


def test_print_core_mask_drops_small_button_fragments():
    """纽扣碎片/小亮点按面积下限过滤, 不进贴印区(防膨胀串联成结构带)。"""
    w, h = 96, 128
    ref = Image.new("RGB", (w, h), (216, 198, 151))
    from PIL import ImageDraw

    d = ImageDraw.Draw(ref)
    d.ellipse((40, 50, 78, 88), fill=(250, 249, 246))  # 主印花
    for cy in (30, 50, 70, 90):  # 门襟小亮点(纽扣)
        d.ellipse((46, cy - 2, 52, cy + 4), fill=(240, 240, 240))
    keep = np.full((h, w), 255, dtype=np.uint8)
    roi = preserve_refit.print_core_mask(ref, keep)
    assert roi is not None
    # 主印花区存在; 顶部亮点区(y<40)必须被清掉
    assert float((roi[:40, :] > 0).mean()) == 0.0
    ys, _xs = np.where(roi > 0)
    assert ys.min() >= 42


def test_print_core_mask_resizes_keep_to_ref():
    ref = Image.new("RGB", (64, 64), (235, 233, 231))
    from PIL import ImageDraw

    d = ImageDraw.Draw(ref)
    d.rectangle((20, 20, 44, 44), fill=(200, 30, 30))
    keep = np.full((128, 128), 255, dtype=np.uint8)
    roi = preserve_refit.print_core_mask(ref, keep)
    assert roi is not None
    ys, xs = np.where(roi > 0)
    # logo (20,20)-(44,44) + 腐蚀/膨胀容差 ±8px, 且不越出画布
    assert 12 <= xs.min() <= 28 and 36 <= xs.max() <= 52
    assert 12 <= ys.min() <= 28 and 36 <= ys.max() <= 52


# ---------- clip_mask_to_garment_alpha ----------


def test_clip_mask_to_garment_alpha_shrinks_without_union():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:50, 10:50] = 255  # 大方框
    alpha = Image.new("L", (64, 64), 0)
    from PIL import ImageDraw

    d = ImageDraw.Draw(alpha)
    d.ellipse((20, 20, 44, 44), fill=255)  # 小圆(衣料)
    out = preserve_refit.clip_mask_to_garment_alpha(m, alpha, grow_px=0)
    arr = np.asarray(out)
    assert arr.shape == (64, 64)
    # 圆外 且 mask=255 的位置必须被裁掉(证明是纯裁剪而非并集)
    assert arr[12, 12] == 0
    assert arr[32, 32] == 255
    assert arr.max() <= 255
    assert not np.array_equal(arr, m)


def test_clip_mask_to_garment_alpha_none_passthrough():
    m = np.full((32, 32), 255, dtype=np.uint8)
    assert preserve_refit.clip_mask_to_garment_alpha(m, None) is m
    assert preserve_refit.clip_mask_to_garment_alpha(None, Image.new("L", (32, 32), 255)) is None


# ---------- refit_cloth_to_drawn_garment ----------


def test_refit_cloth_to_drawn_garment_aligns_content_bbox():
    # 平铺衣: 64x80 白底, 红色衣身在中间 (16,20)-(48,60)
    g = Image.new("RGB", (64, 80), (255, 255, 255))
    from PIL import ImageDraw

    d = ImageDraw.Draw(g)
    d.rectangle((16, 20, 48, 60), fill=(200, 30, 30))
    # 生成图 128x160, 检测衣料区在 (30,40)-(98,120)
    drawn = np.zeros((160, 128), dtype=np.uint8)
    drawn[40:121, 30:99] = 255
    out = preserve_refit.refit_cloth_to_drawn_garment(g, (128, 160), drawn)
    assert out is not None
    rgb, rgba = out
    assert rgb.size == (128, 160) and rgba.size == (128, 160)
    a = np.asarray(rgba.split()[-1])
    ys, xs = np.where(a > 10)
    assert abs(xs.min() - 30) <= 2 and abs(xs.max() - 98) <= 2
    assert abs(ys.min() - 40) <= 2 and abs(ys.max() - 120) <= 2


def test_refit_cloth_to_drawn_garment_solid_garment_uses_full_bbox():
    # 无白底纯色衣: garment_alpha 可能全零 → 回退整图 bbox, 仍应成功重投
    g = Image.new("RGB", (64, 80), (30, 120, 220))
    drawn = np.zeros((160, 128), dtype=np.uint8)
    drawn[40:121, 30:99] = 255
    out = preserve_refit.refit_cloth_to_drawn_garment(g, (128, 160), drawn)
    assert out is not None
    a = np.asarray(out[1].split()[-1])
    assert a.max() > 10
    ys, xs = np.where(a > 10)
    assert abs(xs.min() - 30) <= 2 and abs(xs.max() - 98) <= 2
    assert abs(ys.min() - 40) <= 2 and abs(ys.max() - 120) <= 2


def test_refit_cloth_to_drawn_garment_degenerate_masks_return_none():
    g = Image.new("RGB", (64, 80), (200, 30, 30))
    # 全画布 drawn(比例守卫) → None
    full = np.full((160, 128), 255, dtype=np.uint8)
    assert preserve_refit.refit_cloth_to_drawn_garment(g, (128, 160), full) is None
    # 过小区域 → None
    tiny = np.zeros((160, 128), dtype=np.uint8)
    tiny[70:74, 60:66] = 255
    assert preserve_refit.refit_cloth_to_drawn_garment(g, (128, 160), tiny) is None
    # 尺寸不匹配 → None
    assert preserve_refit.refit_cloth_to_drawn_garment(g, (128, 160), np.full((80, 64), 255, dtype=np.uint8)) is None


def test_tps_warp_large_scale_rect_anchors_stable():
    """服务器尺度回归(r5): 1441x2086 平铺衣 8 锚点 warp 到 576x768 画布。

    像素尺度 Bookstein cond~1e13 跨 LAPACK 版本解发散, 归一化后必须
    覆盖率健康(接近 dst bbox x 源填充率)且 bbox 对齐。
    """
    w, h = 1441, 2086
    rgb = np.full((h, w, 4), (255, 255, 255, 0), dtype=np.uint8)
    rgb[46:2041, 46:1395, :3] = (200, 30, 30)
    rgb[46:2041, 46:1395, 3] = 255
    ref = Image.fromarray(rgb, "RGBA")
    src = [(46, 46), (720, 46), (1395, 46), (46, 1043), (1395, 1043), (46, 2040), (720, 2040), (1395, 2040)]
    dst = [(164, 192), (286, 192), (409, 192), (164, 352), (409, 352), (164, 512), (286, 512), (409, 512)]
    out = preserve_refit.tps_warp(ref, src, dst, (576, 768))
    a = np.asarray(out.split()[-1])
    frac = float((a > 10).mean())
    src_fill = ((1395 - 46) * (2041 - 46)) / (w * h)
    dst_frac = (409 - 164) * (512 - 192) / (576 * 768)
    assert frac >= 0.5 * src_fill * dst_frac, f"warped coverage collapsed: {frac}"
    ys, xs = np.where(a > 10)
    assert abs(xs.min() - 164) <= 3 and abs(xs.max() - 409) <= 3
    assert abs(ys.min() - 192) <= 3 and abs(ys.max() - 512) <= 3


def test_tps_warp_bad_solve_falls_back(monkeypatch):
    """求解守卫: LAPACK 静默出垃圾解时残差检查抛错 → LANCZOS 回退。"""
    ref = Image.new("RGBA", (100, 200), (200, 100, 50, 255))
    src = [(0, 0), (100, 0), (100, 200), (0, 200)]
    dst = [(0, 0), (64, 0), (64, 64), (0, 64)]
    real_solve = np.linalg.solve

    def _garbage_solve(a_, b_):
        sol = real_solve(a_, b_)
        return sol * 1e6 + 123.0

    monkeypatch.setattr(np.linalg, "solve", _garbage_solve)
    out = preserve_refit.tps_warp(ref, src, dst, (64, 64))
    assert out.size == (64, 64)
    # 回退是 LANCZOS resize: 纯色图四角 alpha 仍在
    assert np.asarray(out)[0, 0, 3] > 127


def test_tps_warp_collinear_anchors_fall_back():
    """病态锚点(共线): 系统奇异/残差超限 → 回退 resize 而非垃圾 warp。"""
    ref = Image.new("RGBA", (100, 200), (200, 100, 50, 255))
    src = [(0, 0), (50, 0), (100, 0), (0, 200)]
    dst = [(0, 0), (32, 0), (64, 0), (0, 64)]
    out = preserve_refit.tps_warp(ref, src, dst, (64, 64))
    assert out.size == (64, 64)


def test_refit_cloth_to_drawn_garment_rejects_collapsed_warp(monkeypatch):
    """覆盖率守卫: warp 塌成稀疏斜条时 refit 判失败返回 None(回退模板)。"""
    g = Image.new("RGB", (64, 80), (255, 255, 255))
    from PIL import ImageDraw

    ImageDraw.Draw(g).rectangle((16, 20, 48, 60), fill=(200, 30, 30))
    drawn = np.zeros((160, 128), dtype=np.uint8)
    drawn[40:121, 30:99] = 255

    def _collapse(ref_img, src, dst, canvas_size):
        arr = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
        arr[:, 60:62, 3] = 255
        return Image.fromarray(arr, "RGBA")

    monkeypatch.setattr(preserve_refit, "tps_warp", _collapse)
    assert preserve_refit.refit_cloth_to_drawn_garment(g, (128, 160), drawn) is None
