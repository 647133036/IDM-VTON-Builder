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
    # 边界带 Telea 后, 中间行 28-35 列(边界 ±4px) 应全为 0(源图色,
    # 因为源图全黑, 结果全白, Telea 在边界带填黑)
    h, w = arr.shape[:2]
    mid = w // 2
    band = arr[h // 2, mid - 4 : mid + 4, 0]
    # 边界带被 Telea 平滑后, 不应有 255(结果白) 残留
    assert (band < 128).all(), f"boundary band should be smoothed to source color, got {band}"


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
