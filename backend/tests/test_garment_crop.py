"""crop_garment_to_content 单测:纯 numpy/PIL,无需 torch/GPU。

验证服装平铺图裁到内容边界的算法正确性(去掉白底/透明边),以及
检测不到清晰主体时安全 no-op(不回归)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from engine_sdxl import crop_garment_to_content
from PIL import Image


def _white_bg_with_garment() -> Image.Image:
    """白底 + 中间一块蓝色\"服装\"(方区),四周留白边。"""
    arr = np.full((300, 300, 3), (255, 255, 255), dtype=np.uint8)
    arr[70:230, 70:230] = (30, 120, 200)
    return Image.fromarray(arr, "RGB")


def test_crop_removes_white_border():
    img = _white_bg_with_garment()
    out = crop_garment_to_content(img)
    oh, ow = out.size
    # 原图 300x300,内容区 70:230;裁掉白边后应明显变小
    assert ow < 300 and oh < 300, (ow, oh)
    c = np.array(out)[oh // 2, ow // 2]
    assert tuple(int(x) for x in c) == (30, 120, 200), c


def test_crop_no_op_on_uniform_image():
    """纯色图(与自身四角背景无差异)→ 检测不到内容 → 原样返回。"""
    arr = np.full((200, 200, 3), (120, 120, 120), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    out = crop_garment_to_content(img)
    assert out.size == img.size
    assert np.array_equal(np.array(out), arr)


def test_crop_rgba_alpha_drives_content():
    """透明底 RGBA:alpha 决定内容,裁掉全透明边。"""
    arr = np.zeros((300, 300, 4), dtype=np.uint8)
    arr[:, :, 3] = 0  # 全透明
    arr[80:220, 80:220, :3] = (200, 40, 40)
    arr[80:220, 80:220, 3] = 255  # 中间不透明"服装"
    img = Image.fromarray(arr, "RGBA")
    out = crop_garment_to_content(img)
    oh, ow = out.size
    assert ow < 300 and oh < 300, (ow, oh)
    assert out.mode == "RGBA"
    c = np.array(out)[oh // 2, ow // 2]
    assert tuple(int(x) for x in c[:3]) == (200, 40, 40), c


def test_crop_preserves_detail_in_content():
    """裁剪不能丢内容内部细节:内容区里的白色花纹/深色领口都要保留。"""
    arr = np.full((400, 400, 3), (255, 255, 255), dtype=np.uint8)
    arr[80:320, 80:320] = (90, 90, 160)  # 蓝紫衣身
    arr[140:200, 160:260] = (255, 255, 255)  # 衣身里一块白花纹(领口)
    img = Image.fromarray(arr, "RGB")
    out = crop_garment_to_content(img)
    o = np.array(out)
    oh, ow = o.shape[:2]
    # 内容区中心应是蓝紫,白花纹仍在范围内(未裁掉)
    assert o[oh // 2, ow // 2].tolist() == [90, 90, 160], o[oh // 2, ow // 2]
    # 白花纹仍在裁剪结果里(说明内部细节保留)
    assert (o == 255).any()


def test_crop_real_flatlay_guarded():
    """真实平铺图(若有素材):不崩溃、返回 Image。素材缺失则跳过,不阻断套件。"""
    import glob

    from platform_presets import force_white_background  # noqa: F401  (仅确认模块可导入链)

    # 在常见素材位置找一张服装平铺 jpg
    candidates = []
    for base in [
        "/mnt/workspace/vton/market_analysis",
        "/mnt/workspace/vton/repo/backend/data/uploads",
    ]:
        candidates += glob.glob(base + "/**/*.jpg", recursive=True)
    candidates += glob.glob(base + "/*.png")
    candidates = [c for c in candidates if c and Path(c).stat().st_size > 0]
    if not candidates:
        return  # 无素材,跳过
    for c in candidates:
        img = Image.open(c).convert("RGB")
        out = crop_garment_to_content(img)
        assert isinstance(out, Image.Image)
        assert out.size[0] > 0 and out.size[1] > 0
        break
