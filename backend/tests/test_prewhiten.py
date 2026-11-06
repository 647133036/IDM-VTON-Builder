"""prewhiten.py 单测 (远端复核#4 补缺口): 守卫语义 fail-closed 化后的行为定案。

- changed_frac 双向守卫 (<min = already_white, >max = 拒绝)
- u2net 主体交集守卫: 重叠超阈值 → 拒绝
- **fail-closed**: 取不到主体掩膜(u2net 失败)或掩膜尺寸不符 → 拒绝, 不再裸跑 changed_frac
- 成衣侧可由 engine 入口 VTON_PREWHITEN_GARMENT=0 单独关 (engine 行为, 本文件不测)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prewhiten import prewhiten_guarded


def _bg_block_image(size=64, block=32, bg=235, block_val=30):
    """四角 flood-fill 场景: 近白背景 + 中央深色方块(不被压)。"""
    arr = np.full((size, size, 3), bg, np.uint8)
    off = (size - block) // 2
    arr[off : off + block, off : off + block] = block_val
    return Image.fromarray(arr)


def _subject_block_mask(size=64, block=32):
    m = np.zeros((size, size), np.uint8)
    off = (size - block) // 2
    m[off : off + block, off : off + block] = 255
    return m


def test_prewhiten_applied_with_subject_mask_no_overlap():
    """主体(中央方块)与压白区(背景)不重叠 → applied, 背景变白。"""
    img = _bg_block_image()
    _out, info = prewhiten_guarded(img, tol=20, subject_mask=_subject_block_mask())
    assert info["status"] == "applied", info
    assert info["subject_overlap"] == 0
    arr = np.array(_out)
    assert tuple(arr[0, 0]) == (255, 255, 255)  # 角区已压白
    assert arr[32, 32, 0] < 100  # 中央方块保留


def test_prewhiten_rejected_subject_overlap():
    """压白区与主体显著重叠(主体=全图) → rejected_subject_overlap, 原图返回。"""
    img = _bg_block_image()
    full = np.full((64, 64), 255, np.uint8)
    _out, info = prewhiten_guarded(img, tol=20, subject_mask=full)
    assert info["status"] == "rejected_subject_overlap", info
    assert np.array_equal(np.array(_out), np.array(img))  # 原图返回


def test_prewhiten_already_white():
    """背景本就近白(全 255) → changed_frac < min → already_white, no-op。"""
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    _out, info = prewhiten_guarded(img, tol=20, subject_mask=np.full((64, 64), 255, np.uint8))
    assert info["status"] == "already_white", info


def test_prewhiten_rejected_changed_ratio_high():
    """压白占比 > max(0.85) = 背景非 uniform 或吞了主体 → 拒绝(比例守卫先于主体守卫)。"""
    arr = np.full((64, 64, 3), 235, np.uint8)
    arr[30:34, 30:34] = 30  # 4x4 深色点: 四角 fill 吃掉其余 ~99% → changed > 0.85
    img = Image.fromarray(arr)
    _out, info = prewhiten_guarded(img, tol=20, subject_mask=_subject_block_mask())
    assert info["status"] == "rejected_changed_ratio", info


def test_prewhiten_fail_closed_no_subject_mask(monkeypatch):
    """远端复核#4: u2net 取不到主体掩膜 → fail-closed 拒绝(不再 skip 裸跑 changed_frac)。"""
    import platform_presets

    def _boom(*a, **k):
        raise RuntimeError("u2net unavailable (simulated)")

    monkeypatch.setattr(platform_presets, "foreground_mask", _boom)
    img = _bg_block_image()
    _out, info = prewhiten_guarded(img, tol=20, subject_mask=None)
    assert info["status"] == "rejected_subject_overlap", info
    assert "unavailable" in info["reason"]
    assert np.array_equal(np.array(_out), np.array(img))


def test_prewhiten_fail_closed_shape_mismatch(monkeypatch):
    """掩膜尺寸与图不符 = 掩膜错位, 交集无意义 → fail-closed 拒绝。"""
    img = _bg_block_image()
    tiny = np.full((10, 10), 255, np.uint8)
    _out, info = prewhiten_guarded(img, tol=20, subject_mask=tiny)
    assert info["status"] == "rejected_subject_overlap", info
    assert "shape" in info["reason"]
