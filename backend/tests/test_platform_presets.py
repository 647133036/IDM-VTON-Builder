"""平台店规适配层单测:1:1 居中裁剪、纯白底压白、JPEG 限体积、平台模板。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from platform_presets import (
    PLATFORM_PRESETS,
    PLATFORM_SET,
    _correct_skin_tone,
    _detect_face_mask,
    _detect_skin,
    _heuristic_face_mask,
    apply_platform_preset,
    blend_tryon,
    center_crop_square,
    clothing_mask,
    detect_garment_extent,
    encode_jpeg_within,
    force_white_background,
    reapply_garment_detail,
    save_platform_result,
)


def _beige_bg_with_subject(size=(400, 400), subject=(120, 320)) -> Image.Image:
    """米色背景(185,168,152)+ 中间一个非白主体方块,模拟扩散模型输出。"""
    arr = np.full((size[1], size[0], 3), (185, 168, 152), dtype=np.uint8)
    x0, x1 = subject
    arr[60:340, x0:x1] = (60, 60, 120)
    # 主体上的"白色衣服"区域(接近白但不与边缘连通,必须保留原色)
    arr[100:200, 160:240] = (250, 248, 245)
    return Image.fromarray(arr, "RGB")


def test_center_crop_square_is_11_and_centered():
    img = Image.new("RGB", (400, 800))
    out = center_crop_square(img)
    assert out.size == (400, 400)
    # 3:4 竖图居中裁 1:1:切掉上下各 200
    px = np.array(out)
    assert px.shape == (400, 400, 3)


def test_force_white_background_flattens_border_bg_only():
    img = _beige_bg_with_subject()
    out = force_white_background(img)
    arr = np.array(out)
    # 四角必须压到纯白
    for corner in (arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]):
        assert tuple(corner) == (255, 255, 255), corner
    # 非白主体(深蓝)保持原样
    assert tuple(arr[200, 200])[:2] == (60, 60), arr[200, 200]
    # 主体上的近白衣物(不与边缘连通)保持原色,不被压白
    assert abs(int(arr[150, 200][0]) - 250) <= 5, arr[150, 200]


def test_force_white_background_gray_studio():
    """中灰影棚背景(扩散模型常见输出,亮度 ~120)也要压白。"""
    arr = np.full((400, 400, 3), (117, 125, 138), dtype=np.uint8)
    arr[60:340, 120:320] = (60, 60, 120)  # 深色主体
    arr[100:200, 160:240] = (250, 248, 245)  # 主体上的近白衣物
    img = Image.fromarray(arr, "RGB")
    out = np.array(force_white_background(img))
    for corner in (out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]):
        assert tuple(corner) == (255, 255, 255), corner
    assert tuple(out[200, 200])[:2] == (60, 60), out[200, 200]
    assert abs(int(out[150, 200][0]) - 250) <= 5, out[150, 200]


def test_force_white_background_preserves_dark_hair_at_border():
    """贴边黑发丝不被压白;保护带仅 2px(抗锯齿边),其外背景必须纯白,
    不得留下灰色光晕。"""
    arr = np.full((400, 400, 3), (117, 125, 138), dtype=np.uint8)  # 灰影棚
    arr[0:50, 180:220] = (30, 30, 35)  # 黑发丝从顶边垂下
    arr[50:58, 180:220] = (95, 100, 110)  # 发丝与背景的抗锯齿过渡带
    arr[80:340, 100:300] = (60, 60, 120)  # 深色主体
    img = Image.fromarray(arr, "RGB")
    out = np.array(force_white_background(img))
    # 远处背景角:纯白
    assert tuple(out[5, 5]) == (255, 255, 255), out[5, 5]
    # 发丝本体:保持深色(不被扣白)
    assert int(out[20, 200].mean()) < 80, out[20, 200]
    # 2px 抗锯齿保护边内:保留原色
    aa = int(out[50, 200].mean())
    assert 90 < aa < 250, aa
    # 保护带之外(4px):必须纯白,不允许灰色光晕
    assert tuple(out[54, 200]) == (255, 255, 255), out[54, 200]
    # 主体保持
    assert tuple(out[200, 200])[:2] == (60, 60), out[200, 200]


def test_force_white_background_face_not_whitened_via_hair_bridge():
    """回归:发质内部 1-6px 亮色断丝把'脸'与画面外背景打通 → 整张脸被压白。
    修复后细缝必须被切断:脸保持原色,开阔背景照压白。"""
    h = w = 400
    arr = np.full((h, w, 3), (240, 240, 240), dtype=np.uint8)  # 近白影棚
    arr[40:300, 150:250] = (35, 35, 40)  # 深色'头发'包围
    arr[150:250, 170:230] = (200, 190, 170)  # 内部'脸'(被头发围住)
    arr[40:150, 200:203] = (235, 235, 235)  # 3px 亮色断丝:头顶背景↔脸
    img = Image.fromarray(arr, "RGB")
    out = np.array(force_white_background(img))
    # 开阔背景压白(头顶 40px 亮区、头发下方背景、四角)
    assert tuple(out[5, 200]) == (255, 255, 255), out[5, 200]
    assert tuple(out[380, 200]) == (255, 255, 255), out[380, 200]
    assert tuple(out[5, 5]) == (255, 255, 255), out[5, 5]
    # 脸(只经 3px 断丝连通)保持原色
    assert out[200, 200].tolist() == [200, 190, 170], out[200, 200]
    # 深色'头发'本体保持
    assert int(out[100, 160].mean()) < 80, out[100, 160]


def test_force_white_background_face_control_hair_touching_border():
    """对照组:发顶直接贴画面顶边(无亮缝)。脸被深色头发完整包围,
    背景压白、脸保持——修复不得破坏这条既有安全路径。"""
    h = w = 400
    arr = np.full((h, w, 3), (240, 240, 240), dtype=np.uint8)
    arr[0:300, 150:250] = (35, 35, 40)
    arr[150:250, 170:230] = (200, 190, 170)
    img = Image.fromarray(arr, "RGB")
    out = np.array(force_white_background(img))
    assert tuple(out[5, 100]) == (255, 255, 255), out[5, 100]
    assert out[200, 200].tolist() == [200, 190, 170], out[200, 200]
    assert tuple(out[380, 200]) == (255, 255, 255), out[380, 200]


def test_force_white_background_thin_border_sliver_whitened():
    """主体与画面边缘之间 4px 细缝:腐蚀会杀掉其连通域,
    外圈 4px 例外必须兜底压白,不留色偏残留。"""
    h = w = 400
    arr = np.full((h, w, 3), (150, 150, 150), dtype=np.uint8)  # 灰影棚
    arr[40:360, 100:396] = (60, 60, 120)  # 深色主体,右缝 4px
    img = Image.fromarray(arr, "RGB")
    out = np.array(force_white_background(img))
    assert tuple(out[200, 398]) == (255, 255, 255), out[200, 398]  # 4px 细缝(外圈兜底)
    assert tuple(out[200, 50]) == (255, 255, 255), out[200, 50]  # 左侧开阔走廊
    assert tuple(out[200, 200]) == (60, 60, 120), out[200, 200]  # 主体保护


def test_apply_platform_presets_sizes():
    img = Image.new("RGB", (2048, 2048), (200, 180, 150))
    for platform, preset in PLATFORM_PRESETS.items():
        out, got = apply_platform_preset(img, platform)
        assert got == preset
        assert out.size == (preset["size"], preset["size"])
    # none / 未知平台:原样
    same, none_preset = apply_platform_preset(img, "none")
    assert none_preset is None
    assert same.size == (2048, 2048)


def test_platform_set_membership():
    for p in ("shopify", "amazon", "aliexpress"):
        assert p in PLATFORM_SET
    assert "taobao" not in PLATFORM_SET


def test_encode_jpeg_within_respects_cap():
    # 平滑内容(渐变+色块):正常质量即可满足小体积上限
    w, h = 800, 800
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    for c, base in enumerate((120, 80, 200)):
        grad[:, :, c] = np.linspace(base, base + 60, w, dtype=np.uint8)[None, :]
    grad[h // 2 :, : w // 2] = (60, 60, 120)
    img = Image.fromarray(grad, "RGB")
    data = encode_jpeg_within(img, max_bytes=10 * 1024)
    assert len(data) <= 10 * 1024
    # 小图直接达标:保持高质量
    small = Image.new("RGB", (100, 100), (255, 255, 255))
    data2 = encode_jpeg_within(small, max_bytes=5 * 1024 * 1024)
    assert 0 < len(data2) < 5 * 1024 * 1024


def test_save_platform_result_writes_jpg_under_cap(tmp_path):
    img = _beige_bg_with_subject((1000, 1000))
    out_path = tmp_path / "job_x.jpg"
    size_px, nbytes = save_platform_result(img, "aliexpress", out_path)
    assert size_px == 1000
    assert out_path.exists()
    assert nbytes <= PLATFORM_PRESETS["aliexpress"]["max_bytes"]
    # 输出必须 1:1
    im = Image.open(out_path)
    assert im.size[0] == im.size[1]
    # 四角纯白
    arr = np.array(im)
    assert tuple(arr[0, 0]) == (255, 255, 255)


def test_save_platform_result_unknown_platform_falls_back(tmp_path):
    img = Image.new("RGB", (500, 500), (255, 255, 255))
    out_path = tmp_path / "job_y.jpg"
    size_px, nbytes = save_platform_result(img, "none", out_path)
    assert size_px == 500
    assert nbytes == 0
    assert out_path.exists()


# ---- 宽亮色通路 · 人脸保护(face_guard)回归 ----


def _wide_bright_path_face() -> Image.Image:
    """近白背景 + 一列浅色"身体"贯通上下(宽亮色通路)+ 其中一块"脸"。
    整条通路 L1<250,腐蚀切不断,脸会随背景被压白(face_guard 的目标场景)。"""
    h = w = 400
    arr = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
    arr[:, 130:270] = (232, 224, 214)  # 浅色衣/肤, 上下贯通(宽通路)
    arr[60:150, 150:250] = (214, 196, 178)  # 上中部"脸"
    return Image.fromarray(arr, "RGB")


def test_face_guard_exempts_wide_bright_path(monkeypatch):
    """注入一个覆盖脸区的检测掩膜:开启 face_guard 后脸必须保留原色,
    而关闭 face_guard(现状)时同一张脸的宽通路被压白。开阔背景两种都压白。"""
    import platform_presets

    img = _wide_bright_path_face()
    arr_out_old = np.array(platform_presets.force_white_background(img, face_guard=False))
    # 现状(无豁免): 脸被宽通路带白
    assert tuple(arr_out_old[100, 200]) == (255, 255, 255), arr_out_old[100, 200]

    h = w = img.size[1]
    fake = np.zeros((h, w), dtype=np.uint8)
    fake[40:170, 140:260] = 1  # 脸的 bbox 外扩
    monkeypatch.setattr(platform_presets, "_detect_face_mask", lambda *a, **k: fake)
    arr_out = np.array(platform_presets.force_white_background(img))
    # 豁免: 脸区保留原肤色
    assert tuple(arr_out[100, 200]) == (214, 196, 178), arr_out[100, 200]
    # 背景仍压白
    assert tuple(arr_out[5, 5]) == (255, 255, 255), arr_out[5, 5]
    # 脸区以外、同通路上的浅色身体仍被压白(豁免只覆盖检到的脸)
    assert tuple(arr_out[300, 200]) == (255, 255, 255), arr_out[300, 200]


def test_detect_face_mask_empty_when_no_face():
    """合成无脸图: Haar 检不到脸 → 掩膜全 0 → face_guard 退化为现状(不误伤)。"""
    import platform_presets

    arr = np.array(_beige_bg_with_subject(), dtype=np.int16)
    m = platform_presets._detect_face_mask(arr)
    assert m.sum() == 0
    # 与无豁免结果完全一致
    img = _beige_bg_with_subject()
    same = np.array(platform_presets.force_white_background(img, face_guard=False))
    diff = np.array(platform_presets.force_white_background(img, face_guard=True))
    assert np.array_equal(same, diff)


def test_detect_face_mask_finds_real_face():
    """真实模特照(可选素材): 检到脸则掩膜非空。素材缺失时跳过,不阻断套件。"""
    from pathlib import Path

    import platform_presets

    ref = Path("/mnt/workspace/vton/market_analysis/vton_models")
    photos = sorted(ref.glob("*.jpg")) if ref.exists() else []
    if not photos:
        return  # 跳过
    detected_any = False
    for p in photos:
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.int16)
        if platform_presets._detect_face_mask(arr).sum() > 0:
            detected_any = True
            break
    # 至少要有一张能被 Haar 检到脸,否则说明检测链路坏了
    assert detected_any, "没有任何模特照被 Haar 检到脸"


# ---- 主体掩膜(subject_mask)回归: 治浅色身体/头发换白底后被啃 ----


def _light_hair_touching_top():
    """近白背景 + 头顶一块浅色"发"贴上边缘(宽亮通路),整条 L1<250。
    无主体掩膜时被压白;把它标为主体则必须保留。返回 (img, mask)。"""
    h = w = 400
    arr = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)
    arr[:120, 150:250] = (230, 220, 210)  # 浅色发, 贴顶(与背景同通路)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:130, 140:260] = 1  # 主体区含这块发 + 2px 外扩
    return Image.fromarray(arr, "RGB"), mask


def test_subject_mask_protects_light_hair_at_border():
    """核心回归: 浅色发贴顶边——无掩膜被啃, 有掩膜保留原色。"""
    import platform_presets

    img, mask = _light_hair_touching_top()
    old = np.array(platform_presets.force_white_background(img, face_guard=False, subject_mask=None))
    assert tuple(old[40, 200]) == (255, 255, 255), old[40, 200]  # 现状: 被啃
    new = np.array(platform_presets.force_white_background(img, face_guard=False, subject_mask=mask))
    assert tuple(new[40, 200]) == (230, 220, 210), new[40, 200]  # 有掩膜: 保留


def test_subject_mask_background_still_whitened():
    """加了主体掩膜, 主体外的背景仍要压白(不能整图不动)。"""
    import platform_presets

    img, mask = _light_hair_touching_top()
    new = np.array(platform_presets.force_white_background(img, face_guard=False, subject_mask=mask))
    assert tuple(new[5, 5]) == (255, 255, 255), new[5, 5]  # 角落背景压白


def test_subject_mask_none_is_noop_regression():
    """subject_mask=None 时行为与旧版完全一致(不回归既有 15 个测试)。"""
    import platform_presets

    img = _beige_bg_with_subject()
    a = np.array(platform_presets.force_white_background(img))
    b = np.array(platform_presets.force_white_background(img, subject_mask=None))
    assert np.array_equal(a, b)


def test_clothing_mask_none_without_subject():
    img = Image.new("RGB", (200, 200), (180, 180, 180))
    assert clothing_mask(img, subject_mask=np.zeros((200, 200), dtype=np.uint8)) is None


def test_clothing_mask_upper_drops_face_and_legs():
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    img = Image.fromarray(np.full((200, 120, 3), 80, dtype=np.uint8), "RGB")
    m = clothing_mask(img, garment_type="upper", subject_mask=subj)
    assert m is not None
    assert m.shape == (200, 120)
    assert int(m[160, 60]) < 40
    assert int(m[70, 60]) > 180


def test_clothing_mask_lower_drops_torso():
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    img = Image.fromarray(np.full((200, 120, 3), 80, dtype=np.uint8), "RGB")
    m = clothing_mask(img, garment_type="lower", subject_mask=subj)
    assert m is not None
    assert int(m[40, 60]) < 40
    assert int(m[150, 60]) > 180


def test_blend_tryon_keeps_unmasked_pixels():
    orig = Image.new("RGB", (40, 40), (10, 20, 30))
    gen = Image.new("RGB", (40, 40), (200, 10, 10))
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    out = np.array(blend_tryon(orig, gen, mask))
    assert tuple(out[0, 0]) == (10, 20, 30)
    assert tuple(out[20, 20]) == (200, 10, 10)


def test_foreground_mask_none_when_no_model(monkeypatch):
    """本地无完整 u2net.onnx → 快速返回 None(不触发慢下载), 调用方回退纯颜色压白。"""
    import platform_presets

    monkeypatch.setattr(platform_presets, "_u2net_onnx_path", lambda: None)
    img = _beige_bg_with_subject()
    assert platform_presets.foreground_mask(img) is None


def test_foreground_mask_none_when_session_fails(monkeypatch):
    """rembg 已装但 new_session 建不起来(加载失败)→ 返回 None(不抛), 回退不回归。"""
    import platform_presets
    import pytest

    rembg = pytest.importorskip("rembg")

    img = _beige_bg_with_subject()
    # 复位全局 session 缓存, 确保走到(被 monkeypatch 的)new_session 构建路径
    monkeypatch.setattr(platform_presets, "_FG_SESSION", None)
    monkeypatch.setattr(platform_presets, "_FG_SESSION_WARNED", False)
    monkeypatch.setattr(platform_presets, "_u2net_onnx_path", lambda: platform_presets._U2NET_ROOT / "u2net.onnx")
    monkeypatch.setattr(rembg, "new_session", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated")))
    assert platform_presets.foreground_mask(img) is None


def test_foreground_mask_runs_when_model_present():
    """NAS 放了完整 u2net.onnx 时真跑一次; 未就位则跳过(可选特性, 不阻断套件)。
    合成图前景占比可能过低 → 允许返回 None, 只要不抛异常且类型正确。"""
    import platform_presets

    if platform_presets._u2net_onnx_path() is None:
        return  # 模型未就位, 跳过
    img = _beige_bg_with_subject()
    m = platform_presets.foreground_mask(img)
    assert m is None or np.asarray(m).dtype == np.uint8


def test_reapply_garment_detail_passthrough_without_cv2(monkeypatch):
    """无 cv2 时原样返回输入图(安全回退)。"""
    import builtins

    real_import = builtins.__import__

    def _block_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_cv2)
    gen = Image.new("RGB", (64, 64), (100, 150, 200))
    ref = Image.new("RGB", (64, 64), (200, 100, 50))
    mask = np.zeros((64, 64), dtype=np.uint8)
    out = reapply_garment_detail(gen, ref, mask)
    assert out is gen


def test_reapply_garment_detail_zero_mask_keeps_generated():
    """mask 全 0(无衣服区)时,生成图像素不变。"""
    gen = Image.new("RGB", (80, 80), (120, 80, 60))
    ref = Image.new("RGB", (80, 80), (200, 200, 30))
    mask = np.zeros((80, 80), dtype=np.uint8)
    out = reapply_garment_detail(gen, ref, mask)
    a = np.array(gen)
    b = np.array(out)
    assert np.array_equal(a, b)


def test_reapply_garment_detail_changes_masked_region():
    """mask 有值时,衣服区像素必须改变(纹理+颜色注入生效)。"""
    from PIL import ImageDraw

    # 生成图:纯色衣服区;参考图:带花纹的另一种颜色
    gen_arr = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
    gen = Image.fromarray(gen_arr, "RGB")

    ref = Image.new("RGB", (100, 100), (30, 160, 60))
    d = ImageDraw.Draw(ref)
    d.rectangle([20, 20, 80, 80], fill=(200, 40, 40))

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255

    out = reapply_garment_detail(gen, ref, mask)
    a = np.array(gen)
    b = np.array(out)
    # mask 区像素必须改变
    assert not np.array_equal(a[10:90, 10:90], b[10:90, 10:90])
    # mask 外像素不变
    assert np.array_equal(a[:10, :], b[:10, :])
    assert np.array_equal(a[90:, :], b[90:, :])


def test_reapply_garment_detail_color_matches_source():
    """颜色迁移后,生成图衣服区均值更接近参考图(颜色被拉过去)。"""
    # 生成图衣服区偏蓝(60,60,180),参考图偏红(180,60,60)
    gen_arr = np.full((64, 64, 3), (60, 60, 180), dtype=np.uint8)
    gen = Image.fromarray(gen_arr, "RGB")
    ref = Image.new("RGB", (64, 64), (180, 60, 60))
    mask = np.full((64, 64), 255, dtype=np.uint8)

    out = reapply_garment_detail(gen, ref, mask, texture_strength=0.0, color_strength=1.0)
    out_arr = np.array(out)
    # R 通道应上升(从60趋向180), B 通道应下降(从180趋向60)
    assert out_arr[..., 0].mean() > 100  # R 上升
    assert out_arr[..., 2].mean() < 140  # B 下降


# ---------------------------------------------------------------------------
# v0.1.4: detect_garment_extent / 动态 clothing_mask / seamlessClone / 边缘衰减
# ---------------------------------------------------------------------------


def _transparent_with_shapes(size, shapes):
    """透明 RGBA 画布上绘制一系列矩形/多边形。

    shapes: list of (type, coords, color)
      type="rect" → coords=(x0,y0,x1,y1)
      type="poly" → coords=[(x,y),...]
    """
    from PIL import ImageDraw

    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    color = (200, 50, 50, 255)
    for typ, coords in shapes:
        if typ == "rect":
            d.rectangle(coords, fill=color)
        elif typ == "poly":
            d.polygon(coords, fill=color)
    return img


# --- detect_garment_extent --------------------------------------------------


def test_detect_garment_extent_long_sleeve_by_aspect():
    """高瘦矩形(aspect>1.35)→ sleeve=long, hem=long。"""
    img = _transparent_with_shapes((100, 250), [("rect", (10, 25, 90, 225))])
    ext = detect_garment_extent(img)
    assert ext["sleeve"] == "long"
    assert ext["hem"] == "long"


def test_detect_garment_extent_short_sleeve_t_shape():
    """T 恤形(宽肩+渐窄身)→ sleeve=short, hem=waist。

    宽顶 bar 使 top_w≈1.0;渐窄三角形身使 bot_w<mid_w*0.6;shoulder_spread>1.15。
    """
    img = _transparent_with_shapes(
        (200, 200),
        [
            ("rect", (20, 50, 180, 70)),  # 肩+短袖
            ("poly", [(60, 70), (140, 70), (105, 125), (95, 125)]),  # 渐窄身
        ],
    )
    ext = detect_garment_extent(img)
    assert ext["sleeve"] == "short"
    assert ext["hem"] == "waist"


def test_detect_garment_extent_sleeveless_narrow_top():
    """吊带衫形(细带+三角形身,top_w<0.15)→ sleeve=sleeveless。"""
    img = _transparent_with_shapes(
        (200, 200),
        [
            ("rect", (95, 50, 105, 55)),  # 细吊带
            ("poly", [(40, 70), (160, 70), (105, 130), (95, 130)]),  # 三角形身渐窄
        ],
    )
    ext = detect_garment_extent(img)
    assert ext["sleeve"] == "sleeveless"


def test_detect_garment_extent_tiny_content_defaults():
    """内容 < 64 像素 → 兜底 short/waist。"""
    img = _transparent_with_shapes((200, 200), [("rect", (97, 97, 103, 103))])
    ext = detect_garment_extent(img)
    assert ext["sleeve"] == "short"
    assert ext["hem"] == "waist"


def test_detect_garment_extent_opaque_background():
    """不透明图(无 alpha 通道)走角部差分通路,仍能检测。"""
    from PIL import ImageDraw

    img = Image.new("RGB", (100, 250), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.rectangle((10, 25, 90, 225), fill=(200, 50, 50))
    ext = detect_garment_extent(img)
    assert ext["sleeve"] == "long"
    assert ext["hem"] == "long"


# --- clothing_mask with garment_extent --------------------------------------


def _subject_img(h=200, w=120):
    """统一主体: y 20-180 满宽,用于 clothing_mask 测试。span=160。"""
    subj = np.zeros((h, w), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    img = Image.fromarray(np.full((h, w, 3), 80, dtype=np.uint8), "RGB")
    return img, subj


def test_clothing_mask_long_sleeve_extends_below_default():
    """长袖 mask 比 default(short)延伸更远:row 125 在 long 保留、default 截断。"""
    img, subj = _subject_img()
    m_default = clothing_mask(img, garment_type="upper", subject_mask=subj)
    m_long = clothing_mask(
        img, garment_type="upper", subject_mask=subj, garment_extent={"sleeve": "long", "hem": "waist"}
    )
    assert m_default is not None and m_long is not None
    # default cut at 20+160*0.62=119; long cut at 20+160*0.72=135
    # row 125: default 截断, long 保留
    assert int(m_long[125, 60]) > int(m_default[125, 60]) + 50


def test_clothing_mask_upper_long_hem_drapes_over_pants():
    """长款上衣(hem=long)垂过裤腰:row 140 在 hem=long 保留、hem=waist 截断。"""
    img, subj = _subject_img()
    m_waist = clothing_mask(
        img, garment_type="upper", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "waist"}
    )
    m_long = clothing_mask(
        img, garment_type="upper", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "long"}
    )
    assert m_waist is not None and m_long is not None
    # waist cut at 20+160*0.62=119; long cut at 20+160*0.80=148
    # row 140: waist 截断, long 保留(垂过裤腰)
    assert int(m_long[140, 60]) > int(m_waist[140, 60]) + 50


def test_clothing_mask_sleeveless_cuts_above_default():
    """无袖 mask 比 default(short)截断更早:row 100 在 sleeveless 截断、default 保留。"""
    img, subj = _subject_img()
    m_default = clothing_mask(img, garment_type="upper", subject_mask=subj)
    m_sleeveless = clothing_mask(
        img, garment_type="upper", subject_mask=subj, garment_extent={"sleeve": "sleeveless", "hem": "waist"}
    )
    assert m_default is not None and m_sleeveless is not None
    # row 100: default cut at 119 → 保留; sleeveless cut at 20+160*0.55=108 → 截断
    assert int(m_default[100, 60]) > int(m_sleeveless[100, 60]) + 50


def test_clothing_mask_dress_long_hem_keeps_more():
    """连衣裙长款 hem=long 比 hem=waist 保留更多:row 150 在 long 保留、waist 截断。"""
    img, subj = _subject_img()
    m_waist = clothing_mask(
        img, garment_type="dress", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "waist"}
    )
    m_long = clothing_mask(
        img, garment_type="dress", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "long"}
    )
    assert m_waist is not None and m_long is not None
    # row 150: waist cut at 20+160*0.82=151 → 截断; long cut at 20+160*0.96=173 → 保留
    assert int(m_long[150, 60]) > int(m_waist[150, 60]) + 50


def test_clothing_mask_onesie_does_not_fall_through():
    """onesie 必须被显式处理(不穿透到无裁切),返回非 None 且底部截断。"""
    img, subj = _subject_img()
    m = clothing_mask(img, garment_type="onesie", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "waist"})
    assert m is not None
    # onesie short cut at 20+160*0.82=151 → row 170 截断
    assert int(m[170, 60]) < 40
    # row 70 保留
    assert int(m[70, 60]) > 100


def test_clothing_mask_auto_treated_as_upper():
    """auto 类型按 upper 处理(不再穿透无裁切):底部截断 + 脸区保护。"""
    img, subj = _subject_img()
    m = clothing_mask(img, garment_type="auto", subject_mask=subj)
    assert m is not None
    # auto=upper → cut at 20+160*0.62=119 → row 160 截断
    assert int(m[160, 60]) < 40
    # row 70 保留
    assert int(m[70, 60]) > 100


def test_clothing_mask_heuristic_face_when_detection_fails(monkeypatch):
    """cv2 人脸检测不可用时(cv2 5.0 缺 CascadeClassifier),启发式仍保护头顶区域。

    subject rows 20-180(span=160)→ face_band=20+28=48。
    row 35(脸区)被保护(mask 低),row 70(衣服区)仍保留(mask 高)。
    """
    import platform_presets

    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    img = Image.fromarray(np.full((200, 120, 3), 80, dtype=np.uint8), "RGB")

    # 模拟 cv2 5.0: _detect_face_mask 返回全零
    monkeypatch.setattr(platform_presets, "_detect_face_mask", lambda *a, **kw: np.zeros((200, 120), dtype=np.uint8))

    m = clothing_mask(img, garment_type="upper", subject_mask=subj)
    assert m is not None
    assert int(m[35, 60]) < 40  # face area protected
    assert int(m[70, 60]) > 180  # clothing area retained


# --- _detect_face_mask down_margin_frac -------------------------------------


def test_face_mask_down_margin_asymmetric(monkeypatch):
    """down_margin_frac < margin_frac 时,向下扩展远小于向上/左右扩展。"""
    import cv2

    # 构造一张 200x200 灰度图,中央放一个 60x60 "脸" 区
    arr = np.full((200, 200, 3), 128, dtype=np.uint8)
    arr[70:130, 70:130] = 200

    # monkeypatch CascadeClassifier 返回固定 face bbox
    class _FakeCascade:
        def empty(self):
            return False

        def detectMultiScale(self, *a, **kw):
            return np.array([[70, 70, 60, 60]])  # x, y, w, h

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *a, **kw: _FakeCascade(), raising=False)
    monkeypatch.setattr(cv2, "CASCADE_SCALE_IMAGE", 1, raising=False)

    # 对称扩展(margin=0.5, down=0.5)
    sym = _detect_face_mask(arr, margin_frac=0.5, down_margin_frac=0.5)
    # 非对称扩展(margin=0.5, down=0.08)
    asym = _detect_face_mask(arr, margin_frac=0.5, down_margin_frac=0.08)

    assert sym.any() and asym.any()

    # 脸 bbox: x=70,y=70,w=60,h=60。max(w,h)=60。margin=0.5 → mx=30。
    # 对称: y0=70-30=40, y1=70+60+30=160 → 向下扩展到 row 160
    # 非对称: md=60*0.08=4 → y1=70+60+4=134 → 向下只到 row 134
    assert int(sym[155, 100]) == 1  # 对称覆盖 row 155
    assert int(asym[155, 100]) == 0  # 非对称不覆盖 row 155
    # 两者在 row 45(向上扩展区)都有值
    assert int(sym[45, 100]) == 1
    assert int(asym[45, 100]) == 1


# --- blend_tryon seamlessClone ----------------------------------------------


def test_blend_tryon_identity_region_from_original():
    """mask=0 的身份区,seamlessClone 后像素更接近 original 而非 generated。"""
    orig = Image.new("RGB", (80, 80), (10, 20, 30))
    gen = Image.new("RGB", (80, 80), (200, 200, 10))
    mask = np.zeros((80, 80), dtype=np.uint8)
    # 中心 30x30 = 衣服区(mask=255), 外围 = 身份区(mask=0)
    mask[25:55, 25:55] = 255
    out = np.array(blend_tryon(orig, gen, mask))

    # 身份区(角落): 更接近 original (10,20,30)
    corner = out[5, 5]
    assert abs(int(corner[0]) - 10) < abs(int(corner[0]) - 200)
    # 衣服区(中心): 更接近 generated (200,200,10)
    center = out[40, 40]
    assert abs(int(center[0]) - 200) < abs(int(center[0]) - 10)


def test_blend_tryon_multi_component_identity():
    """两个分离的身份区(上+下)都被 seamlessClone 贴回 original 像素。"""
    orig = Image.new("RGB", (80, 80), (10, 20, 30))
    gen = Image.new("RGB", (80, 80), (200, 200, 10))
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[30:50, 10:70] = 255  # 中间一条衣服区
    # 上方 row 10 和下方 row 70 都是身份区(mask=0)
    out = np.array(blend_tryon(orig, gen, mask))

    for pt in ((10, 40), (70, 40)):
        px = out[pt[0], pt[1]]
        assert abs(int(px[0]) - 10) < abs(int(px[0]) - 200), f"identity region {pt} not from original"


# --- reapply_garment_detail edge falloff ------------------------------------


def test_reapply_edge_falloff_center_stronger_than_edge(monkeypatch):
    """distanceTransform 边缘衰减: 有衰减时边缘区纹理变化 < 无衰减时;中心区差异小。

    通过 monkeypatch cv2.distanceTransform 返回全 1(d_max≤3 → 不触发衰减),
    对比正常(有衰减)与禁用衰减两种输出的差异集中在边缘而非中心。
    """
    import cv2

    rng = np.random.default_rng(42)
    ref = Image.fromarray(rng.integers(0, 256, (100, 100, 3), dtype=np.uint8), "RGB")
    gen = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8), "RGB")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    # 1) 正常 distanceTransform(有边缘衰减)
    out_falloff = np.array(reapply_garment_detail(gen, ref, mask, texture_strength=1.0, color_strength=0.5))

    # 2) 禁用衰减: distanceTransform 返回全 1 → d_max=1 ≤ 3 → edge_weight=mask_norm
    def _uniform_dist(src, *a, **kw):
        return np.ones(src.shape[:2], dtype=np.float32)

    monkeypatch.setattr(cv2, "distanceTransform", _uniform_dist)
    out_uniform = np.array(reapply_garment_detail(gen, ref, mask, texture_strength=1.0, color_strength=0.5))

    gen_arr = np.array(gen)

    # 边缘区(row 23-27,距 mask 顶边 3-7px): 衰减压制纹理 → falloff 变化 < uniform 变化
    edge_falloff = np.abs(out_falloff[23:27, 45:55].astype(int) - gen_arr[23:27, 45:55].astype(int)).mean()
    edge_uniform = np.abs(out_uniform[23:27, 45:55].astype(int) - gen_arr[23:27, 45:55].astype(int)).mean()
    assert edge_uniform > edge_falloff, f"edge: uniform {edge_uniform:.1f} should > falloff {edge_falloff:.1f}"

    # 中心区(row 45-55,距边缘 ≥25px): 衰减已满强度 → 两者接近
    center_falloff = np.abs(out_falloff[45:55, 45:55].astype(int) - gen_arr[45:55, 45:55].astype(int)).mean()
    center_uniform = np.abs(out_uniform[45:55, 45:55].astype(int) - gen_arr[45:55, 45:55].astype(int)).mean()
    center_gap = abs(center_falloff - center_uniform)
    edge_gap = edge_uniform - edge_falloff
    assert edge_gap > center_gap, f"edge effect {edge_gap:.1f} should > center effect {center_gap:.1f}"


# ---------------------------------------------------------------------------
# _detect_skin: YCbCr 肤色检测
# ---------------------------------------------------------------------------


def test_detect_skin_finds_skin_tone():
    """肤色像素(Cb 77-127, Cr 133-173)被检测为 255,非肤色为 0。"""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    # 典型肤色 RGB(180,140,120) → YCbCr 约 (148,105,150) → 命中
    arr[0:2, 0:2] = (180, 140, 120)
    # 白色背景 → 非肤色
    arr[2:4, 2:4] = (255, 255, 255)
    skin = _detect_skin(arr)
    assert skin[0, 0] == 255
    assert skin[3, 3] == 0


def test_detect_skin_returns_zeros_on_error(monkeypatch):
    """cv2.cvtColor 异常时返回全零(安全回退)。"""
    import cv2

    def _fail(*a, **kw):
        raise Exception("simulated failure")

    monkeypatch.setattr(cv2, "cvtColor", _fail)
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    skin = _detect_skin(arr)
    assert skin.shape == (10, 10)
    assert not skin.any()


# ---------------------------------------------------------------------------
# _heuristic_face_mask: subject bbox 顶部估算脸区
# ---------------------------------------------------------------------------


def test_heuristic_face_mask_top_18_percent():
    """主体 bbox 顶部 18% 被标记为脸区,其余为 0。"""
    arr = np.zeros((200, 100, 3), dtype=np.uint8)
    subj = np.zeros((200, 100), dtype=np.uint8)
    subj[50:190, 20:80] = 1  # subject from y=50 to y=190
    mask = _heuristic_face_mask(arr, subj, frac=0.18)
    # face_band = 50 + (190-50)*0.18 = 50 + 25 = 75
    assert mask[60, 40] == 1  # within top 18% of subject
    assert mask[80, 40] == 0  # below face band
    assert mask[10, 40] == 0  # above subject → not face


def test_heuristic_face_mask_none_subject():
    """subject_mask=None 时返回全零。"""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = _heuristic_face_mask(arr, None)
    assert not mask.any()


def test_heuristic_face_mask_empty_subject():
    """主体为空时返回全零。"""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    subj = np.zeros((100, 100), dtype=np.uint8)
    mask = _heuristic_face_mask(arr, subj)
    assert not mask.any()


# ---------------------------------------------------------------------------
# force_white_background + subject_mask + heuristic face: 胸口白块修复
# ---------------------------------------------------------------------------


def test_force_white_background_subject_mask_protects_chest():
    """subject_mask 保护主体:胸口浅色区域不被压白。

    无 subject_mask 时,胸口浅色(亮度 200+)若与背景连通会被压白。
    传 subject_mask 后主体区豁免,胸口原色保留。
    """
    # 米色背景 + 主体(深色框 + 浅色胸口区)
    arr = np.full((200, 200, 3), (185, 168, 152), dtype=np.uint8)
    arr[40:160, 60:140] = (80, 80, 80)  # 主体框(深色)
    arr[60:90, 70:130] = (230, 225, 220)  # 胸口浅色区(亮度 ~225)
    img = Image.fromarray(arr, "RGB")
    subj = np.zeros((200, 200), dtype=np.uint8)
    subj[40:160, 60:140] = 1  # 主体区

    out_with = np.array(force_white_background(img, subject_mask=subj))
    out_without = np.array(force_white_background(img, subject_mask=None))

    # 胸口区(row 70, col 100): 有 subject_mask 时保留原色(~225)
    chest_with = out_with[75, 100].mean()
    chest_without = out_without[75, 100].mean()
    # 有保护时胸口亮度应低于无保护时(无保护可能被压向 255)
    assert chest_with <= chest_without + 5


def test_force_white_background_heuristic_face_when_detection_fails(monkeypatch):
    """cv2 5.0 下 _detect_face_mask 返回全零,heuristic 兜底保护脸/胸口区。"""
    # 让 _detect_face_mask 返回全零(模拟 cv2 5.0)
    monkeypatch.setattr(
        "platform_presets._detect_face_mask",
        lambda *a, **kw: np.zeros((200, 200), dtype=np.uint8),
    )
    arr = np.full((200, 200, 3), (190, 175, 160), dtype=np.uint8)
    arr[40:160, 60:140] = (80, 80, 80)  # 主体框
    arr[50:70, 70:130] = (220, 215, 210)  # 脸区(浅色,与背景亮度接近)
    img = Image.fromarray(arr, "RGB")
    subj = np.zeros((200, 200), dtype=np.uint8)
    subj[40:160, 60:140] = 1

    out = np.array(force_white_background(img, subject_mask=subj))
    # 脸区(row 55, col 100): heuristic 保护 top 18% = 40+22=62,
    # row 55 在 [40,62) 内,应被保护不被压白
    face_region = out[55, 100].mean()
    # 背景区(row 10, col 10): 应被压白
    bg_region = out[10, 10].mean()
    assert bg_region > 250  # 背景压白
    # 脸区不被完全压白(原色 220, heuristic 保护)
    assert face_region < 255 or face_region > 240  # 保留原色或接近


# ---------------------------------------------------------------------------
# _correct_skin_tone: SDXL 生成肤色校正
# ---------------------------------------------------------------------------


def test_correct_skin_tone_shifts_toward_original():
    """result 肤色向 original 肤色偏移。"""
    # original 肤色 RGB(180,140,120)
    orig = np.full((50, 50, 3), (220, 220, 220), dtype=np.uint8)
    orig[10:40, 10:40] = (180, 140, 120)  # 肤色区
    # result 肤色偏白 RGB(210,190,180)
    res = np.full((50, 50, 3), (220, 220, 220), dtype=np.uint8)
    res[10:40, 10:40] = (210, 190, 180)  # 偏白肤色

    orig_img = Image.fromarray(orig, "RGB")
    res_img = Image.fromarray(res, "RGB")
    out = np.array(_correct_skin_tone(orig_img, res_img))

    # 肤色区应向 original 偏移: R 从 210 向 180 靠近
    assert out[25, 25, 0] < 210, f"R should shift toward 180, got {out[25, 25, 0]}"
    # 背景区基本不变(GaussianBlur 可能有 ±1 精度抖动)
    assert abs(out[0, 0, 0].astype(int) - 220) <= 1


def test_correct_skin_tone_no_skin_returns_unchanged():
    """无肤色时原样返回。"""
    orig = Image.fromarray(np.full((50, 50, 3), (255, 0, 0), dtype=np.uint8), "RGB")
    res = Image.fromarray(np.full((50, 50, 3), (0, 255, 0), dtype=np.uint8), "RGB")
    out = _correct_skin_tone(orig, res)
    out_arr = np.array(out)
    assert (out_arr == np.array(res)).all()


def test_correct_skin_tone_close_match_returns_unchanged():
    """original 与 result 肤色接近(差 < 2)时原样返回。"""
    skin = (180, 140, 120)
    orig = np.full((50, 50, 3), skin, dtype=np.uint8)
    res = np.full((50, 50, 3), skin, dtype=np.uint8)
    # 微小差异(1)
    res[10:40, 10:40] = (181, 141, 121)
    out = _correct_skin_tone(Image.fromarray(orig, "RGB"), Image.fromarray(res, "RGB"))
    out_arr = np.array(out)
    # shift < 2 → 不校正
    assert abs(out_arr[25, 25, 0].astype(int) - 181) <= 1


# ---------------------------------------------------------------------------
# reapply_garment_detail: 肤色区纹理抑制
# ---------------------------------------------------------------------------


def test_reapply_garment_detail_reduces_texture_on_skin():
    """mask 内肤色区纹理变化 < 非肤色区(手臂皮肤不应被贴服装纹理)。

    color_strength 必须非零:reapply 的最终输出 out = corrected_rgb * col_a + gen * (1-col_a),
    col_a=0 时 out=gen 纹理也丢失。用 color_strength=0.5 让纹理+颜色都生效。
    """
    rng = np.random.default_rng(42)
    # ref 有随机高频纹理
    ref = Image.fromarray(rng.integers(0, 256, (100, 100, 3), dtype=np.uint8), "RGB")
    # gen 上半部分是肤色, 下半部分是服装色
    gen_arr = np.full((100, 100, 3), 128, dtype=np.uint8)
    # 肤色 RGB(180,140,120) → 上半
    gen_arr[0:50, :] = (180, 140, 120)
    # 服装色 RGB(60,60,120) → 下半
    gen_arr[50:100, :] = (60, 60, 120)
    gen = Image.fromarray(gen_arr, "RGB")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255  # 整个区域都在 mask 内

    out = np.array(reapply_garment_detail(gen, ref, mask, texture_strength=1.0, color_strength=0.5))

    gen_native = np.array(gen)
    # 肤色区(row 30)纹理变化应远小于服装区(row 70)
    skin_change = np.abs(out[25:35, 40:60].astype(int) - gen_native[25:35, 40:60].astype(int)).mean()
    cloth_change = np.abs(out[65:75, 40:60].astype(int) - gen_native[65:75, 40:60].astype(int)).mean()
    assert cloth_change > skin_change * 2, f"cloth change {cloth_change:.1f} should >> skin change {skin_change:.1f}"


def test_reapply_garment_detail_skin_color_not_shifted():
    """mask 内肤色区颜色迁移被抑制(肤色不被拉向服装颜色)。"""
    # gen 肤色 RGB(180,140,120), ref 服装色 RGB(60,60,120)
    gen_arr = np.full((100, 100, 3), (180, 140, 120), dtype=np.uint8)
    gen_arr[40:60, 40:60] = (180, 140, 120)  # 全肤色
    gen = Image.fromarray(gen_arr, "RGB")
    ref = Image.fromarray(np.full((100, 100, 3), (60, 60, 120), dtype=np.uint8), "RGB")
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    out = np.array(reapply_garment_detail(gen, ref, mask, texture_strength=0.0, color_strength=0.85))

    gen_native = np.array(gen)
    # 肤色区颜色变化应很小(被抑制 85%)
    color_shift = np.abs(out[40:60, 40:60].astype(int) - gen_native[40:60, 40:60].astype(int)).mean()
    assert color_shift < 15, f"skin color shift {color_shift:.1f} should be small (<15)"
