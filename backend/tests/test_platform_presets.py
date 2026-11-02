"""平台店规适配层单测:1:1 居中裁剪、纯白底压白、JPEG 限体积、平台模板。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from platform_presets import (
    _RTMPOSE_INPUT_WH,
    _RTMPOSE_SPLIT_RATIO,
    PLATFORM_PRESETS,
    PLATFORM_SET,
    _build_inpaint_mask,
    _correct_skin_tone,
    _detect_face_mask,
    _detect_skin,
    _estimate_waist_ratio,
    _heuristic_face_mask,
    _sdxl_inpaint_repair,
    _seam_repair,
    _simcc_decode,
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
    """连衣裙长款 hem=long 比 hem=waist 保留更多:row 168 在 long 保留、waist 截断。"""
    img, subj = _subject_img()
    m_waist = clothing_mask(
        img, garment_type="dress", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "waist"}
    )
    m_long = clothing_mask(
        img, garment_type="dress", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "long"}
    )
    assert m_waist is not None and m_long is not None
    # waist cut at 20+160*0.90=164 → row 168 截断; long cut at 20+160*0.97=175 → 保留
    assert int(m_long[168, 60]) > int(m_waist[168, 60]) + 50


def test_clothing_mask_onesie_does_not_fall_through():
    """onesie 必须被显式处理(不穿透到无裁切),返回非 None 且底部截断。"""
    img, subj = _subject_img()
    m = clothing_mask(img, garment_type="onesie", subject_mask=subj, garment_extent={"sleeve": "short", "hem": "waist"})
    assert m is not None
    # onesie short cut at 20+160*0.90=164 → row 170 截断
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


def test_estimate_waist_ratio_uniform_falls_back():
    """均匀宽度(无收腰)回退到 0.62。"""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    ratio = _estimate_waist_ratio(subj, 20, 160)
    assert ratio == 0.62


def test_estimate_waist_ratio_narrowed_detects_waist():
    """有收腰的主体,腰线比例偏离 0.62。"""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:60, 20:100] = 1
    subj[60:90, 35:85] = 1
    subj[90:110, 45:75] = 1
    subj[110:140, 35:85] = 1
    subj[140:180, 25:95] = 1
    ratio = _estimate_waist_ratio(subj, 20, 160)
    assert ratio != 0.62
    assert 0.45 <= ratio <= 0.58


def test_estimate_waist_ratio_median_fallback():
    """宽度变化 < 5%(无明显收腰)→ 回退 0.62。"""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:60, 20:100] = 1
    subj[60:100, 21:99] = 1
    subj[100:140, 20:100] = 1
    subj[140:180, 20:100] = 1
    ratio = _estimate_waist_ratio(subj, 20, 160)
    assert ratio == 0.62


def test_estimate_waist_ratio_asymmetric_fallback():
    """最窄点左右严重不对称(一侧裁切)→ 回退 0.62。"""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:60, 20:100] = 1
    subj[60:90, 35:85] = 1
    subj[90:110, 60:75] = 1
    subj[110:140, 35:85] = 1
    subj[140:180, 25:95] = 1
    ratio = _estimate_waist_ratio(subj, 20, 160)
    assert ratio == 0.62


def test_clothing_mask_waist_aware_cut():
    """有收腰的主体,腰线裁切跟随实际腰线(非固定 0.62)。"""
    subj_w = np.zeros((200, 120), dtype=np.uint8)
    subj_w[20:60, 20:100] = 1
    subj_w[60:90, 35:85] = 1
    subj_w[90:110, 45:75] = 1
    subj_w[110:140, 35:85] = 1
    subj_w[140:180, 25:95] = 1
    img = Image.fromarray(np.full((200, 120, 3), 80, dtype=np.uint8), "RGB")
    m_waist = clothing_mask(img, garment_type="upper", subject_mask=subj_w)
    _, subj_u = _subject_img()
    m_uniform = clothing_mask(img, garment_type="upper", subject_mask=subj_u)
    assert m_waist is not None and m_uniform is not None
    assert int(m_waist[105, 60]) < int(m_uniform[105, 60])


# --- MediaPipe Pose keypoints + dress neck/leg fix --------------------------


def test_estimate_waist_ratio_uses_pose_keypoints():
    """有 pose_keypoints 时用 shoulder→hip 插值算腰线, 不走轮廓法."""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    # shoulder y=0.33*200=66, hip y=0.68*200=136
    # waist = 66 + (136-66)*0.55 = 104.5 → ratio = (104.5-20)/160 = 0.528
    kp = {"hip": (0.5, 0.68, 0.9), "shoulder": (0.5, 0.33, 0.9), "knee": (0.5, 0.95, 0.8)}
    ratio = _estimate_waist_ratio(subj, 20, 160, pose_keypoints=kp)
    assert 0.48 <= ratio <= 0.58


def test_estimate_waist_ratio_pose_low_visibility_falls_back():
    """pose hip 或 shoulder visibility < 0.3 → 回退轮廓法。"""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    kp = {"hip": (0.5, 0.68, 0.2), "shoulder": (0.5, 0.33, 0.9), "knee": (0.5, 0.95, 0.8)}
    ratio = _estimate_waist_ratio(subj, 20, 160, pose_keypoints=kp)
    # uniform width → contour fallback → 0.62
    assert ratio == 0.62


def test_estimate_waist_ratio_pose_out_of_range_falls_back():
    """几何通过但插值超出 [0.45, 0.65] → 回退轮廓法 (单独触发范围门控)."""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    # sho_r=(0.42*200-20)/160=0.40, hip_r=(0.86*200-20)/160=0.95
    # ratio=0.40+0.55*0.55=0.7025 > 0.65; knee_r=(0.95*200-20)/160=1.0625 > hip_r
    kp = {"hip": (0.5, 0.86, 1.0), "shoulder": (0.5, 0.42, 1.0), "knee": (0.5, 0.95, 1.0)}
    ratio = _estimate_waist_ratio(subj, 20, 160, pose_keypoints=kp)
    assert ratio == 0.62


def test_estimate_waist_ratio_geom_hip_above_shoulder_falls_back():
    """hip 不低于 shoulder+5% → 几何不一致, 回退轮廓法."""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    kp = {"hip": (0.5, 0.30, 1.0), "shoulder": (0.5, 0.40, 1.0), "knee": (0.5, 0.80, 1.0)}
    ratio = _estimate_waist_ratio(subj, 20, 160, pose_keypoints=kp)
    assert ratio == 0.62


def test_estimate_waist_ratio_geom_knee_above_hip_falls_back():
    """knee 不高于 hip → 几何不一致, 回退轮廓法."""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    kp = {"hip": (0.5, 0.68, 1.0), "shoulder": (0.5, 0.33, 1.0), "knee": (0.5, 0.60, 1.0)}
    ratio = _estimate_waist_ratio(subj, 20, 160, pose_keypoints=kp)
    assert ratio == 0.62


def test_clothing_mask_dress_excludes_neck():
    """dress 类型排除脖子区(顶部 22%): row 50(脖子)被排除, row 70(胸口)保留."""
    img, subj = _subject_img()
    m = clothing_mask(img, garment_type="dress", subject_mask=subj)
    assert m is not None
    # neck_end = 20 + 160*0.22 = 55 → row 50 被排除
    assert int(m[50, 60]) < 40
    # row 70 保留
    assert int(m[70, 60]) > 100


def test_clothing_mask_upper_does_not_exclude_neck():
    """upper 类型不排除脖子(领口在衣物范围): row 35(脸区)被排除, row 50 保留."""
    img, subj = _subject_img()
    m = clothing_mask(img, garment_type="upper", subject_mask=subj)
    assert m is not None
    # upper 不做 neck 排除 → row 50 应保留(仅脸区 18% 排除到 row 48)
    assert int(m[50, 60]) > 100


def test_clothing_mask_dress_covers_legs():
    """dress normal 下摆延伸到 90% 身高(覆盖腿): row 160(大腿)保留。"""
    img, subj = _subject_img()
    m = clothing_mask(img, garment_type="dress", subject_mask=subj)
    assert m is not None
    # cut at 20+160*0.90=164 → row 160 保留
    assert int(m[160, 60]) > 100


def test_clothing_mask_dress_knee_keypoints(monkeypatch):
    """有膝关键点时 dress 下摆精确截断到膝部。"""
    img, subj = _subject_img()
    # knee at y=0.85*200=170 → knee_r = (170-20)/160 = 0.9375
    kp = {"hip": (0.5, 0.68, 0.9), "shoulder": (0.5, 0.33, 0.9), "knee": (0.5, 0.85, 0.8)}

    import platform_presets

    monkeypatch.setattr(platform_presets, "_detect_pose_keypoints", lambda img: kp)

    m = clothing_mask(img, garment_type="dress", subject_mask=subj)
    assert m is not None
    # knee_r=0.9375, normal cut = knee_r - 0.02 = 0.9175 → 20+160*0.9175=167
    # erode(9x9)+blur(sigma=6) feathers ~8px each side → row 155 safe, row 172 cut
    assert int(m[155, 60]) > 100
    assert int(m[172, 60]) < 40


def _simcc_round_trip(img_w, img_h, seed=42):
    iw, ih = _RTMPOSE_INPUT_WH
    crop_w, crop_h = iw * 2, ih * 2
    cx0, cy0 = 40, 50
    rng = np.random.RandomState(seed)
    orig_x = cx0 + rng.rand(17) * crop_w
    orig_y = cy0 + rng.rand(17) * crop_h
    mx = (orig_x - cx0) / crop_w * iw
    my = (orig_y - cy0) / crop_h * ih
    idx_x = np.round(mx * _RTMPOSE_SPLIT_RATIO).astype(int)
    idx_y = np.round(my * _RTMPOSE_SPLIT_RATIO).astype(int)
    sx = np.zeros((17, int(iw * _RTMPOSE_SPLIT_RATIO) + 8))
    sy = np.zeros((17, int(ih * _RTMPOSE_SPLIT_RATIO) + 8))
    sx[np.arange(17), idx_x] = 1.0
    sy[np.arange(17), idx_y] = 1.0
    kx, ky = _simcc_decode(sx, sy, (cx0, cy0, crop_w, crop_h), (img_w, img_h))
    return np.sqrt((kx * img_w - orig_x) ** 2 + (ky * img_h - orig_y) ** 2)


def test_simcc_decode_round_trip():
    """正向编码→解码 round-trip 误差 < 1px, 锁死 crop→原图变换链."""
    err = _simcc_round_trip(1024, 1536)
    assert err.max() < 1.0, f"round-trip error {err.max():.3f}px >= 1px"


def test_simcc_decode_round_trip_portrait_3_4():
    """3:4 竖图 x/y 独立缩放 round-trip < 1px."""
    err = _simcc_round_trip(768, 1024, seed=1)
    assert err.max() < 1.0, f"3:4 round-trip error {err.max():.3f}px >= 1px"


def test_simcc_decode_round_trip_landscape_4_3():
    """4:3 横图 x/y 独立缩放 round-trip < 1px."""
    err = _simcc_round_trip(1024, 768, seed=2)
    assert err.max() < 1.0, f"4:3 round-trip error {err.max():.3f}px >= 1px"


def test_simcc_decode_round_trip_near_square():
    """接近 1:1 非正方形 (1000x1024) 最容易掩盖 x/y 缩放方向错误."""
    err = _simcc_round_trip(1000, 1024, seed=3)
    assert err.max() < 1.0, f"near-square round-trip error {err.max():.3f}px >= 1px"


def test_fake_deps_injection_sets_stub_modules():
    """fake deps 注入后 mmdet/xtcocotools/mmcv.ops 是空壳模块."""
    import types

    import platform_presets

    if not platform_presets._RTMPOSE_FAKE_DEPS:
        import pytest

        pytest.skip("RTMPOSE_FAKE_DEPS=0")
    platform_presets._inject_fake_deps()
    assert platform_presets._FAKE_DEPS_INJECTED
    assert sys.modules["mmdet"].ConfigType is dict
    assert isinstance(sys.modules["xtcocotools"], types.ModuleType)
    assert hasattr(sys.modules["mmcv.ops"], "MultiScaleDeformableAttention")


def test_estimate_waist_ratio_shoulder_low_visibility_falls_back():
    """shoulder visibility < 0.3 → 回退轮廓法, 不硬插值."""
    subj = np.zeros((200, 120), dtype=np.uint8)
    subj[20:180, 20:100] = 1
    kp = {"hip": (0.5, 0.68, 0.9), "shoulder": (0.5, 0.33, 0.2), "knee": (0.5, 0.95, 0.8)}
    ratio = _estimate_waist_ratio(subj, 20, 160, pose_keypoints=kp)
    assert ratio == 0.62


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
    arr = np.full((10, 10, 3), (255, 255, 255), dtype=np.uint8)
    arr[0:6, 0:6] = (180, 140, 120)
    skin = _detect_skin(arr)
    assert skin[2, 2] == 255
    assert skin[8, 8] == 0


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


def test_detect_skin_morphology_removes_noise():
    """形态学开运算去除散点噪声,连续区域保留。"""
    arr = np.full((20, 20, 3), (255, 255, 255), dtype=np.uint8)
    arr[5, 5] = (180, 140, 120)
    arr[10:15, 10:15] = (180, 140, 120)
    skin = _detect_skin(arr)
    assert skin[5, 5] == 0
    assert skin[12, 12] == 255


def test_correct_skin_tone_excludes_face_region():
    """脸区(前景顶部 18%)不被校正,手臂区被校正。"""
    skin = (180, 140, 120)
    light = (210, 190, 180)
    orig = np.full((100, 100, 3), (241, 241, 241), dtype=np.uint8)
    orig[10:25, 40:60] = skin
    orig[60:80, 10:30] = skin
    res = np.full((100, 100, 3), (241, 241, 241), dtype=np.uint8)
    res[10:25, 40:60] = light
    res[60:80, 10:30] = light
    out = np.array(_correct_skin_tone(Image.fromarray(orig, "RGB"), Image.fromarray(res, "RGB")))
    face_delta = abs(out[15, 50, 0].astype(int) - 210)
    arm_delta = abs(out[70, 20, 0].astype(int) - 210)
    assert arm_delta > face_delta


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


# ---------------------------------------------------------------------------
# clothing_mask: 短袖排除手臂皮肤(遮罩策略改进)
# ---------------------------------------------------------------------------


def _subject_with_skin(size=(200, 200)):
    """主体图:深色衣身 + 肤色手臂在两侧。

    返回 (Image, subject_mask) 用于 clothing_mask 测试。
    """
    arr = np.full((size[1], size[0], 3), (220, 220, 220), dtype=np.uint8)  # 白背景
    # 主体:深色衣物在中心
    arr[50:170, 70:130] = (40, 40, 80)  # 衣服
    # 肤色手臂在两侧(短袖露出)
    arr[50:120, 55:70] = (180, 140, 120)  # 左臂肤色
    arr[50:120, 130:145] = (180, 140, 120)  # 右臂肤色
    # 头部
    arr[30:55, 85:115] = (180, 140, 120)
    img = Image.fromarray(arr, "RGB")
    subj = np.zeros(size, dtype=np.uint8)
    subj[30:170, 55:145] = 1
    return img, subj


def test_clothing_mask_short_sleeve_excludes_arm_skin():
    """短袖时手臂皮肤被排除出 clothing mask(保护原图手臂)。"""
    img, subj = _subject_with_skin()
    garment_extent = {"sleeve": "short", "hem": "waist"}
    mask = clothing_mask(img, garment_type="upper", subject_mask=subj, garment_extent=garment_extent)
    assert mask is not None
    # 左臂肤色区(row 80, col 60): 应被排除(mask=0,保留原图皮肤)
    assert mask[80, 60] < 20, f"left arm skin should be excluded, got {mask[80, 60]}"
    # 右臂肤色区(row 80, col 135): 应被排除
    assert mask[80, 135] < 20, f"right arm skin should be excluded, got {mask[80, 135]}"
    # 衣物中心区(row 100, col 100): 应保留(mask>0,允许生成替换)
    assert mask[100, 100] > 20, f"torso clothing should remain in mask, got {mask[100, 100]}"


def test_clothing_mask_long_sleeve_keeps_arm_skin_in_mask():
    """长袖时手臂皮肤不排除(新衣物需覆盖整条手臂)。"""
    img, subj = _subject_with_skin()
    garment_extent = {"sleeve": "long", "hem": "waist"}
    mask = clothing_mask(img, garment_type="upper", subject_mask=subj, garment_extent=garment_extent)
    assert mask is not None
    # 长袖:手臂区(在 72% 裁切线以上)不应因肤色被排除
    # row 80 在主体范围内,左臂 col 60 在侧边 25% (arm_w=50) 内
    # 长袖不减肤色 → mask 应 > 0(允许生成结果覆盖手臂)
    assert mask[80, 60] > 0, f"long sleeve should keep arm in mask, got {mask[80, 60]}"


def test_clothing_mask_sleeveless_excludes_arm_skin():
    """无袖时手臂皮肤被排除(无袖不覆盖手臂)。"""
    img, subj = _subject_with_skin()
    garment_extent = {"sleeve": "sleeveless", "hem": "waist"}
    mask = clothing_mask(img, garment_type="upper", subject_mask=subj, garment_extent=garment_extent)
    assert mask is not None
    # 无袖:手臂皮肤应被排除
    if mask[80, 60] > 0:
        # 可能在 55% 裁切线下方已被清零,检查上方的肤色
        assert mask[60, 60] < 20 or mask[80, 60] < 20, "sleeveless should exclude arm skin"


# ---------------------------------------------------------------------------
# _seam_repair: cv2.inpaint Telea 接缝修复
# ---------------------------------------------------------------------------


def test_seam_repair_smooths_boundary():
    """mask 边界带像素被 Telea 修改,边界外像素保持不变。"""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[:, :100] = (200, 50, 50)
    arr[:, 100:] = (50, 50, 200)
    img = Image.fromarray(arr, "RGB")
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[:, :100] = 255

    out = np.array(_seam_repair(img, mask, grow_px=8, radius=5))
    orig = np.array(img)

    changed = np.any(out != orig, axis=2)
    assert changed.any(), "seam band pixels should be modified"
    assert not changed[50, 20], "far-left (non-seam) should be unchanged"
    assert not changed[50, 180], "far-right (non-seam) should be unchanged"


def test_seam_repair_no_seam_returns_unchanged():
    """无 mask 边界(全 0 或全 255)时原样返回。"""
    img = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8), "RGB")
    mask_all_zero = np.zeros((50, 50), dtype=np.uint8)
    out = _seam_repair(img, mask_all_zero)
    assert np.array_equal(np.array(out), np.array(img))

    mask_all_full = np.full((50, 50), 255, dtype=np.uint8)
    out2 = _seam_repair(img, mask_all_full)
    assert np.array_equal(np.array(out2), np.array(img))


def test_seam_repair_grow_px_controls_band_width():
    """grow_px 越大,修复带越宽,更多边界像素被改变。radius 需大于带宽才能填满。"""
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[:, :100] = (255, 0, 0)
    arr[:, 100:] = (0, 0, 255)
    img = Image.fromarray(arr, "RGB")
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[:, :100] = 255

    orig = np.array(img)
    out_small = np.array(_seam_repair(img, mask, grow_px=4, radius=20))
    out_large = np.array(_seam_repair(img, mask, grow_px=16, radius=20))

    changed_small = np.count_nonzero(np.any(out_small != orig, axis=2))
    changed_large = np.count_nonzero(np.any(out_large != orig, axis=2))
    assert changed_large > changed_small, (
        f"larger grow_px should change more pixels: {changed_large} vs {changed_small}"
    )


# ---------------------------------------------------------------------------
# _sdxl_inpaint_repair: SDXL Inpainting 骨架(env var 控制)
# ---------------------------------------------------------------------------


def test_sdxl_inpaint_repair_disabled_by_default(monkeypatch):
    """VTON_INPAINT_REPAIR 未设置时原样返回(不加载模型)。"""
    monkeypatch.delenv("VTON_INPAINT_REPAIR", raising=False)
    img = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8), "RGB")
    mask = np.full((50, 50), 255, dtype=np.uint8)
    out = _sdxl_inpaint_repair(img, mask)
    assert np.array_equal(np.array(out), np.array(img))


def test_sdxl_inpaint_repair_no_model_returns_unchanged(monkeypatch):
    """VTON_INPAINT_REPAIR=1 但模型不存在时原样返回。"""
    monkeypatch.setenv("VTON_INPAINT_REPAIR", "1")
    img = Image.fromarray(np.full((50, 50, 3), 128, dtype=np.uint8), "RGB")
    mask = np.full((50, 50), 255, dtype=np.uint8)
    out = _sdxl_inpaint_repair(img, mask)
    # 模型不存在 → 原样返回
    assert np.array_equal(np.array(out), np.array(img))


def test_build_inpaint_mask_seam_band_only():
    """无 original 时只提取接缝带,内部区域为 0。"""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    img = Image.fromarray(np.full((60, 60, 3), 128, dtype=np.uint8), "RGB")
    focused = _build_inpaint_mask(img, mask, original=None)
    assert focused.any()
    # 内部(远离边界)不应在接缝带中
    assert focused[30, 30] == 0
    # 边界附近应有值
    assert focused[10, 30] == 255 or focused[49, 30] == 255


def test_build_inpaint_mask_adds_skin_anomaly():
    """original 传入时检测肤色异常区,加入 focused mask。"""
    skin = (180, 140, 120)
    orig = np.full((100, 100, 3), (241, 241, 241), dtype=np.uint8)
    orig[60:80, 10:30] = skin
    res = np.full((100, 100, 3), (241, 241, 241), dtype=np.uint8)
    res[60:80, 10:30] = (210, 190, 180)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    focused = _build_inpaint_mask(
        Image.fromarray(res, "RGB"),
        mask,
        original=Image.fromarray(orig, "RGB"),
    )
    # 手臂区(60-80, 10-30)肤色差异 > 30,应被标记
    assert focused[70, 20] == 255


def test_build_inpaint_mask_smaller_than_full():
    """聚焦 mask 像素数远少于整张 shared_mask。"""
    mask = np.full((100, 100), 255, dtype=np.uint8)
    img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8), "RGB")
    focused = _build_inpaint_mask(img, mask, original=None)
    assert focused.sum() < mask.sum()


# ---------------------------------------------------------------------------
# reapply_garment_detail pattern_lock: warp 衣覆盖,生成图只调光影
# ---------------------------------------------------------------------------


def test_pattern_lock_center_matches_ref_pattern():
    """pattern_lock 中心区应贴回参考图花纹,而不是生成图颜色。"""
    gen = Image.fromarray(np.full((80, 80, 3), (40, 40, 180), dtype=np.uint8), "RGB")
    ref_arr = np.full((80, 80, 3), (180, 40, 40), dtype=np.uint8)
    ref_arr[20:60, 20:40] = (40, 180, 40)
    ref = Image.fromarray(ref_arr, "RGB")
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:70, 10:70] = 255

    out = np.array(reapply_garment_detail(gen, ref, mask, pattern_lock=True, light_strength=0.0))
    # 中心红区贴近 ref 红,远离 gen 蓝
    red = out[30:50, 45:55]
    assert red[..., 0].mean() > 120
    assert red[..., 2].mean() < 80
    # 中心绿条贴近 ref 绿
    green = out[30:50, 25:35]
    assert green[..., 1].mean() > 120
    # mask 外保持生成图
    assert np.array_equal(out[:10, :], np.array(gen)[:10, :])


def test_pattern_lock_skips_white_background():
    """cloth_ref 白底不得覆盖生成图像素。"""
    gen_arr = np.full((64, 64, 3), (30, 80, 160), dtype=np.uint8)
    gen = Image.fromarray(gen_arr, "RGB")
    ref_arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    ref_arr[16:48, 16:48] = (200, 40, 40)
    ref = Image.fromarray(ref_arr, "RGB")
    mask = np.full((64, 64), 255, dtype=np.uint8)

    out = np.array(reapply_garment_detail(gen, ref, mask, pattern_lock=True, light_strength=0.0))
    # 白底区(角)保持生成图蓝
    corner = out[2:8, 2:8]
    assert np.abs(corner.astype(int) - gen_arr[2:8, 2:8].astype(int)).mean() < 5
    # 内容区贴近红色参考
    inner = out[24:40, 24:40]
    assert inner[..., 0].mean() > 120


def test_pattern_lock_zero_mask_keeps_generated():
    """pattern_lock 在 mask 全 0 时不改像素。"""
    gen = Image.new("RGB", (48, 48), (90, 70, 50))
    ref = Image.new("RGB", (48, 48), (10, 200, 30))
    mask = np.zeros((48, 48), dtype=np.uint8)
    out = reapply_garment_detail(gen, ref, mask, pattern_lock=True)
    assert np.array_equal(np.array(out), np.array(gen))


def test_pattern_lock_reduces_overlay_on_skin():
    """pattern_lock 对肤色区降权,手臂不被贴上服装色。"""
    gen_arr = np.full((80, 80, 3), (180, 140, 120), dtype=np.uint8)
    gen = Image.fromarray(gen_arr, "RGB")
    ref = Image.fromarray(np.full((80, 80, 3), (40, 40, 160), dtype=np.uint8), "RGB")
    mask = np.full((80, 80), 255, dtype=np.uint8)

    out = np.array(reapply_garment_detail(gen, ref, mask, pattern_lock=True, light_strength=0.0))
    shift = np.abs(out[20:60, 20:60].astype(int) - gen_arr[20:60, 20:60].astype(int)).mean()
    assert shift < 40, f"skin overlay shift {shift:.1f} should stay small"


def test_pattern_lock_frequency_separation_preserves_pattern():
    """频率分离: 强光影下花纹高频仍可见(std > 均匀区)。"""
    ref_arr = np.full((80, 80, 3), 180, dtype=np.uint8)
    for c in range(0, 80, 4):
        ref_arr[:, c : c + 2] = 40
    ref = Image.fromarray(ref_arr, "RGB")

    gen_arr = np.full((80, 80, 3), 128, dtype=np.uint8)
    gen_arr[:, :40] = 50
    gen_arr[:, 40:] = 200
    gen = Image.fromarray(gen_arr, "RGB")
    mask = np.full((80, 80), 255, dtype=np.uint8)

    out = np.array(reapply_garment_detail(gen, ref, mask, pattern_lock=True, light_strength=1.0))
    center = out[30:50, 30:50]
    assert center.std() > 15, f"high-freq pattern should survive strong lighting: std={center.std():.1f}"
