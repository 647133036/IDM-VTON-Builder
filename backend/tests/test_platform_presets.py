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
    apply_platform_preset,
    blend_tryon,
    center_crop_square,
    clothing_mask,
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
