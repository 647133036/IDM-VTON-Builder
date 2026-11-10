import numpy as np
import pytest
from engine import (
    ETHNICITY_FIT,
    TORSOS,
    _garment_bbox,
    _silhouette_scale,
    _warp_garment,
    compose_tryon,
    engine_status,
    fit_garment_to_body,
    guess_audience,
    guess_garment_type,
    remove_watermark,
    torso_box,
)


def test_torso_box_defaults_to_western_europe_without_fit():
    box = torso_box("women", "no_such_ethnicity")
    assert box["shoulder"] == pytest.approx(TORSOS["women"]["shoulder"], rel=1e-6)


def test_torso_box_east_asia_narrow_shoulder_and_longer_torso():
    west = torso_box("women", "western_europe")
    east = torso_box("women", "east_asia")
    assert east["shoulder"] < west["shoulder"]
    assert east["w"] < west["w"]
    assert east["h"] > west["h"]


def test_torso_box_north_america_wider():
    west = torso_box("men", "western_europe")
    na = torso_box("men", "north_america")
    assert na["shoulder"] > west["shoulder"]
    assert na["w"] > west["w"]


def test_torso_box_centered_when_width_changes():
    west = torso_box("women", "western_europe")
    east = torso_box("women", "east_asia")
    center_west = west["x"] + west["w"] / 2
    center_east = east["x"] + east["w"] / 2
    assert center_east == pytest.approx(center_west, abs=1e-9)


def test_ethnicity_fit_covers_all_default_values():
    values = [
        "western_europe",
        "eastern_europe",
        "west_asia",
        "central_asia",
        "east_asia",
        "southeast_asia",
        "africa",
        "north_america",
        "south_america",
        "oceania",
    ]
    for v in values:
        assert v in ETHNICITY_FIT
        keys = {"w", "h", "shoulder"}
        assert set(ETHNICITY_FIT[v]) >= keys


def test_guess_audience():
    assert guess_audience("ladies_top.jpg") == "women"
    assert guess_audience("mens_tshirt.png") == "men"
    assert guess_audience("girl_dress.jpg") == "girls"
    assert guess_audience("boy_shirt.png") == "boys"
    assert guess_audience("baby_onesie.jpg") == "baby"
    assert guess_audience("unknown.jpg") == "women"
    assert guess_audience("unknown.jpg", "men") == "men"


def test_guess_garment_type():
    assert guess_garment_type("dress_a.jpg") == "dress"
    assert guess_garment_type("trousers.jpg") == "lower"
    assert guess_garment_type("onesie.jpg") == "onesie"
    assert guess_garment_type("t_shirt.jpg") == "upper"


def test_compose_tryon_outputs_image(base_image, garment_image):
    from PIL import Image as PILImage

    result = compose_tryon(
        base_image,
        garment_image,
        "women",
        "upper",
        long_side=256,
        background="white",
        ethnicity="east_asia",
    )
    assert isinstance(result, PILImage.Image)
    assert max(result.size) == 256


def test_compose_tryon_transparent_background(base_image, garment_image):
    result = compose_tryon(
        base_image,
        garment_image,
        "men",
        "dress",
        long_side=256,
        background="transparent",
        ethnicity="north_america",
    )
    assert result.mode in ("RGBA", "LA")


def test_compose_tryon_east_vs_north_pixels_differ(tmp_path):
    from PIL import Image as PILImage
    from PIL import ImageDraw

    base = tmp_path / "base_green.png"
    garment = tmp_path / "garment_panel.png"
    PILImage.new("RGB", (220, 360), (40, 160, 40)).save(base)

    gimg = PILImage.new("RGB", (280, 380), (255, 255, 255))
    d = ImageDraw.Draw(gimg)
    d.rectangle([40, 60, 240, 320], fill=(220, 40, 40))
    d.rectangle([60, 80, 220, 300], fill=(20, 90, 200))
    gimg.save(garment)

    east = compose_tryon(
        base,
        garment,
        "women",
        "upper",
        long_side=256,
        background="white",
        ethnicity="east_asia",
    )
    north = compose_tryon(
        base,
        garment,
        "women",
        "upper",
        long_side=256,
        background="white",
        ethnicity="north_america",
    )
    ea, na = east.convert("RGB"), north.convert("RGB")
    diff = sum(
        1
        for ex, nx in zip(ea.getdata(), na.getdata(), strict=True)
        if abs(ex[0] - nx[0]) + abs(ex[1] - nx[1]) + abs(ex[2] - nx[2]) > 3
    )
    total = ea.width * ea.height
    assert diff / total > 0.01


def test_engine_status_shape():
    st = engine_status()
    assert set(st) >= {
        "engine",
        "license_mode",
        "gpu_available",
        "idm_vton_available",
        "sdxl_available",
        "fashn_available",
        "sdxl_inpaint",
        "commercial_compliant",
        "compliance_advice",
    }
    assert st["engine"] in (
        "composite-fallback",
        "idm-vton",
        "sdxl-commercial",
        "fashn-vton",
    )


def test_silhouette_scale_is_smooth_and_bounded():
    import numpy as np

    box = torso_box("women", "east_asia")
    ys = np.linspace(0, 1, 101)
    s = _silhouette_scale(ys, box, 0.42)
    assert s.min() >= 0.72
    assert s.max() <= 1.06
    # 腰部应比肩部略窄(曲线贴合躯干)。
    shoulder = float(np.interp(0.10, ys, s))
    waist = float(np.interp(0.5, ys, s))
    assert waist < shoulder


def test_silhouette_scale_zero_strength_is_identity():
    import numpy as np

    box = torso_box("women", "western_europe")
    ys = np.linspace(0, 1, 8)
    assert np.allclose(_silhouette_scale(ys, box, 0.0), 1.0)


def test_warp_garment_changes_pixels():
    import numpy as np
    from PIL import Image

    # 纯色渐变衣料:无变形时左右对称,变形后中心列采样点偏移,像素必不同。
    w, h = 200, 260
    x = np.tile(np.linspace(0, 255, w), (h, 1))
    img = Image.fromarray(x.astype(np.uint8), "L").convert("RGBA")
    box = torso_box("women", "east_asia")
    out = _warp_garment(img, box, 0.42)
    assert out.size == img.size
    assert out.mode == "RGBA"
    a = np.asarray(img.convert("RGBA"))
    b = np.asarray(out.convert("RGBA"))
    assert not np.array_equal(a, b)


def test_warp_garment_noop_without_cv2_or_zero_strength():
    from PIL import Image

    img = Image.new("RGBA", (80, 100), (200, 40, 40, 255))
    box = torso_box("women", "east_asia")
    out = _warp_garment(img, box, 0.0)
    assert out is img or out.size == img.size


def test_warp_garment_vertical_aligns_content_to_torso():
    """VTON_WARP_VERTICAL=1 时,衣料内容中心向躯干框中心对齐。"""
    import os

    import numpy as np
    from PIL import Image

    w, h = 200, 300
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[200:280, 50:150, :3] = 128
    arr[200:280, 50:150, 3] = 255
    img = Image.fromarray(arr, "RGBA")
    box = torso_box("women", "east_asia")

    old = os.environ.get("VTON_WARP_VERTICAL")
    os.environ["VTON_WARP_VERTICAL"] = "1.0"
    try:
        out = _warp_garment(img, box, 0.42)
    finally:
        if old is None:
            os.environ.pop("VTON_WARP_VERTICAL", None)
        else:
            os.environ["VTON_WARP_VERTICAL"] = old

    out_arr = np.asarray(out)
    content_before = np.where(arr[..., 3].sum(axis=1) > 20)[0]
    content_after = np.where(out_arr[..., 3].sum(axis=1) > 20)[0]
    assert len(content_before) > 0
    assert len(content_after) > 0
    center_before = (content_before[0] + content_before[-1]) / 2
    center_after = (content_after[0] + content_after[-1]) / 2
    assert center_after < center_before


def test_warp_garment_vertical_disabled_preserves_rows():
    """VTON_WARP_VERTICAL=0 时,垂直方向不变形(仅水平缩放)。"""
    import os

    import numpy as np
    from PIL import Image

    w, h = 200, 300
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[100:180, 50:150, :3] = 128
    arr[100:180, 50:150, 3] = 255
    img = Image.fromarray(arr, "RGBA")
    box = torso_box("women", "east_asia")

    old = os.environ.get("VTON_WARP_VERTICAL")
    os.environ["VTON_WARP_VERTICAL"] = "0.0"
    try:
        out = _warp_garment(img, box, 0.42)
    finally:
        if old is None:
            os.environ.pop("VTON_WARP_VERTICAL", None)
        else:
            os.environ["VTON_WARP_VERTICAL"] = old

    out_arr = np.asarray(out)
    content_before = np.where(arr[..., 3].sum(axis=1) > 20)[0]
    content_after = np.where(out_arr[..., 3].sum(axis=1) > 20)[0]
    assert len(content_before) > 0
    assert len(content_after) > 0
    assert content_after[0] == content_before[0]
    assert content_after[-1] == content_before[-1]


def test_warp_garment_anchors_hem_to_central_band_not_wings():
    """垂直锚点取中央带: 侧垂翼(袖尖/垂带)不得当 hem, 锚行以下拖影清零。

    c10 04469: 全 bbox 锚把垂带当下摆 → 衣身整体上提(hem 截在肚脐、结压胸口);
    remap 尾段退化把下摆边缘行竖直拖影到画布底(09263 alpha 内容到 1023)。
    """
    import numpy as np
    from PIL import Image, ImageDraw

    g = Image.new("RGBA", (400, 800), (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    d.rectangle([120, 60, 280, 560], fill=(70, 120, 160, 255))  # 衣身, 下摆 560
    d.rectangle([80, 60, 118, 780], fill=(70, 120, 160, 255))  # 左垂翼到 780
    d.rectangle([282, 60, 320, 780], fill=(70, 120, 160, 255))  # 右垂翼
    box = {"y": 0.2, "h": 0.4, "shoulder": 0.9, "hem": 0.6, "neck": 0.2, "sleeve": 0.05}
    a = np.asarray(_warp_garment(g, box, 0.4))[..., 3]
    # 中央带 [0.32, 0.68]w: 下摆须锚在 box 底 0.6*800=480 (而非翼底 ~773)
    rows = np.where((a[:, 128:272] > 100).sum(axis=1) > 4)[0]
    assert abs(int(rows.max()) - 480) <= 6, f"中央带下摆锚到 {rows.max()}, 期望 ~480"
    # 下摆锚行以下清零(拖影修复): 485 行起不得有实料
    assert not (a[485:, :] > 40).any(), "remap 尾段拖影未清零"


def test_remove_watermark_reduces_corner_highlights():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (360, 360), (90, 120, 150))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    d.text((18, 18), "WATERMARK", fill=(245, 248, 250), font=font)
    before = np.asarray(img).astype(np.float32)
    after = np.asarray(remove_watermark(img)).astype(np.float32)
    # 亮水印像素(亮度>210)在文本区域内应大幅减少。
    region = (slice(0, 60), slice(0, 220))
    bright_before = (before[region].mean(axis=2) > 210).sum()
    bright_after = (after[region].mean(axis=2) > 210).sum()
    assert bright_before >= 100
    assert bright_after < bright_before * 0.5


def test_remove_watermark_noop_when_no_corner_feature():
    import numpy as np
    from PIL import Image

    img = Image.new("RGB", (300, 300), (70, 90, 110))
    out = remove_watermark(img)
    # 无角部特征时基本不变(允许轻微抗锯齿差异)。
    assert np.abs(np.asarray(img).astype(np.int16) - np.asarray(out).astype(np.int16)).mean() < 2


def test_fit_garment_to_body_maps_wide_garment_to_shoulder_width():
    from PIL import Image, ImageDraw

    box = torso_box("women", "east_asia")
    tw, th = 400, 300
    wide = Image.new("RGBA", (600, 300), (0, 0, 0, 0))
    ImageDraw.Draw(wide).rectangle([0, 0, 600, 300], fill=(200, 40, 40, 255))
    fitted = fit_garment_to_body(wide, box, tw, th)
    bb = _garment_bbox(fitted)
    gw = bb[2] - bb[0] + 1
    assert abs(gw - box["shoulder"] * tw) <= 3
    # 衣料高度不超过躯干框 1.6 倍(防止拉伸过大)。
    gh = bb[3] - bb[1] + 1
    assert gh <= th * 1.6 + 2


def test_fit_garment_to_body_falls_back_when_no_alpha():
    from PIL import Image

    # 无 alpha 通道时回退到 fit_garment,仍返回 (tw,th) 尺寸。
    box = torso_box("men", "western_europe")
    img = Image.new("RGB", (100, 120), (10, 20, 30))
    out = fit_garment_to_body(img, box, 200, 260)
    assert out.size == (200, 260)


def test_engine_status_includes_ip_adapter_flag():
    st = engine_status()
    assert "ip_adapter_available" in st
    assert isinstance(st["ip_adapter_available"], bool)


def test_sdxl_available_checks_key_files(monkeypatch, tmp_path):
    # 目录存在但缺 model_index.json 时必须判为不可用(NEW-M4)。
    import engine_sdxl

    for d in ("sdxl_base", "controlnet_openpose", "controlnet_canny"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "sdxl_base" / "unet").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "_IDM_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "_TRAINING_DIR", tmp_path)
    avail, reason = engine_sdxl.sdxl_available()
    assert avail is False
    assert "model_index.json" in reason


def test_compose_tryon_prefers_fashn_in_commercial(monkeypatch, base_image, garment_image):
    import engine as eng
    import engine_fashn
    from PIL import Image as PILImage

    monkeypatch.setattr(eng, "_LICENSE_MODE", "commercial")
    monkeypatch.setattr(eng, "_license_allows_idm", lambda: (False, "commercial"))
    monkeypatch.setattr(engine_fashn, "fashn_available", lambda: (True, "ok"))

    def _fashn(*_a, **_k):
        img = PILImage.new("RGB", (64, 64), (12, 34, 56))
        return img

    monkeypatch.setattr(engine_fashn, "compose_tryon_fashn", _fashn)
    result = eng.compose_tryon(base_image, garment_image, "women", "upper", 64, "white", "western_europe")
    assert result.info.get("engine") == "fashn-vton"
    assert result.info.get("engine_degraded") is False


def test_compose_tryon_fashn_failure_falls_back_to_sdxl(monkeypatch, base_image, garment_image):
    import engine as eng
    import engine_fashn
    import engine_sdxl
    from PIL import Image as PILImage

    monkeypatch.setattr(eng, "_LICENSE_MODE", "commercial")
    monkeypatch.setattr(eng, "_license_allows_idm", lambda: (False, "commercial"))
    monkeypatch.setattr(engine_fashn, "fashn_available", lambda: (True, "ok"))
    monkeypatch.setattr(
        engine_fashn,
        "compose_tryon_fashn",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom fashn")),
    )
    monkeypatch.setattr(engine_sdxl, "sdxl_available", lambda: (True, "ok"))

    def _sdxl(*_a, **_k):
        img = PILImage.new("RGB", (64, 64), (90, 90, 90))
        return img

    monkeypatch.setattr(engine_sdxl, "compose_tryon_sdxl", _sdxl)
    result = eng.compose_tryon(base_image, garment_image, "women", "upper", 64, "white", "western_europe")
    assert result.info.get("engine") == "sdxl-commercial"
    assert result.info.get("engine_degraded") is True
    assert "FASHN" in result.info.get("degraded_reason", "")


def test_compose_tryon_sdxl_failure_falls_back(monkeypatch, base_image, garment_image):
    # SDXL 运行时抛异常必须回退合成引擎并在图上标注降级(NEW-H4)。
    import engine as eng
    import engine_sdxl

    monkeypatch.setattr(eng, "_LICENSE_MODE", "commercial")

    def _no_gpu():
        return False, "No GPU available"

    monkeypatch.setattr(engine_sdxl, "sdxl_available", lambda: (True, "ok"))
    monkeypatch.setattr(
        engine_sdxl,
        "compose_tryon_sdxl",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = eng.compose_tryon(base_image, garment_image, "women", "upper", 512, "white", "western_europe")
    assert result.size[0] > 0
    assert result.info.get("engine") == "composite-fallback"
    assert result.info.get("engine_degraded") is True
    assert "SDXL" in result.info.get("degraded_reason", "")


def test_compose_tryon_marks_engine_when_composite(base_image, garment_image):
    # 无 GPU 环境:合成引擎正常返回且 engine 标记为 composite-fallback。
    import engine as eng

    result = eng.compose_tryon(base_image, garment_image, "women", "upper", 512, "white", "western_europe")
    assert result.info.get("engine") == "composite-fallback"
    assert result.info.get("engine_degraded") is False


# ---- refit_sleeves_to_arms: 袖条重挂(修"长袖/短袖变无袖") ----
def _flat_with_folded_sleeves(tmp_path, sleeve="long", flare=False):
    """768x1024 白底平铺图: 中央灰衣身 + 左右向下折放袖(左红/右蓝), 可选低位绿散段。

    袖带宽 ~60px(带内占 <22%/78% 列), 衣身 240-528 不越带界, 与
    09290/09263/04469 实拍平铺(袖向下折放)同构。
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (768, 1024), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((240, 200, 528, 700), fill=(128, 128, 128))
    s_bottom = 700 if sleeve == "long" else 420
    d.rectangle((120, 260, 180, s_bottom), fill=(200, 60, 60))
    d.rectangle((588, 260, 648, s_bottom), fill=(60, 60, 200))
    if flare:
        d.rectangle((90, 800, 150, 890), fill=(30, 200, 60))
    img.save(tmp_path / f"flat_{sleeve}_{flare}.jpg")
    return img


def _blank_warped():
    from PIL import Image

    return Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))


def test_refit_pastes_both_sleeves_folded_layout(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path)
    out = refit_sleeves_to_arms(_blank_warped(), garment, 1024, None, "long")
    arr = np.asarray(out)
    # 左袖列 ~x297-363: 红色袖料; 右袖列 ~x660-726: 蓝色袖料
    # (右袖旧实现 -=tc1 坐标全负, 裁片越界成全透明 → 空贴)。
    for x0, x1, want in ((310, 350, (200, 60, 60)), (670, 710, (60, 60, 200))):
        zone = arr[300:600, x0:x1]
        solid = zone[(zone[:, :, 3] > 200)]
        assert solid.size, f"袖列 x{x0}-{x1} 无实心袖料"
        med = solid[:, :3].mean(axis=0)
        assert abs(int(med[0]) - want[0]) < 90 and abs(int(med[2]) - want[2]) < 90, f"袖色 {med} != {want}"


def test_refit_does_not_paste_white_background(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path)
    out = refit_sleeves_to_arms(_blank_warped(), garment, 1024, None, "long")
    arr = np.asarray(out)
    for x0, x1 in ((297, 363), (660, 726)):
        zone = arr[210:630, x0:x1]
        opaque = zone[zone[:, :, 3] > 200][:, :3]
        # 白底在裁片里 alpha≈0 不贴入: 贴入的实心像素不得是纯白块
        # (旧实现: 不透明 JPEG 直接裁 → 整条 51x430 白矩形贴上画布)。
        assert opaque.size > 500, "袖条实心区过小(疑似没贴上)"
        assert not np.any((opaque >= 248).all(axis=1))


def test_refit_short_sleeve_strip_ends_above_elbow(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path, sleeve="short")
    out = refit_sleeves_to_arms(_blank_warped(), garment, 1024, None, "short")
    arr = np.asarray(out)
    col = arr[:, 310:350, 3] > 60
    # 肩下袖段(短)在 y 300 有料, 肘下 y 500 无料(长袖才会到腕 634)
    assert col[280:330].any(), "短袖上臂段缺失"
    assert not col[480:530].any(), "短袖条贴过了肘(应为 ~0.40 臂长)"


def test_refit_ignores_detached_lower_segment(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path, flare=True)
    out = refit_sleeves_to_arms(_blank_warped(), garment, 1024, None, "long")
    arr = np.asarray(out)
    visible = arr[arr[:, :, 3] > 200][:, :3]
    # 低位散段(衣摆侧缘/垂带, 绿)不连肩, 不应被当袖贴上
    assert visible.size
    assert not np.any((visible[:, 1] > 150) & (visible[:, 0] < 100) & (visible[:, 2] < 100))


def test_refit_horizontal_sleeve_layout_rotates(tmp_path):
    from engine import refit_sleeves_to_arms
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (768, 1024), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((240, 200, 528, 700), fill=(128, 128, 128))
    d.rectangle((60, 300, 200, 360), fill=(200, 60, 60))
    d.rectangle((568, 300, 708, 360), fill=(60, 60, 200))
    out = refit_sleeves_to_arms(_blank_warped(), img, 1024, None, "long")
    arr = np.asarray(out)
    for x0, x1 in ((310, 350), (670, 710)):
        zone = arr[300:600, x0:x1]
        assert (zone[:, :, 3] > 200).sum() > 500, f"横向摊开袖未旋转贴上 x{x0}"
    left = arr[300:600, 310:350]
    solid = left[left[:, :, 3] > 200][:, :3]
    assert (solid[:, 0] > 120).mean() > 0.9, "左袖旋转后应为红色袖料"


def test_refit_passthrough_unsupported_sleeve_len(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path)
    for sl in (None, "sleeveless", "three_quarter"):
        warped = _blank_warped()
        out = refit_sleeves_to_arms(warped, garment, 1024, None, sl)
        assert np.asarray(out).tolist() == np.asarray(warped).tolist()


def test_refit_clamps_noisy_wrist_to_forearm(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path)
    pose = {
        "shoulder": (0.5, 0.20, 1.0),
        "wrist": (0.5, 0.99, 1.0),
    }
    out = refit_sleeves_to_arms(_blank_warped(), garment, 1024, pose, "long")
    arr = np.asarray(out)
    col = arr[:, 310:350, 3] > 60
    assert col[220:700].any(), "夹紧后袖条应仍在肩腕之间"
    assert not col[850:1020].any(), "腕点噪声不得把袖条拉到画布底"


# ---- _arm_column_bands: 剪影自适应手臂列(修"袖条贴到白背景上") ----
def test_arm_column_bands_detects_side_arms():
    """白底人物图: 最左/最右剪影段 = 左/右臂列带。"""
    from engine import _arm_column_bands
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((344, 200, 375, 700), fill=(150, 110, 90))
    d.rectangle((430, 180, 590, 800), fill=(200, 200, 200))
    d.rectangle((630, 200, 677, 700), fill=(150, 110, 90))
    bands = _arm_column_bands(img)
    assert bands is not None, "分臂剪影应检出臂列带"
    l0, l1, r0, r1 = bands
    assert abs(l0 - 344) <= 3 and abs(l1 - 375) <= 3, f"左臂带 {l0}-{l1} != 344-375"
    assert abs(r0 - 630) <= 3 and abs(r1 - 677) <= 3, f"右臂带 {r0}-{r1} != 630-677"


def test_arm_column_bands_single_blob_and_full_bleed():
    """臂贴躯干(单块剪影)按剪影宽 0.18 截两端臂带; 背景铺满画布返回 None。"""
    from engine import _arm_column_bands
    from PIL import Image, ImageDraw

    blob = Image.new("RGB", (1024, 1024), (255, 255, 255))
    d = ImageDraw.Draw(blob)
    d.rectangle((300, 180, 700, 820), fill=(180, 180, 180))
    l0, l1, r0, r1 = _arm_column_bands(blob)
    assert l0 == 300 and l1 == 300 + 72  # 0.18x剪影宽(400)=72
    assert r0 == 700 - 72 and r1 == 700
    full = Image.new("RGB", (1024, 1024), (120, 130, 140))
    assert _arm_column_bands(full) is None, "非白背景不得误判臂列"


def test_refit_uses_silhouette_arm_columns(tmp_path):
    """arm_cols 落位: 袖料贴进给定臂列, 硬编码画布比例列不再贴料。"""
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path)
    out = refit_sleeves_to_arms(_blank_warped(), garment, 1024, None, "long", arm_cols=(500, 540, 780, 830))
    arr = np.asarray(out)
    assert (arr[300:600, 500:540, 3] > 200).any(), "arm_cols 左袖未落位"
    assert (arr[300:600, 780:820, 3] > 200).any(), "arm_cols 右袖未落位"
    assert not (arr[300:600, 300:360, 3] > 200).any(), "左硬编码列不应再有袖料"
    assert not (arr[300:600, 660:720, 3] > 200).any(), "右硬编码列不应再有袖料"


def test_sleeve_strip_overlay_matches_refit_columns(tmp_path):
    """生成后覆盖层与 refit 同源同列: 臂列有料、别处透明、无袖型 None。"""
    from engine import sleeve_strip_overlay

    garment = _flat_with_folded_sleeves(tmp_path)
    cols = (500, 540, 780, 830)
    ov = sleeve_strip_overlay(garment, 1024, None, "long", arm_cols=cols)
    assert ov is not None, "长袖应产出覆盖层"
    a = np.asarray(ov)
    assert (a[300:600, 500:540, 3] > 200).any(), "覆盖层左臂列无袖料"
    assert (a[300:600, 780:820, 3] > 200).any(), "覆盖层右臂列无袖料"
    assert not (a[300:600, 300:360, 3] > 0).any(), "覆盖层不得贴到硬编码列"
    assert sleeve_strip_overlay(garment, 1024, None, "sleeveless", arm_cols=cols) is None


def test_sleeve_strip_top_is_solid(tmp_path):
    """袖条顶部必须实心: alpha 渐变会让袖看起来"从中段才开始"(c8 09290)。"""
    from engine import sleeve_strip_overlay

    garment = _flat_with_folded_sleeves(tmp_path)
    ov = sleeve_strip_overlay(garment, 1024, None, "long", arm_cols=(500, 540, 780, 830))
    a = np.asarray(ov)
    # 肩锚(0.20c=204)起 40 行内, 臂列中点 alpha 应已实心
    assert a[210:250, 505:535, 3].mean() > 200, "袖条顶部半透明(读作中袖)"


def test_sleeve_strip_shading_cylinder_and_cuff(tmp_path):
    """袖条写实明暗: 柱面(边暗中亮) + 袖口罗纹压暗(修"纯色填充/铁上去")。

    c9 09290 用户反馈袖口是"纯色": 平铺衣影棚平光直贴无立体感。
    """
    from engine import sleeve_strip_overlay

    garment = _flat_with_folded_sleeves(tmp_path)
    pose = {
        "shoulder": (0.34, 0.281, 1.0),
        "elbow_l": (0.325, 0.489, 1.0),
        "wrist_l": (0.299, 0.675, 1.0),
        "elbow_r": (0.668, 0.489, 1.0),
        "wrist_r": (0.690, 0.675, 1.0),
        "wrist": (0.5, 0.672, 1.0),
    }
    ov = sleeve_strip_overlay(garment, 1024, pose, "long", arm_cols=(345, 404, 617, 676))
    a = np.asarray(ov)
    # r=530(v=0.625, 竖褶项过零): 柱面 → 左袖条中段亮于外缘 ≥4%
    sel = np.where(a[530, :, 3] > 200)[0]
    assert len(sel) > 20, "r=530 无实心袖料"
    runs = []
    _start = _prev = int(sel[0])
    for c in sel[1:]:
        if c - _prev > 4:
            runs.append((_start, _prev))
            _start = c
        _prev = c
    runs.append((_start, _prev))
    l0, l1 = runs[0]  # 最左连续段 = 左袖条
    edge_px = a[530, l0 + 2 : l0 + 10, :3].astype(float)
    _mid = (l0 + l1) // 2
    center_px = a[530, _mid - 8 : _mid + 8, :3].astype(float)
    assert center_px.mean() > edge_px.mean() * 1.04, (
        f"柱面缺失: center={center_px.mean():.1f} edge={edge_px.mean():.1f}"
    )
    # 袖口段(罗纹)均亮低于前臂中段, 且缝线行显著更暗
    mid = a[560:600, :, :3][a[560:600, :, 3] > 200].astype(float).mean()
    cuff_means = []
    for r in range(664, 688):
        sel = a[r, :, 3] > 200
        if sel.any():
            cuff_means.append(a[r, :, :3][sel].astype(float).mean())
    assert cuff_means, "袖口段无实心料"
    assert np.mean(cuff_means) < 0.96 * mid, f"罗纹未压暗: cuff={np.mean(cuff_means):.1f} mid={mid:.1f}"
    assert min(cuff_means) < 0.87 * mid, "袖口缝线行未压暗"


def test_long_sleeve_ribbon_follows_forearm(tmp_path):
    """前臂斜出臂带外: 长袖须沿肩-肘-腕折线盖到腕, 臂带内下段不得铺料。

    c9 实测: 前臂肘下斜出臂带(腕 x=0.30c/0.69c 在带 0.34-0.40c/0.60-0.66c
    外), 竖条在肘以下会盖到躯干/空隙而漏掉真臂 —— 缎带沿折线才盖得住。
    """
    from engine import sleeve_strip_overlay

    garment = _flat_with_folded_sleeves(tmp_path)
    pose = {
        "shoulder": (0.34, 0.281, 1.0),
        "elbow_l": (0.325, 0.489, 1.0),
        "wrist_l": (0.299, 0.675, 1.0),
        "elbow_r": (0.668, 0.489, 1.0),
        "wrist_r": (0.690, 0.675, 1.0),
        "wrist": (0.5, 0.672, 1.0),
    }
    ov = sleeve_strip_overlay(garment, 1024, pose, "long", arm_cols=(345, 404, 617, 676))
    assert ov is not None
    a = np.asarray(ov)
    # 左腕 (306, 691): 腕行附近 650-685, 前臂折线 x≈307-312 宽~38 → 布料在 292-330
    assert a[650:685, 292:330, 3].mean() > 200, "长袖未沿前臂折线盖到腕"
    # 臂带内段 372-404(带内靠躯干侧)在腕行不应有布料(臂已在带外)
    assert a[650:685, 372:404, 3].mean() < 80, "袖条竖在臂带里, 盖到躯干/空隙"
    # 右腕 (707, 691): 前臂折线 x≈702-706 → 布料在 688-724
    assert a[650:685, 688:724, 3].mean() > 200, "右臂前臂未盖料"


def test_arm_refit_overlay_restores_forearm_along_polyline(tmp_path):
    """短袖: 臂带外的真前臂(肘→腕折线)须贴回原图像素, 修 SDXL 灰白前臂。"""
    from engine import arm_refit_overlay
    from PIL import Image, ImageDraw

    human = Image.new("RGB", (1024, 1024), (255, 255, 255))
    d = ImageDraw.Draw(human)
    # 左臂: 上臂 345-404 竖直, 前臂肘下斜出到 285-340, 手 285-330 到 ~750
    d.rectangle((345, 250, 404, 501), fill=(185, 136, 109))
    d.polygon([(320, 501), (345, 501), (330, 691), (296, 691)], fill=(185, 136, 109))
    d.rectangle((285, 691, 340, 750), fill=(185, 136, 109))
    d.rectangle((617, 250, 676, 501), fill=(185, 136, 109))
    garment = _flat_with_folded_sleeves(tmp_path)
    pose = {
        "shoulder": (0.34, 0.281, 1.0),
        "elbow_l": (0.325, 0.489, 1.0),
        "wrist_l": (0.299, 0.675, 1.0),
        "elbow_r": (0.668, 0.489, 1.0),
        "wrist_r": (0.690, 0.675, 1.0),
        "wrist": (0.5, 0.672, 1.0),
    }
    ov = arm_refit_overlay(garment, human, 1024, pose, "short", arm_cols=(345, 404, 617, 676))
    a = np.asarray(ov)
    # 短袖端(0.281+0.4*0.394=0.439c≈450)以下前臂 ~600 行, 折线 x≈319 宽~43
    px = a[590:610, 312:334]
    assert px[:, :, 3].mean() > 150, "带外前臂未贴回原图"
    assert np.abs(px[:, :, :3].astype(int) - np.array([185, 136, 109])).sum(axis=2).mean() < 40, "贴回的不是原图前臂色"


def test_garment_alpha_uses_corner_background():
    """衣料铺到上/左边缘的平铺图: 背景取四角 patch, alpha 保持实心。

    旧"顶行+左列中值"把衣料当背景 → alpha 全图塌软: pattern-lock 只贴
    两三成、胸口被 SDXL 灰白洗掉, 袖条顶部成半透明渐变。
    """
    from engine import garment_alpha
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 200), (205, 210, 215))  # 浅灰背景
    d = ImageDraw.Draw(img)
    # 粉上衣铺满上/左边缘(顶行/左列全是衣料), 四角留背景
    d.rectangle((20, 0, 180, 160), fill=(220, 160, 130))
    a = np.asarray(garment_alpha(img))
    center = a[40:120, 60:140].mean()
    bg = a[170:195, 20:95].mean()  # 下缘背景(下角 patch 区附近)
    assert center > 200, f"衣料中心 alpha 塌软: {center:.0f}"
    assert bg < 60, f"背景误判为衣料: {bg:.0f}"


def test_arm_refit_overlay_restores_base_below_strip(tmp_path):
    """臂区覆盖层: 袖端行以上=袖料, 以下=原图臂肤(修 SDXL 洗白/多画袖)。"""
    from engine import arm_refit_overlay
    from PIL import Image, ImageDraw

    human = Image.new("RGB", (1024, 1024), (255, 255, 255))
    d = ImageDraw.Draw(human)
    d.rectangle((500, 200, 540, 1023), fill=(185, 136, 109))  # 左臂(到画布底)
    d.rectangle((780, 200, 820, 1023), fill=(185, 136, 109))  # 右臂
    garment = _flat_with_folded_sleeves(tmp_path)
    cols = (500, 540, 780, 820)
    ov = arm_refit_overlay(garment, human, 1024, None, "long", arm_cols=cols)
    assert ov is not None
    a = np.asarray(ov)
    y_strip_end = int(0.20 * 1024) + int((0.62 - 0.20) * 1024)  # 肩0.20 + 臂长
    # 袖端以上: 布料(alpha 实心); 袖端以下: 原图臂肤
    assert a[260 : y_strip_end - 10, 505:535, 3].mean() > 200, "袖段无布料"
    below = a[y_strip_end + 20 : y_strip_end + 80, 505:535]
    assert below[:, :, 3].mean() > 200, "袖端以下未贴回原图臂肤"
    assert np.abs(below[:, :, :3].astype(int) - np.array([185, 136, 109])).sum(axis=2).mean() < 30, (
        "贴回的不是原图臂肤色"
    )
    # 臂列外不得有内容
    assert a[y_strip_end + 20 : 700, 560:770, 3].max() == 0, "覆盖层越出臂列"
    # 短袖: 袖端在肩下 0.40 臂长, 以下同样贴回原图
    ov_s = arm_refit_overlay(garment, human, 1024, None, "short", arm_cols=cols)
    a_s = np.asarray(ov_s)
    ys_end = int(0.20 * 1024) + int((0.62 - 0.20) * 1024 * 0.40)
    assert a_s[ys_end + 20 : ys_end + 80, 505:535, 3].mean() > 200, "短袖袖端以下未贴回原图"
