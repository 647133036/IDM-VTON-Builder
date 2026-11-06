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
        "commercial_compliant",
        "compliance_advice",
    }
    assert st["engine"] in (
        "composite-fallback",
        "idm-vton",
        "sdxl-commercial",
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
    # 左袖列 x 307-358: 红色袖料; 右袖列 x 665-716: 蓝色袖料
    # (右袖旧实现 -=tc1 坐标全负, 裁片越界成全透明 → 空贴)。
    for x0, x1, want in ((315, 350, (200, 60, 60)), (673, 708, (60, 60, 200))):
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
    for x0, x1 in ((307, 358), (665, 716)):
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
    col = arr[:, 315:350, 3] > 60
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
    for x0, x1 in ((315, 350), (673, 708)):
        zone = arr[300:600, x0:x1]
        assert (zone[:, :, 3] > 200).sum() > 500, f"横向摊开袖未旋转贴上 x{x0}"
    left = arr[300:600, 315:350]
    solid = left[left[:, :, 3] > 200][:, :3]
    assert (solid[:, 0] > 120).mean() > 0.9, "左袖旋转后应为红色袖料"


def test_refit_passthrough_unsupported_sleeve_len(tmp_path):
    from engine import refit_sleeves_to_arms

    garment = _flat_with_folded_sleeves(tmp_path)
    for sl in (None, "sleeveless", "three_quarter"):
        warped = _blank_warped()
        out = refit_sleeves_to_arms(warped, garment, 1024, None, sl)
        assert np.asarray(out).tolist() == np.asarray(warped).tolist()
