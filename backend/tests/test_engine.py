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
