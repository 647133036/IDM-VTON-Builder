from pathlib import Path

import pytest
from PIL import Image

from engine import (
    ETHNICITY_FIT,
    TORSOS,
    compose_tryon,
    engine_status,
    guess_audience,
    guess_garment_type,
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
    from PIL import Image as PILImage, ImageDraw

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
        for ex, nx in zip(ea.getdata(), na.getdata())
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
        "no-compliant-engine",
    )