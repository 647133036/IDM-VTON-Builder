import sys
import types

from PIL import Image


def test_fashn_category_mapping():
    from engine_fashn import fashn_category

    assert fashn_category("upper") == "tops"
    assert fashn_category("tunic") == "tops"
    assert fashn_category("lower") == "bottoms"
    assert fashn_category("dress") == "one-pieces"
    assert fashn_category("onesie") == "one-pieces"
    assert fashn_category("auto") == "tops"


def test_fashn_available_missing_weights(monkeypatch, tmp_path):
    import engine_fashn

    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path / "fashn")
    monkeypatch.setattr(engine_fashn, "_REPO_ROOT", tmp_path)
    avail, reason = engine_fashn.fashn_available()
    assert avail is False
    assert "Missing FASHN weights" in reason


def test_parser_stub_skips_real_human_parser(monkeypatch):
    import engine_fashn

    sys.modules.pop("fashn_human_parser", None)
    engine_fashn._ensure_parser_stub()
    stub = sys.modules["fashn_human_parser"]
    assert getattr(stub, "_atelier_commercial_stub", False) is True
    pred = stub.FashnHumanParser().predict(Image.new("RGB", (8, 12), (1, 2, 3)))
    assert pred.shape == (12, 8)
    assert int(pred.sum()) == 0


def test_parser_stub_satisfies_fashn_vton_v15_import_surface():
    """fashn_vton v1.5 模块导入期需要的全部名字(GPU 冒烟曾因缺 3 个名字炸在 import)。"""
    import engine_fashn

    sys.modules.pop("fashn_human_parser", None)
    engine_fashn._ensure_parser_stub()
    stub = sys.modules["fashn_human_parser"]
    # pipeline.py 导入: CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
    for name in ("CATEGORY_TO_BODY_COVERAGE", "FashnHumanParser"):
        assert hasattr(stub, name), name
    # preprocessing/agnostic.py 导入: BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS
    bctl = stub.BODY_COVERAGE_TO_LABELS
    assert set(bctl) == {"upper", "lower", "full"}
    assert set(bctl) == set(stub.CATEGORY_TO_BODY_COVERAGE.values())
    identity = tuple(stub.IDENTITY_LABELS)
    assert identity and all(isinstance(x, str) for x in identity)
    l2i = stub.LABELS_TO_IDS
    assert l2i and all(isinstance(v, int) for v in l2i.values())
    # 每个覆盖档的 label 必须都在 LABELS_TO_IDS 里(结构自洽)
    for labels in bctl.values():
        for label in labels:
            assert label in l2i, label
    for label in identity:
        assert label in l2i, label


def test_get_fashn_pipe_uses_stub_and_weights_dir(monkeypatch, tmp_path):
    import engine_fashn

    calls = {}

    class FakePipeline:
        def __init__(self, weights_dir):
            calls["weights_dir"] = weights_dir

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            img = Image.new("RGB", (16, 16), (4, 5, 6))
            return types.SimpleNamespace(images=[img])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})

    pipe = engine_fashn._get_fashn_pipe()
    assert calls["weights_dir"] == str(tmp_path)
    assert pipe is engine_fashn._get_fashn_pipe()
    assert getattr(sys.modules["fashn_human_parser"], "_atelier_commercial_stub", False)


def test_compose_tryon_fashn_calls_segmentation_free(monkeypatch, tmp_path):
    import engine_fashn
    from PIL import Image as PILImage

    calls = {}

    class FakePipeline:
        def __init__(self, weights_dir):
            self.weights_dir = weights_dir

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            return types.SimpleNamespace(images=[PILImage.new("RGB", (32, 32), (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_PATTERN_LOCK", "0")
    monkeypatch.setenv("VTON_WARP", "0")

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    PILImage.new("RGB", (40, 60), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 64, "white", "western_europe")
    kwargs = calls["kwargs"]
    assert kwargs["category"] == "one-pieces"
    assert kwargs["garment_photo_type"] == "flat-lay"
    assert kwargs["segmentation_free"] is True
    assert out.info["engine"] == "fashn-vton"
    assert out.info["fashn_category"] == "one-pieces"
    assert max(out.size) == 64


def test_compose_tryon_fashn_fidelity_survives_bg_and_resize(monkeypatch, tmp_path):
    """GPU 冒烟实测: 白底转换/缩放替换 Image 后 .info 丢失, print_fidelity 变 null。

    保真分在内容阶段计算, 但必须写入尾段元数据块(背景/缩放之后)才能落库。
    """
    import engine_fashn
    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod
    from PIL import Image as PILImage

    class FakePipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[PILImage.new("RGB", (32, 32), (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")
    monkeypatch.setenv("VTON_PROTECT_FEATHER", "4")

    ref = PILImage.new("RGBA", (32, 32), (200, 10, 10, 255))
    alpha = PILImage.new("L", (32, 32), 255)
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(engine_sdxl, "align_garment_to_pose", lambda *a, **k: (ref, alpha))
    monkeypatch.setattr(engine_sdxl, "intersect_mask_with_garment_alpha", lambda m, a: (m, 1))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets,
        "foreground_mask",
        lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        platform_presets,
        "clothing_mask",
        lambda result, **k: np.full((result.size[1], result.size[0]), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(platform_presets, "limit_mask_to_subject", lambda m, s: m)
    monkeypatch.setattr(platform_presets, "reapply_garment_detail", lambda result, ref, mask, **k: result)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)
    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", lambda r, b, m, feather_px=4: r)

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    PILImage.new("RGB", (40, 60), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 48, "white", "western_europe")
    assert out.info["engine"] == "fashn-vton"
    assert out.info["detail_applied"] is True
    # long_side=48 强制缩放(32->48), 白底路径换新 Image; 保真分必须仍在
    assert max(out.size) == 48
    assert "print_fidelity" in out.info, "fidelity lost across background/resize transforms"
    assert out.info["print_fidelity"] == 1.0
    assert out.info["delivery_tier"] == "auto"


def _fake_fashn(monkeypatch, tmp_path):
    """共享夹具: FakePipeline 输出 32x32, 全部平台钩子可控。"""
    import engine_fashn
    from PIL import Image as PILImage

    class FakePipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[PILImage.new("RGB", (32, 32), (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    return engine_fashn, PILImage


def test_compose_fashn_drawn_align_uses_refit_and_decoupled_masks(monkeypatch, tmp_path):
    """VTON_FASHN_PRINT_ALIGN=drawn(默认): 贴印几何走 TPS 重投, 三 mask 解耦。

    贴印区(paste_mask)=印花核心 ∩ keep; 保护(protect)拿 keep_mask(纯裁剪);
    模板路径(align_garment_to_pose)必须不被触碰。退化 drawn_mask(全画布)
    自动回退模板路径由既有 fidelity 测试覆盖。
    """
    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod

    engine_fashn, PILImage = _fake_fashn(monkeypatch, tmp_path)
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")
    monkeypatch.setenv("VTON_PROTECT_FEATHER", "4")

    def _no_template(*a, **k):
        raise AssertionError("drawn path must not call align_garment_to_pose")

    monkeypatch.setattr(engine_sdxl, "align_garment_to_pose", _no_template)
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets, "foreground_mask", lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8)
    )

    drawn = np.zeros((32, 32), dtype=np.uint8)
    drawn[10:26, 8:24] = 255
    monkeypatch.setattr(platform_presets, "clothing_mask", lambda result, **k: drawn.copy())

    reapply_masks = []

    def _reapply(result, ref, mask, **k):
        reapply_masks.append(np.asarray(mask).copy())
        return ref.convert("RGB")

    monkeypatch.setattr(platform_presets, "reapply_garment_detail", _reapply)

    protect_masks = []

    def _protect(result, base, mask, feather_px=4):
        protect_masks.append(np.asarray(mask).copy())
        return result

    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", _protect)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)

    garment = tmp_path / "g.png"
    flat = PILImage.new("RGB", (24, 32), (255, 255, 255))
    from PIL import ImageDraw

    ImageDraw.Draw(flat).rectangle((6, 8, 18, 24), fill=(200, 30, 30))
    flat.save(garment)
    person = tmp_path / "p.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 48, "white", "western_europe")
    assert out.info["engine"] == "fashn-vton"
    assert out.info["detail_applied"] is True
    # 三 mask 解耦: 贴印区来自印花核心(实心红块内容平坦 → 边界环核心),
    # 且不越出 keep; protect 拿到的整件衣料 keep(裁到 alpha)
    assert len(reapply_masks) == 1 and len(protect_masks) == 1
    paste = reapply_masks[0]
    keep = protect_masks[0]
    assert paste.sum() > 0
    assert np.asarray(paste & keep).sum() == paste.sum()
    assert keep.sum() < 255 * 32 * 32, "keep must be clipped, not full canvas"
    # 贴印区与 keep 在 drawn bbox 内
    ys, xs = np.where(keep > 0)
    assert xs.min() >= 8 and xs.max() < 24 and ys.min() >= 10 and ys.max() < 26
    assert out.info["print_fidelity"] == 1.0
    assert out.info["delivery_tier"] == "auto"
    assert out.info["lock_identity"] is True


def test_compose_fashn_template_align_env_restores_legacy(monkeypatch, tmp_path):
    """VTON_FASHN_PRINT_ALIGN=template: 显式回退模板路径, 语义与旧版一致。"""
    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod

    engine_fashn, PILImage = _fake_fashn(monkeypatch, tmp_path)
    monkeypatch.setenv("VTON_FASHN_PRINT_ALIGN", "template")
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")

    ref = PILImage.new("RGBA", (32, 32), (200, 10, 10, 255))
    alpha = PILImage.new("L", (32, 32), 255)
    calls = []

    def _align(*a, **k):
        calls.append(1)
        return ref, alpha

    monkeypatch.setattr(engine_sdxl, "align_garment_to_pose", _align)
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets, "foreground_mask", lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8)
    )
    drawn = np.zeros((32, 32), dtype=np.uint8)
    drawn[10:26, 8:24] = 255
    monkeypatch.setattr(platform_presets, "clothing_mask", lambda result, **k: drawn.copy())
    monkeypatch.setattr(engine_sdxl, "intersect_mask_with_garment_alpha", lambda m, a: (m, 1))
    monkeypatch.setattr(platform_presets, "reapply_garment_detail", lambda result, r, mask, **k: r.convert("RGB"))
    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", lambda r, b, m, feather_px=4: r)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    PILImage.new("RGB", (24, 32), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 48, "white", "western_europe")
    assert calls, "template env must route through align_garment_to_pose"
    assert out.info["print_fidelity"] == 1.0
    assert out.info["delivery_tier"] == "auto"
