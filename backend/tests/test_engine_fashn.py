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
