import sys
import types


def _install_fake_diffusers(monkeypatch, calls):
    class FakeControlNetModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.setdefault("controlnet", []).append((args[0], kwargs))
            return object()

        @classmethod
        def from_single_file(cls, *args, **kwargs):
            return object()

    class FakeDDPMScheduler:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    class FakeVae:
        def to(self, *args, **kwargs):
            return self

    class FakePipe:
        def __init__(self):
            self.vae = FakeVae()

        def enable_attention_slicing(self, *args, **kwargs):
            return None

        def enable_model_cpu_offload(self, *args, **kwargs):
            return None

        def enable_vae_slicing(self, *args, **kwargs):
            return None

        def enable_vae_tiling(self, *args, **kwargs):
            return None

        def load_ip_adapter(self, *args, **kwargs):
            calls["ip_adapter"] = (args, kwargs)

        def set_ip_adapter_scale(self, *args, **kwargs):
            return None

    class FakePipeline:
        tag = "controlnet"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["pipeline"] = (args[0], kwargs, cls.tag)
            return FakePipe()

    class FakeInpaintPipeline(FakePipeline):
        tag = "inpaint"

    diffusers = types.ModuleType("diffusers")
    diffusers.ControlNetModel = FakeControlNetModel
    diffusers.DDPMScheduler = FakeDDPMScheduler
    diffusers.StableDiffusionXLControlNetPipeline = FakePipeline
    diffusers.StableDiffusionXLControlNetInpaintPipeline = FakeInpaintPipeline
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(empty_cache=staticmethod(lambda: None))
    monkeypatch.setitem(sys.modules, "torch", torch)


def _fake_safetensors(path, *, dim=1024):
    path.write_bytes(b"fake-safetensors")


def _make_lora_weights(tmp_path):
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"fake-lora")
    (lora_dir / "adapter_model.safetensors").write_bytes(b"fake-lora")
    return lora_dir


def _make_sdxl_dirs(tmp_path, *, fp16, with_inpaint=False):
    base = tmp_path / "sdxl_base"
    for subfolder, stem in (
        ("unet", "diffusion_pytorch_model"),
        ("vae", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
        ("text_encoder_2", "model"),
    ):
        component = base / subfolder
        component.mkdir(parents=True)
        _fake_safetensors(component / f"{stem}.safetensors")
        if fp16:
            _fake_safetensors(component / f"{stem}.fp16.safetensors")

    canny = tmp_path / "controlnet_canny"
    canny.mkdir()
    _fake_safetensors(canny / "diffusion_pytorch_model.safetensors")
    if fp16:
        _fake_safetensors(canny / "diffusion_pytorch_model.fp16.safetensors")

    (tmp_path / "controlnet_openpose").mkdir()
    if with_inpaint:
        inpaint = tmp_path / "sdxl_inpaint"
        inpaint.mkdir()
        (inpaint / "model_index.json").write_text("{}")
        for subfolder, stem in (("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model")):
            component = inpaint / subfolder
            component.mkdir(parents=True)
            _fake_safetensors(component / f"{stem}.safetensors")
            (inpaint / subfolder / "config.json").write_text("{}")
        (inpaint / "scheduler").mkdir()
    return base


def _empty_sdxl_state():
    return {
        "pipe": None,
        "ip_adapter_loaded": False,
        "ip_adapter_reason": "",
        "depth_loaded": False,
        "lora_loaded": False,
        "lora_install_failed": False,
        "inpaint_mode": False,
    }


def _load_pipe(monkeypatch, tmp_path, fp16):
    import engine_sdxl

    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    _make_sdxl_dirs(tmp_path, fp16=fp16)
    _make_lora_weights(tmp_path)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda pipe, lora_dir: None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "engine_status", lambda force=False: {})
    monkeypatch.setattr(engine_sdxl, "_engine_status_cache", None)
    monkeypatch.setattr(engine_sdxl, "_sdxl_state", _empty_sdxl_state())

    _pipe, _loaded, _reason = engine_sdxl._get_sdxl_pipe("cpu", None)
    return calls


def _controlnet_kwargs(calls, name):
    return next(kwargs for path, kwargs in calls["controlnet"] if path.endswith(name))


def test_sdxl_non_variant_weights_omit_variant(monkeypatch, tmp_path):
    calls = _load_pipe(monkeypatch, tmp_path, fp16=False)

    canny_kwargs = _controlnet_kwargs(calls, "controlnet_canny")
    assert "variant" not in canny_kwargs
    assert canny_kwargs["use_safetensors"] is True
    assert canny_kwargs["local_files_only"] is True

    _pipe_path, pipe_kwargs, _pipe_tag = calls["pipeline"]
    assert "variant" not in pipe_kwargs
    assert pipe_kwargs["use_safetensors"] is True
    assert pipe_kwargs["local_files_only"] is True


def test_sdxl_fp16_variant_weights_pass_variant(monkeypatch, tmp_path):
    calls = _load_pipe(monkeypatch, tmp_path, fp16=True)

    assert _controlnet_kwargs(calls, "controlnet_canny")["variant"] == "fp16"
    _pipe_path, pipe_kwargs, _pipe_tag = calls["pipeline"]
    assert pipe_kwargs["variant"] == "fp16"
    assert pipe_kwargs["use_safetensors"] is True
    assert pipe_kwargs["local_files_only"] is True


def test_pipeline_partial_fp16_variant_falls_back(monkeypatch, tmp_path):
    import engine_sdxl

    base = _make_sdxl_dirs(tmp_path, fp16=False)
    _fake_safetensors(base / "unet" / "diffusion_pytorch_model.fp16.safetensors")
    _fake_safetensors(base / "vae" / "diffusion_pytorch_model.fp16.safetensors")
    _fake_safetensors(base / "text_encoder" / "model.fp16.safetensors")
    assert engine_sdxl._pipeline_fp16_variant_kwargs(base) == {}


def test_single_component_variant_detection_accepts_both_filenames(tmp_path):
    import engine_sdxl

    controlnet_dir = tmp_path / "controlnet"
    controlnet_dir.mkdir()
    _fake_safetensors(controlnet_dir / "diffusion_pytorch_model.fp16.safetensors")
    assert engine_sdxl._fp16_variant_kwargs(controlnet_dir) == {"variant": "fp16"}

    text_encoder_dir = tmp_path / "text_encoder"
    text_encoder_dir.mkdir()
    _fake_safetensors(text_encoder_dir / "model.fp16.safetensors")
    assert engine_sdxl._fp16_variant_kwargs(text_encoder_dir) == {"variant": "fp16"}

    non_variant_dir = tmp_path / "non_variant"
    non_variant_dir.mkdir()
    _fake_safetensors(non_variant_dir / "model.safetensors")
    assert engine_sdxl._fp16_variant_kwargs(non_variant_dir) == {}


def _write_encoder_config(ip_adapter_dir, dim, *, key="projection_dim"):
    encoder_dir = ip_adapter_dir / "image_encoder"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    (encoder_dir / "config.json").write_text(f'{{"{key}": {dim}}}', encoding="utf-8")


def _write_ip_adapter_weight(path, *, dim, variant="base"):
    import torch
    from safetensors.torch import save_file

    if variant == "base":
        tensors = {"image_proj.proj.weight": torch.zeros((16, dim))}
    else:
        tensors = {"image_proj.layers.0.0.norm1.weight": torch.zeros(dim)}
    save_file(tensors, str(path))


def _make_ip_adapter_dir(tmp_path, *, encoder_dim, key="projection_dim"):
    ip_adapter_dir = tmp_path / "ip_adapter"
    _write_encoder_config(ip_adapter_dir, encoder_dim, key=key)
    return ip_adapter_dir


def test_pick_ip_adapter_weight_prefers_plus_for_1280_encoder(tmp_path):
    import engine_sdxl

    ip_adapter_dir = _make_ip_adapter_dir(tmp_path, encoder_dim=1280)
    base = ip_adapter_dir / "ip-adapter_sdxl_vit-h.safetensors"
    plus = ip_adapter_dir / "ip-adapter-plus_sdxl_vit-h.safetensors"
    _write_ip_adapter_weight(base, dim=1024, variant="base")
    _write_ip_adapter_weight(plus, dim=1280, variant="plus")

    assert engine_sdxl._pick_ip_adapter_weight(ip_adapter_dir) == plus


def test_pick_ip_adapter_weight_selects_base_for_1024_encoder(tmp_path):
    import engine_sdxl

    ip_adapter_dir = _make_ip_adapter_dir(tmp_path, encoder_dim=1024)
    base = ip_adapter_dir / "ip-adapter_sdxl_vit-h.safetensors"
    plus = ip_adapter_dir / "ip-adapter-plus_sdxl_vit-h.safetensors"
    _write_ip_adapter_weight(base, dim=1024, variant="base")
    _write_ip_adapter_weight(plus, dim=1280, variant="plus")

    assert engine_sdxl._pick_ip_adapter_weight(ip_adapter_dir) == base


def test_pick_ip_adapter_weight_returns_none_and_reason_on_mismatch(tmp_path):
    import engine_sdxl

    ip_adapter_dir = _make_ip_adapter_dir(tmp_path, encoder_dim=1280)
    base = ip_adapter_dir / "ip-adapter_sdxl_vit-h.safetensors"
    _write_ip_adapter_weight(base, dim=1024, variant="base")

    assert engine_sdxl._pick_ip_adapter_weight(ip_adapter_dir) is None
    weight, reason = engine_sdxl._select_ip_adapter_weight(ip_adapter_dir)
    assert weight is None
    assert reason == ("IP-Adapter weight image_proj expects 1024-dim embeddings but local image_encoder outputs 1280")


def test_pick_ip_adapter_weight_uses_hidden_size_fallback(tmp_path):
    import engine_sdxl

    ip_adapter_dir = _make_ip_adapter_dir(tmp_path, encoder_dim=1280, key="hidden_size")
    plus_face = ip_adapter_dir / "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    _write_ip_adapter_weight(plus_face, dim=1280, variant="plus")

    assert engine_sdxl._pick_ip_adapter_weight(ip_adapter_dir) == plus_face


def test_apply_lora_scale_uses_set_adapters(monkeypatch):
    import engine_sdxl

    monkeypatch.delenv("VTON_LORA", raising=False)
    calls = {}

    class _Pipe:
        def set_adapters(self, names, adapter_weights=None):
            calls["names"] = names
            calls["weights"] = adapter_weights

    engine_sdxl._apply_lora_scale(_Pipe())
    assert calls["names"] == ["default"]
    assert calls["weights"] == [0.7]


def test_apply_lora_scale_honors_env(monkeypatch):
    import engine_sdxl

    monkeypatch.setenv("VTON_LORA", "0.4")
    calls = {}

    class _Pipe:
        def set_adapters(self, names, adapter_weights=None):
            calls["weights"] = adapter_weights

    engine_sdxl._apply_lora_scale(_Pipe())
    assert calls["weights"] == [0.4]


def test_pipeline_loads_weight_matching_local_encoder(monkeypatch, tmp_path):
    import engine_sdxl

    calls = {}
    _make_sdxl_dirs(tmp_path, fp16=False)
    _make_lora_weights(tmp_path)
    ip_adapter_dir = _make_ip_adapter_dir(tmp_path, encoder_dim=1280)
    _write_ip_adapter_weight(ip_adapter_dir / "ip-adapter_sdxl_vit-h.safetensors", dim=1024, variant="base")
    _write_ip_adapter_weight(ip_adapter_dir / "ip-adapter-plus_sdxl_vit-h.safetensors", dim=1280, variant="plus")
    _install_fake_diffusers(monkeypatch, calls)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda pipe, lora_dir: None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "engine_status", lambda force=False: {})
    monkeypatch.setattr(engine_sdxl, "_engine_status_cache", None)
    monkeypatch.setattr(engine_sdxl, "_sdxl_state", _empty_sdxl_state())

    _pipe, loaded, reason = engine_sdxl._get_sdxl_pipe("cpu", None)

    assert loaded is True
    assert reason == ""
    args, kwargs = calls["ip_adapter"]
    assert args[0] == str(ip_adapter_dir)
    assert kwargs["subfolder"] == ""
    assert kwargs["weight_name"] == "ip-adapter-plus_sdxl_vit-h.safetensors"


def test_align_garment_to_pose_returns_square_rgb(monkeypatch):
    import engine_sdxl
    from PIL import Image

    garment = Image.new("RGB", (600, 800), (180, 60, 60))
    out, alpha = engine_sdxl.align_garment_to_pose(garment, "women", "east_asia", "upper", 1024)
    assert out.size == (1024, 1024)
    assert out.mode == "RGB"
    assert alpha is None or alpha.mode in ("RGBA", "RGB")


def test_align_garment_to_pose_disabled_returns_original(monkeypatch):
    import engine_sdxl
    from PIL import Image

    monkeypatch.setenv("VTON_WARP", "0")
    garment = Image.new("RGB", (400, 500), (90, 90, 120))
    out, alpha = engine_sdxl.align_garment_to_pose(garment, "men", "western_europe", "upper", 1024)
    assert out is garment
    assert alpha is None


def test_align_garment_to_pose_changes_pixels_vs_flat(monkeypatch):
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGBA", (500, 700), (0, 0, 0, 0))
    d = ImageDraw.Draw(garment)
    d.rectangle([120, 100, 380, 620], fill=(60, 140, 90, 255))
    flat = engine_sdxl.fit_square(garment.convert("RGB"), 1024)
    aligned, _alpha = engine_sdxl.align_garment_to_pose(garment.convert("RGB"), "women", "east_asia", "upper", 1024)
    a = np.asarray(flat)
    b = np.asarray(aligned)
    assert a.shape == b.shape
    assert not np.array_equal(a, b)


def test_align_garment_to_pose_dress_differs_from_upper():
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGBA", (500, 700), (0, 0, 0, 0))
    d = ImageDraw.Draw(garment)
    d.rectangle([120, 100, 380, 620], fill=(60, 140, 90, 255))
    g = garment.convert("RGB")
    upper = np.asarray(engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "upper", 1024)[0])
    dress = np.asarray(engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "dress", 1024)[0])
    assert not np.array_equal(upper, dress)


def test_align_garment_to_pose_lower_narrower_than_upper():
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGBA", (500, 700), (0, 0, 0, 0))
    d = ImageDraw.Draw(garment)
    d.rectangle([120, 100, 380, 620], fill=(60, 140, 90, 255))
    g = garment.convert("RGB")
    upper = np.asarray(engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "upper", 1024)[0])
    lower = np.asarray(engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "lower", 1024)[0])

    upper_cols = np.where(upper.mean(axis=(0, 2)) < 240)[0]
    lower_cols = np.where(lower.mean(axis=(0, 2)) < 240)[0]
    assert len(upper_cols) > 0
    assert len(lower_cols) > 0
    upper_width = upper_cols[-1] - upper_cols[0]
    lower_width = lower_cols[-1] - lower_cols[0]
    assert lower_width < upper_width


def test_align_garment_to_pose_mid_hem_extends_below_waist():
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGBA", (500, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(garment)
    d.rectangle([140, 80, 360, 820], fill=(180, 80, 80, 255))
    g = garment.convert("RGB")
    waist_a = engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "upper", 1024)[1]
    mid_a = engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "upper", 1024, hem="mid")[1]
    assert waist_a is not None and mid_a is not None

    def last_row(im):
        a = np.asarray(im.getchannel("A"))
        rows = np.where(a.max(axis=1) > 40)[0]
        return int(rows[-1]) if len(rows) else 0

    assert last_row(mid_a) > last_row(waist_a) + 40


def test_align_garment_to_pose_tall_flat_ref_and_alpha_share_rows():
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGB", (300, 1000), (255, 255, 255))
    d = ImageDraw.Draw(garment)
    d.rectangle([60, 40, 240, 960], fill=(90, 60, 140))
    pose = {"hip": (0.5, 0.643, 1.0), "shoulder": (0.5, 0.28, 1.0)}
    out, alpha = engine_sdxl.align_garment_to_pose(garment, "women", "east_asia", "upper", 1024, pose=pose, hem="waist")
    assert alpha is not None
    assert alpha.size == (1024, 1024), f"alpha 画布 {alpha.size}, 期望方画布"
    a = np.asarray(alpha)[..., 3]
    rgb = np.asarray(out)
    rgb_bottom = int(np.where((rgb < 240).any(axis=(1, 2)))[0].max())
    a_bottom = int(np.where((a > 40).any(axis=1))[0].max())
    assert abs(rgb_bottom - a_bottom) <= 2, f"RGB 底行 {rgb_bottom} vs alpha 底行 {a_bottom} 错位"


def test_align_garment_to_pose_upper_hem_floor_covers_hip():
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGB", (300, 1000), (255, 255, 255))
    d = ImageDraw.Draw(garment)
    d.rectangle([60, 40, 240, 960], fill=(90, 60, 140))
    pose = {"hip": (0.5, 0.643, 1.0), "shoulder": (0.5, 0.28, 1.0)}
    _out, alpha = engine_sdxl.align_garment_to_pose(
        garment, "women", "east_asia", "upper", 1024, pose=pose, hem="waist"
    )
    a = np.asarray(alpha)[..., 3]
    center = a[:, 340:684]
    rows = np.where((center > 40).sum(axis=1) > 8)[0]
    assert rows.max() >= 674, f"中央带下摆 {int(rows.max())} 未盖过髋线 674"


# --- auto 品类推断 + 袖型反向约束 ------------------------------------------------


def test_infer_auto_garment_type_body_ratio_gates_tunic_and_dress():
    import engine_sdxl

    infer = engine_sdxl.infer_auto_garment_type
    assert infer({"aspect": 1.36, "hem": "mid", "body_ratio": 0.84}) == "upper"
    assert infer({"aspect": 1.2, "hem": "mid", "body_ratio": 1.0}) == "tunic"
    assert infer({"aspect": 1.1, "hem": "long", "body_ratio": 0.95}) == "tunic"
    assert infer({"aspect": 1.8, "hem": "mid", "body_ratio": 0.6}) == "upper"
    assert infer({"aspect": 1.8, "hem": "long", "body_ratio": 1.0}) == "dress"
    assert infer({"aspect": 0.5, "hem": "waist", "body_ratio": 1.0}) == "lower"
    assert infer(None) == "upper"


def test_infer_auto_garment_type_missing_body_ratio_keeps_legacy_behavior():
    import engine_sdxl

    infer = engine_sdxl.infer_auto_garment_type
    assert infer({"aspect": 1.8}) == "dress"
    assert infer({"hem": "mid"}) == "tunic"
    assert infer({}) == "upper"


def test_sleeve_negative_prompts_pin_opposite_sleeve_terms():
    import engine_sdxl

    table = engine_sdxl.SLEEVE_NEGATIVE_PROMPTS
    assert set(table) == {"long", "short", "sleeveless"}
    assert "bare forearms" in table["long"] and "short sleeves" in table["long"]
    assert "long sleeves" in table["short"] and "sleeveless" in table["short"]
    assert "long sleeves" in table["sleeveless"] and "short sleeves" in table["sleeveless"]


def test_dedupe_negative_drops_repeat_phrases_and_keeps_order():
    import engine_sdxl

    out = engine_sdxl._dedupe_negative(
        [
            "bare forearms, short sleeves",
            "bad anatomy, extra limbs",
            "plastic, bad anatomy, extra limbs",
            "",
            "floor shadow",
        ]
    )
    assert out == "bare forearms, short sleeves, bad anatomy, extra limbs, plastic, floor shadow"


def test_negative_assembly_fits_clip_budget_with_sleeve_first():
    import engine_sdxl as eng

    for aud in eng.AUDIENCE_NEGATIVE:
        for sl, sv in eng.SLEEVE_NEGATIVE_PROMPTS.items():
            text = eng._dedupe_negative([sv, eng.AUDIENCE_NEGATIVE[aud], eng.FIT_NEGATIVE, eng.COMMON_NEGATIVE])
            phrases = [p.strip() for p in text.split(",")]
            assert text.startswith(sv.split(",")[0].strip() + ","), f"{aud}/{sl}: sleeve terms not first"
            assert len(phrases) == len({p.lower() for p in phrases}), f"{aud}/{sl}: dup phrases"
            assert all(phrases), f"{aud}/{sl}: empty phrase"


# ---- intersect_mask_with_garment_alpha ----


def _intersect_fixture():
    import numpy as np
    from PIL import Image, ImageDraw

    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[50:462, 150:362] = 255
    alpha = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    d = ImageDraw.Draw(alpha)
    d.rectangle((200, 100, 311, 400), fill=(255, 128, 128, 255))
    return mask, alpha


def test_intersect_mask_trims_outside_garment_shape():
    import engine_sdxl as eng
    import numpy as np

    mask, alpha = _intersect_fixture()
    out, kept = eng.intersect_mask_with_garment_alpha(mask, alpha)
    assert kept is not None and kept > 0.3
    arr = np.asarray(out)
    assert arr[50, 50] == 0
    assert arr[250, 250] == 255
    assert arr[250, 190] > 0
    assert arr[250, 120] == 0


def test_intersect_mask_skips_on_low_coverage():
    import engine_sdxl as eng
    import numpy as np
    from PIL import Image, ImageDraw

    mask = np.full((512, 512), 255, dtype=np.uint8)
    alpha = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    ImageDraw.Draw(alpha).rectangle((240, 240, 260, 260), fill=(255, 0, 0, 255))
    out, kept = eng.intersect_mask_with_garment_alpha(mask, alpha)
    assert kept is None
    assert np.asarray(out).mean() == 255


def test_intersect_mask_env_switch_and_none_passthrough(monkeypatch):
    import engine_sdxl as eng

    mask, alpha = _intersect_fixture()
    monkeypatch.setenv("VTON_MASK_ALPHA_INTERSECT", "0")
    out, kept = eng.intersect_mask_with_garment_alpha(mask, alpha)
    assert kept is None and out is mask
    monkeypatch.setenv("VTON_MASK_ALPHA_INTERSECT", "1")
    out, kept = eng.intersect_mask_with_garment_alpha(mask, None)
    assert kept is None
    assert out is mask or out is None
    out, kept = eng.intersect_mask_with_garment_alpha(None, alpha)
    assert kept is None
    assert out is None


def test_intersect_mask_unions_strict_fabric_beyond_subject():
    import engine_sdxl as eng
    import numpy as np
    from PIL import Image

    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:400, 150:310] = 255
    a = np.zeros((512, 512), dtype=np.uint8)
    a[100:400, 150:390] = 255
    alpha = Image.merge(
        "RGBA",
        (*Image.new("RGB", (512, 512), (90, 90, 90)).split(), Image.fromarray(a)),
    )
    out, kept = eng.intersect_mask_with_garment_alpha(mask, alpha)
    assert kept is not None and kept > 0.5
    arr = np.asarray(out)
    assert arr[250, 350] == 255
    assert arr[50, 200] == 0
    assert arr[250, 250] == 255


def test_intersect_mask_preserves_input_type():
    import engine_sdxl as eng
    from PIL import Image

    mask, alpha = _intersect_fixture()
    pil_mask = Image.fromarray(mask)
    out_pil, kept = eng.intersect_mask_with_garment_alpha(pil_mask, alpha)
    assert kept is not None and isinstance(out_pil, Image.Image)
    out_arr, kept2 = eng.intersect_mask_with_garment_alpha(mask, alpha)
    assert kept2 is not None and not isinstance(out_arr, Image.Image)


def test_inpaint_weights_ready(tmp_path, monkeypatch):
    import engine_sdxl

    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    assert engine_sdxl._inpaint_weights_ready() is False
    (tmp_path / "sdxl_inpaint").mkdir()
    (tmp_path / "sdxl_inpaint" / "model_index.json").write_text("{}")
    assert engine_sdxl._inpaint_weights_ready() is True


def test_get_sdxl_pipe_selects_inpaint_when_weights_ready(tmp_path, monkeypatch):
    import engine_sdxl

    _make_sdxl_dirs(tmp_path, fp16=False, with_inpaint=True)
    _make_lora_weights(tmp_path)
    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda pipe, lora_dir: None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "engine_status", lambda force=False: {})
    monkeypatch.setattr(engine_sdxl, "_engine_status_cache", None)
    monkeypatch.setattr(engine_sdxl, "_sdxl_state", _empty_sdxl_state())

    _pipe, _loaded, _reason = engine_sdxl._get_sdxl_pipe("cpu", None)
    assert engine_sdxl._sdxl_state["inpaint_mode"] is True
    pipe_path, _kwargs, pipe_tag = calls["pipeline"]
    assert pipe_path.endswith("sdxl_inpaint")
    assert pipe_tag == "inpaint"


def test_get_sdxl_pipe_falls_back_to_base_without_inpaint(tmp_path, monkeypatch):
    import engine_sdxl

    _make_sdxl_dirs(tmp_path, fp16=False)
    _make_lora_weights(tmp_path)
    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda pipe, lora_dir: None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "engine_status", lambda force=False: {})
    monkeypatch.setattr(engine_sdxl, "_engine_status_cache", None)
    monkeypatch.setattr(engine_sdxl, "_sdxl_state", _empty_sdxl_state())

    _pipe, _loaded, _reason = engine_sdxl._get_sdxl_pipe("cpu", None)
    assert engine_sdxl._sdxl_state["inpaint_mode"] is False
    pipe_path, _kwargs, pipe_tag = calls["pipeline"]
    assert pipe_path.endswith("sdxl_base")
    assert pipe_tag == "controlnet"


def test_build_sdxl_infer_kwargs_inpaint_mode():
    import engine_sdxl
    from PIL import Image

    init_img = Image.new("RGB", (16, 16))
    mask_img = Image.new("L", (16, 16), 0)
    control_img = Image.new("RGB", (16, 16))
    kwargs = engine_sdxl._build_sdxl_infer_kwargs(
        prompt="p",
        negative_prompt="n",
        control_images=control_img,
        control_scales=[1.0],
        width=16,
        height=16,
        num_inference_steps=20,
        guidance_scale=7.0,
        generator=object(),
        inpaint_mode=True,
        init_image=init_img,
        mask_image=mask_img,
        strength=0.99,
    )
    assert kwargs["image"] is init_img
    assert kwargs["mask_image"] is mask_img
    assert kwargs["control_image"] is control_img
    assert kwargs["strength"] == 0.99


def test_build_sdxl_infer_kwargs_full_frame_mode():
    import engine_sdxl
    from PIL import Image

    control_img = Image.new("RGB", (16, 16))
    kwargs = engine_sdxl._build_sdxl_infer_kwargs(
        prompt="p",
        negative_prompt="n",
        control_images=control_img,
        control_scales=[1.0],
        width=16,
        height=16,
        num_inference_steps=20,
        guidance_scale=7.0,
        generator=object(),
        inpaint_mode=False,
    )
    assert kwargs["image"] is control_img
    assert "mask_image" not in kwargs
    assert "strength" not in kwargs


def test_engine_status_reports_inpaint_fields(tmp_path, monkeypatch):
    import engine_sdxl

    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    st = engine_sdxl.engine_status()
    assert "sdxl_inpaint" in st
    assert st["sdxl_inpaint"] in (True, False)
