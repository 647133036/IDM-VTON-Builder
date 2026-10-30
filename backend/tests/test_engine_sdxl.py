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
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["pipeline"] = (args[0], kwargs)
            return FakePipe()

    diffusers = types.ModuleType("diffusers")
    diffusers.ControlNetModel = FakeControlNetModel
    diffusers.DDPMScheduler = FakeDDPMScheduler
    diffusers.StableDiffusionXLControlNetPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(empty_cache=staticmethod(lambda: None))
    monkeypatch.setitem(sys.modules, "torch", torch)


def _make_sdxl_dirs(tmp_path, *, fp16):
    base = tmp_path / "sdxl_base"
    for subfolder, stem in (
        ("unet", "diffusion_pytorch_model"),
        ("vae", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
        ("text_encoder_2", "model"),
    ):
        component = base / subfolder
        component.mkdir(parents=True)
        (component / f"{stem}.safetensors").write_bytes(b"x")
        if fp16:
            (component / f"{stem}.fp16.safetensors").write_bytes(b"x")

    canny = tmp_path / "controlnet_canny"
    canny.mkdir()
    (canny / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    if fp16:
        (canny / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")

    (tmp_path / "controlnet_openpose").mkdir()
    return base


def _load_pipe(monkeypatch, tmp_path, fp16):
    import engine_sdxl

    calls = {}
    _install_fake_diffusers(monkeypatch, calls)
    _make_sdxl_dirs(tmp_path, fp16=fp16)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda pipe, lora_dir: None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(
        engine_sdxl,
        "_sdxl_state",
        {"pipe": None, "ip_adapter_loaded": False, "ip_adapter_reason": "", "depth_loaded": False},
    )
    engine_sdxl._get_sdxl_pipe("cpu", None)
    return calls


def _controlnet_kwargs(calls, name):
    return next(kwargs for path, kwargs in calls["controlnet"] if path.endswith(name))


def test_sdxl_non_variant_weights_omit_variant(monkeypatch, tmp_path):
    calls = _load_pipe(monkeypatch, tmp_path, fp16=False)

    canny_kwargs = _controlnet_kwargs(calls, "controlnet_canny")
    assert "variant" not in canny_kwargs
    assert canny_kwargs["use_safetensors"] is True
    assert canny_kwargs["local_files_only"] is True

    _, pipe_kwargs = calls["pipeline"]
    assert "variant" not in pipe_kwargs
    assert pipe_kwargs["use_safetensors"] is True
    assert pipe_kwargs["local_files_only"] is True


def test_sdxl_fp16_variant_weights_pass_variant(monkeypatch, tmp_path):
    calls = _load_pipe(monkeypatch, tmp_path, fp16=True)

    assert _controlnet_kwargs(calls, "controlnet_canny")["variant"] == "fp16"
    _, pipe_kwargs = calls["pipeline"]
    assert pipe_kwargs["variant"] == "fp16"
    assert pipe_kwargs["use_safetensors"] is True
    assert pipe_kwargs["local_files_only"] is True


def test_pipeline_partial_fp16_variant_falls_back(monkeypatch, tmp_path):
    import engine_sdxl

    base = _make_sdxl_dirs(tmp_path, fp16=False)
    (base / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    (base / "vae" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    (base / "text_encoder" / "model.fp16.safetensors").write_bytes(b"x")
    # text_encoder_2 只有非 variant 权重 -> 整个 pipeline 不能传 variant。
    assert engine_sdxl._pipeline_fp16_variant_kwargs(base) == {}


def test_single_component_variant_detection_accepts_both_filenames(tmp_path):
    import engine_sdxl

    controlnet_dir = tmp_path / "controlnet"
    controlnet_dir.mkdir()
    (controlnet_dir / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    assert engine_sdxl._fp16_variant_kwargs(controlnet_dir) == {"variant": "fp16"}

    text_encoder_dir = tmp_path / "text_encoder"
    text_encoder_dir.mkdir()
    (text_encoder_dir / "model.fp16.safetensors").write_bytes(b"x")
    assert engine_sdxl._fp16_variant_kwargs(text_encoder_dir) == {"variant": "fp16"}

    non_variant_dir = tmp_path / "non_variant"
    non_variant_dir.mkdir()
    (non_variant_dir / "model.safetensors").write_bytes(b"x")
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


def test_pipeline_loads_weight_matching_local_encoder(monkeypatch, tmp_path):
    import engine_sdxl

    calls = {}
    _make_sdxl_dirs(tmp_path, fp16=False)
    ip_adapter_dir = _make_ip_adapter_dir(tmp_path, encoder_dim=1280)
    # 必须在 fake torch 注入前落盘权重,否则 save_file 拿到的是测试桩。
    _write_ip_adapter_weight(ip_adapter_dir / "ip-adapter_sdxl_vit-h.safetensors", dim=1024, variant="base")
    _write_ip_adapter_weight(ip_adapter_dir / "ip-adapter-plus_sdxl_vit-h.safetensors", dim=1280, variant="plus")
    _install_fake_diffusers(monkeypatch, calls)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda pipe, lora_dir: None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(
        engine_sdxl,
        "_sdxl_state",
        {"pipe": None, "ip_adapter_loaded": False, "ip_adapter_reason": "", "depth_loaded": False},
    )

    _, loaded, reason = engine_sdxl._get_sdxl_pipe("cpu", None)

    assert loaded is True
    assert reason == ""
    args, kwargs = calls["ip_adapter"]
    assert args[0] == str(ip_adapter_dir)
    assert kwargs["subfolder"] == ""
    assert kwargs["weight_name"] == "ip-adapter-plus_sdxl_vit-h.safetensors"
