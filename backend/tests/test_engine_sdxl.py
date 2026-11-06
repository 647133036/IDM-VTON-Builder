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


def test_align_garment_to_pose_returns_square_rgb(monkeypatch):
    import engine_sdxl
    from PIL import Image

    # VTON_WARP 默认开启;输入一张纯色平铺衣,输出应为 1024x1024 RGB。
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

    # 带透明背景的服装图(让 fit_garment_to_body 能取到 alpha)。
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
    """dress 和 upper 应用不同 GARMENT_BOX, warp 结果不同。"""
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
    """lower 用 hem 宽度(髋宽)而非 shoulder, 结果比 upper 窄。"""
    import engine_sdxl
    import numpy as np
    from PIL import Image, ImageDraw

    garment = Image.new("RGBA", (500, 700), (0, 0, 0, 0))
    d = ImageDraw.Draw(garment)
    d.rectangle([120, 100, 380, 620], fill=(60, 140, 90, 255))
    g = garment.convert("RGB")
    upper = np.asarray(engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "upper", 1024)[0])
    lower = np.asarray(engine_sdxl.align_garment_to_pose(g, "women", "east_asia", "lower", 1024)[0])

    # 非白内容列
    upper_cols = np.where(upper.mean(axis=(0, 2)) < 240)[0]
    lower_cols = np.where(lower.mean(axis=(0, 2)) < 240)[0]
    assert len(upper_cols) > 0
    assert len(lower_cols) > 0
    upper_width = upper_cols[-1] - upper_cols[0]
    lower_width = lower_cols[-1] - lower_cols[0]
    assert lower_width < upper_width


# --- auto 品类推断 + 袖型反向约束 ------------------------------------------------


def test_infer_auto_garment_type_body_ratio_gates_tunic_and_dress():
    """body_ratio 门槛挡"垂带撑高的短款": 裹身衣垂带把 bbox 撑高, hem/aspect
    像中长款, 但衣身本体只到 ~0.8(04469 实测 0.84) —— 按 tunic 出图会画成
    过髋幽灵长衫, 必须挡回 upper。真中长款(衣身宽到底)不受影响。"""
    import engine_sdxl

    infer = engine_sdxl.infer_auto_garment_type
    # 04469 实测几何: hem=mid 但 body_ratio 0.84 → upper
    assert infer({"aspect": 1.36, "hem": "mid", "body_ratio": 0.84}) == "upper"
    # 真中长款: 衣身宽到底 → tunic
    assert infer({"aspect": 1.2, "hem": "mid", "body_ratio": 1.0}) == "tunic"
    # hem=long + 衣身宽到底 → tunic
    assert infer({"aspect": 1.1, "hem": "long", "body_ratio": 0.95}) == "tunic"
    # 垂带撑高的"假连衣裙": 高瘦 + 衣身本体短 → upper, 不按 dress 出图
    assert infer({"aspect": 1.8, "hem": "mid", "body_ratio": 0.6}) == "upper"
    # 真连衣裙: 高瘦 + 衣身长 → dress
    assert infer({"aspect": 1.8, "hem": "long", "body_ratio": 1.0}) == "dress"
    # 矮宽 → lower
    assert infer({"aspect": 0.5, "hem": "waist", "body_ratio": 1.0}) == "lower"
    # 无 extent → upper
    assert infer(None) == "upper"


def test_infer_auto_garment_type_missing_body_ratio_keeps_legacy_behavior():
    """extent 缺 body_ratio 键(旧调用方/手工构造 dict)→ 默认 1.0, 门槛
    等价旧逻辑: 高瘦→dress、hem 过腰→tunic 不回退。"""
    import engine_sdxl

    infer = engine_sdxl.infer_auto_garment_type
    assert infer({"aspect": 1.8}) == "dress"
    assert infer({"hem": "mid"}) == "tunic"
    assert infer({}) == "upper"


def test_sleeve_negative_prompts_pin_opposite_sleeve_terms():
    """袖型反向约束表: 每个袖型类都有对立袖型压制词(04469 短袖被画无袖、
    09290 长袖被画肘部截断的实测修复), 词表被意外清空/改名时报警。"""
    import engine_sdxl

    table = engine_sdxl.SLEEVE_NEGATIVE_PROMPTS
    assert set(table) == {"long", "short", "sleeveless"}
    # 长袖: 必须压"露小臂/短袖/中袖"
    assert "bare forearms" in table["long"] and "short sleeves" in table["long"]
    # 短袖: 必须压"长袖/无袖"
    assert "long sleeves" in table["short"] and "sleeveless" in table["short"]
    # 无袖: 必须压"长袖/短袖"
    assert "long sleeves" in table["sleeveless"] and "short sleeves" in table["sleeveless"]


def test_dedupe_negative_drops_repeat_phrases_and_keeps_order():
    """negative 短语去重: CLIP 77-token 截断下重复词条白烧预算(AUDIENCE 与
    COMMON 曾同时带 "bad anatomy, extra limbs"), 保留首次出现并保序。"""
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
    """negative 组装契约: 袖型词必须最前(CLIP 按 77 token 截断, 旧序追加末尾
    实测袖型词 encoded=0 → C3 与 C2 逐像素相同); 全 audience 与全袖型的
    组合短语无重复、袖型词在首位。token 数用服务器真实 CLIPTokenizer 验证
    (women+long=75, 全组合 ≤77), 这里钉字符串结构防回归。"""
    import engine_sdxl as eng

    for aud in eng.AUDIENCE_NEGATIVE:
        for sl, sv in eng.SLEEVE_NEGATIVE_PROMPTS.items():
            text = eng._dedupe_negative([sv, eng.AUDIENCE_NEGATIVE[aud], eng.FIT_NEGATIVE, eng.COMMON_NEGATIVE])
            phrases = [p.strip() for p in text.split(",")]
            assert text.startswith(sv.split(",")[0].strip() + ","), f"{aud}/{sl}: sleeve terms not first"
            assert len(phrases) == len({p.lower() for p in phrases}), f"{aud}/{sl}: dup phrases"
            assert all(phrases), f"{aud}/{sl}: empty phrase"
