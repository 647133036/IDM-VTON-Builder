import sys
import threading
import types

# ---------------------------------------------------------------------------
# N4-L6: IP-Adapter 权重存在但 load_ip_adapter 抛错时,必须标记降级而非静默继续。
# ---------------------------------------------------------------------------


def _install_fake_diffusers(monkeypatch, pipe, load_raises=False):
    def _from_pretrained(cls, *a, **k):
        return object()

    diffusers = types.ModuleType("diffusers")
    diffusers.ControlNetModel = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    diffusers.DDPMScheduler = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))

    def _pipe_from_pretrained(cls, *a, **k):
        return pipe

    diffusers.StableDiffusionXLControlNetPipeline = types.SimpleNamespace(
        from_pretrained=staticmethod(_pipe_from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(empty_cache=staticmethod(lambda: None))
    monkeypatch.setitem(sys.modules, "torch", torch)


def _make_sdxl_pipe(monkeypatch, ip_raises):
    calls = {"load_ip_adapter": 0, "load_ip_adapter_args": None, "load_ip_adapter_kwargs": None}

    class FakePipe:
        def __init__(self):
            self.vae = types.SimpleNamespace(to=staticmethod(lambda self, *a, **k: self))

        def to(self, *a, **k):
            return self

        def enable_attention_slicing(self, *a, **k):
            return None

        def set_ip_adapter_scale(self, *a, **k):
            return None

        def load_ip_adapter(self, *a, **k):
            calls["load_ip_adapter"] += 1
            calls["load_ip_adapter_args"] = a
            calls["load_ip_adapter_kwargs"] = k
            if ip_raises:
                raise RuntimeError("boom ip adapter")

    return FakePipe(), calls


def test_sdxl_ip_adapter_load_failure_marks_degraded(monkeypatch, tmp_path):
    import engine_sdxl

    pipe, calls = _make_sdxl_pipe(monkeypatch, ip_raises=True)
    _install_fake_diffusers(monkeypatch, pipe)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda p, d: None)

    for d in ("sdxl_base", "controlnet_openpose", "controlnet_canny"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    ipdir = tmp_path / "ip_adapter"
    ipdir.mkdir()
    (ipdir / "ip-adapter-plus_sdxl_vit-h.safetensors").write_bytes(b"x")

    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "_IDM_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "_TRAINING_DIR", tmp_path)
    monkeypatch.setattr(
        engine_sdxl,
        "_sdxl_state",
        {"pipe": None, "ip_adapter_loaded": False, "ip_adapter_reason": ""},
    )

    got_pipe, loaded, reason = engine_sdxl._get_sdxl_pipe("cpu", None)
    assert got_pipe is pipe
    assert loaded is False
    assert "failed to load" in reason
    assert "ip-adapter-plus_sdxl_vit-h.safetensors" in reason
    assert calls["load_ip_adapter"] == 1
    # NEW-H1:必须显式传 subfolder="" + weight_name + local_files_only,防 0.34 签名回归。
    assert calls["load_ip_adapter_kwargs"]["subfolder"] == ""
    assert calls["load_ip_adapter_kwargs"]["weight_name"] == "ip-adapter-plus_sdxl_vit-h.safetensors"
    assert calls["load_ip_adapter_kwargs"]["download_kwargs"] == {"local_files_only": True}


def test_sdxl_ip_adapter_absent_is_not_degraded(monkeypatch, tmp_path):
    import engine_sdxl

    pipe, calls = _make_sdxl_pipe(monkeypatch, ip_raises=False)
    _install_fake_diffusers(monkeypatch, pipe)
    monkeypatch.setattr(engine_sdxl, "_load_trained_lora", lambda p, d: None)

    for d in ("sdxl_base", "controlnet_openpose", "controlnet_canny"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    # 无 ip_adapter 权重目录 -> 未安装,不视为降级。
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "_IDM_DIR", tmp_path)
    monkeypatch.setattr(engine_sdxl, "_TRAINING_DIR", tmp_path)
    monkeypatch.setattr(
        engine_sdxl,
        "_sdxl_state",
        {"pipe": None, "ip_adapter_loaded": False, "ip_adapter_reason": ""},
    )

    got_pipe, loaded, reason = engine_sdxl._get_sdxl_pipe("cpu", None)
    assert got_pipe is pipe
    assert loaded is False
    assert reason == ""
    assert calls["load_ip_adapter"] == 0


# ---------------------------------------------------------------------------
# NEW-H3 残留:检测器懒加载并发首载必须单飞。
# ---------------------------------------------------------------------------


def test_openpose_detector_concurrent_first_load_single_flight(monkeypatch, tmp_path):
    import engine_sdxl

    built = {"n": 0}

    class FakeOpenpose:
        @classmethod
        def from_pretrained(cls, *a, **k):
            built["n"] += 1
            return FakeOpenpose()

        def __call__(self, image):
            return image

    controlnet_aux = types.ModuleType("controlnet_aux")
    controlnet_aux.OpenposeDetector = FakeOpenpose
    monkeypatch.setitem(sys.modules, "controlnet_aux", controlnet_aux)
    monkeypatch.setattr(engine_sdxl, "_pose_detector", None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)

    results = []

    def run():
        results.append(engine_sdxl._get_openpose_control("img"))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert built["n"] == 1
    assert len(results) == 8


def test_canny_detector_concurrent_first_load_single_flight(monkeypatch, tmp_path):
    import engine_sdxl

    built = {"n": 0}

    class FakeCanny:
        def __init__(self):
            built["n"] += 1

        def __call__(self, image, **k):
            return image

    controlnet_aux = types.ModuleType("controlnet_aux")
    controlnet_aux.CannyDetector = FakeCanny
    monkeypatch.setitem(sys.modules, "controlnet_aux", controlnet_aux)
    monkeypatch.setattr(engine_sdxl, "_canny_detector", None)
    monkeypatch.setattr(engine_sdxl, "SDXL_DIR", tmp_path)

    results = []

    def run():
        results.append(engine_sdxl._get_canny_control("img"))

    threads = [threading.Thread(target=run) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert built["n"] == 1
    assert len(results) == 8


# ---------------------------------------------------------------------------
# NEW-H3 残留:IDM-VTON 管线首载并发必须单飞(否则重复构建撑爆显存)。
# ---------------------------------------------------------------------------


def _install_idm_fakes(monkeypatch, pipe, build_counter):
    torch = types.ModuleType("torch")
    torch.device = staticmethod(lambda *a, **k: "cuda:0")
    torch.float16 = "float16"
    monkeypatch.setitem(sys.modules, "torch", torch)

    def _from_pretrained(cls, *a, **k):
        return object()

    diffusers = types.ModuleType("diffusers")
    diffusers.AutoencoderKL = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    diffusers.DDPMScheduler = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    transformers.CLIPTextModel = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    transformers.CLIPTextModelWithProjection = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class FakeTryonPipeline:
        def __init__(self, **kwargs):
            build_counter["n"] += 1
            self._pipe = pipe

        def to(self, *a, **k):
            return self

        def enable_vae_slicing(self, *a, **k):
            return None

        def enable_attention_slicing(self, *a, **k):
            return None

    src = types.ModuleType("src")
    tryon = types.ModuleType("src.tryon_pipeline")
    tryon.StableDiffusionXLInpaintPipeline = FakeTryonPipeline
    garmnet = types.ModuleType("src.unet_hacked_garmnet")
    garmnet.UNet2DConditionModel = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    tryon_mod = types.ModuleType("src.unet_hacked_tryon")
    tryon_mod.UNet2DConditionModel = types.SimpleNamespace(from_pretrained=staticmethod(_from_pretrained))
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.tryon_pipeline", tryon)
    monkeypatch.setitem(sys.modules, "src.unet_hacked_garmnet", garmnet)
    monkeypatch.setitem(sys.modules, "src.unet_hacked_tryon", tryon_mod)


def test_idm_pipeline_concurrent_first_load_single_flight(monkeypatch, tmp_path):
    import engine as eng

    build_counter = {"n": 0}
    _install_idm_fakes(monkeypatch, object(), build_counter)
    monkeypatch.setattr(eng, "_idm_vton_available", lambda: (True, ""))
    monkeypatch.setattr(eng, "_engine_state", {"pipe": None, "available": None, "reason": ""})
    monkeypatch.setattr(eng, "WEIGHTS_DIR", tmp_path)
    monkeypatch.setattr(eng, "_IDM_DIR", tmp_path)

    results = []

    def run():
        results.append(eng._load_idm_vton_pipeline())

    threads = [threading.Thread(target=run) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # 全部线程拿到同一管线,且只构建一次。
    assert build_counter["n"] == 1
    assert all(r is not None for r in results)
    assert all(r is results[0] for r in results)
