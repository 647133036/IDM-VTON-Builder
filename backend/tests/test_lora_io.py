import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from lora_io import (  # noqa: E402
    DIFFUSERS_LORA,
    PEFT_LORA,
    lora_weight_path,
    peft_state_to_diffusers_lora,
)


def test_lora_weight_path_prefers_diffusers(tmp_path):
    (tmp_path / DIFFUSERS_LORA).write_bytes(b"x")
    (tmp_path / PEFT_LORA).write_bytes(b"y")
    assert lora_weight_path(tmp_path).name == DIFFUSERS_LORA


def test_lora_weight_path_falls_back_to_peft(tmp_path):
    (tmp_path / PEFT_LORA).write_bytes(b"y")
    assert lora_weight_path(tmp_path).name == PEFT_LORA


def test_lora_weight_path_empty(tmp_path):
    assert lora_weight_path(tmp_path) is None


def test_peft_state_gets_unet_prefix_and_diffusers_lora_keys():
    state = {
        "base_model.model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.weight": 1,
        "base_model.model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.weight": 2,
        "base_model.model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.weight": 3,
        "base_model.model.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.weight": 4,
    }
    out = peft_state_to_diffusers_lora(state)
    assert len(out) == 4
    assert all(k.startswith("unet.") for k in out)
    assert all("base_model" not in k for k in out)
    joined = " ".join(out)
    assert "lora_A" not in joined
    assert "to_k.lora.down.weight" in joined
    assert "to_k.lora.up.weight" in joined
    assert "to_out.0.lora.down.weight" in joined
    assert "to_out.0.lora.up.weight" in joined


def test_lora_weight_path_finds_family_subdir(tmp_path):
    # NEW-M8: family 权重放在 lora_dir/<group>/ 子目录,loader 必须能找到。
    sub = tmp_path / "cloth"
    sub.mkdir()
    (sub / DIFFUSERS_LORA).write_bytes(b"x")
    assert lora_weight_path(tmp_path).name == DIFFUSERS_LORA


def test_load_trained_lora_family_loads_each_group(tmp_path):
    # NEW-M8: family 目录下每个含权重的子目录都应被加载。
    from lora_io import load_trained_lora

    for group in ("cloth", "human"):
        d = tmp_path / group
        d.mkdir()
        (d / DIFFUSERS_LORA).write_bytes(b"w")
    calls = []

    class _FakePipe:
        def load_lora_weights(self, *a, **k):
            calls.append((a, k))

    label = load_trained_lora(_FakePipe(), tmp_path)
    assert label.startswith("family:")
    assert "cloth" in label and "human" in label
    assert len(calls) == 2


def test_strict_mode_raises_on_convert_failure(monkeypatch):
    # NEW-M10: LORA_IO_STRICT=1 时转换失败必须 fail-fast。
    import lora_io

    monkeypatch.setattr(lora_io, "STRICT", True)

    class _Boom:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "diffusers.utils":
            raise ImportError("no diffusers")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import pytest

    with pytest.raises(ImportError):
        lora_io.peft_state_to_diffusers_lora({"base_model.model.x.lora_A.weight": 1})
