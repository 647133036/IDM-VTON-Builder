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
