from __future__ import annotations

from pathlib import Path

DIFFUSERS_LORA = "pytorch_lora_weights.safetensors"
PEFT_LORA = "adapter_model.safetensors"
_PEFT_PREFIX = "base_model.model."


def lora_weight_path(lora_dir: Path) -> Path | None:
    for name in (DIFFUSERS_LORA, PEFT_LORA):
        path = lora_dir / name
        if path.exists():
            return path
    return None


def peft_state_to_diffusers_lora(state: dict) -> dict:
    stripped = {}
    for key, value in state.items():
        name = key[len(_PEFT_PREFIX) :] if key.startswith(_PEFT_PREFIX) else key
        stripped[name] = value
    try:
        from diffusers.utils import convert_state_dict_to_diffusers

        converted = convert_state_dict_to_diffusers(stripped)
    except Exception:  # noqa: BLE001
        converted = _rewrite_peft_lora_keys(stripped)
    return {k if k.startswith("unet.") else f"unet.{k}": v for k, v in converted.items()}


def _rewrite_peft_lora_keys(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        name = key.replace(".lora_A.weight", ".lora.down.weight")
        name = name.replace(".lora_B.weight", ".lora.up.weight")
        name = name.replace(".lora_A.", ".lora.down.")
        name = name.replace(".lora_B.", ".lora.up.")
        out[name] = value
    return out


def load_trained_lora(pipe, lora_dir: Path) -> str:
    diffusers_w = lora_dir / DIFFUSERS_LORA
    if diffusers_w.exists():
        pipe.load_lora_weights(str(lora_dir), weight_name=DIFFUSERS_LORA)
        return "diffusers"
    peft_w = lora_dir / PEFT_LORA
    if not peft_w.exists():
        raise FileNotFoundError(f"no LoRA weights in {lora_dir}")
    from safetensors.torch import load_file

    state = load_file(str(peft_w))
    pipe.load_lora_weights(peft_state_to_diffusers_lora(state))
    return "peft"
