from __future__ import annotations

import logging
from pathlib import Path

DIFFUSERS_LORA = "pytorch_lora_weights.safetensors"
PEFT_LORA = "adapter_model.safetensors"
_PEFT_PREFIX = "base_model.model."
logger = logging.getLogger("lora_io")


def lora_weight_path(lora_dir: Path) -> Path | None:
    for name in (DIFFUSERS_LORA, PEFT_LORA):
        path = lora_dir / name
        if path.is_file():
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
    except ImportError:
        # diffusers 未安装或工具函数缺失时才回退到手动重写。
        logger.warning("diffusers utils unavailable; falling back to manual key rewrite")
        converted = _rewrite_peft_lora_keys(stripped)
    except Exception as exc:  # noqa: BLE001
        # 转换失败必须留痕,不能静默吞掉——否则权重格式错误会被掩盖。
        logger.warning("convert_state_dict_to_diffusers failed (%s); using manual key rewrite", exc)
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
    if diffusers_w.is_file():
        pipe.load_lora_weights(str(lora_dir), weight_name=DIFFUSERS_LORA)
        return "diffusers"
    peft_w = lora_dir / PEFT_LORA
    if not peft_w.is_file():
        raise FileNotFoundError(f"no LoRA weights in {lora_dir}")
    from safetensors.torch import load_file

    state = load_file(str(peft_w))
    if not any("lora_" in key for key in state):
        raise ValueError(f"{peft_w} has no LoRA keys; not a valid adapter")
    converted = peft_state_to_diffusers_lora(state)
    if not converted:
        raise ValueError(f"{peft_w} converted to empty state dict; invalid adapter")
    pipe.load_lora_weights(converted)
    return "peft"
