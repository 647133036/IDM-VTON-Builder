from __future__ import annotations

import logging
import os
from pathlib import Path

DIFFUSERS_LORA = "pytorch_lora_weights.safetensors"
PEFT_LORA = "adapter_model.safetensors"
_PEFT_PREFIX = "base_model.model."
logger = logging.getLogger("lora_io")

# LORA_IO_STRICT=1 时:转换失败直接抛错(fail-fast),不再回退手动改写(NEW-M10)。
STRICT = os.getenv("LORA_IO_STRICT", "").strip().lower() in ("1", "true", "yes", "on")


# 优先 PEFT adapter_model.safetensors:train_log 记录的 800 步产物是这份 93MB;
# 同目录下的 pytorch_lora_weights.safetensors 是另一条保存路径,数值对不上。
_WEIGHT_NAMES = (PEFT_LORA, DIFFUSERS_LORA)


def lora_weight_path(lora_dir: Path) -> Path | None:
    # 支持单适配器(lora_dir 直接含权重)与 family(lora_dir/<group>/ 各含权重)。
    for name in _WEIGHT_NAMES:
        path = lora_dir / name
        if path.is_file():
            return path
    for sub in sorted(lora_dir.iterdir()) if lora_dir.is_dir() else []:
        if not sub.is_dir():
            continue
        for name in _WEIGHT_NAMES:
            path = sub / name
            if path.is_file():
                return path
    return None


def _load_single_adapter(pipe, lora_dir: Path) -> str:
    peft_w = lora_dir / PEFT_LORA
    if peft_w.is_file():
        from safetensors.torch import load_file

        state = load_file(str(peft_w))
        if not any("lora_" in key for key in state):
            raise ValueError(f"{peft_w} has no LoRA keys; not a valid adapter")
        converted = peft_state_to_diffusers_lora(state)
        if not converted:
            raise ValueError(f"{peft_w} converted to empty state dict; invalid adapter")
        pipe.load_lora_weights(converted)
        return "peft"
    diffusers_w = lora_dir / DIFFUSERS_LORA
    if diffusers_w.is_file():
        pipe.load_lora_weights(str(lora_dir), weight_name=DIFFUSERS_LORA)
        return "diffusers"
    raise FileNotFoundError(f"no LoRA weights in {lora_dir}")


def load_trained_lora(pipe, lora_dir: Path) -> str:
    if lora_weight_path(lora_dir) is None:
        raise FileNotFoundError(f"no LoRA weights in {lora_dir}")
    # 直接目录含权重:加载单个适配器。
    if (lora_dir / PEFT_LORA).is_file() or (lora_dir / DIFFUSERS_LORA).is_file():
        return _load_single_adapter(pipe, lora_dir)
    # 否则按 family 加载每个子目录的适配器(NEW-M8:输出/加载路径对齐)。
    loaded = []
    for sub in sorted(lora_dir.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / DIFFUSERS_LORA).is_file() or (sub / PEFT_LORA).is_file():
            loaded.append(f"{sub.name}:{_load_single_adapter(pipe, sub)}")
    if not loaded:
        raise FileNotFoundError(f"no LoRA weights in {lora_dir}")
    return "family:" + ",".join(loaded)


def peft_state_to_diffusers_lora(state: dict) -> dict:
    stripped = {}
    for key, value in state.items():
        name = key[len(_PEFT_PREFIX) :] if key.startswith(_PEFT_PREFIX) else key
        stripped[name] = value
    try:
        from diffusers.utils import convert_state_dict_to_diffusers

        converted = convert_state_dict_to_diffusers(stripped)
    except ImportError:
        if STRICT:
            raise
        logger.warning("diffusers utils unavailable; falling back to manual key rewrite")
        converted = _rewrite_peft_lora_keys(stripped)
    except Exception as exc:  # 转换失败必须留痕,不能静默吞掉——否则权重格式错误会被掩盖。
        if STRICT:
            raise
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
