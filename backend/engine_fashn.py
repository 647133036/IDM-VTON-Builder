from __future__ import annotations

import logging
import os
import sys
import threading
import types
from pathlib import Path

from PIL import Image

logger = logging.getLogger("atelier.engine_fashn")

_GPU_LOCK = threading.Lock()
_PIPE_LOCK = threading.Lock()
_fashn_state: dict = {"pipe": None}

_REPO_ROOT = Path(__file__).resolve().parent.parent
FASHN_DIR = _REPO_ROOT / "models" / "fashn"

GARMENT_TYPE_TO_FASHN = {
    "upper": "tops",
    "tunic": "tops",
    "lower": "bottoms",
    "dress": "one-pieces",
    "onesie": "one-pieces",
}


def fashn_category(garment_type: str) -> str:
    return GARMENT_TYPE_TO_FASHN.get(garment_type, "tops")


def _ensure_parser_stub() -> None:
    """Commercial path must not load fashn-human-parser (NVIDIA SegFormer license)."""
    existing = sys.modules.get("fashn_human_parser")
    if existing is not None and getattr(existing, "_atelier_commercial_stub", False):
        _patch_loaded_pipeline(existing)
        return

    stub = types.ModuleType("fashn_human_parser")
    stub.CATEGORY_TO_BODY_COVERAGE = {
        "tops": "upper",
        "bottoms": "lower",
        "one-pieces": "full",
    }
    stub.BODY_COVERAGE_TO_FASHN_LABELS = {}
    stub.FASHN_LABELS_TO_IDS = {}

    class FashnHumanParser:
        def __init__(self, device: str = "cpu", **_kwargs):
            self.device = device

        def predict(self, image):
            import numpy as np

            arr = np.asarray(image)
            h, w = arr.shape[:2]
            return np.zeros((h, w), dtype=np.int32)

    stub.FashnHumanParser = FashnHumanParser
    stub._atelier_commercial_stub = True
    sys.modules["fashn_human_parser"] = stub
    _patch_loaded_pipeline(stub)


def _patch_loaded_pipeline(stub) -> None:
    pipe_mod = sys.modules.get("fashn_vton.pipeline")
    if pipe_mod is None:
        return
    pipe_mod.FashnHumanParser = stub.FashnHumanParser
    pipe_mod.CATEGORY_TO_BODY_COVERAGE = stub.CATEGORY_TO_BODY_COVERAGE


def fashn_available() -> tuple[bool, str]:
    required = [
        FASHN_DIR / "model.safetensors",
        FASHN_DIR / "dwpose" / "yolox_l.onnx",
        FASHN_DIR / "dwpose" / "dw-ll_ucoco_384.onnx",
    ]
    missing = []
    for p in required:
        if p.is_file():
            continue
        try:
            missing.append(str(p.relative_to(_REPO_ROOT)))
        except ValueError:
            missing.append(str(p))
    if missing:
        return False, f"Missing FASHN weights: {', '.join(missing)}"
    try:
        import importlib.util

        if importlib.util.find_spec("fashn_vton") is None:
            return False, "fashn_vton not installed"
    except Exception:  # noqa: BLE001
        return False, "fashn_vton not installed"
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "No GPU available"
    return True, "ok"


def _get_fashn_pipe():
    with _PIPE_LOCK:
        if _fashn_state["pipe"] is not None:
            return _fashn_state["pipe"]
        _ensure_parser_stub()
        from fashn_vton import TryOnPipeline

        pipe = TryOnPipeline(weights_dir=str(FASHN_DIR))
        _fashn_state["pipe"] = pipe
        logger.info("FASHN VTON v1.5 loaded from %s (segmentation_free, parser stubbed)", FASHN_DIR)
        return pipe


def compose_tryon_fashn(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
    ethnicity: str = "western_europe",
) -> Image.Image:
    from engine_sdxl import (
        _remove_background,
        align_garment_to_pose,
        crop_garment_to_content,
        infer_auto_garment_type,
        intersect_mask_with_garment_alpha,
    )
    from platform_presets import (
        clothing_mask,
        detect_garment_extent,
        force_white_background,
        foreground_mask,
        limit_mask_to_subject,
        reapply_garment_detail,
    )
    from prewhiten import prewhiten_input

    base_img = Image.open(base_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")
    base_img, pw_base = prewhiten_input(base_img)
    if pw_base.get("status") not in ("applied", "already_white", "disabled"):
        logger.warning("base prewhiten not applied: %s", pw_base.get("reason"))
    if os.environ.get("VTON_PREWHITEN_GARMENT", "1") == "0":
        pw_garment = {"status": "disabled", "reason": "VTON_PREWHITEN_GARMENT=0"}
    else:
        garment_img, pw_garment = prewhiten_input(garment_img)

    garment_img = crop_garment_to_content(garment_img)

    garment_extent = None
    try:
        garment_extent = detect_garment_extent(garment_img)
    except Exception:  # noqa: BLE001
        logger.debug("detect_garment_extent failed; using defaults")
    if garment_type == "auto":
        garment_type = infer_auto_garment_type(garment_extent)

    category = fashn_category(garment_type)
    pipe = _get_fashn_pipe()
    seed = int(os.environ.get("VTON_SEED", "42"))
    steps = int(os.environ.get("VTON_FASHN_STEPS", "30"))
    guidance = float(os.environ.get("VTON_FASHN_GUIDANCE", "1.5"))

    with _GPU_LOCK:
        output = pipe(
            person_image=base_img,
            garment_image=garment_img,
            category=category,
            garment_photo_type="flat-lay",
            num_timesteps=steps,
            guidance_scale=guidance,
            seed=seed,
            segmentation_free=True,
        )
    result = output.images[0]
    if result.mode != "RGB":
        result = result.convert("RGB")

    degraded_reasons: list[str] = []
    warp_enabled = os.environ.get("VTON_WARP", "1") != "0"
    pattern_on = os.environ.get("VTON_PATTERN_LOCK", "1") != "0"
    detail_applied = False
    if pattern_on and warp_enabled:
        try:
            warp_canvas = max(result.size)
            cloth_ref, cloth_ref_alpha = align_garment_to_pose(
                garment_img,
                audience,
                ethnicity,
                garment_type,
                warp_canvas,
            )
            if cloth_ref.size != result.size:
                cloth_ref = cloth_ref.resize(result.size, Image.Resampling.LANCZOS)
                if cloth_ref_alpha is not None:
                    cloth_ref_alpha = cloth_ref_alpha.resize(result.size, Image.Resampling.BILINEAR)
            subj = foreground_mask(result)
            shared = clothing_mask(
                result,
                garment_type=garment_type,
                garment_extent=garment_extent,
                subject_mask=subj,
            )
            apply_mask = limit_mask_to_subject(shared, subj) if shared is not None else None
            apply_mask, _kept = intersect_mask_with_garment_alpha(apply_mask, cloth_ref_alpha)
            if apply_mask is not None:
                result = reapply_garment_detail(
                    result,
                    cloth_ref,
                    apply_mask,
                    pattern_lock=True,
                    ref_alpha=cloth_ref_alpha,
                )
                detail_applied = True
        except Exception:
            logger.exception("FASHN pattern_lock failed; keeping FASHN pixels")
            degraded_reasons.append("FASHN pattern_lock failed")

    if background == "transparent":
        result = _remove_background(result)
    elif background == "white":
        if result.mode == "RGBA":
            bg = Image.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[-1])
            result = bg
        else:
            result = result.convert("RGB")
        subj_mask = foreground_mask(result)
        result = force_white_background(result, subject_mask=subj_mask)

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), Image.Resampling.LANCZOS)

    degraded_reason = "; ".join(degraded_reasons)
    result.info["engine"] = "fashn-vton"
    result.info["engine_degraded"] = bool(degraded_reason)
    result.info["degraded_reason"] = degraded_reason
    result.info["detail_applied"] = detail_applied
    result.info["fashn_category"] = category
    result.info["prewhiten"] = {"base": pw_base, "garment": pw_garment}
    return result
