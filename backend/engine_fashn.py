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
    # fashn_vton v1.5 在模块导入期还需要以下三个常量(preprocessing/agnostic.py)。
    # segmentation_free=True 时运行期不触解析器, 但导入必须成功; 值按 SCHP/LIP
    # 语义给出(仅供类型/结构兼容, 商用路径永不消费)。
    stub.BODY_COVERAGE_TO_LABELS = {
        "upper": ("Upper-clothes",),
        "lower": ("Pants", "Skirt"),
        "full": ("Dress",),
    }
    stub.IDENTITY_LABELS = (
        "Face",
        "Hair",
        "Sunglasses",
        "Hat",
        "Left-arm",
        "Right-arm",
        "Left-leg",
        "Right-leg",
        "Left-shoe",
        "Right-shoe",
    )
    stub.LABELS_TO_IDS = {
        "Background": 0,
        "Hat": 1,
        "Hair": 2,
        "Sunglasses": 3,
        "Upper-clothes": 4,
        "Skirt": 5,
        "Pants": 6,
        "Dress": 7,
        "Belt": 8,
        "Left-shoe": 9,
        "Right-shoe": 10,
        "Face": 11,
        "Left-leg": 12,
        "Right-leg": 13,
        "Left-arm": 14,
        "Right-arm": 15,
        "Bag": 16,
        "Scarf": 17,
        # agnostic.py 深路径索引的部位键 (seg_free=True 不消费, 结构兼容)
        "arms": 18,
        "torso": 19,
        "legs": 20,
        "hands": 21,
        "feet": 22,
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
    from preserve_refit import protect_inverse_refit
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
    lock_on = os.environ.get("VTON_LOCK_ID", "1") != "0"
    detail_applied = False
    keep_mask = None
    paste_mask = None
    cloth_ref = None
    cloth_ref_alpha = None
    if (pattern_on or lock_on) and warp_enabled:
        try:
            subj = foreground_mask(result)
            shared = clothing_mask(
                result,
                garment_type=garment_type,
                garment_extent=garment_extent,
                subject_mask=subj,
            )
            drawn_mask = limit_mask_to_subject(shared, subj) if shared is not None else None
            refit = None
            if os.environ.get("VTON_FASHN_PRINT_ALIGN", "drawn") == "drawn" and drawn_mask is not None:
                from preserve_refit import refit_cloth_to_drawn_garment

                refit = refit_cloth_to_drawn_garment(garment_img, result.size, drawn_mask)
            if refit is not None:
                # 贴印对齐生成几何: 平铺衣 TPS 重投到 FASHN 画的衣服 bbox;
                # 检测区即衣服边界, 纯裁剪(clip, 无并集); 贴印区=印花核心,
                # 全衣料款回退整件贴回。贴印/保真分与衣服保护三 mask 解耦。
                cloth_ref, cloth_ref_alpha = refit
                from preserve_refit import clip_mask_to_garment_alpha, print_core_mask

                keep_mask = clip_mask_to_garment_alpha(drawn_mask, cloth_ref_alpha)
                paste_mask = print_core_mask(cloth_ref, keep_mask)
                if paste_mask is None:
                    paste_mask = keep_mask
            else:
                # 模板对齐(legacy 回退): warp 平铺衣到身体模板先验,
                # 并集再裁剪(intersect)兜住宽松款悬垂面料。
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
                keep_mask, _kept = intersect_mask_with_garment_alpha(drawn_mask, cloth_ref_alpha)
                paste_mask = keep_mask
            if pattern_on and paste_mask is not None:
                result = reapply_garment_detail(
                    result,
                    cloth_ref,
                    paste_mask,
                    pattern_lock=True,
                    ref_alpha=cloth_ref_alpha,
                )
                detail_applied = True
        except Exception:
            logger.exception("FASHN pattern_lock/lock failed; keeping FASHN pixels")
            degraded_reasons.append("FASHN pattern_lock/lock failed")

    # 保护贴回: keep_mask 外(人脸/发丝/手/露肤手臂/背景)贴回源图像素,
    # keep_mask 内保留 FASHN 像素 + 贴印。手臂不会变白、胸口不会变白、发丝不会糊。
    if lock_on and keep_mask is not None:
        try:
            _feather = int(os.environ.get("VTON_PROTECT_FEATHER", "4"))
            result = protect_inverse_refit(result, base_img, keep_mask, feather_px=_feather)
        except Exception:
            logger.exception("FASHN protect_inverse_refit failed; keeping FASHN pixels")
            degraded_reasons.append("FASHN protect_inverse_refit failed")

    # 印花保真分 + 交付档位: 贴印区(paste_mask)内 warp 参考与结果的灰度相关。
    # pattern_lock 生效 → 高分 auto; 花纹/字母被重画或错位 → 低分 review。
    # 只算不写: 后续背景/缩放会替换 Image 对象丢掉 .info, 元数据统一在
    # 尾部元数据块写入(与 engine.py 同序), 否则 print_fidelity 会变 null。
    _pf_val = None
    _tier_val = None
    if paste_mask is not None and cloth_ref is not None:
        try:
            from quality_gate import delivery_tier, print_fidelity_score

            _pf_val = print_fidelity_score(result, cloth_ref, paste_mask)
            _tier_val = delivery_tier(_pf_val, sku_has_text=False)
        except Exception:
            logger.exception("FASHN print fidelity scoring failed; tier not set")

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
    result.info["lock_identity"] = lock_on and keep_mask is not None
    result.info["fashn_category"] = category
    result.info["prewhiten"] = {"base": pw_base, "garment": pw_garment}
    if _pf_val is not None:
        result.info["print_fidelity"] = _pf_val
    if _tier_val is not None:
        result.info["delivery_tier"] = _tier_val
    return result
