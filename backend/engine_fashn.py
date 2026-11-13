from __future__ import annotations

import logging
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
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
    pending_print_warp = None
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
                # 检测区即衣服边界, 纯裁剪(clip, 无并集); 贴印/保真分与衣服
                # 保护三 mask 解耦。
                cloth_ref, cloth_ref_alpha = refit
                from preserve_refit import clip_mask_to_garment_alpha

                keep_mask = clip_mask_to_garment_alpha(drawn_mask, cloth_ref_alpha)
                if pattern_on:
                    # 印花贴回推迟到最终分辨率: (a) 576 原生画布贴印会被
                    # 末尾 2.67x LANCZOS 放大拉糊(r8); (b) 印花核心检测也
                    # 必须在最终分辨率做 —— 576 下印花笔触被 8x 降采样混进
                    # 底色, 检测只剩空心环(真实数据内区 NCC -0.69)。实心
                    # 款回退同样在全分辨率整件贴回, 质量不降级。
                    _dm = np.asarray(drawn_mask)
                    if _dm.ndim == 3:
                        _dm = _dm[..., 0]
                    pending_print_warp = _dm
                    paste_mask = None
                else:
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

    # 印花核心直贴(最终分辨率): 放大后的成图上重投 + 核心检测 + 贴印,
    # 印花笔触保参考原样清晰度(hf/folds 关, 不受生成纹理污染); 保真分
    # 同步在成图分辨率上度量。核心检测只在最终分辨率做一次(576 下印花
    # 笔触被 8x 降采样混进底色只剩空心环); 实心款(无核心)回退整件贴回
    # 同样在全分辨率, 质量较 r8 低分辨率贴印只升不降。模板路径不变。
    if pending_print_warp is not None:
        try:
            from preserve_refit import clip_mask_to_garment_alpha as _clip_fn
            from preserve_refit import print_core_mask as _core_fn
            from preserve_refit import refit_cloth_to_drawn_garment as _refit_fn

            _dm_low = pending_print_warp
            if _dm_low.shape[::-1] != result.size:
                _dm_full = np.array(
                    Image.fromarray(_dm_low.astype(np.uint8)).resize(result.size, Image.Resampling.NEAREST)
                )
                if _dm_full.ndim == 3:
                    _dm_full = _dm_full[..., 0]
            else:
                _dm_full = _dm_low
            refit_full = _refit_fn(garment_img, result.size, _dm_full)
            if refit_full is not None:
                cloth_ref_full, alpha_full = refit_full
                keep_full = _clip_fn(_dm_full, alpha_full)
                core_full = _core_fn(cloth_ref_full, keep_full)
                if core_full is not None and np.asarray(core_full).max() > 0:
                    # 印花核心=参考原样: light/hf/folds 全关 —— 印花场内 FASHN
                    # 的鬼影笔触让 LF 明暗场尖峰(NCC 实测 -0.4), 恒等贴回
                    # 保真分=1.0(本地全链路实证), 场外仍保留身体光影褶皱。
                    # content_mask=keep: 白墨印花在 alpha 上是透明洞(garment_alpha
                    # 与画布同色判定, 满印墨迹网触边, 拓扑内外判定无解), alpha
                    # 内容门会跳过笔触留鬼影; keep(drawn 裁 alpha 膨胀)衣料
                    # 语义完整。
                    paste_full, core_kwargs = (
                        core_full,
                        {
                            "light_strength": 0.0,
                            "hf_strength": 0.0,
                            "fold_gain": 0.0,
                            "content_mask": keep_full,
                            # 细笔触印花羽化 1px: 鬼场/整件贴回 5px 羽化平滑
                            # 大块接缝, 但核心场恒等贴印的印花笔触本身仅 3-4px
                            # 宽, 5px 距离场把笔触 alpha 压到 0.2-0.4, 白墨
                            # 印花半透发灰(r10 GPU 逐成分 NCC 实测)。identity
                            # 贴印的 crisp 印花边缘语义本来就该干净利落。
                            "feather_px": 1.0,
                        },
                    )
                else:
                    paste_full, core_kwargs = keep_full, {}
                # transparent 路径此时已是 RGBA: 贴回在 RGB 上做再装回
                # alpha, 透明信息不因贴印丢失。
                _alpha_ch = result.split()[-1] if result.mode == "RGBA" else None
                _rgb_result = result.convert("RGB")
                if core_full is not None:
                    # 色调匹配(r10 GPU 实测: gen 衣身 [117,103,69] vs 平铺 ref
                    # [220,206,168], 差百余灰阶): 直接贴参考, 鬼场擦除带与
                    # 核心场内衣料都比生成图衣身亮 +100 → 视觉验收"印花周
                    # 围一圈发白发光轮廓带"。先统一把参考逐通道平移到生成
                    # 图衣身色调; 鬼场/核心场分别按各自语义取用。
                    try:
                        from preserve_refit import match_ref_tone as _tone_fn

                        _ref_matched = _tone_fn(
                            _rgb_result, cloth_ref_full, keep_full, avoid_mask=np.asarray(core_full)
                        )
                    except Exception:
                        logger.exception("FASHN tone match failed; using raw ref")
                        _ref_matched = cloth_ref_full
                    # 鬼影擦除(核心场外): FASHN 按自己的几何摆放印花(实测
                    # 偏上), 核心场只盖 TPS 位置, 生成图的整套鬼影笔触
                    # 暴露在场外(视觉验收: 上方半透明重影)。生成图上同色轴
                    # 检出鬼场, 用色调匹配参考+恒等参数贴回 —— 恒等 (光/褶
                    # 全关) 防 gen 鬼影区的明暗场/褶皱带通把鬼影形状搬回。
                    # 实心款回退分支整件已贴回, 无需此步。
                    try:
                        from preserve_refit import ghost_print_mask as _ghost_fn

                        ghost_full = _ghost_fn(_rgb_result, cloth_ref_full, keep_full, exclude_mask=core_full)
                        if ghost_full is not None and np.asarray(ghost_full).max() > 0:
                            _rgb_result = reapply_garment_detail(
                                _rgb_result,
                                _ref_matched,
                                ghost_full,
                                pattern_lock=True,
                                ref_alpha=alpha_full,
                                light_strength=0.0,
                                hf_strength=0.0,
                                fold_gain=0.0,
                                content_mask=np.asarray(keep_full),
                            )
                    except Exception:
                        logger.exception("FASHN ghost print erase failed; keeping current pixels")
                        degraded_reasons.append("FASHN ghost print erase failed")
                    # 核心场贴回参考混合: 白墨笔触保持原样(客观亮色), 场内
                    # SKU 衣料像素换色调匹配版 —— 整场恒等会把凸包里的亮
                    # 衣料(198+)拍上暗衣身(96), 印花一圈发光。
                    try:
                        from preserve_refit import blend_core_ref_tone as _blend_fn

                        _core_ref = _blend_fn(cloth_ref_full, core_full, _ref_matched)
                    except Exception:
                        logger.exception("FASHN core ref blend failed; using raw ref")
                        _core_ref = cloth_ref_full
                else:
                    _ref_matched = None
                    _core_ref = cloth_ref_full
                result = reapply_garment_detail(
                    _rgb_result,
                    _core_ref,
                    paste_full,
                    pattern_lock=True,
                    ref_alpha=alpha_full,
                    **core_kwargs,
                )
                if _alpha_ch is not None:
                    result = result.copy()
                    result.putalpha(_alpha_ch)
                detail_applied = True
                from quality_gate import delivery_tier, print_fidelity_score

                _pf_val = print_fidelity_score(result, cloth_ref_full, paste_full)
                _tier_val = delivery_tier(_pf_val, sku_has_text=False)
        except Exception:
            logger.exception("FASHN full-res print paste failed; keeping current pixels")
            degraded_reasons.append("FASHN full-res print paste failed")

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
