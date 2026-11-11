"""FASHN 阶段转储调试: 定位躯干丢失发生在 FASHN 裸生成还是我们的后处理链.

dump 顺序:
  s0_fashn_raw.png        复刻 compose 的输入预处理 + segmentation_free=True
  s0b_seg_free_false.png  同输入但 segmentation_free=False (stub parser 进链)
  s0c_no_prewhiten.png    原始输入不经 prewhiten, segmentation_free=True
  s1_after_patternlock.png / s2_after_protect.png / s3_final.png
  m_*.png 全部中间 mask (u2net 前景/共享服装区/apply 区/cloth_ref/alpha)
  log 内含 PipelineOutput 字段内省 + segmentation_free 源码行.
"""

import inspect
import os
import sys
from pathlib import Path

REPO = "/root/vton/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/backend")
os.chdir(REPO)

OUT = Path("/root/autodl-tmp/vton_combo/out_debug")
OUT.mkdir(parents=True, exist_ok=True)
BASE = Path("/root/autodl-tmp/vton_combo/bases/women_01.png")
GARM = Path("/root/autodl-tmp/vton_combo/garments/01_khaite_goya-silk-top_None_main.jpg")

for k, v in {
    "LICENSE_MODE": "commercial",
    "VTON_REFIT_ENGINE": "tps",
    "VTON_PROTECT_FEATHER": "4",
    "VTON_PRINT_INPAINT": "1",
    "VTON_INPAINT_REPAIR": "1",
    "VTON_WARP": "1",
    "VTON_PATTERN_LOCK": "1",
    "VTON_LOCK_ID": "1",
    "VTON_SEED": "42",
}.items():
    os.environ.setdefault(k, v)


def note(msg):
    print(msg, flush=True)


import numpy as np
from PIL import Image

import engine_fashn as ef
from engine_sdxl import (
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
from preserve_refit import protect_inverse_refit
from quality_gate import print_fidelity_score


def save_mask(obj, name):
    if obj is None:
        note("%s: None" % name)
        return None
    if isinstance(obj, Image.Image):
        img = obj.convert("L") if obj.mode in ("L", "1") else obj.convert("RGB")
        img.save(OUT / (name + ".png"))
        arr = np.asarray(img.convert("L"))
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2:
            if arr.dtype != np.uint8:
                arr = (np.clip(arr.astype(float) / max(float(arr.max()), 1.0), 0, 1) * 255).astype("uint8")
            Image.fromarray(arr).save(OUT / (name + ".png"))
        else:
            Image.fromarray(arr.astype("uint8")).save(OUT / (name + ".png"))
    note("%s saved frac>127=%.4f" % (name, float((arr > 127).mean()) if arr.ndim == 2 else -1))
    return obj


# --- 0) fashn v1.5 源码内省: segmentation_free 流程与 PipelineOutput 字段 ---
import engine_fashn as _ef_for_stub

_ef_for_stub._ensure_parser_stub()
import fashn_vton

pipe_src = inspect.getsourcefile(fashn_vton.TryOnPipeline)
note("== fashn pipeline.py: %s" % pipe_src)
src_lines = open(pipe_src).read().splitlines()
for idx, line in enumerate(src_lines, 1):
    if "segmentation_free" in line or ("parser" in line.lower() and "import" not in line.lower()):
        note("SRC %d: %s" % (idx, line.rstrip()))
try:
    note("== PipelineOutput: %s" % inspect.getsource(fashn_vton.PipelineOutput))
except Exception as e:  # noqa: BLE001
    note("PipelineOutput introspect failed: %s" % e)
try:
    note("== TryOnPipeline.run signature: %s" % inspect.signature(fashn_vton.TryOnPipeline.run))
except Exception as e:  # noqa: BLE001
    note("run signature introspect failed: %s" % e)

# --- 1) 输入准备 (复刻 compose) ---
base_img = Image.open(BASE).convert("RGB")
garment_img = Image.open(GARM).convert("RGB")
base_img, pw_base = prewhiten_input(base_img)
garment_img, pw_garment = prewhiten_input(garment_img)
garment_img = crop_garment_to_content(garment_img)
note("prewhiten: base=%s garment=%s" % (pw_base.get("status"), pw_garment.get("status")))
extent = detect_garment_extent(garment_img)
gt = infer_auto_garment_type(extent)
category = ef.fashn_category(gt)
note("garment_type=%s category=%s extent=%s" % (gt, category, extent))

pipe = ef._get_fashn_pipe()


def raw_call(person, garment, seg_free=True, tag=""):
    out = pipe(
        person_image=person,
        garment_image=garment,
        category=category,
        garment_photo_type="flat-lay",
        num_timesteps=30,
        guidance_scale=1.5,
        seed=42,
        segmentation_free=seg_free,
    )
    fields = [f for f in dir(out) if not f.startswith("_")]
    note("%s PipelineOutput fields: %s" % (tag, fields))
    for attr in fields:
        v = getattr(out, attr)
        if isinstance(v, Image.Image):
            v.convert("RGB").save(OUT / ("s0%sx_out_%s.png" % (tag, attr)))
            note("%s saved out.%s %s" % (tag, attr, v.size))
    img = out.images[0].convert("RGB")
    img.save(OUT / ("s0%s_fashn_raw.png" % tag))
    note("%s raw saved %s" % (tag, img.size))
    return img


# --- 2) 三种裸调用 (s0b/s0c 为对照实验, 失败不阻塞主链) ---
img_s0 = raw_call(base_img, garment_img, seg_free=True, tag="")
try:
    img_s0b = raw_call(base_img, garment_img, seg_free=False, tag="b")
except Exception as e:  # noqa: BLE001
    note("s0b (seg_free=False) failed (expected with stub): %s" % e)
base_raw = Image.open(BASE).convert("RGB")
garm_raw = crop_garment_to_content(Image.open(GARM).convert("RGB"))
try:
    img_s0c = raw_call(base_raw, garm_raw, seg_free=True, tag="c")
except Exception as e:  # noqa: BLE001
    note("s0c (no prewhiten) failed: %s" % e)

# --- 3) 后处理链逐阶段 (基于 s0) ---
result = img_s0
subj0 = foreground_mask(result)
save_mask(subj0, "m_fg_on_s0")

cloth_ref, cloth_ref_alpha = align_garment_to_pose(garment_img, "women", "east_asian", gt, max(result.size))
if cloth_ref.size != result.size:
    cloth_ref = cloth_ref.resize(result.size, Image.Resampling.LANCZOS)
    if cloth_ref_alpha is not None:
        cloth_ref_alpha = cloth_ref_alpha.resize(result.size, Image.Resampling.BILINEAR)
cloth_ref.convert("RGB").save(OUT / "m_cloth_ref.png")
note("cloth_ref saved %s alpha=%s" % (cloth_ref.size, None if cloth_ref_alpha is None else cloth_ref_alpha.size))
save_mask(cloth_ref_alpha, "m_cloth_ref_alpha")

subj = foreground_mask(result)
shared = clothing_mask(result, garment_type=gt, garment_extent=extent, subject_mask=subj)
save_mask(shared, "m_clothing_shared")
apply_mask = limit_mask_to_subject(shared, subj) if shared is not None else None
save_mask(apply_mask, "m_apply_limited")
apply_mask, kept = intersect_mask_with_garment_alpha(apply_mask, cloth_ref_alpha)
note("alpha-intersect kept=%s" % kept)
save_mask(apply_mask, "m_apply")

if apply_mask is not None:
    result = reapply_garment_detail(result, cloth_ref, apply_mask, pattern_lock=True, ref_alpha=cloth_ref_alpha)
    result.save(OUT / "s1_after_patternlock.png")
    note("s1 saved")
    result = protect_inverse_refit(result, base_img, apply_mask, feather_px=4)
    result.save(OUT / "s2_after_protect.png")
    note("s2 saved")
    pf = print_fidelity_score(result, cloth_ref, apply_mask)
    note("fidelity at s2: %.4f" % pf)

if result.mode == "RGBA":
    bg = Image.new("RGB", result.size, (255, 255, 255))
    bg.paste(result, mask=result.split()[-1])
    result = bg
else:
    result = result.convert("RGB")
sm = foreground_mask(result)
save_mask(sm, "m_fg_on_s2")
result = force_white_background(result, subject_mask=sm)
result.save(OUT / "s3_final.png")
note("s3 saved")

print("== debug end ==")
