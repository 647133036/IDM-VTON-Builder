# User Instruction Memory

This file records user instructions, preferences, and teachings for reference in future interactions.

## Format

### User Instruction Entry
User instruction entries should follow this format:

[User Instruction Summary]
- Date: [YYYY-MM-DD]
- Context: [Mentioned scenario or time]
- Instructions:
  - [Content of user teaching or instruction, described line by line]

### Project Knowledge Entry
Entries discovered by the Agent during task execution should follow this format:

[Project Knowledge Summary]
- Date: [YYYY-MM-DD]
- Context: Discovered by Agent while performing [specific task description]
- Category: [Operations & Deployment|Build Methods|Testing Methods|Troubleshooting & Debugging|Workflow & Collaboration|Environment Configuration]
- Instructions:
  - [Specific knowledge points, described line by line]

## Deduplication Strategy
- Before adding a new entry, check for similar or identical instructions.
- If a duplicate is found, skip the new entry or merge it with the existing one.
- When merging, update the context or date information.
- This helps avoid redundant entries and keeps the memory file tidy.

## Entries

[Project Knowledge Summary]
- Date: 2026-10-27
- Context: Discovered by Agent while closing the second-round security review
- Category: Workflow & Collaboration
- Instructions:
  - Before pushing, all four checks must be green, with a baseline of 99 pytest +
    11 vitest and a clean ruff run:
      * `python3 -m ruff check backend/ training/` (from the repo root)
      * `python3 -m ruff format --check backend/ training/` (from the repo root)
      * `python3 -m pytest tests/ -q` (run from the `backend/` directory)
      * `npm test` (from the repo root; needs `npm install` first)
  - `npm run build` (from the repo root) MUST also pass before pushing: it is the only
    check that catches JS module syntax errors. `npm test` only covers `src/i18n.js`,
    not `src/main.js`, so a stray paren in `main.js` will not fail `npm test` but will
    blank out the SPA (vite dev server logs it, but nothing surfaces that log).
  - `ruff` only sees the repo layout when both `backend/` and `training/` are passed as
    explicit path arguments from the repo root; `training/` does not exist under `backend/`.
  - Security fixes must ship with a regression test that fails before the fix is applied
    (requestor requirement); add it to `backend/tests/test_api.py` unless the layer differs.

[Project Knowledge Summary]
- Date: 2026-10-28
- Context: Discovered by Agent while debugging IP-Adapter on the RTX 4090 (diffusers 0.42.0, SDXL ControlNet pipeline)
- Category: Troubleshooting & Debugging
- Instructions:
  - IP-Adapter Plus (`ip-adapter-plus_sdxl_vit-h`) is incompatible with diffusers 0.42:
    the plus `image_proj` returns a tuple the attention processor cannot consume, and
    plus needs last_hidden (patch) features while the pipeline's `encode_image` uses
    pooled embeddings. Use the NON-plus `ip-adapter_sdxl_vit-h` instead; its `image_proj`
    is a simple MLP (`image_proj.norm`+`image_proj.proj`), works with the h94 `models/
    image_encoder` (hidden 1280, proj 1024).
  - `enable_attention_slicing()` MUST NOT be called when IP-Adapter is loaded:
    `SlicedAttnProcessor` replaces the IPAdapter processors and crashes with
    `AttributeError: 'tuple' object has no attribute 'shape'` because `unet_2d_condition`
    (ip_image_proj branch) passes `encoder_hidden_states` as a `(text, image)` tuple.
  - `run_test.sh`/`run_app.sh` need `export HF_ENDPOINT=https://hf-mirror.com` first, and
    all local component loads use `local_files_only=True` to avoid HF network hangs.
  - The IP-Adapter weight dir is `app/models/sdxl/ip_adapter/`: weight file plus an
    `image_encoder/` (config.json + model.safetensors) must both exist for it to load.

[Project Knowledge Summary]
- Date: 2026-10-31
- Context: Discovered by Agent while writing tests for v0.1.4 _detect_face_mask
- Category: Troubleshooting & Debugging
- Instructions:
  - cv2 5.0.0 removed `CascadeClassifier` and `CASCADE_SCALE_IMAGE` from the main
    namespace. `_detect_face_mask` in `platform_presets.py` silently fails (broad
    `except Exception` catches `AttributeError`) and returns all zeros.
  - This means `clothing_mask`'s face subtraction was dead code — the face was never
    excluded from the clothing replacement region.
  - Fix applied in `clothing_mask`: when `_detect_face_mask` returns all zeros,
    estimate face as top 18% of subject bbox and subtract from cloth mask.
  - Tests that monkeypatch cv2 face detection must use `raising=False` on
    `monkeypatch.setattr(cv2, "CascadeClassifier", ...)` and also patch
    `cv2.CASCADE_SCALE_IMAGE` (both absent in cv2 5.0).
  - cv2 5.0 has `FaceDetectorYN`/`FaceDetectorYN_create` (YuNet DNN detector) as
    the modern replacement, but it requires a separate ONNX model download.

[Project Knowledge Summary]
- Date: 2026-11-01
- Context: Discovered by Agent while fixing v0.1.4 胸口白块 + 手臂变白 defects
- Category: Troubleshooting & Debugging
- Instructions:
  - `force_white_background` had the SAME cv2 5.0 bug as `clothing_mask`:
    `_detect_face_mask` returns all zeros, so `face_guard` was dead code.
    Fixed by adding `_heuristic_face_mask` helper (subject bbox top 18%) used
    by both `clothing_mask` and `force_white_background` when detection fails.
  - `force_white_background` was called WITHOUT `subject_mask` in
    `engine_sdxl.py:compose_tryon_sdxl` — light chest pixels connected to
    background through clothing edges got pressed to white (胸口白块 root cause).
    Fixed by computing `foreground_mask(human_img)` once and passing to both
    `clothing_mask` and `force_white_background`.
  - OpenCV `cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)` returns channels in order
    [Y, Cr, Cb] — NOT [Y, Cb, Cr]. Swapping these breaks skin detection silently.
  - `reapply_garment_detail` applies texture AND color to the entire mask=255
    area including arm skin (手臂变白 root cause). Fixed by detecting skin via
    YCbCr (Cb 77-127, Cr 133-173) and reducing tex_a/col_a by 85% on skin pixels.
  - `reapply_garment_detail` final blend `out = corrected_rgb * col_a + gen * (1-col_a)`
    means `color_strength=0` discards texture too (col_a=0 → out=gen). Tests
    must use non-zero color_strength to verify texture effects.
  - `clothing_mask` now excludes arm skin for short/sleeveless: YCbCr skin in
    the outer 25% of the *subject* bbox (not image width) is subtracted from
    the mask. Long sleeve keeps arm skin in the mask so generated sleeves cover it.
  - `_seam_repair` uses `cv2.inpaint(..., INPAINT_TELEA)` on the mask boundary
    band (dilate-erode, grow_px=8). Telea only fills pixels within `radius` of
    known pixels; tests that compare changed-pixel counts must set radius larger
    than the band width.
  - `_sdxl_inpaint_repair` is gated by `VTON_INPAINT_REPAIR=1` and requires
    `models/sdxl/sdxl_inpaint/model_index.json`. Without the model it is a no-op.
    Official weight: `stable-diffusion-xl-1.0-inpainting-0.1` (~7GB).

[Project Knowledge Summary]
- Date: 2026-11-01
- Context: Discovered by Agent while implementing DualFit-style pattern lock
- Category: Troubleshooting & Debugging
- Instructions:
  - `VTON_PATTERN_LOCK=1` (default) overlays warped `cloth_ref` onto the garment
    mask. PLTON-inspired frequency separation: `shade_full = clamp(L_gen/L_ref,
    0.75, 1.25)`; `lit = LF_ref * shade_full + HF_ref * shade_half` where
    `shade_half` scales deviation by `VTON_PATTERN_LOCK_HF_SHADE` (default 0.5).
    LF (base color) gets full lighting modulation; HF (pattern/logo) gets half,
    so shadows darken base color but pattern contrast survives.
  - Pattern lock env vars: `VTON_PATTERN_LOCK_LIGHT_STRENGTH` (1.0, total
    lighting dial: 0=exact ref, 1=full), `VTON_PATTERN_LOCK_LF_STRENGTH` (1.0),
    `VTON_PATTERN_LOCK_HF_STRENGTH` (0.5), `VTON_PATTERN_LOCK_LF_BLUR_SIGMA`
    (8.0), `VTON_PATTERN_LOCK_LIGHTING_MIN/MAX` (0.75/1.25, clamp on lighting
    ratio not on shade), `VTON_PATTERN_LOCK_SKIN_WEIGHT` (0.85),
    `VTON_PATTERN_LOCK_FEATHER_PX` (5), `VTON_PATTERN_LOCK_SKIP_WHITE_BG` (1),
    `VTON_PATTERN_LOCK_WHITE_THRESH` (245).
  - White pixels (RGB>245) connected to image border are skipped (canvas white);
    interior near-white still overlays.
  - Residual risks: seam feather zone HF discontinuity may break thin stripes
    at cuffs/collars; extreme body proportions (very long/short torso) still
    misalign since vertical warp uses statistical torso_box, not keypoints.
  - `VTON_WARP_VERTICAL` (default 1.0) controls vertical anchor alignment in
    `_warp_garment`: maps garment content (collar→hem from alpha) to torso box
    (neck→waist). 0=horizontal-only (v0.1.6 behavior), 1=full vertical align.
  - `align_garment_to_pose` now applies `GARMENT_BOX` before warp so dress/
    lower/onesie target the correct vertical body region (dress h=1.55→knee,
    lower y=0.92→hip). Previously only `_composite_engine` applied it.
  - `_detect_skin` uses morphological open+close (5x5 ellipse kernel) to remove
    scatter noise and fill holes. Tests must use >= 6x6 skin regions (smaller
    ones get eroded by the open operation).
  - `_correct_skin_tone` excludes face region (foreground bbox top 18%) from
    both shift statistics and correction application. Only arm/neck skin is
    corrected; face stays untouched to avoid unnatural color shifts.
  - `_sdxl_inpaint_repair` v0.1.7: focused mask (_build_inpaint_mask = seam band
    + skin anomaly vs original), pipeline singleton (_INPAINT_PIPE +
    _INPAINT_LOCK, no per-call from_pretrained/del), `original` param for skin
    anomaly detection. Still requires VTON_INPAINT_REPAIR=1 + model download.
  - `_estimate_waist_ratio` detects waist from subject_mask row-width profile
    (torso 30%-75% height, min width via valid-mode convolve). Falls back to
    0.62 when no narrowing (min/max > 0.85). clothing_mask cut ratios changed
    from fixed values to `waist_r + offset` preserving original deltas.
  - `VTON_PATTERN_LOCK=0` keeps the old high-frequency softlight + Reinhard path.
  - PLTON / PhysDiff-VTON style high-frequency injection during denoising is not
    available on this SDXL ControlNet pipeline; do not claim generation-time
    pattern constraints.

[User Instruction Summary]
- Date: 2026-11-06
- Context: GPU 服务器(AutoDL)切换为"无卡开机"(CPU-only)模式期间的协作流程
- Category: Workflow & Collaboration
- Instructions:
  - 服务器可能处于无卡开机状态(无 GPU, nvidia-smi 报 No devices found), SSH/scp/CPU 推理仍可用
  - 先干完 CPU 能做的事: 本地修码+测试+推送、scp 同步、sha256 校验、CPU 上跑 detect/推断验证、备好重跑 harness
  - 需要 GPU 的任务(如 compose_tryon_sdxl 出图)明确告知用户, 等用户换有卡开机后再执行
