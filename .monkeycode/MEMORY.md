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
- Date: 2026-11-09
- Context: GPU 服务器(AutoDL)按开机时长计费(用不用都收费); 用户对"按需多次往返 SSH 拉取产物导致实例空转计费"强烈不满
- Category: Workflow & Collaboration
- Instructions:
  - AutoDL 实例开机即计费(含无卡开机, 但费率低): 每轮 GPU 推理完成后, 必须 ONE GO 用一条
    tar+scp 命令批量拉取全部产物(结果图、c*_run.log、c*_debug/、run_cX.py、输入), 禁止按需多次往返拉取
  - 拉取完成后立即提醒用户关机停止计费; AutoDL 关机保留磁盘数据, 漏拉的日志/debug 可在下次开机时补拉,
    不值得为补拉让实例空转计费
  - 服务器可能处于无卡开机状态(无 GPU, nvidia-smi 报 No devices found), SSH/scp/CPU 推理仍可用
  - 先干完 CPU 能做的事: 本地修码+测试+推送、scp 同步、sha256 校验、CPU 上跑 detect/推断验证、备好重跑 harness
  - 需要 GPU 的任务(如 compose_tryon_sdxl 出图)明确告知用户, 等用户换有卡开机后再执行

[Project Knowledge Summary]
- Date: 2026-11-10
- Context: Discovered by Agent while testing FASHN-first commercial engine path
- Category: Testing Methods
- Instructions:
  - `engine_sdxl._get_sdxl_pipe` 内部调用模块级 `engine_status()`(depth 降级检查),
    测试中必须同时 `monkeypatch.setattr(engine_sdxl, "engine_status", lambda force=False: {})`
    和 `monkeypatch.setattr(engine_sdxl, "_engine_status_cache", None)`,否则进程缓存的
    真实 `engine_status` 结果会让 depth 降级分支干扰断言。
  - 测试中构造 fake SDXL 权重目录时, `*_safetensors` 文件写 `b"x"` 即可(文件存在性检查),
    但 IP-Adapter 维度推断测试必须用真 `safetensors.torch.save_file` 写合法张量
    (`_infer_ip_adapter_input_dim` 会 `safe_open` 读 shape)。
  - `models/sdxl/lora/` 下有真实 LoRA 权重(`adapter_model.safetensors`),
    `_get_sdxl_pipe` 的 lora_install_failed 复检(`lora_weight_path`)会命中;
    测试中 mock `_load_trained_lora` 后, `lora_install_failed` 复检走真实 `lora_io`,
    若 tmp_path 无 lora 目录则返回 None → 不误标降级。
  - FASHN 商用路径: `engine_fashn._ensure_parser_stub()` 注入 `sys.modules["fashn_human_parser"]`
    零张量 stub; `_patch_loaded_pipeline` 同步替换已加载的 `fashn_vton.pipeline` 中
    `FashnHumanParser` 与 `CATEGORY_TO_BODY_COVERAGE` 引用, 确保 NVIDIA SegFormer 权重
    永不下载。测试 `test_engine_fashn.py` 验证 stub 标志 `_atelier_commercial_stub=True`。

[Project Knowledge Summary]
- Date: 2026-11-10
- Context: Discovered by Agent while implementing print-preservation refit pipeline
- Category: Workflow & Collaboration
- Instructions:
  - 商用主路径保护贴回: `protect_inverse_refit(result, source, apply_mask, feather_px)` 替代
    旧 `blend_tryon` + `_seam_repair` 两步。遮罩外贴源图, 遮罩内保留生成像素,
    边界带用 cv2.inpaint TELEA 平滑。`VTON_PROTECT_FEATHER`(默认 4) 控制边界带宽度。
  - 确定性贴印升级: `VTON_REFIT_ENGINE` 选 `grid`(默认)/`tps`/`optical_flow`。
    `tps_warp` 用 RBF thin-plate-spline 把 SKU 平铺图投到人体;
    `optical_flow_refit` 在 TPS 结果上叠加 Farneback 光流微移(×0.5 保守系数)。
    贴印在 `reapply_garment_detail`(pattern_lock) 之前执行, 让 pattern_lock 用更准的贴印像素。
  - 多 seed 排序: `VTON_MULTI_SEED`(默认 3, 纯色=1)。`worker.py` 调 `run_multiseed`,
    按 `quality_gate.score_for_sku`(SSIM/OCR) 排序取最高, 写 `job.multiseed` 元数据。
  - 质检分档: `quality_gate.delivery_tier` 按 `VTON_QG_SSIM`(0.70) / `VTON_QG_OCR`(0.80)
    分 `auto`/`review`。scikit-image 不可用时 `ssim_score` 回退到 numpy 灰度均值/方差 SSIM。
  - `engine_status()` 新增 `refit_engine` 字段; `engine.py` 商用分支读 `VTON_REFIT_ENGINE`。
  - 测试 `test_preserve_refit.py` 22 用例覆盖 protect_inverse_refit / tps_warp /
    optical_flow_refit / run_multiseed / quality_gate。全量回归 378 passed。

[Project Knowledge Summary]
- Date: 2026-11-10
- Context: Discovered by Agent while implementing print-ROI inpaint + fidelity tier
- Category: Workflow & Collaboration
- Instructions:
  - 印花 ROI 局部 inpaint: `VTON_PRINT_INPAINT=1` 时 `detect_print_roi(cloth_ref, apply_mask)`
    取贴回区内高频区(印花/文字边缘)的 dilate-erode 边界带; `_sdxl_inpaint_repair`
    门禁为 `VTON_INPAINT_REPAIR=1 OR VTON_PRINT_INPAINT=1`, 边界带并入 `_build_inpaint_mask`
    的 `print_roi` 参数(bitwise_or)。字母/图案内部像素保持确定性贴印, 不重绘。
  - 印花保真分: `quality_gate.print_fidelity_score(result, cloth_ref, apply_mask)` 用贴回区内
    warp 参考 vs 结果的灰度 NCC 映射 `(ncc+1)/2`; 双方灰度 std<6(纯色款)返回 1.0。
    engine_sdxl/engine_fashn 写入 `result.info["print_fidelity"]` + `delivery_tier`,
    worker 落库 job 元数据; 多 seed scorer 首选 print_fidelity(跨 seed 可比)。
  - 测试注意: Telea/dilate-erode 边界带的逐像素位置对输入敏感, 断言要用
    "边缘附近有带"(or 组合) + "块中心无带", 不要断言精确行号。
  - worker 多 seed 测试用 `monkeypatch.setattr(worker_mod, "compose_tryon", stub)` 替换
    模块级导入的 compose, stub 读 `os.environ["VTON_SEED"]` 按 seed 返回不同
    print_fidelity, 验证择优与 multiseed 元数据(含 scores dict)落库。

[Project Knowledge Summary]
- Date: 2026-11-10
- Context: Discovered by Agent while wiring print fidelity into CPU composite path
- Category: Workflow & Collaboration
- Instructions:
  - CPU composite 兜底路径(engine.py `_composite_engine`)同样写 print_fidelity/
    delivery_tier: 参考层=无阴影原始贴放层(ref_layer), mask=layer alpha。三引擎
    (FASHN/SDXL/composite)的 job 元数据字段一致。
  - `print_fidelity_score` 纯色判定只看参考侧 std<6(composite 接触阴影会抬高
    结果侧方差, 不算失真)。
  - 测试环境变量必须清理: `test_refit_seeds_default` 曾泄漏 VTON_MULTI_SEED=99
    使后续真 compose 测试跑 8 次; 用 try/finally pop 或 monkeypatch.setenv。
  - worker 读 VTON_MULTI_SEED 有 int() 护栏: 非法值回退 3, 上限封顶 8。
  - 测试用纯色方图做 SKU 时 composite 的 `garment_alpha` 返回全 0(无背景对比
    → 衣服不贴), 测试样图必须白底+色块(见 test_worker.py `_make_sku_png`)。
  - GPU 冒烟脚本: `scripts/gpu/smoke_print_pipeline.py`(服务器布局 /root/vton/repo
    + /root/autodl-tmp/vton_combo 样图), 一条命令验证 engine_status + FASHN 出图
    + 保真分档 + 多 seed 择优。

[Project Knowledge Summary]
- Date: 2026-11-13
- Context: Discovered by Agent while fixing GPU smoke r9 (fidelity 0.871 → 0.9663 PASS)
- Category: Troubleshooting & Debugging
- Instructions:
  - AutoDL 服务器 conda numpy 1.23.5 (OpenBLAS ILP64) 的 `np.linalg.solve`(LAPACK
    gesv) 在良态矩阵上返回垃圾: Bookstein 归一化 8 锚点 TPS 的 11x11 系统 cond 20,
    solve 残差 2.71175554767862(与冒烟日志逐位一致), 同库 `np.linalg.lstsq`(SVD
    路径) 残差 1.3e-15 正常。沙箱 numpy 2.4.6 两者都正常。结论: 服务器端数值求解
    必须带残差守卫 + lstsq 换道重试(已实装 preserve_refit.tps_warp), 全库唯一
    solve 调用点就是 tps_warp。r6-r8 GPU fidelity 卡 0.85 平台的根因即此(守卫
    触发后静默回退 LANCZOS, 印花贴在几何错位位置), 非算法问题。
  - 服务器脚本必须显式用 `/root/miniconda3/bin/python`: 裸 `python3` 是系统
    python, 无 numpy/torch, 冒烟秒退 ModuleNotFoundError。nohup setsid bash
    非登录 shell 不会加载 conda PATH, sed 替换脚本内 python3 为绝对路径是
    最稳做法(smoke_launch_261110.sh / collect_smoke_261110.sh 均已替换)。
  - 冒烟成败判据: 日志结论行 `print_fidelity >= 0.9 auto 档, PASS`; 结果以
    `result.info.print_fidelity` 为准。GPU 冒烟全程 4-6 分钟, 显存峰值 ~6-8G,
    用户看到"显存才用 6g 现已停"= 正常跑完(非失败), 先查日志再下结论。
  - 服务器 python 探针/TCP 往返偶发 SSH 节流(kex_exchange_identification):
    轮询间隔 ≥100s; 快速连续 SSH 会拿空响应, 误判成任务未完成。
  - 收集流程: 服务器侧 `collect_smoke_261110.sh` 打一个 tar(日志+结果+输入+环境),
    单 scp 拉回 /tmp/opencode/gpu_results/<round>/, 符合用户一次性拉取要求。
  - 冒烟脚本第二段 `VTON_MULTI_SEED=3` 直调 `engine.compose_tryon` 输出
    `multiseed: {}` 是结构使然(multiseed 由 worker.run_multiseed 写 info,
    引擎层不写), 非失败信号; PASS 只看单 seed 段。
