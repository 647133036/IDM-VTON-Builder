# 需求实施计划

- [x] 1. FASHN VTON v1.5 作为商用主引擎（用户覆盖原「禁 FASHN」规格）
  - [x] 1.1 `backend/engine_fashn.py`：`_ensure_parser_stub()` 向 `sys.modules` 注入零张量 `FashnHumanParser`，NVIDIA SegFormer 权重永不下载；`_patch_loaded_pipeline` 同步替换已加载的 `fashn_vton.pipeline` 引用
  - [x] 1.2 `fashn_available()`：校验 `models/fashn/model.safetensors` + `dwpose/yolox_l.onnx` + `dwpose/dw-ll_ucoco_384.onnx` + `fashn_vton` 已安装 + torch + GPU
  - [x] 1.3 `compose_tryon_fashn()`：调用 `TryOnPipeline(weights_dir=models/fashn)`，强制 `segmentation_free=True`、`garment_photo_type="flat-lay"`；后处理 white bg + resize + `pattern_lock`（`VTON_PATTERN_LOCK=1`）
  - [x] 1.4 `engine.py` 商用分支：FASHN 优先，运行时异常降级 SDXL，再降级合成引擎
  - [x] 1.5 `engine_status()` 新增 `fashn_available`、`fashn_reason`、`sdxl_inpaint`、`fashn_dir`；`/api/license` 暴露 `fashn_available`、`sdxl_inpaint`；`/api/health` 隐藏 `fashn_dir`
  - [x] 1.6 `scripts/download_weights.sh` 新增 `fashn` 组（fashn-vton-1.5 model.safetensors + DWPose ONNX，不拉 human parser）
  - [x] 1.7 测试：`tests/test_engine_fashn.py`（category 映射、parser stub、缺权重、`compose_tryon_fashn` segmentation_free 契约）+ `tests/test_engine.py` 引擎优先级/降级
  - [x] 1.8 文档同步：`docs/LICENSE_COMPLIANCE.md`、`docs/SDXL_PIPELINE.md`、`README.md` 引擎表、本 spec 的 FASHN 禁令改为「Apache-2.0 允许，parser 仍禁」

- [x] 2. SDXL ControlNet Inpaint 衣身 hole 兜底
  - [x] 2.1 `_get_sdxl_pipe` 在 `models/sdxl/sdxl_inpaint/model_index.json` 存在且 `StableDiffusionXLControlNetInpaintPipeline` 可导入时从 inpaint 目录加载，`_sdxl_state["inpaint_mode"]=True`；权重缺失或类不存在时回退 base ControlNet，`inpaint_mode=False`
  - [x] 2.2 `compose_tryon_sdxl` 在 warp 之后、采样之前用 `clothing_mask(human_img)` ∩ warp alpha（`cloth_ref_alpha`）作为 hole；hole 均值 < 4.0 或异常时退回全图 ControlNet 并写 `degraded_reason`
  - [x] 2.3 `_build_sdxl_infer_kwargs`：inpaint 时 `image=init_image` / `mask_image=hole` / `control_image=control_images` / `strength`（`VTON_INPAINT_STRENGTH` 默认 0.99）；全图模式 `image=control_images`
  - [x] 2.4 `arm_cols` 按 `warp_canvas / target_w` 缩放到目标画布；`ip_ref` 从高分辨率 warp canvas resize 到 1024 供 IP-Adapter
  - [x] 2.5 测试：`test_engine_sdxl.py` 新增 `_build_sdxl_infer_kwargs` 单测、inpaint 权重存在时 from_pretrained 指向 `sdxl_inpaint`、`_inpaint_weights_ready`、`engine_status` 含 `sdxl_inpaint`；`test_engine_concurrency.py` 同步 InpaintPipeline stub

- [x] 3. 回归
  - [x] 3.1 `ruff check backend/ training/` 全绿
  - [x] 3.2 `pytest tests/ -q` 全绿（基线 99+，新增 FASHN/Inpaint 用例）
