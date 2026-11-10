# Requirements Document

## Introduction

印花/字母保真当前卡在生成端：SDXL 全图重绘会改写遮罩外的发丝、胸口、手臂，1024 潜空间把字母高频重解释一遍。本规格把 FASHN VTON v1.5（Apache-2.0，像素空间 TPS + 扩散）定为重绘部分的主引擎，并把后处理从「遮罩内贴回 + 遮罩外锁身份」升级为「遮罩外贴回 + 遮罩内确定性贴印 + 遮罩外保护」，让未重绘区域（脸、发丝、手臂、胸口背景）保持源图像素，遮罩内的印花/字母由 SKU 平铺图像素直接构成，不再经扩散记字。

## Glossary

- **System**: Atelier 商用换衣管线（`LICENSE_MODE=commercial`）。
- **重绘区域 / hole**: 需要扩散生成的区域 = `clothing_mask` ∩ 生成主体 ∩ warp alpha，收紧到衣身本体。
- **保护区域 / inverse mask**: 源图像素贴回区域 = 全图 − 重绘区域（人脸、发丝、手、露肤手臂、背景）。
- **确定性贴印 / refit**: 用几何变换（网格 remap / TPS / 光流）把 SKU 平铺图投到人体表面，字母与图案由源图像素构成。
- **高频印花**: logo、字母、大面积图案、细密重复印花。
- **FASHN VTON v1.5**: 像素空间 TPS 几何对齐 + 人体解析 + 扩散生成，约 972M 参数，Apache 2.0，`segmentation_free=True`。
- **多 seed 排序**: 同一任务用不同 seed 出图，按质检分排序取最高。
- **质检分**: OCR（文字类）或 SSIM（图案类）对印花 ROI 的打分。

## Requirements

### Requirement 1: 重绘部分用 FASHN VTON v1.5

**User Story:** AS 管线负责人, I want 把扩散重绘从 SDXL 全图切换到像素空间的 FASHN, so that 印花保真问题从潜空间挪到像素空间，重绘区域更小、高频损失更少。

#### Acceptance Criteria

1. WHEN `LICENSE_MODE=commercial` 且 FASHN 权重在位, the System SHALL 优先调用 `engine_fashn.compose_tryon_fashn`，FASHN 运行时异常才降级 SDXL ControlNet Inpaint。
2. WHEN 调用 FASHN, the System SHALL 保持 `segmentation_free=True`、`garment_photo_type="flat-lay"`、`fashn-human-parser` 零张量 stub（NVIDIA SegFormer 权重永不下载）。
3. WHEN FASHN 权重缺失或无 GPU, the System SHALL 走 SDXL ControlNet Inpaint 兜底路径，并在 job 元数据标记 `engine_degraded`。
4. WHEN 设计重绘范围, the System SHALL 把扩散生成限定在收紧后的 `clothing_mask` ∩ 生成主体 ∩ warp alpha 区域内，遮罩外不发扩散。

### Requirement 2: 遮罩外保护贴回

**User Story:** AS 品牌商, I want 脸、发丝、手臂、胸口背景保持源图像素, so that 手臂不会变白、胸口不会变白、发丝不会糊。

#### Acceptance Criteria

1. WHEN 生成完成, the System SHALL 计算保护区域 = 全图 − 重绘区域（含膨胀羽化带）。
2. WHEN 保护区域存在, the System SHALL 把源图（human_img）对应像素贴回结果图，仅重绘区域内保留生成像素。
3. WHEN 贴回跨越重绘边界, the System SHALL 用 cv2.inpaint Telea 在边界带平滑过渡，消除色阶断层。
4. WHEN `VTON_LOCK_ID=0`, the System SHALL 跳过保护贴回，保留全生成帧（用于对比实验）。
5. WHEN 保护贴回运行, the System SHALL 保持 `force_white_background` 主体保护掩膜 = 源图主体 ∪ 生成图主体，避免压白误伤。

### Requirement 3: 遮罩内确定性贴印（refit）

**User Story:** AS 品牌商, I want 字母与图案由 SKU 平铺图像素构成, so that 拼写、间距、图案排布不随扩散漂。

#### Acceptance Criteria

1. WHEN SKU 含高频印花或文字, the System SHALL 用几何变换把 SKU 平铺图（长边 ≥ 2048）投到人体表面，控制点来自 RTMPose 关键点与服装轮廓。
2. WHEN 贴印完成, the System SHALL 在重绘区域内用贴印结果替换生成像素，重绘结果只贡献低频光影与褶皱，高频由贴印像素构成。
3. WHEN 贴印使用网格 remap, the System SHALL 把网格升级到 TPS 或 OpenCV 光流，现有 `_warp_garment` 作为回退。
4. WHEN 关键点或轮廓失败, the System SHALL 回退到现有网格 remap + `pattern_lock`，并把任务标为复核档。
5. WHEN 贴印参考图分辨率, the System SHALL 使用未缩到 1024 的 SKU 平铺图（长边跟输出 2048），禁止先缩到 1024 再 warp。

### Requirement 4: 多 seed 排序

**User Story:** AS 运营人员, I want 印花款多 seed 出图按质检分取最高, so that 单次生成的随机漂字被筛掉。

#### Acceptance Criteria

1. WHEN SKU 被分为含印花/文字档, the System SHALL 用 3 个不同 seed 跑同一任务（默认 `VTON_MULTI_SEED=3`，0 或 1 时单 seed）。
2. WHEN 多 seed 完成, the System SHALL 按质检分（OCR 逐字比对 或 SSIM）排序，只暴露最高分结果。
3. WHEN 纯色 SKU, the System SHALL 单 seed 以控制成本。
4. WHEN 排序完成, the System SHALL 把各 seed 质检分与最终选择写入 job 元数据。

### Requirement 5: 参考图长边跟 2048

**User Story:** AS 管线工程师, I want 贴印/ warp / IP-Adapter 参考图保持源图长边, so that 高频印花不被 1024 压缩丢一次。

#### Acceptance Criteria

1. WHEN 预处理 SKU 平铺图, the System SHALL 把长边保持到不低于输出长边（默认 2048），低层 warp / 贴印 / IP-Adapter 参考共用同一张未额外缩小的图。
2. WHEN 生成参考锁 1024（SDXL 原生）, the System SHALL 仅对 IP-Adapter 语义路缩到 1024，贴印路保持 2048。
3. WHEN 输出缩放, the System SHALL 把结果长边缩回 `long_side`（默认 2048），贴印已在高分辨率完成，缩放只损失一次。

### Requirement 6: 质检分档 auto / review

**User Story:** AS 质检负责人, I want 出图后自动核对文字与图案, so that quality screening 有实测分。

#### Acceptance Criteria

1. WHEN 出图完成且 SKU 含文字, the System SHALL 裁印花 ROI，用 Apache 2.0 的 Tesseract 或 PaddleOCR 识别，并与平铺图 OCR 逐字比对，不一致触发下一 seed 重跑（默认上限 3）。
2. WHEN SKU 为图案类, the System SHALL 用 SSIM 比较贴印参考与结果 ROI，低于阈值进复核档。
3. WHEN 质检分可用, the System SHALL 把分写入 job 元数据，标记 `delivery_tier=auto` 或 `review`，review 须人工确认后才推店。

### Requirement 7: 与现有管线兼容

**User Story:** AS 维护者, I want 新管线挂在现有 compose 流程上, so that 身份锁、腰线、纯白底、商用门禁继续生效。

#### Acceptance Criteria

1. WHEN 保护贴回运行, the System SHALL 保持 `LICENSE_MODE=commercial` 门禁、`force_white_background`、RTMPose 腰线插值、输出长边 ≥ 2048。
2. WHEN `VTON_PATTERN_LOCK=0`, the System SHALL 仍允许遮罩收紧与保护贴回生效，确定性贴印关闭。
3. WHEN FASHN 不可用且 GPU 可用, the System SHALL 在 SDXL Inpaint 上跑保护贴回 + 贴印，遮罩外仍贴源图。
4. WHEN GPU 不可用, the System SHALL 在合成引擎上只跑几何贴印 + 保护贴回，job 标有限质量。
