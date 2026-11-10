# Requirements Document

## Introduction

纯色换衣能过、印花/logo/文字崩，是扩散试衣的结构性失效：VAE 潜空间把高频细节压缩掉，扩散过程按语义「重画」衣服，而不是把 SKU 图搬到人体上。本规格把这条诊断固化为商用技术路线：在现有 SDXL + ControlNet + IP-Adapter 管线上，按成本分层修复印花保真，并把「pixel-accurate」承诺收进可兑现的 SKU 档位。

实现栈限定为已获商用授权或明确允许商用的组件，以及本仓库自研的 CPU 几何/后处理。学术论文只作为失效机理与分层思路的出处，权重、网络结构和第三方闭源/禁商用模型一律不进入生产路径。

## Glossary

- **System**: Atelier 商用换衣管线（`LICENSE_MODE=commercial` 下的 SDXL 引擎 + CPU 后处理）。
- **SKU 平铺图**: 商家上传的服装正拍/平铺/幽灵模特图，是印花与文字的唯一真值。
- **高频印花**: logo、字母、大面积图案、细密重复印花、缝合线等 VAE 8x 压缩后会丢失的细节。
- **低频版型**: 品类、廓形、领型、袖长、底色、光影褶皱。
- **遮罩**: `clothing_mask` 得到的服装重绘区域；仅该区域允许被生成或贴回改写。
- **确定性贴印**: 用几何变换（TPS / 光流 / 现有网格 warp 的升级）把 SKU 平铺图投到人体表面，再乘以生成图的光影，字母与图案由原图像素构成。
- **可商用组件**: 本仓库 README「授权说明」表中标注允许商用、且作者已获授权或许可证本身允许商用的模型与库。
- **自动出图档**: 纯色、小胸标、规则重复印花，质检通过后可直接发布。
- **复核档**: 大面积图案、成段文字、不规则印花，须质检打分或人工精修后发布。

## Requirements

### Requirement 1: 印花失效作为产品约束

**User Story:** AS 管线负责人, I want 把「纯色能过、印花崩」写成明确的质量约束, so that 后续改动按失效层次修，而不是当参数没调好来撞。

#### Acceptance Criteria

1. WHEN 商用引擎处理一件服装, the System SHALL 把输出质量按三层分别评估：低频版型（品类/廓形/领型/袖长/底色）、中频贴合（遮罩边界/褶皱光影）、高频印花（图案排布/logo/文字）。
2. WHEN 低频版型与参考平铺图一致而高频印花与参考不一致, the System SHALL 将该结果归类为印花保真失效，并走本规格的分层修复。
3. WHEN 诊断印花失效, the System SHALL 按三种可区分模式记录：文字/logo 拼写变形、图案拉伸错位或重复、印花边缘糊且与衣身轮廓不对齐。
4. WHEN 三种模式同时出现, the System SHALL 按「文字拼写 → 图案几何 → 边缘对齐」的优先级选择主修路径。

### Requirement 2: 商用实现边界

**User Story:** AS 产品负责人, I want 生产路径只使用可商用组件, so that 印花保真方案可以上线售卖。

#### Acceptance Criteria

1. WHEN `LICENSE_MODE=commercial`, the System SHALL 加载 FASHN VTON v1.5 主模型与 `fashn_vton` 代码（Apache 2.0，商用主引擎，`segmentation_free=True`）、FASHN DWPose / YOLOX 权重（Apache 2.0）、SDXL base、SDXL Inpainting、ControlNet（OpenPose / Canny / Depth）、IP-Adapter、RTMPose、Rembg/U2Net、MediaPipe 回退、自研 LoRA，以及 OpenCV/NumPy 自研后处理。fashn-human-parser（NVIDIA SegFormer 权重，含商用限制）SHALL 由 `engine_fashn._ensure_parser_stub()` 注入零张量 stub，权重永不下载。
2. WHEN 设计或实现印花通路, the System SHALL 用自研几何 warp + pattern_lock + 可选 SDXL Inpainting 局部重绘承担「底层细节搬运」，FASHN VTON 承担像素空间 TPS + 扩散生成，IP-Adapter 只承担高层语义（品类、廓形、整体识别度）。
3. WHEN 引用 IDM-VTON、DensePose、GarmentNet、fashn-human-parser（NVIDIA SegFormer 权重）、DS-VTON 代码、POVNet 代码或任何 CC BY-NC / 未获商用授权的权重, the System SHALL 把它们留在文档的「机理对照」节，运行时加载列表保持为空。FASHN VTON v1.5 主模型与 DWPose / YOLOX 权重为 Apache 2.0，允许商用。
4. WHEN 姿态条件需要人体结构, the System SHALL 使用 RTMPose 关键点 + ControlNet OpenPose SDXL 权重 + Depth ControlNet，姿态通路与 IDM-VTON 的 DensePose 潜变量解耦。

### Requirement 3: 零重训杠杆（Phase 0）

**User Story:** AS 运营人员, I want 不重训就能先抬印花上限, so that 现有 GPU 配方立刻少崩一批。

#### Acceptance Criteria

1. WHEN 预处理服装图, the System SHALL 把平铺图长边保持到不低于输出长边（默认 2048），低层 warp / pattern_lock / IP-Adapter 参考使用同一张未额外缩小的图。
2. WHEN 计算服装遮罩, the System SHALL 把重绘区域收紧到服装本体（现有 `clothing_mask` ∩ 生成主体 ∩ warp alpha），背景、脸、手、短袖露出的臂肤保持在遮罩外。
3. WHEN 提交印花或文字类 SKU, the System SHALL 支持同一任务多 seed 出图，并按质检分排序后只暴露最高分结果。
4. WHEN 任务被标为纯色自动出图档, the System SHALL 允许单 seed 以控制成本。

### Requirement 4: 两段式 coarse-to-fine（Phase 1）

**User Story:** AS 管线工程师, I want 先出版型再只重绘印花区, so that 低分辨率阶段丢掉的高频可以在高分辨率局部补回。

#### Acceptance Criteria

1. WHEN 启用两段式生成, the System SHALL 第一段只约束版型、贴合和光影（现有 SDXL + ControlNet + IP-Adapter，步数可低于终稿）。
2. WHEN 第一段完成且 SKU 被判为含印花/文字, the System SHALL 在高分辨率下仅对印花区域做局部 inpaint，参考图为 warp 后的 SKU 平铺图。
3. WHEN 局部 inpaint 运行, the System SHALL 复用已落地的 SDXL Inpainting 单例（`_sdxl_inpaint_repair` 的模型与锁），把聚焦遮罩从接缝带扩展为「接缝带 ∪ 印花 ROI」。
4. WHEN 第一段已给出可用光影, the System SHALL 保持人体身份锁（`blend_tryon`）与肤色校正不变，第二段只改印花 ROI 内像素。

### Requirement 5: 确定性印花投递（Phase 2）

**User Story:** AS 品牌商, I want 大图案和文字由原图像素构成, so that 字母数、间距、拼写不再随扩散漂。

#### Acceptance Criteria

1. WHEN SKU 被分为大面积图案或含文字, the System SHALL 用确定性几何把平铺图印花投到人体表面：在现有 `_warp_garment` 网格上升级为 TPS 或 OpenCV 光流，控制点来自 RTMPose 关键点与服装分割轮廓。
2. WHEN 印花已经投到人体, the System SHALL 从第一段生成结果提取低频光影与褶皱，按现有频率分离公式乘到投递结果上（LF 全强度、HF 半强度）。
3. WHEN 确定性投递完成, the System SHALL 把该区域的 RGB 来源记为 `sku_warp`，扩散输出在该区域只提供 shading。
4. WHEN 关键点或分割失败, the System SHALL 回退到现有 `_warp_garment` + `pattern_lock`，并把任务标为复核档。

### Requirement 6: 印花质检与 OCR 回环（Phase 3）

**User Story:** AS 质检负责人, I want 出图后自动核对文字和图案, so that quality screening 有实测分而不是口头承诺。

#### Acceptance Criteria

1. WHEN 出图完成且 SKU 含文字, the System SHALL 裁印花 ROI，用本产品自有 OCR（缺省时用 Apache 2.0 的 Tesseract 或 PaddleOCR）识别文字，并与平铺图 OCR 结果做逐字比对。
2. WHEN 识别文本与设计稿不一致, the System SHALL 给该结果低分并触发同任务下一 seed 重跑，重跑次数上限由设置给出（默认 3）。
3. WHEN SKU 为图案类（无文字）, the System SHALL 用结构相似度（SSIM 或等价）比较 warp 后参考与结果的印花 ROI，低于阈值的结果进入复核档。
4. WHEN 质检分可用, the System SHALL 把该分写入 job 元数据，供队列排序与「自动出图 / 复核」分流。

### Requirement 7: 交付档位与文案范围

**User Story:** AS 法务/增长, I want 「pixel-accurate」只覆盖能兑现的 SKU 类, so that 英文官网不会变成无条件质量担保。

#### Acceptance Criteria

1. WHEN 对外描述商用引擎能力, the System SHALL 把自动出图档限定为纯色、小胸标、规则重复印花。
2. WHEN 对外描述大面积图案或文字排版款, the System SHALL 将其归入「含人工复核/精修」档，并与自动出图档分开陈述。
3. WHEN 产品文案使用 pixel-accurate / actual SKU 表述, the System SHALL 同时给出档位条件（自动出图档，且质检分高于阈值）。
4. WHEN 任务完成, the System SHALL 在 job 上标记 `delivery_tier` 为 `auto` 或 `review`，下载与推送默认允许 auto；review 须人工确认后才能推店。

### Requirement 8: 与现有管线兼容

**User Story:** AS 维护者, I want 印花路线挂在现有 compose 流程上, so that 身份锁、腰线、纯白底、商用门禁继续生效。

#### Acceptance Criteria

1. WHEN 印花分层修复运行, the System SHALL 保持 `LICENSE_MODE=commercial` 门禁、身份锁、`force_white_background`、RTMPose 腰线插值与输出长边 ≥ 2048。
2. WHEN `VTON_PATTERN_LOCK=0`, the System SHALL 仍允许 Phase 0 的分辨率与遮罩收紧生效，确定性贴印关闭。
3. WHEN GPU 不可用, the System SHALL 在合成引擎上只跑几何 warp + 贴印回退，并把 job 标为有限质量。
