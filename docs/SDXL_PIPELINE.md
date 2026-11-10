# SDXL 商用合规 Pipeline

本文档说明如何使用 Stable Diffusion XL + ControlNet + IP-Adapter 构建商用合规的换衣 pipeline。

## 许可对比

| 组件 | 许可 | 商用 |
|------|------|------|
| Stable Diffusion XL | OpenRAIL-M | 允许 |
| ControlNet | OpenRAIL-M | 允许 |
| IP-Adapter | Apache 2.0 | 允许 |
| IDM-VTON（微调权重） | CC BY-NC-SA | 禁止 |

使用 SDXL pipeline 完全合规，无 NC 限制。

## 快速部署

### 1. 下载权重

```bash
bash scripts/download_sdxl_weights.sh
```

权重约 15-20GB。

### 2. 设置商用模式

```bash
export LICENSE_MODE=commercial
```

### 3. 验证引擎状态

```bash
curl http://localhost:8000/api/license
```

期望输出：

```json
{
  "license_mode": "commercial",
  "commercial_compliant": true,
  "current_engine": "sdxl-commercial"
}
```

## 架构对比

```
IDM-VTON pipeline:
  SDXL base (OpenRAIL-M) + 微调 UNet (CC BY-NC-SA) + ControlNet + IP-Adapter
  → 商用禁止（NC 限制来自微调 UNet）

SDXL pipeline:
  SDXL base (OpenRAIL-M) + ControlNet (OpenRAIL-M) + IP-Adapter (Apache 2.0)
  → 商用允许（全部组件可商用）
```

## 引擎选择逻辑

`backend/engine.py` 的 `compose_tryon()` 自动选择：

1. `LICENSE_MODE=research` → 优先 IDM-VTON（如果有 GPU + 权重）
2. `LICENSE_MODE=commercial` → 优先 SDXL pipeline（如果有 GPU + 权重）
3. 都不可用 → 回退合成引擎

## 性能参考

| 硬件 | IDM-VTON | SDXL Pipeline |
|------|----------|---------------|
| RTX 4090 | 3-5 秒 | 5-8 秒 |
| A100 | 2-3 秒 | 3-5 秒 |

SDXL pipeline 略慢，因为少了 IDM-VTON 的针对性优化。

## 效果差异

| 维度 | IDM-VTON | SDXL Pipeline |
|------|----------|---------------|
| 服装贴合度 | 高（微调优化） | 中等 |
| 身体结构保持 | 高 | 高（ControlNet） |
| 服装特征保留 | 高 | 中等（IP-Adapter） |
| 速度 | 快 | 略慢 |

SDXL pipeline 效果略逊于 IDM-VTON，但完全合规。

## 改进方向

LoRA / ControlNet 自训仍然有效，但印花失效的主因是扩散在潜空间「重画」衣服，不是缺一次微调。完整分层见当前工作区 `.monkeycode/specs/print-fidelity-commercial-roadmap/`。下面是写入商用管线的执行顺序。

## 印花保真商用技术路线

纯色能过、印花/logo/文字崩，是扩散试衣的结构性失效。VAE 8x 压缩几乎不影响纯色；印花、字母、细密图案是高频，一压缩一 warp 就被重新解释。IP-Adapter 只锁品类/廓形/底色；字母间距和拼写没有「把这块图贴过去」的算子。生产路径用自研几何 + 后处理承担底层细节，禁止把禁商用权重接进来「补这一路」。

### 商用双通路（对应学术结构，实现全部已在仓库里）

| 职责 | 学术对照（不加载） | 本管线 |
|------|-------------------|--------|
| 高层语义 | IP-Adapter / CLIP | `ip-adapter_sdxl_vit-h`（Apache 2.0） |
| 底层细节 | GarmentNet 拼自注意力 | `_warp_garment` + `pattern_lock` 频率分离 |
| 姿态 | DensePose 潜变量 | RTMPose + ControlNet OpenPose SDXL + Depth |
| 重绘范围 | 服装二值遮罩 | `clothing_mask` ∩ 生成主体 ∩ warp alpha |
| 像素级文字/大图案 | 像素空间生成 | Phase 2 确定性贴印，扩散只出 shading |

### 引擎优先级（`LICENSE_MODE=commercial`）

| 优先级 | 引擎 | 说明 |
|--------|------|------|
| 1 | FASHN VTON v1.5（`engine_fashn.py`） | Apache 2.0，像素空间 + TPS，无需 mask；`segmentation_free=True`，parser 注入零张量 stub |
| 2 | SDXL ControlNet Inpaint（`engine_sdxl.py`） | 衣身 hole 重绘兜底；Inpaint 权重缺失时全图 ControlNet + degraded 标记 |
| 3 | 仓库内合成引擎（`engine.py`） | 纯 PIL，无第三方权重，保真度有限 |

### 禁止进入 `LICENSE_MODE=commercial` 的加载集

IDM-VTON 微调 UNet / GarmentNet（CC BY-NC-SA）、DensePose、fashn-human-parser（NVIDIA SegFormer 权重，含商用限制 §3.3）、DS-VTON / POVNet 官方代码、任何 NC 或未获书面授权的权重。FASHN VTON v1.5 主模型与 `fashn_vton` 代码为 Apache 2.0，允许商用；DWPose / YOLOX 权重同为 Apache 2.0。OpenPose 原版人体估计（CC BY-NC-SA）不作为生产检测器。

姿态条件图继续用已获商用授权的 ControlNet OpenPose SDXL 权重；人体关键点用 RTMPose（Apache 2.0）。OpenPose 原版人体估计（CC BY-NC-SA）不作为生产检测器。

### 三种崩法，三种修法

| 崩法 | 主修 |
|------|------|
| 文字 / logo 拼写变形 | 确定性贴印 + OCR 逐字比对 |
| 图案拉伸、错位、重复 | 参考图分辨率门闩 + TPS/光流升级 |
| 印花边缘糊、和衣身不对齐 | 遮罩收紧 + Telea / 印花 ROI inpaint |

### 按成本的处理顺序

**Phase 0（不重训，先做）**

- 平铺图长边保持到不低于输出长边（默认 2048），禁止预处理先缩到 512 再 warp；低层通路能拿到的细节由这张图决定
- 遮罩收紧到服装本体（已有 apply_mask 求交继续作为硬约束）
- 印花/文字款多 seed 出图，按质检分排序；纯色可单 seed

**Phase 1（结构，复用已落地 inpaint）**

- 第一段：现有 SDXL 只管版型、贴合、光影（步数可降到 20）
- 第二段：高分辨率只对印花 ROI 做 `_sdxl_inpaint_repair`，参考为 warp 后的 SKU 图；接缝带 ∪ 印花 ROI，管线单例，身份锁不变

**Phase 2（pixel-accurate 的支点）**

- 大图案和文字用确定性几何投到人体：在 `_warp_garment` 上升级 OpenCV TPS 或 Farneback 光流，控制点来自 RTMPose 与衣身轮廓
- 光影从生成图低频取回，按现有 `lit = LF_ref * shade_lf + HF_ref * shade_hf` 相乘
- 该区域 RGB 来源记 `print_source=sku_warp`；失败回退现有网格 warp 并标复核档

**Phase 3（质检，把筛选变成实测）**

- 文字款：裁印花区 → 自有 OCR（缺省 Tesseract / PaddleOCR，均为 Apache 2.0）→ 与平铺图逐字比 → 不符低分踢回重跑（默认最多 3 seed）
- 图案款：warp 参考 vs 结果做 SSIM
- job 写入 `quality_score` 与 `delivery_tier=auto|review`

### 交付档位（文案必须跟路线走）

- **自动出图**：纯色、小胸标、规则重复印花，且质检分过线
- **含人工复核/精修**：大面积图案、成段文字、不规则印花

对外「pixel-accurate / actual SKU」只覆盖自动出图档。无条件担保印花款会把交付质量写成英文语境下的无条件承诺。

规格正文：当前工作区 `.monkeycode/specs/print-fidelity-commercial-roadmap/requirements.md` 与 `design.md`。

## 实操调试小贴士

### 1. ControlNet 双路控制

只用 IP-Adapter 换衣服，可能会把原平铺图的背景也带过去，或者衣服位置不对。

必须用 ControlNet 双路控制：

- **OpenPose**（从模特底图提取）— 锁定姿势，防止身体变形
- **Canny**（从平铺图提取）— 提取服装边缘，保持形状

```python
control_pose = _get_openpose_control(human_img)
control_canny = _get_canny_control(cloth_img)

pipe(
    control_image=[control_pose, control_canny],
    controlnet_conditioning_scale=[0.8, 0.6],  # 调整权重
)
```

### 2. 童装比例修复

SDXL 生成童装时，容易生成"大头娃娃"。

已在 `AUDIENCE_PROMPTS` 和 `AUDIENCE_NEGATIVE` 中针对童装优化：

```python
# 正向 prompt
"girls": "a child model wearing this garment, correct proportions, realistic body"

# 负向 prompt
"girls": "adult body, distorted proportions, big head, small body"
```

### 3. 背景处理

客户要求纯色底时：

- **White Background** — 自动合成纯白底
- **Transparent** — 使用 Rembg 一键去背景

```python
if background == "transparent":
    result = _remove_background(result)  # 使用 rembg
elif background == "white":
    # 合成纯白底
```

Rembg 是开源工具，效果较好，无需复杂抠图代码。

### 4. 权重调优

ControlNet 权重需要调整：

```python
# OpenPose 权重高，保持身体结构
# Canny 权重低，只保留服装边缘
controlnet_conditioning_scale = [0.8, 0.6]
```

如果衣服"飘"在空中，提高 Canny 权重到 0.8。

如果身体变形，提高 OpenPose 权重到 0.9。

### 5. 生成参数

```python
num_inference_steps = 30  # 质量优先
guidance_scale = 7.5  # 适中
```

速度优先可降到 20 步。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LICENSE_MODE` | `unset` | `research` 或 `commercial` |
| `SDXL_MODEL_DIR` | `models/sdxl` | SDXL 权重目录 |

## 常见问题

### 效果不如 IDM-VTON

这是预期的。IDM-VTON 的微调 UNet 专门针对换衣优化，SDXL pipeline 用的是通用组件。如需提升效果，需要自行训练微调模型。

### 速度慢

- 确保使用 `torch.float16` 精度
- 降低 `num_inference_steps`（默认 30，可降到 20）
- 启用 `pipe.enable_attention_slicing()`

### 权重下载失败

检查 Hugging Face 访问权限。大陆用户可配置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_sdxl_weights.sh
```
