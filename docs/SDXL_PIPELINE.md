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

如需提升 SDXL pipeline 效果：

1. **训练自定义 ControlNet** — 用自有服装数据微调
2. **训练自定义 IP-Adapter** — 针对服装风格优化
3. **增加后处理** — 色彩校正、边缘融合

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
controlnet_conditioning_scale=[0.8, 0.6]
```

如果衣服"飘"在空中，提高 Canny 权重到 0.8。

如果身体变形，提高 OpenPose 权重到 0.9。

### 5. 生成参数

```python
num_inference_steps=30  # 质量优先
guidance_scale=7.5      # 适中
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
