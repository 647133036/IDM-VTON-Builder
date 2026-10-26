# SDXL LoRA 训练工程（验证 pipeline）

用于在 GPU 环境（如魔塔 ModelScope Notebook 免费 GPU）上对整个 LoRA 训练链路做端到端验证。默认用仓库现有素材（`public/examples/cloth/` 成衣平铺图、`public/examples/human/` 人像图），产出可被 `backend/engine_sdxl.py` 加载的 LoRA 权重。

本工程完全不依赖 IDM-VTON（CC BY-NC-SA 禁止商用），只做 SDXL LoRA 微调，商用合规。

## 目录

```
training/
  prepare_dataset.py   # 把 public/examples 整理成 LoRA 训练集（图片 + caption）
  train_lora_sdxl.py   # 精简 SDXL LoRA 训练脚本（单卡，魔塔 Notebook 可跑）
  infer_lora.py        # 用训练好的 LoRA 出图验证
  run_train.sh         # 一键链路：prepare → train → infer
  requirements.txt
  README.md
```

## 在魔塔 Notebook 上运行

```bash
# 1. 拉取仓库
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder

# 2. 安装依赖（Notebook 通常已预装 torch/diffusers）
pip install -r training/requirements.txt

# 3. 一键训练（默认 25 张现成素材，约几分钟跑完）
bash training/run_train.sh
```

产物：
- LoRA 权重：`models/sdxl/lora/`（adapter_model.safetensors + adapter_config.json）
- 验证样图：`training/output/`（训练前 / 训练后对比）

## 在本地复现

`train_lora_sdxl.py` 自动检测设备：有 CUDA 用 fp16，否则退化为 CPU fp32（很慢，仅验证逻辑）。默认从魔搭下载 SDXL base：

```
AI-ModelScope/stable-diffusion-xl-base-1.0
```

也可传 `--base_model <本地目录>` 指向已下载的 `models/sdxl/sdxl_base`。

## 手动步骤

```bash
# 单独整理数据
python training/prepare_dataset.py \
  --cloth public/examples/cloth \
  --human public/examples/human \
  --out training/dataset

# 单独训练
python training/train_lora_sdxl.py \
  --dataset_dir training/dataset \
  --base_model AI-ModelScope/stable-diffusion-xl-base-1.0 \
  --output_dir models/sdxl/lora \
  --resolution 1024 \
  --train_batch_size 1 \
  --max_train_steps 800 \
  --lora_rank 16

# 单独出图
python training/infer_lora.py \
  --base_model AI-ModelScope/stable-diffusion-xl-base-1.0 \
  --lora_dir models/sdxl/lora \
  --output_dir training/output
```

## 说明

- 当前素材仅 25 张，目的是验证 pipeline（下载、编码、LoRA、保存、回载、出图全链路），产出质量有限
- caption 由目录自动生成（cloth → 平铺商品图视角，human → 全身人像视角），存放在 `training/dataset/*.txt`，可编辑后重新训练
- 训练产物（dataset / output / models/sdxl）不入库