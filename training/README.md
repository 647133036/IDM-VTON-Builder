# SDXL 训练工程（验证 pipeline）

一组用于在 GPU 环境（如魔塔 ModelScope Notebook 免费 GPU）上端到端验证训练链路的脚本。默认使用仓库现有素材（`public/examples/cloth/` 成衣平铺图、`public/examples/human/` 人像图），产物可被 `backend/engine_sdxl.py` 加载。

完全不依赖 IDM-VTON（CC BY-NC-SA 禁止商用），只做 SDXL LoRA / ControlNet 微调，商用合规。

## 目录

```
training/
  prepare_dataset.py          # 成衣/人像 → LoRA 训练集（图片 + caption）
  train_lora_sdxl.py          # 精简 SDXL LoRA 训练（单卡，魔塔可用）
  train_lora_family.py        # 按分组 (cloth/human) 各训一个 LoRA
  prepare_controlnet_data.py  # 人像 → ControlNet 条件图 (openpose/canny)
  train_controlnet_sdxl.py    # 精简 SDXL ControlNet 微调
  infer_lora.py               # 用训练好的 LoRA 出图验证
  calibrate_ethnicity_fit.py  # 用实测比例拟合 ETHNICITY_FIT 乘子（无需 GPU）
  run_train.sh                # 一键：MODEL=lora|lora_family|controlnet
  requirements.txt
  README.md
```

## 在魔塔 Notebook 上运行

```bash
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder
pip install -r training/requirements.txt

# LoRA（默认，25 张素材验证链路）
bash training/run_train.sh

# 分组建 LoRA 族
MODEL=lora_family GROUPS="cloth human" bash training/run_train.sh

# ControlNet 微调（openpose 或 canny 条件）
MODEL=controlnet CONDITION=canny bash training/run_train.sh
```

产物：
- LoRA：`models/sdxl/lora/`（引擎 `engine_sdxl.py` 自动加载）
- LoRA 族：`models/sdxl/lora/{cloth,human}/`（按分组子目录）
- ControlNet：`models/sdxl/controlnet/`
- 样图：`training/output/`

## ETHNICITY_FIT 校准（无需 GPU）

用 GPU 出图后量测的真实体型比例，拟合 `backend/engine.py` 的 ETHNICITY_FIT 乘子：

```bash
python3 training/calibrate_ethnicity_fit.py --measurements meas.csv
python3 training/calibrate_ethnicity_fit.py --measurements meas.csv --diff
```

`meas.csv` 格式：`ethnicity,w,h,shoulder`（w/shoulder 为相对基线的宽度/肩宽比例，h 为躯干高比例）。`--diff` 只打印发生变化的条目，输出可直接贴回 `ETHNICITY_FIT`。

## 手动步骤

```bash
python3 training/prepare_dataset.py --out training/dataset

python3 training/train_lora_sdxl.py \
  --dataset_dir training/dataset \
  --base_model AI-ModelScope/stable-diffusion-xl-base-1.0 \
  --output_dir models/sdxl/lora \
  --max_train_steps 800

python3 training/prepare_controlnet_data.py \
  --source public/examples/human --condition canny --out training/cn_dataset

python3 training/train_controlnet_sdxl.py \
  --dataset_dir training/cn_dataset \
  --base_model AI-ModelScope/stable-diffusion-xl-base-1.0 \
  --output_dir models/sdxl/controlnet

python3 training/infer_lora.py \
  --base_model AI-ModelScope/stable-diffusion-xl-base-1.0 \
  --lora_dir models/sdxl/lora --output_dir training/output
```

## 说明

- 现有素材仅 25 张，目的是验证全链路（下载、编码、LoRA/ControlNet、保存、回载、出图），产出质量有限
- `train_lora_sdxl.py` / `train_controlnet_sdxl.py` 自动检测设备：CUDA 用 bf16（SDXL UNet fp16 训练会 NaN），否则退化 CPU fp32（很慢，仅验证）
- transformers 5.x 下 CLIP text_encoder_2 取 `text_embeds`（无 `pooler_output` 时回退）
- ControlNet 的 `captions.csv` 用 csv.writer 写，caption 含逗号不会列错位
- ControlNet 条件图默认 openpose（需联网下载 annotator），离线可用 `--condition canny`
- caption 自动生成于 `training/dataset/*.txt`，可编辑后重训
- 训练产物（dataset / output / models）不入库