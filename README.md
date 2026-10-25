# Atelier 0.1.0

店铺后台固定底图换衣工作台。选一张模特底图，批量拖入服装平铺图，队列生成正面成品，下载或推送到商品页。

## 功能

- 女装 / 男装 / 女童 / 男童 / 婴幼儿正面底座，可随时替换或上传自家模特
- 批量拖入几十到上百张服装图，只出正面，长边 2048px
- 白底 JPEG 或透明 PNG
- 前端嵌进现有店铺 admin，后端跑在你的云账号
- 示例服装库，点击即加入批量队列
- 成品大图预览弹窗，左右对照原图与成品
- 双引擎策略：IDM-VTON（研究模式）/ SDXL（商用模式）/ 合成引擎（兜底）

## 技术栈

**前端**
- Vite 4 + React 18 + TypeScript
- Zustand 状态管理
- Tailwind CSS

**后端**
- FastAPI + Uvicorn
- PyJWT 认证
- PIL / NumPy 图像处理

**AI 引擎**
- IDM-VTON：基于知识蒸馏的虚拟试衣模型（ECCV 2024）
- SDXL + ControlNet + IP-Adapter：商用合规 pipeline
- 合成引擎：无 GPU 环境下的兜底方案

**部署**
- Git submodule 管理依赖
- 环境变量控制引擎选择
- 许可证合规检查

## 快速启动（无 GPU）

```bash
# 后端
pip3 install --break-system-packages -r backend/requirements.txt
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 前端
npm install
npm run dev
```

前端把 `/api` 和 `/files` 反代到 `8000`。

演示账号：`admin / admin123`，`staff / staff123`。

## GPU 部署（商用模式）

有 NVIDIA GPU（最低 12GB VRAM）时，可启用 SDXL 商用引擎。

```bash
# 1. 克隆仓库（含 submodule）
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder
git submodule update --init --recursive

# 2. 下载 SDXL 权重（约 12-15GB）
bash scripts/download_sdxl_weights.sh

# 3. 安装后端依赖
cd backend
pip install -r requirements.txt
pip install torch>=2.1.0 diffusers>=0.27.0 transformers>=4.40.0 accelerate>=0.30.0 controlnet-aux rembg

# 4. 启动后端（商用模式）
LICENSE_MODE=commercial python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 5. 验证引擎状态
curl http://localhost:8000/api/license
# 期望 commercial_compliant: true
```

完整 GPU 部署指南见 `docs/GPU_DEPLOY.md`。

## GPU 部署（研究模式）

```bash
# 下载 IDM-VTON 权重（约 20-30GB）
bash scripts/download_weights.sh

# 启动后端（研究模式）
LICENSE_MODE=research python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 嵌入店铺

```html
<div id="atelier-tryon"></div>
<script src="/widget.js" data-src="/embed.html"></script>
```

或直接 iframe：

```html
<iframe src="/embed.html" style="width:100%;min-height:720px;border:0"></iframe>
```

## 目录结构

```text
backend/          FastAPI 后端、引擎包装器
src/              前端 SPA
public/           嵌入页、挂件、示例图
docs/             文档
scripts/          权重下载脚本
IDM-VTON/         官方虚拟试穿仓库（git submodule）
```

## 环境变量

```bash
STORE_JWT_SECRET=replace-me
STORE_ADMIN_PASSWORD=replace-me
STORE_STAFF_PASSWORD=replace-me
LICENSE_MODE=commercial  # commercial（生产）或 research（测试）
IDM_WEIGHTS_DIR=/path/to/IDM-VTON/ckpt  # 可选
```

## 引擎选择

| LICENSE_MODE | 引擎优先级 | 商用合规 |
|---|---|---|
| `commercial` | SDXL → 合成引擎 | 合规 |
| `research` | IDM-VTON → 合成引擎 | 不合规 |
| 未设置 | 合成引擎 | 需确认 |

访问 `/api/license` 查看当前引擎状态。

## 感谢

本项目使用了以下开源项目：

- [IDM-VTON](https://github.com/yisol/IDM-VTON) - 虚拟试衣模型（ECCV 2024）
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) - 图像生成基础模型
- [ControlNet](https://github.com/lllyasviel/ControlNet) - 空间控制网络
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) - 图像提示适配器
- [Rembg](https://github.com/danielgatis/rembg) - 背景移除工具

## 文档

- 安装部署：`docs/INSTALL.md`
- GPU 部署：`docs/GPU_DEPLOY.md`
- 商用 Pipeline：`docs/SDXL_PIPELINE.md`
- 合规说明：`docs/LICENSE_COMPLIANCE.md`

## 许可

**Atelier 核心代码**：MIT License，允许商用。

**IDM-VTON submodule**：遵循上游 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)，禁止商用。

商用部署请设置 `LICENSE_MODE=commercial`，系统会自动禁用 IDM-VTON，使用 SDXL 商用 pipeline。

```bash
# 初始化 submodule
git clone <本仓库>
git submodule update --init --recursive
```
