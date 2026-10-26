# Atelier 0.1.0

店铺后台固定底图换衣工作台。选一张模特底图，批量拖入服装平铺图，队列生成正面成品，下载或推送到商品页。界面内置中英文一键切换。

## 功能

- 女装 / 男装 / 女童 / 男童 / 婴幼儿正面底座，可随时替换或上传自家模特
- 批量拖入几十到上百张服装图，只出正面，长边 2048px
- 白底 JPEG 或透明 PNG
- 已上传图片形成可复用图片库，支持服饰分类管理与批量调整分类
- 底座分类（受众 / 年龄带 / 人种）可管理、可批量调整
- 人种标签可自定义（默认 10 个地理区域：西欧 / 东欧 / 西亚 / 中亚 / 东亚 / 东南亚 / 非洲 / 北美 / 南美 / 澳洲），换衣引擎按底座人种修正体型（肩宽、躯干比例）
- 多店铺管理：每店铺独立 embed 嵌入链接，商品标识通过 `store` 参数区分
- 真实推送：Shopify Admin API 上传商品图，WordPress REST 媒体 + WooCommerce 商品关联
- 成品大图预览弹窗，左右对照原图与成品
- 界面中英文一键切换（登录页 / 导航栏按钮），语言偏好本地持久化
- 三层引擎体系：IDM-VTON（需授权）/ SDXL（商用合规）/ 合成引擎（无 GPU 兜底），UI 横幅实时显示当前引擎与 GPU 状态

## 图片规格

**上传（服装图 / 底座图）**
- 文件大小：无硬性限制
- 格式：仅校验 MIME 是否为图片，支持 JPG / PNG / WEBP 等
- 分辨率与宽高比：不拦截，任意尺寸均可上传

**生成处理**
- 合成引擎：服装图先自动裁剪主体，再按目标躯干框等比缩放并居中适配
- SDXL 商用引擎：底图与服装图统一缩放到 1024×1024 后送 ControlNet + IP-Adapter

**输出成品**
- 长边统一 2048px（`output_long_side`，可配置），宽高比跟随底座图（预置底座为 3:4 竖版，约 1536×2048）
- 背景支持白底 JPEG 或透明 PNG

> 建议：输入图尽量不小于 512px 短边，避免放大后细节模糊。

## 技术栈

**前端**
- 原生 JavaScript（ES 模块，无框架）+ CSS
- Vite 5 构建
- 自研轻量 i18n（`src/i18n.js`，zh/en 字典）

**后端**
- Python 3 + FastAPI + Uvicorn
- PyJWT 认证（admin / staff 双账号）
- PIL / NumPy 图像处理（合成引擎）
- 数据持久化：JSON 文件 + 文件锁（无数据库）

**AI 引擎**
- IDM-VTON：基于知识蒸馏的虚拟试衣模型（ECCV 2024），**禁止商用，商用需购买授权**
- SDXL + ControlNet + IP-Adapter：商用合规 pipeline（有 GPU 时启用）
- 合成引擎：无 GPU 环境下的兜底方案

**部署**
- Git submodule 管理依赖
- 环境变量控制引擎选择
- 许可证合规检查（研究 / 商用模式分流）

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

有 NVIDIA GPU（最低 12GB VRAM）时，可启用 SDXL 商用引擎。页面横幅会随环境自动显示「SDXL + ControlNet + IP-Adapter（商用合规 pipeline）」。

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
# 期望 commercial_compliant: true, gpu_available: true
```

完整 GPU 部署指南见 `docs/GPU_DEPLOY.md`。

## GPU 部署（研究模式）

```bash
# 下载 IDM-VTON 权重（约 20-30GB）
bash scripts/download_weights.sh

# 启动后端（研究模式）
LICENSE_MODE=research python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

研究模式下 IDM-VTON 仅限测试 / 学习，禁止商用。

## 多店铺与推送

### 添加店铺

「店铺嵌入」页可添加多个店铺，每个店铺有独立 embed 链接与推送目标：

| 字段 | 说明 |
|---|---|
| 店铺名称 | 展示名，例如 旗舰店 |
| 平台 | Shopify / WordPress(WooCommerce) / TikTok Shop / 淘宝 / 其他（仅打标） |
| 店铺地址 | Shopify: `{shop}.myshopify.com` · WordPress: `https://你的站点.com` |
| API 凭证 | Shopify Admin token · WordPress: `用户名:应用密码` |
| 默认商品 ID | 可选，推送时优先取请求里的 `product_id` |

### 推送

完成的任务可勾选一个或多个店铺推送，推送记录（每店 ok / url / message）写入任务。

- **Shopify**：Admin API `POST /products/{id}/images.json` 上传商品图
- **WordPress**：REST `POST /wp-json/wp/v2/media` 上传媒体 → `POST /wc/v3/products/{id}` 关联商品
- **其他平台**：仅打标，不调用外部接口
- 凭证缺失或外部返回错误时，推送失败会写回任务的 `push_results`，不会误标为已推送

### 嵌入店铺

```html
<div id="atelier-tryon"></div>
<script src="/widget.js" data-src="/embed.html?store=STORE_ID"></script>
```

或直接 iframe：

```html
<iframe src="/embed.html?store=STORE_ID" style="width:100%;min-height:720px;border:0"></iframe>
```

## 图片库与分类

- 上传的图片自动保存，可复用、可加入生成队列
- 「管理分类」面板：新建 / 改名 / 删除分类
- 「移动到分类」：勾选多张图片后批量移动，或直接新建分类
- 删除分类时其中图片自动移回「未分类」

## 底座管理

- 上传自家模特底座：名称 / 受众 / 年龄带 / 人种 / 备注 / 底图
- 底座列表显示完整分类标签，支持多选后「批量调整分类」
- 「人种管理」面板：新增 / 改名 / 删除人种；删除人种时引用底座回退到「东亚」

## 测试

后端单元 / 集成 / 系统测试，使用 pytest 与 FastAPI TestClient，state 与图片全部隔离在临时目录，不触碰真实 `backend/data`。前端组件测试用 vitest 覆盖 i18n 字典与人种翻译。

```bash
pip3 install --break-system-packages pytest httpx
cd backend
python3 -m pytest tests/ -q

# 前端组件测试
cd /workspace
npm test
```

覆盖分层：

- **单元测试**（`backend/tests/test_engine.py` / `test_store.py` / `test_auth.py` / `test_push.py` / `test_worker.py`）：引擎按人种修正体型、底座人种迁移、CRUD、JWT、Shopify/WordPress 推送适配器（网络层 mock）、worker 任务处理分支
- **组件测试**（`src/i18n.test.js`）：中英文切换、占位符插值、10 地域人种翻译、自定义人种回退原文
- **集成测试**（`backend/tests/test_api.py`）：FastAPI 全 API 链路——登录鉴权、底座 / 人种 / 店铺 CRUD、批量任务创建、图片库、推送、路径穿越防护、健康与许可接口
- **系统测试**（`backend/tests/test_system.py`）：真实 worker 线程领取任务并跑完一张成品到 `done`，可下载；缺失源图正确标记 `failed`

## 目录结构

```text
backend/          FastAPI 后端、引擎包装器、推送适配器
  tests/          pytest 测试（单元 / 集成 / 系统）
src/              前端 SPA + i18n 字典
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

访问 `/api/license` 查看当前引擎、GPU 与合规状态；页面顶部横幅同步展示。

## API 概览

```text
POST /api/auth/login
GET  /api/bases
POST /api/bases                        # 上传底座
PATCH /api/bases/{id}
POST /api/bases/batch/patch            # 批量调整底座分类
DELETE /api/bases/{id}
GET|POST /api/ethnicities              # 人种管理
PATCH|DELETE /api/ethnicities/{value}
GET|POST /api/stores                   # 多店铺管理
PATCH|DELETE /api/stores/{id}
POST /api/jobs/batch
GET  /api/jobs
POST /api/jobs/{id}/push               # body: {"store_ids":["x"],"product_id":"123"}
GET  /api/download/{id}
GET  /api/uploads                      # 图片库
POST /api/uploads/{filename}/move      # 调整图片分类
GET|POST /api/uploads/categories       # 服饰分类管理
PATCH|DELETE /api/uploads/categories/{category}
GET  /api/health | /api/license        # 引擎 / GPU / 合规状态
```

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

**IDM-VTON submodule**：遵循上游 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)，**禁止商用，商用需购买授权**。

**SDXL 基础模型**：遵循 **Stability AI Community License**。商用必须：
1. 在 Stability AI 完成**免费注册**（商用目的使用或分发前）
2. 随分发附带协议副本与仓库根目录 `NOTICE` 文件
3. 在界面与文档**显著展示 `Powered by Stability AI`**（本项目已在前端页脚落实）

收入门槛：你与关联公司合计年收入低于 $100 万时商用免费（仍需注册）；超过门槛需申请付费 Enterprise License。完整核对见 `docs/LICENSE_COMPLIANCE.md`。

商用部署请设置 `LICENSE_MODE=commercial`，系统会自动禁用 IDM-VTON，使用 SDXL 商用 pipeline。

```bash
# 初始化 submodule
git clone <本仓库>
git submodule update --init --recursive
```
