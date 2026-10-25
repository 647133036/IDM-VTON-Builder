# Atelier 0.1.0

店铺后台里的固定底图换衣工作台（预览版）。

选一张正面模特底图，批量拖入平铺图或幽灵模特图，队列生成正面成品，再下载或推送到商品页。

## 版本定位

`0.1.0` 是可运行的预发骨架，用来验证「固定底图 + 换衣 + 店铺嵌入」这条链路。

已具备：登录、可更换底座、批量队列、下载 / 推送、iframe 嵌入、成品大图预览。

引擎采用双轨策略：有 GPU + 权重时走 IDM-VTON 扩散模型（出真图），无 GPU 时自动回退合成引擎（兜底）。引擎选择全自动，无需手动切换。

## 能做什么

- 女装 / 男装 / 女童 / 男童 / 婴幼儿正面底座，可随时替换或上传自家模特
- 一次拖入几十到上百张服装图，只出正面，长边 2048px
- 白底 JPEG 或透明 PNG
- 前端嵌进现有店铺 admin，后端跑在你的云账号
- 示例服装库，点击即加入批量队列
- 成品大图预览弹窗，左右对照原图与成品

## 目录

```text
backend/          FastAPI 登录、底座、队列、IDM-VTON 引擎包装器
src/               店铺后台 SPA
public/embed.html  iframe 嵌入页
public/widget.js   店铺挂件
public/examples/   示例服装图与模特图
docs/              安装、GPU 部署、员工上手
scripts/           权重下载脚本
IDM-VTON/          官方虚拟试穿仓库（git submodule）
```

## 快速启动（无 GPU）

```bash
pip3 install --break-system-packages -r backend/requirements.txt
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

```bash
npm install
npm run dev
```

前端把 `/api` 和 `/files` 反代到 `8000`。

演示账号：`admin / admin123`，`staff / staff123`。

## GPU 部署（IDM-VTON 扩散引擎）

有 NVIDIA GPU（最低 24GB VRAM）时，可启用 IDM-VTON 扩散模型引擎，获得官方级画质。

```bash
# 1. 克隆仓库（含 submodule）
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder
git submodule update --init --recursive

# 2. 下载 IDM-VTON 权重（约 20-30GB）
bash scripts/download_weights.sh

# 3. 安装后端依赖
cd backend
pip install -r requirements.txt
pip install torch>=2.1.0 diffusers>=0.27.0 transformers>=4.40.0 accelerate>=0.30.0

# 4. 启动后端
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 5. 验证引擎状态
curl http://localhost:8000/api/health
# 期望 idm_vton_available: true
```

完整 GPU 部署指南见 `docs/GPU_DEPLOY.md`，含硬件要求、性能参考、Docker 部署、常见问题。

## 嵌进店铺

```html
<div id="atelier-tryon"></div>
<script src="/widget.js" data-src="/embed.html"></script>
```

或直接 iframe：

```html
<iframe src="/embed.html" style="width:100%;min-height:720px;border:0"></iframe>
```

## 文档

- 安装部署：`docs/INSTALL.md`
- GPU 部署：`docs/GPU_DEPLOY.md`
- 员工上手：`docs/STAFF_QUICKSTART.md`
- 可打印页：`docs/STAFF_QUICKSTART.html`

## 环境变量

```bash
STORE_JWT_SECRET=replace-me
STORE_ADMIN_PASSWORD=replace-me
STORE_STAFF_PASSWORD=replace-me
IDM_WEIGHTS_DIR=/path/to/IDM-VTON/ckpt  # 可选，默认 IDM-VTON/ckpt
LICENSE_MODE=research  # research（测试）或 commercial（生产）
```

### 许可证合规

IDM-VTON 采用 **CC BY-NC-SA 4.0**，禁止商业用途。电商换衣是商业用途，与 IDM-VTON 许可冲突。

通过 `LICENSE_MODE` 环境变量控制引擎可用性：

| 模式 | IDM-VTON | 商用合规 | 适用场景 |
|------|----------|----------|----------|
| `research` | 可用（需 GPU） | 不合规 | 测试、学习、内部验证 |
| `commercial` | 禁用 | 合规 | 生产环境（需商用模型） |
| 未设置 | 禁用 | 需确认 | 默认状态 |

- 研究模式：`export LICENSE_MODE=research`，IDM-VTON 可用但不得用于生产
- 商业模式：`export LICENSE_MODE=commercial`，IDM-VTON 禁用，必须使用商用模型

完整合规说明见 `docs/LICENSE_COMPLIANCE.md`。

访问 `/api/license` 可查看当前合规状态。

## 许可

本项目（含 Atelier 后台、引擎、文档）采用 **CC BY-NC-SA 4.0**，与 `IDM-VTON/` 官方仓库授权保持一致。见根目录 `LICENSE.txt`。

`IDM-VTON/` 以 git submodule 挂载，源码遵循上游 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)（[yisol/IDM-VTON](https://github.com/yisol/IDM-VTON)）。

```bash
git clone <本仓库>
git submodule update --init --recursive
```

商用前请自行完成授权与内容合规确认。
