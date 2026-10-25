# Atelier 0.1.0

店铺后台里的固定底图换衣工作台（预览版）。

选一张正面模特底图，批量拖入平铺图或幽灵模特图，队列生成正面成品，再下载或推送到商品页。

## 版本定位

`0.1.0` 是可运行的预发骨架，用来验证「固定底图 + 换衣 + 店铺嵌入」这条链路。

已具备：登录、可更换底座、批量队列、下载 / 推送、iframe 嵌入。

当前引擎是服务端合成流水线。有 GPU 后，把 `backend/engine.py` 换成仓库内 `IDM-VTON/` 官方推理。

## 能做什么

- 女装 / 男装 / 女童 / 男童 / 婴幼儿正面底座，可随时替换或上传自家模特
- 一次拖入几十到上百张服装图，只出正面，长边 2048px
- 白底 JPEG 或透明 PNG
- 前端嵌进现有店铺 admin，后端跑在你的云账号

## 目录

```text
backend/          FastAPI 登录、底座、队列、合成引擎
src/               店铺后台 SPA
public/embed.html  iframe 嵌入页
public/widget.js   店铺挂件
docs/              安装与员工上手
IDM-VTON/          官方虚拟试穿仓库（GPU 推理用）
```

## 启动

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
- 员工上手：`docs/STAFF_QUICKSTART.md`
- 可打印页：`docs/STAFF_QUICKSTART.html`

## 环境变量

```bash
STORE_JWT_SECRET=replace-me
STORE_ADMIN_PASSWORD=replace-me
STORE_STAFF_PASSWORD=replace-me
```

## 许可

本项目（含 Atelier 后台、引擎、文档）采用 **CC BY-NC-SA 4.0**，与 `IDM-VTON/` 官方仓库授权保持一致。见根目录 `LICENSE.txt`。

`IDM-VTON/` 以 git submodule 挂载，源码遵循上游 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)（[yisol/IDM-VTON](https://github.com/yisol/IDM-VTON)）。

```bash
git clone <本仓库>
git submodule update --init --recursive
```

商用前请自行完成授权与内容合规确认。
