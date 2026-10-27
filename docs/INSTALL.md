# Atelier 安装与部署

固定底图 + 换衣工作台：前端嵌进店铺后台，后端跑在你的云账号。

## 架构

- 前端：Vite SPA，路径 `/`，可 iframe 嵌入 admin
- 后端：FastAPI `backend/main.py`，端口 8000
- 队列：进程内 worker，任务落盘 `backend/data/`
- 引擎：当前为服务端合成流水线（接口与 IDM-VTON 对齐）。有 GPU 时把 `engine.compose_tryon` 换成 `gradio_demo/app.py` 同款推理

## 本机 / 预发

```bash
pip3 install --break-system-packages -r backend/requirements.txt
cd /workspace
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
npm install
npm run dev
```

前端已把 `/api` 和 `/files` 反代到 8000。

## 环境变量

```bash
STORE_JWT_SECRET=replace-me
STORE_ADMIN_PASSWORD=replace-me
STORE_STAFF_PASSWORD=replace-me
```

默认账号口令由环境变量 `STORE_ADMIN_PASSWORD` / `STORE_STAFF_PASSWORD` 设置；未设置时服务端随机生成并打印到启动日志。

## 云上部署（AWS / GCP / Azure）

1. 把 `backend/` 放到 GPU 或 CPU 实例，systemd / docker 跑 uvicorn
2. 对象存储挂 `backend/data/uploads` 与 `backend/data/results`（可选）
3. 店铺 admin 增加菜单，iframe 指向本 SPA
4. 有 NVIDIA GPU 时安装 `IDM-VTON/environment.yaml`，用官方权重 `yisol/IDM-VTON`

## 底座

女装 / 男装 / 女童 / 男童 / 婴幼儿默认底图在 `backend/data/bases/`。后台「模特底座」可随时上传替换。
