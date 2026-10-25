# IDM-VTON GPU 部署指南

本文档说明如何在有 GPU 的机器上部署 Atelier，启用 IDM-VTON 扩散模型引擎。

## 硬件要求

- GPU: NVIDIA，最低 24GB VRAM（推荐 A100 / RTX 4090）
- CPU: 8 核以上
- 内存: 32GB 以上
- 磁盘: 50GB 可用空间（权重约 20-30GB）
- 系统: Ubuntu 20.04+ / Linux with CUDA

## 软件要求

- Python 3.10+
- CUDA 11.8+ 或 12.x
- NVIDIA 驱动 >= 520

## 快速部署

### 1. 克隆仓库

```bash
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder
git submodule update --init --recursive
```

### 2. 下载权重

```bash
# 需要 huggingface-cli
pip install huggingface_hub

# 下载 IDM-VTON 权重（约 20-30GB）
bash scripts/download_weights.sh
```

### 3. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt

# 安装 GPU 推理依赖
pip install torch>=2.1.0 diffusers>=0.27.0 transformers>=4.40.0 accelerate>=0.30.0
```

### 4. 启动后端

```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. 验证引擎状态

```bash
curl http://localhost:8000/api/health
```

期望输出：

```json
{
  "ok": true,
  "engine": "idm-vton-composite",
  "engine_detail": {
    "idm_vton_available": true,
    "reason": "",
    "engine": "idm-vton",
    "weights_dir": "/path/to/IDM-VTON/ckpt",
    "idm_vton_dir": "/path/to/IDM-VTON"
  }
}
```

当 `idm_vton_available` 为 `true` 时，工具将使用 IDM-VTON 扩散模型生成图片。

### 6. 启动前端

```bash
cd ..
npm install
npm run dev
```

访问 `http://localhost:5173` 使用工具。

## 引擎选择逻辑

`backend/engine.py` 中的 `compose_tryon()` 函数自动选择引擎：

1. 检查 GPU 是否可用
2. 检查权重是否完整
3. 如果都满足，加载 IDM-VTON pipeline 并推理
4. 如果任一条件不满足，回退到合成引擎

无需手动切换，引擎自动选择。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IDM_WEIGHTS_DIR` | `IDM-VTON/ckpt` | 权重目录路径 |
| `STORE_JWT_SECRET` | `idm-vton-store-demo-secret` | JWT 密钥（生产环境必须修改） |
| `STORE_ADMIN_PASSWORD` | `admin123` | 管理员密码 |
| `STORE_STAFF_PASSWORD` | `staff123` | 员工密码 |

## 性能参考

| 硬件 | 单张生成时间 | 显存占用 |
|------|-------------|----------|
| RTX 4090 | 3-5 秒 | 18-20GB |
| A100 40GB | 2-3 秒 | 18-20GB |
| A100 80GB | 2-3 秒 | 18-20GB |

## 常见问题

### 权重下载失败

检查网络连接和 HuggingFace 访问权限。如果在大陆，可能需要配置代理：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_weights.sh
```

### GPU 内存不足

- 降低 `num_inference_steps`（默认 30，可降到 20）
- 使用 `pipe.enable_vae_tiling()` 减少显存占用
- 确保没有其他进程占用 GPU

### 推理速度慢

- 确保使用 `torch.float16` 精度
- 检查 CUDA 版本是否匹配
- 考虑使用 TensorRT 优化（需要额外配置）

## 生产部署

生产环境建议使用 Docker 部署：

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# ... 安装依赖 ...
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

使用 `nvidia-docker` 运行：

```bash
docker run --gpus all -p 8000:8000 atelier-idm-vton
```
