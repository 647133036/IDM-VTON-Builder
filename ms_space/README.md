# AI 换衣工作台（魔搭创空间）

上传模特图 + 衣服图，生成**完美贴合、有立体感**的换衣效果。

- 技术：SDXL 基座 + OpenPose/Depth ControlNet + IP-Adapter（衣服色彩/纹理一致性）。
- 贴合：Depth ControlNet 用人体深度图约束衣服包裹身体轮廓，产生真实褶皱与立体感，避免"平面贴纸"。

## 文件说明

| 文件 | 用途 |
|------|------|
| `app.py` | Gradio 应用唯一入口（创空间运行入口）。 |
| `engine_sdxl.py` | 换衣引擎（SDXL + ControlNet + IP-Adapter 推理）。 |
| `dl_weights.py` | 权重下载：ModelScope 镜像优先（SDXL/Depth），hf-mirror 补缺（OpenPose/IP-Adapter/image_encoder/Annotators）。 |
| `requirements.txt` | Python 依赖（torch/torchvision 由创空间 base_image 预装）。 |

## 部署到魔搭创空间

1. 在 [modelscope.cn/studios](https://modelscope.cn/studios) 创建创空间，运行方式选 **Gradio**，硬件选择 **xGPU**（建议 L20 48GB 档）。
2. 上传本目录全部文件到创空间仓库根目录（`app.py` 必须在根目录）。
3. 确认 `requirements.txt` 已提交（创空间构建时自动安装依赖）。
4. 部署后确认应用状态为「运行中」，并实际点开体验一次（首次会下载约 17GB 权重，冷启动时间较长，属正常）。
5. 模型权重默认下载到 `/mnt/workspace/models/sdxl`，可通过环境变量 `MODELS_DIR` 覆盖。

## 本地运行（可选）

有 GPU + 已下载权重时可本地跑：

```bash
export MODELS_DIR=/path/to/models/sdxl
python -m dl_weights        # 首次下载权重
python app.py               # http://127.0.0.1:7860
```

## 注意事项

- 权重下载源：ModelScope 内网高速（SDXL 基座、Depth ControlNet），其余走 `hf-mirror.com`（大陆可达）。
- 生成分辨率 1024×1024，单次约 1 分钟（xGPU）。
- 本应用仅用于服装搭配演示，请勿用于不当用途。
