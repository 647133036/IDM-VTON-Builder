"""魔搭创空间换衣 Demo 入口。

启动时预下载权重(首次),首次生成时懒加载 SDXL 管线。观摩者可上传
模特图 + 衣服图,得到完美贴合、有立体感的换衣结果。
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import dl_weights
import gradio as gr
from engine_sdxl import compose_tryon_sdxl, ip_adapter_available, sdxl_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ms_space.app")

# 创空间环境:启动时确保权重就绪(首次下载,已存在则秒回)。
dl_weights.download_all()

AUDIENCES = [
    ("women", "女士"),
    ("men", "男士"),
    ("girls", "女孩"),
    ("boys", "男孩"),
    ("baby", "婴儿"),
]
GARMENT_TYPES = [
    ("upper", "上衣"),
    ("dress", "连衣裙"),
    ("lower", "下装"),
    ("onesie", "连体衣"),
]
ETHNICITIES = [
    ("western_europe", "欧洲"),
    ("east_asia", "东亚"),
    ("southeast_asia", "东南亚"),
    ("africa", "非洲"),
    ("north_america", "北美"),
    ("south_america", "南美"),
    ("oceania", "大洋洲"),
]
BACKGROUNDS = [
    ("white", "白色背景"),
    ("transparent", "透明背景"),
]

# 状态显示(启动检查 + 懒加载提示)。
_ready, _reason = sdxl_available()
_ip_ok = ip_adapter_available()
STATUS = "权重未就绪: " + _reason
if _ready:
    STATUS = "引擎就绪 | IP-Adapter 贴合: " + ("可用" if _ip_ok else "不可用")


def generate(base_img, garment_img, audience, garment_type, ethnicity, background, progress=gr.Progress()):
    if base_img is None or garment_img is None:
        raise gr.Error("请同时上传模特图和衣服图")
    if not _ready:
        raise gr.Error(f"引擎未就绪: {_reason}")

    progress(0.0, desc="加载模型(首次约1-2分钟)...")
    tmp = Path(tempfile.mkdtemp(prefix="ms_tryon_"))
    base_p = tmp / "base.png"
    garment_p = tmp / "garment.png"
    base_img.save(base_p)
    garment_img.save(garment_p)

    progress(0.5, desc="生成换衣结果(约1分钟)...")
    result = compose_tryon_sdxl(
        base_p,
        garment_p,
        audience,
        garment_type,
        long_side=1024,
        background=background,
        ethnicity=ethnicity,
    )
    out = tmp / "out.png"
    result.save(out)
    progress(1.0, desc="完成")
    return str(out), STATUS


with gr.Blocks(title="AI 换衣工作台 - 魔搭创空间", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AI 换衣工作台\n"
        "上传模特图与衣服图,生成**完美贴合、有立体感**的换衣效果。\n"
        "技术:SDXL + OpenPose/Depth ControlNet + IP-Adapter,衣服真实包裹身体轮廓。"
    )
    with gr.Row():
        with gr.Column():
            base_in = gr.Image(label="模特图", type="pil", height=360)
            garment_in = gr.Image(label="衣服图", type="pil", height=360)
        with gr.Column():
            audience_dd = gr.Dropdown(AUDIENCES, value="women", label="模特类型")
            garment_dd = gr.Dropdown(GARMENT_TYPES, value="upper", label="服装类型")
            eth_dd = gr.Dropdown(ETHNICITIES, value="western_europe", label="体型参考")
            bg_dd = gr.Dropdown(BACKGROUNDS, value="white", label="背景")
            run_btn = gr.Button("生成换衣", variant="primary")
            status = gr.Textbox(label="状态", value=STATUS, interactive=False)
    with gr.Column():
        out_img = gr.Image(label="换衣结果", type="filepath", height=460)

    run_btn.click(
        generate,
        inputs=[base_in, garment_in, audience_dd, garment_dd, eth_dd, bg_dd],
        outputs=[out_img, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
