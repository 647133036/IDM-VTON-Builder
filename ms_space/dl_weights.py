"""下载换衣引擎所有权重到 MODELS_DIR(默认 /mnt/workspace/models/sdxl)。

创空间环境在中国大陆,ModelScope 有镜像的组件走 ModelScope(内网高速),
其余走 hf-mirror(大陆可达)。仅首次运行需下载;已存在则跳过。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SDXL_DIR = Path(os.environ.get("MODELS_DIR", "/mnt/workspace/models/sdxl"))

# ModelScope 镜像(大陆内网,优先;已确认存在)
MS_MIRRORS = {
    "sdxl_base": "AI-ModelScope/stable-diffusion-xl-base-1.0",
    "controlnet_depth": "AI-ModelScope/controlnet-depth-sdxl-1.0",
}
# hf-mirror 补缺(大陆可达;ModelScope 无对应镜像)
HF_MIRRORS = {
    "controlnet_openpose": "lllyasviel/controlnet-openpose",
    "annotators": "lllyasviel/Annotators",
}


def _dl_ms(model_id: str, target: Path) -> None:
    from modelscope import snapshot_download

    snapshot_download(model_id, local_dir=str(target))


def _dl_hf(model_id: str, target: Path, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    snapshot_download(model_id, local_dir=str(target), allow_patterns=allow_patterns)


def _dir_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def download_all() -> None:
    SDXL_DIR.mkdir(parents=True, exist_ok=True)

    for name, mid in MS_MIRRORS.items():
        target = SDXL_DIR / name
        marker = target / ("model_index.json" if name == "sdxl_base" else "config.json")
        if marker.exists():
            print(f"[skip] {name} already present", flush=True)
            continue
        print(f"[download] {name} <- ModelScope {mid}", flush=True)
        _dl_ms(mid, target)

    for name, mid in HF_MIRRORS.items():
        target = SDXL_DIR / name
        if _dir_nonempty(target):
            print(f"[skip] {name} already present", flush=True)
            continue
        print(f"[download] {name} <- hf-mirror {mid}", flush=True)
        _dl_hf(mid, target)

    # IP-Adapter 权重(h94/IP-Adapter,repo 内 ip_adapter/ 子目录含 vit-h)。
    ipa = SDXL_DIR / "ip_adapter"
    if not (ipa / "ip-adapter_sdxl_vit-h.safetensors").exists():
        tmp = SDXL_DIR / "_ipa_tmp"
        print("[download] ip_adapter weight <- hf-mirror h94/IP-Adapter", flush=True)
        _dl_hf("h94/IP-Adapter", tmp, allow_patterns=["ip_adapter/ip-adapter_sdxl_vit-h.safetensors"])
        ipa.mkdir(parents=True, exist_ok=True)
        for f in (tmp / "ip_adapter").glob("ip-adapter_sdxl_vit-h.*"):
            shutil.move(str(f), str(ipa / f.name))
        shutil.rmtree(tmp, ignore_errors=True)

    # IP-Adapter image_encoder(CLIP vision,h94/models repo 的 image_encoder/ 子目录)。
    if not (ipa / "image_encoder" / "config.json").exists():
        print("[download] image_encoder <- hf-mirror h94/models", flush=True)
        _dl_hf("h94/models", ipa, allow_patterns=["image_encoder/*"])

    print(f"[done] weights ready at {SDXL_DIR}", flush=True)


if __name__ == "__main__":
    download_all()
