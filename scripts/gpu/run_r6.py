#!/usr/bin/env python3
"""r6: 在 GPU 机重生成 r5 全 5 张
- 先由 women_01.png 现场生成 1024 base white
- 再喂 5 张 garment，换 seed 重跑
输出: /root/vton/outputs/
"""
import os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/root/vton")
sys.path.insert(0, str(ROOT / "repo"))
sys.path.insert(0, str(ROOT / "repo" / "backend"))
os.environ["VTON_AR"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import backend.engine_sdxl as eng
from platform_presets import foreground_mask

IN  = ROOT / "inputs"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 生成 base white 1024
SIDE = 1024
base_raw = Image.open(IN / "women_01.png").convert("RGB")
w, h = base_raw.size
s = SIDE / max(w, h)
nw, nh = max(1, int(w * s)), max(1, int(h * s))
canvas = Image.new("RGB", (SIDE, SIDE), (255, 255, 255))
canvas.paste(base_raw.resize((nw, nh), Image.Resampling.LANCZOS),
             ((SIDE - nw) // 2, (SIDE - nh) // 2))
m = foreground_mask(canvas)
if m is not None:
    arr = np.asarray(canvas, np.uint8).copy()
    keep = (np.asarray(m) > 0)
    out_arr = np.full_like(arr, 255)
    out_arr[keep] = arr[keep]
    base_white = Image.fromarray(out_arr)
    print("[0] base white generated with u2net subject mask", flush=True)
else:
    base_white = canvas
    print("[0] u2net unavailable, plain letterbox base", flush=True)
base_path = OUT / "_base_white_1024.png"
base_white.save(base_path)
print(f"[0] saved {base_path}", flush=True)

PAIRS = [
    ("01_khaite_goya-silk-top_None_main.jpg", "upper",
     "sheer mustard-gold long-sleeved shirt with monkey print on chest, vertical stripes, relaxed cocoon silhouette",
     "31", "1.1"),
    ("07_khaite_briar-silk-top_None_main.jpg", "upper",
     "structured satin top with wide flared bell-shaped short sleeves, open front, sculptural silhouette",
     "31", "0.8"),
    ("50-laporte.jpg", "upper", "", "31", "0.8"),
    ("51-laporte.jpg", "dress",
     "the garment is a dress with A-line skirt hem ending at mid-thigh, NOT pants, NO split legs, no trousers",
     "42", "0.8"),
    ("52-laporte.jpg", "upper", "", "42", "0.8"),
]

t0 = time.time()
for stem, gtype, hint, seed, ipa in PAIRS:
    garment = IN / stem
    if not garment.is_file():
        print(f"[skip] missing {stem}", flush=True); continue
    os.environ["VTON_SEED"] = seed
    os.environ["VTON_IPA"] = ipa
    os.environ["VTON_PROMPT_HINT"] = hint
    out_path = OUT / f"r6_women01_{stem.rsplit('.', 1)[0]}.png"
    t1 = time.time()
    print(f"\n[{stem}] seed={seed} ipa={ipa} ...", flush=True)
    img = eng.compose_tryon_sdxl(
        base_path=base_path,
        garment_path=garment,
        audience="women",
        garment_type=gtype,
        background="white",
        fit="regular",
        num_inference_steps=50,
    )
    img.save(out_path)
    print(f"  -> {out_path}  {time.time()-t1:.1f}s", flush=True)

print(f"\nr6 done in {time.time()-t0:.1f}s; 输出在 {OUT}", flush=True)
