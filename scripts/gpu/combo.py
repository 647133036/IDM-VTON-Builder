import os, sys, time
from pathlib import Path
os.environ["LICENSE_MODE"] = "commercial"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VTON_SEED"] = "42"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# good-recipe: no mangling warp/reproject, keep the real model body
os.environ["VTON_WARP"] = "0"
os.environ["VTON_DETAIL"] = "0"
os.environ["VTON_LOCK_ID"] = "1"
sys.path.insert(0, "/root/vton/repo/backend")
sys.path.insert(0, "/root/vton/repo")
import backend.engine_sdxl as eng

base = Path("/root/autodl-tmp/vton_combo/bases/women_01.png")
garments = [
    "/root/autodl-tmp/vton_combo/garments/01_khaite_goya-silk-top_None_main.jpg",
    "/root/autodl-tmp/vton_combo/garments/04469_00.jpg",
    "/root/autodl-tmp/vton_combo/garments/09263_00.jpg",
    "/root/autodl-tmp/vton_combo/garments/09290_00.jpg",
]
outdir = Path("/root/autodl-tmp/vton_combo/out")
outdir.mkdir(parents=True, exist_ok=True)
for i, g in enumerate(garments):
    t0 = time.time()
    img = eng.compose_tryon_sdxl(
        base_path=base, garment_path=Path(g),
        audience="women", garment_type="auto", ethnicity="east_asian",
        num_inference_steps=30, background="white", fit="regular",
    )
    dest = outdir / f"tryon_{i}_{Path(g).stem[:24]}.jpg"
    img.convert("RGB").save(dest, quality=92)
    st = eng._sdxl_state
    print(f"C[{i}] {Path(g).name} in {time.time()-t0:.1f}s ip_adapter={st.get('ip_adapter_loaded')} -> {dest.name}", flush=True)
print("COMBO_DONE")
