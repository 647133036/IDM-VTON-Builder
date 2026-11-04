import os, sys, time
from pathlib import Path
os.environ["LICENSE_MODE"] = "commercial"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VTON_SEED"] = "42"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# VARIANT C target recipe: clean garment (no mangling reproject) + real body (identity lock) + NO pose-warp
os.environ["VTON_DETAIL"] = "0"
os.environ["VTON_LOCK_ID"] = "1"
os.environ["VTON_WARP"] = "0"
sys.path.insert(0, "/root/vton/repo/backend")
sys.path.insert(0, "/root/vton/repo")
import backend.engine_sdxl as eng

models = sorted(Path("/root/autodl-tmp/vton_inputs/vton_models").glob("*.jpg"))
flat = sorted(Path("/root/autodl-tmp/vton_inputs/vton_flatlays").glob("*.jpg"))
for idx in (0, 1, 2):
    base, garm = models[idx], flat[idx]
    out = Path(f"/root/autodl-tmp/vton_outputs/varC_{idx}.jpg")
    t0 = time.time()
    img = eng.compose_tryon_sdxl(
        base_path=base, garment_path=garm, audience="women", garment_type="dress",
        ethnicity="east_asian", num_inference_steps=30, background="white", fit="regular",
    )
    img.convert("RGB").save(out, quality=92)
    print(f"C[{idx}] in {time.time()-t0:.1f}s ip_adapter={eng._sdxl_state.get('ip_adapter_loaded')}", flush=True)
print("VTON_DONE_C")
