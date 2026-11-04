import os, sys, time
from pathlib import Path
os.environ["LICENSE_MODE"] = "commercial"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VTON_SEED"] = "42"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# VARIANT A: isolate the raw diffusion. Turn off the post-process warp/blend.
os.environ["VTON_DETAIL"] = "0"
os.environ["VTON_LOCK_ID"] = "0"
os.environ["VTON_WARP"] = "0"
sys.path.insert(0, "/root/vton/repo/backend")
sys.path.insert(0, "/root/vton/repo")
import backend.engine_sdxl as eng

models = sorted(Path("/root/autodl-tmp/vton_inputs/vton_models").glob("*.jpg"))
flat = sorted(Path("/root/autodl-tmp/vton_inputs/vton_flatlays").glob("*.jpg"))
base, garm = models[0], flat[0]
print(f"pair: {base.name} x {garm.name}", flush=True)
out = Path("/root/autodl-tmp/vton_outputs/varA_raw.jpg")
st = eng._sdxl_state
t0 = time.time()
img = eng.compose_tryon_sdxl(
    base_path=base, garment_path=garm, audience="women", garment_type="dress",
    ethnicity="east_asian", num_inference_steps=30, background="white", fit="regular",
)
img.convert("RGB").save(out, quality=92)
print(f"A (raw, post off) done in {time.time()-t0:.1f}s -> {out}", flush=True)
print("depth_loaded=", st.get("depth_loaded"), "ip_adapter_loaded=", st.get("ip_adapter_loaded"))
