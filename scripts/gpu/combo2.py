import os, sys, time, cv2, numpy as np
from pathlib import Path
os.environ["LICENSE_MODE"]="commercial"; os.environ["OMP_NUM_THREADS"]="8"
os.environ["VTON_SEED"]="42"; os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["VTON_WARP"]="0"; os.environ["VTON_DETAIL"]="0"; os.environ["VTON_LOCK_ID"]="1"
sys.path.insert(0,"/root/vton/repo/backend"); sys.path.insert(0,"/root/vton/repo")

GNAMES = ["01_khaite_goya-silk-top_None_main.jpg","04469_00.jpg","09263_00.jpg","09290_00.jpg"]
GD = Path("/root/autodl-tmp/vton_combo/garments")
base = "/root/autodl-tmp/vton_combo/bases/women_01.png"
garments = [str(GD / n) for n in GNAMES]

def prewhiten(p, out, tol=45):
    im = cv2.imread(p)
    h, w = im.shape[:2]
    m = np.zeros((h + 2, w + 2), np.uint8)
    for sx, sy in [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]:
        if m[sy, sx]:
            continue
        cv2.floodFill(im, m, (sx, sy), (255,255,255), (tol,tol,tol), (tol,tol,tol), cv2.FLOODFILL_FIXED_RANGE)
    cv2.imwrite(out, im)

clean = Path("/root/autodl-tmp/vton_combo/clean"); clean.mkdir(exist_ok=True)
prewhiten(base, str(clean / "base.png"))
cgar = []
for n in GNAMES:
    dst = clean / ("g_" + n)
    prewhiten(str(GD / n), str(dst))
    cgar.append(dst)
print("prewhitened base +", len(cgar), "garments", flush=True)

import backend.engine_sdxl as eng
out = Path("/root/autodl-tmp/vton_combo/out2"); out.mkdir(exist_ok=True)
for i, dst in enumerate(cgar):
    t0 = time.time()
    img = eng.compose_tryon_sdxl(base_path=clean / "base.png", garment_path=dst,
        audience="women", garment_type="auto", ethnicity="east_asian",
        num_inference_steps=30, background="white", fit="regular")
    dest = out / ("tryon_" + str(i) + "_" + GNAMES[i][:22] + ".jpg")
    img.convert("RGB").save(dest, quality=92)
    print("C2[%d] %s in %.1fs -> %s" % (i, GNAMES[i], time.time()-t0, dest.name), flush=True)
print("COMBO2_DONE")
