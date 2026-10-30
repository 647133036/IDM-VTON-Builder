import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from train_lora_sdxl import build_time_ids, letterbox_rgb, min_snr_weights  # noqa: E402


def test_build_time_ids_sdxl_order():
    ids = build_time_ids(1024, 2)
    assert tuple(ids[0].tolist()) == (1024, 1024, 0, 0, 1024, 1024)
    assert ids.shape == (2, 6)


def test_letterbox_rgb_pads_short_side():
    img = Image.new("RGB", (50, 100), (7, 8, 9))
    out = letterbox_rgb(img, 100)
    assert out.size == (100, 100)
    arr = np.array(out)
    assert tuple(int(x) for x in arr[50, 50]) == (7, 8, 9)
    assert tuple(int(x) for x in arr[0, 0]) == (255, 255, 255)


def test_min_snr_weights_downweight_high_noise():
    import torch

    class _Sched:
        def __init__(self):
            t = torch.linspace(0.999, 0.001, 1000)
            self.alphas_cumprod = t

    w = min_snr_weights(_Sched(), torch.tensor([10, 900]), gamma=5.0)
    assert float(w[0]) < float(w[1])
    assert float(w[0]) < 1.0
    assert float(w[1]) == 1.0
