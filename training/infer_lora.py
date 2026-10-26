from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROMPT = "a full-body photo of a person wearing a simple t-shirt and jeans, studio lighting, professional fashion photography"
NEGATIVE = "worst quality, low quality, blurry, lowres, bad anatomy, extra limbs, watermark, text"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sample images from a trained SDXL LoRA")
    parser.add_argument("--base_model", default="AI-ModelScope/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_dir", type=Path, default=Path("models/sdxl/lora"))
    parser.add_argument("--output_dir", type=Path, default=Path("training/output"))
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.lora_dir / "adapter_model.safetensors").exists():
        print(f"no trained LoRA found at {args.lora_dir}")
        return 1

    import torch
    from diffusers import StableDiffusionXLPipeline

    base = args.base_model
    local = Path(base).expanduser()
    if not local.exists():
        from modelscope import snapshot_download
        base = snapshot_download(base)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.load_lora_weights(str(args.lora_dir))
    pipe.to(device)
    pipe.enable_attention_slicing()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for i in range(args.num_samples):
        img = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            num_inference_steps=args.steps,
            guidance_scale=7.0,
            generator=generator,
        ).images[0]
        out = args.output_dir / f"sample_{i:02d}.png"
        img.save(out)
        print(f"saved {out}", flush=True)

    print(f"done: {args.num_samples} samples -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())