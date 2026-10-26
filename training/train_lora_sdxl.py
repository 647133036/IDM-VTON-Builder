from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_base_model(base_model: str) -> str:
    candidate = Path(base_model).expanduser()
    if candidate.exists():
        return str(candidate)
    try:
        from modelscope import snapshot_download
    except ImportError:
        return base_model
    local = snapshot_download(base_model)
    return str(local)


def _make_train_state(dataset_dir: Path):
    import torch

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def _load_components(base_model: str, dtype):
    import torch

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    base = _resolve_base_model(base_model)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    return {
        "base": base,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
    }


def _load_prompts(dataset_dir: Path):
    txts = sorted(dataset_dir.rglob("*.txt"))
    prompts = []
    for txt in txts:
        text = txt.read_text(encoding="utf-8").strip()
        if text:
            prompts.append(text)
    return prompts


class _CaptionDataset:
    def __init__(self, dataset_dir: Path, resolution: int):
        import torch
        from PIL import Image
        from torch.utils.data import Dataset

        img_exts = {".jpg", ".jpeg", ".png"}
        self.items = []
        for img in sorted(dataset_dir.rglob("*.png")):
            txt = img.with_suffix(".txt")
            caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            self.items.append((img, caption))
        self.resolution = resolution

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch
        from PIL import Image
        from torchvision import transforms

        path, caption = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        tensor = transforms.ToTensor()(img) * 2.0 - 1.0
        return tensor, caption


def _encode_text(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompts, device):
    import torch

    def tok(tokenizer_cls, text):
        return tokenizer_cls(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)

    ids = tok(tokenizer, prompts)
    ids_2 = tok(tokenizer_2, prompts)
    with torch.no_grad():
        enc = text_encoder(ids.input_ids).last_hidden_state
        out_2 = text_encoder_2(ids_2.input_ids)
    prompt_embeds = torch.cat([enc, out_2.last_hidden_state], dim=-1)
    pooled_embeds = out_2.pooler_output
    return prompt_embeds, pooled_embeds


def train(args) -> int:
    import torch
    from peft import LoraConfig, get_peft_model

    device, dtype = _make_train_state(args.dataset_dir)
    comps = _load_components(args.base_model, dtype)

    unet = comps["unet"]
    vae = comps["vae"]
    text_encoder = comps["text_encoder"]
    text_encoder_2 = comps["text_encoder_2"]
    tokenizer = comps["tokenizer"]
    tokenizer_2 = comps["tokenizer_2"]
    noise_scheduler = comps["noise_scheduler"]

    for model in (vae, text_encoder, text_encoder_2):
        model.to(device)
        model.requires_grad_(False)
    unet.to(device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    writer = Path(args.output_dir)
    writer.mkdir(parents=True, exist_ok=True)

    from torch.utils.data import DataLoader

    dataset = _CaptionDataset(args.dataset_dir, args.resolution)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    print(f"trainable samples: {len(dataset)} / steps: {args.max_train_steps}", flush=True)

    global_step = 0
    while global_step < args.max_train_steps:
        for images, prompts in loader:
            images = images.to(device, dtype=dtype)
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            prompt_embeds, pooled = _encode_text(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompts, device)
            prompt_embeds = prompt_embeds.to(dtype)
            pooled = pooled.to(dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            added = {
                "text_embeds": pooled,
                "time_ids": torch.tensor(
                    [args.resolution, args.resolution, args.resolution, args.resolution, 0, 0],
                    device=device,
                    dtype=torch.long,
                ).repeat(bsz, 1),
            }

            noise_pred = unet(
                noisy,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}", flush=True)
            if global_step >= args.max_train_steps:
                break

    unet.save_pretrained(str(writer))
    print(f"LoRA saved to {writer}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny SDXL LoRA trainer")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--base_model", default="AI-ModelScope/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", default="models/sdxl/lora", type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    if not args.dataset_dir.exists() or not list(args.dataset_dir.rglob("*.png")):
        print(f"dataset dir empty: {args.dataset_dir}. run prepare_dataset.py first")
        return 1

    try:
        return train(args)
    except Exception as exc:
        print(f"train failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())