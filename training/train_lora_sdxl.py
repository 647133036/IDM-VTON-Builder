from __future__ import annotations

import argparse
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

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # bf16 仅当 GPU 支持时使用;不支持则回退 fp32(fp16 在此训练中会产生 NaN loss)。
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def _load_components(base_model: str, dtype):

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


def letterbox_rgb(img, side: int, pad: int = 255):
    """把图缩进 side x side,短边补白,不压扁。"""
    from PIL import Image

    w, h = img.size
    scale = side / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = Image.new("RGB", (side, side), (pad, pad, pad))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((side - nw) // 2, (side - nh) // 2))
    return canvas


def build_time_ids(resolution: int, batch_size: int, device=None):
    """SDXL micro-conditioning:[orig_h, orig_w, crop_y, crop_x, target_h, target_w]。"""
    import torch

    ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=torch.long,
        device=device,
    )
    return ids.repeat(batch_size, 1)


def min_snr_weights(noise_scheduler, timesteps, gamma: float):
    """Kohya/diffusers min-SNR:epsilon 预测用 min(snr, gamma)/snr,压低高 SNR(低噪)步。"""
    import torch

    alphas = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha = alphas[timesteps]
    sigma2 = (1.0 - alpha).clamp(min=1e-8)
    snr = alpha / sigma2
    return torch.minimum(snr, snr.new_full(snr.shape, gamma)) / snr.clamp(min=1e-8)


class _CaptionDataset:
    def __init__(self, dataset_dir: Path, resolution: int):
        self.items = []
        for img in sorted(dataset_dir.rglob("*.png")):
            txt = img.with_suffix(".txt")
            caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            self.items.append((img, caption))
        self.resolution = resolution

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        from torchvision import transforms

        path, caption = self.items[idx]
        img = letterbox_rgb(Image.open(path).convert("RGB"), self.resolution)
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
        # SDXL 约定:text_encoder_2 取 hidden_states[-2],而非 last_hidden_state。
        out_2 = text_encoder_2(ids_2.input_ids, output_hidden_states=True)
    prompt_embeds = torch.cat([enc, out_2.hidden_states[-2]], dim=-1)
    pooled_embeds = getattr(out_2, "text_embeds", None)
    if pooled_embeds is None:
        pooled_embeds = out_2.pooler_output
    return prompt_embeds, pooled_embeds


def train(args) -> int:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    device, dtype = _make_train_state(args.dataset_dir)
    comps = _load_components(args.base_model, dtype)

    unet = comps["unet"]
    vae = comps["vae"]
    text_encoder = comps["text_encoder"]
    text_encoder_2 = comps["text_encoder_2"]
    tokenizer = comps["tokenizer"]
    tokenizer_2 = comps["tokenizer_2"]
    noise_scheduler = comps["noise_scheduler"]

    for model in (text_encoder, text_encoder_2):
        model.to(device)
        model.requires_grad_(False)
    # VAE 编码必须 fp32,bf16 会给 latent 加噪声。
    vae.to(device=device, dtype=torch.float32)
    vae.requires_grad_(False)
    unet.to(device)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    unet.add_adapter(lora_config)
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
            images = images.to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(dtype)

            prompt_embeds, pooled = _encode_text(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompts, device)
            prompt_embeds = prompt_embeds.to(dtype)
            pooled = pooled.to(dtype)

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            noisy = noise_scheduler.add_noise(latents, noise, timesteps)
            added = {
                "text_embeds": pooled,
                "time_ids": build_time_ids(args.resolution, bsz, device),
            }

            noise_pred = unet(
                noisy,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added,
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, loss.ndim)))
            if args.snr_gamma > 0:
                loss = loss * min_snr_weights(noise_scheduler, timesteps, args.snr_gamma)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}", flush=True)
            if global_step >= args.max_train_steps:
                break

    peft_state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(unet).items()}
    unet_lora_layers = convert_state_dict_to_diffusers(peft_state)
    StableDiffusionXLPipeline.save_lora_weights(
        str(writer),
        unet_lora_layers=unet_lora_layers,
    )
    from safetensors.torch import save_file

    save_file(peft_state, str(writer / "adapter_model.safetensors"))
    print(f"LoRA saved to {writer}/pytorch_lora_weights.safetensors + adapter_model.safetensors", flush=True)
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
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="min-SNR gamma; 0 disables")
    args = parser.parse_args()

    if not args.dataset_dir.exists() or not list(args.dataset_dir.rglob("*.png")):
        print(f"dataset dir empty: {args.dataset_dir}. run prepare_dataset.py first")
        return 1

    try:
        return train(args)
    except Exception as exc:  # noqa: BLE001
        print(f"train failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
