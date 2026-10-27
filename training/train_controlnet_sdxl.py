from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image


def _resolve_base_model(base_model: str) -> str:
    candidate = Path(base_model).expanduser()
    if candidate.exists():
        return str(candidate)
    try:
        from modelscope import snapshot_download
    except ImportError:
        return base_model
    return str(snapshot_download(base_model))


def _build_time_ids(resolution: int, bs: int):
    import torch

    return torch.tensor(
        [resolution, resolution, resolution, resolution, 0, 0],
        dtype=torch.long,
    ).repeat(bs, 1)


class _ControlNetCaptionDataset:
    def __init__(self, dataset_dir: Path, resolution: int):
        import pandas as pd

        csv_path = dataset_dir / "captions.csv"
        rows = pd.read_csv(csv_path)
        self.pairs = []
        for _, row in rows.iterrows():
            img = dataset_dir / row["image"]
            cond = dataset_dir / row["conditioning"]
            if img.exists() and cond.exists():
                self.pairs.append((img, cond, str(row["caption"])))
        self.resolution = resolution

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        from torchvision import transforms

        img_path, cond_path, caption = self.pairs[idx]
        img = Image.open(img_path).convert("RGB").resize((self.resolution, self.resolution), Image.LANCZOS)
        cond = Image.open(cond_path).convert("RGB").resize((self.resolution, self.resolution), Image.LANCZOS)
        t = transforms.ToTensor()
        # 训练目标图归一化到 [-1,1] 供 VAE;ControlNet 条件图保持 [0,1](与推理端一致)。
        return t(img) * 2.0 - 1.0, t(cond), caption


def _encode_prompts(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompts, device, dtype):
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
        hidden = text_encoder(ids.input_ids).last_hidden_state
        # SDXL 约定:text_encoder_2 取 hidden_states[-2],而非 last_hidden_state。
        out_2 = text_encoder_2(ids_2.input_ids, output_hidden_states=True)
    prompt_embeds = torch.cat([hidden, out_2.hidden_states[-2]], dim=-1).to(dtype)
    pooled = getattr(out_2, "text_embeds", None)
    if pooled is None:
        pooled = out_2.pooler_output
    return prompt_embeds, pooled.to(dtype)


def _build_models(base_model: str, dtype):

    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    base = _resolve_base_model(base_model)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
    controlnet = ControlNetModel.from_unet(unet)
    controlnet = controlnet.to(dtype=dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", torch_dtype=dtype)
    return {
        "base": base,
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
    }


def train(args) -> int:
    import torch
    from diffusers.utils import convert_all_state_dict_to_peft  # noqa: F401
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # bf16 仅当 GPU 支持时使用;不支持则回退 fp32(fp16 在此训练中会产生 NaN loss)。
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    components = _build_models(args.base_model, dtype)
    vae = components["vae"]
    unet = components["unet"]
    controlnet = components["controlnet"]
    noise_scheduler = components["noise_scheduler"]
    tokenizer = components["tokenizer"]
    tokenizer_2 = components["tokenizer_2"]
    text_encoder = components["text_encoder"]
    text_encoder_2 = components["text_encoder_2"]

    vae = vae.to(device=device, dtype=dtype)
    unet = unet.to(device=device, dtype=dtype)
    controlnet = controlnet.to(device=device, dtype=dtype)
    text_encoder.to(device=device, dtype=dtype)
    text_encoder_2.to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    params = [p for p in controlnet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    dataset = _ControlNetCaptionDataset(args.dataset_dir, args.resolution)
    if len(dataset) == 0:
        print("empty dataset; run prepare_controlnet_data.py first")
        return 1
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    print(f"controlnet steps: {args.max_train_steps} on {len(dataset)} pairs", flush=True)

    writer = Path(args.output_dir)
    writer.mkdir(parents=True, exist_ok=True)

    global_step = 0
    while global_step < args.max_train_steps:
        for images, cond_images, captions in loader:
            images = images.to(device=device, dtype=dtype)
            cond_images = cond_images.to(device=device, dtype=dtype)

            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(dtype)

            prompt_embeds, pooled = _encode_prompts(
                tokenizer, tokenizer_2, text_encoder, text_encoder_2, captions, device, dtype
            )
            time_ids = _build_time_ids(args.resolution, images.shape[0]).to(device)
            added_cond = {"text_embeds": pooled, "time_ids": time_ids}

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond,
                controlnet_cond=cond_images,
                return_dict=False,
            )

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond,
                down_block_additional_residuals=[sample.to(dtype=dtype) for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=dtype),
                return_dict=False,
            )[0]

            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % args.log_interval == 0:
                print(f"step {global_step}/{args.max_train_steps} loss={loss.item():.4f}", flush=True)
            if global_step >= args.max_train_steps:
                break

    controlnet.save_pretrained(str(writer))
    print(f"controlnet saved to {writer}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny SDXL ControlNet trainer")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--base_model", default="AI-ModelScope/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", default="models/sdxl/controlnet_openpose", type=Path)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=800)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    if not (args.dataset_dir / "captions.csv").exists():
        print(f"dataset meta missing: {args.dataset_dir}/captions.csv")
        return 1

    try:
        return train(args)
    except Exception as exc:  # noqa: BLE001
        print(f"train failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
