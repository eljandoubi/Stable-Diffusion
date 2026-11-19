"""Sample images from a trained latent diffusion model and optionally score them with FID."""

import argparse
import json
import os
import time

import torch
from safetensors.torch import load_file
from torchvision.utils import save_image

from helpers import (
    DATASETS,
    compute_fid,
    count_num_params,
    generate_images,
    load_real_images,
    load_vae,
    save_image_folder,
)
from modules import LatentDiffusion, LDMConfig, NoiseScheduler


def load_ldm(config: LDMConfig, path_to_checkpoint: str, use_ema: bool = True):
    model = LatentDiffusion(config)
    name = "ema.safetensors" if use_ema else "model.safetensors"
    model.load_state_dict(load_file(os.path.join(path_to_checkpoint, name)))
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent diffusion sampling and FID")
    parser.add_argument("--model_config", required=True, type=str)
    parser.add_argument(
        "--vae_checkpoint",
        required=True,
        type=str,
        help="Folder with the VAE model.safetensors",
    )
    parser.add_argument(
        "--ldm_checkpoint", required=True, type=str, help="Folder with ema.safetensors"
    )
    parser.add_argument(
        "--scale_factor_file",
        default=None,
        type=str,
        help="Defaults to scale_factor.json next to the checkpoint folder",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Use the raw weights instead of the EMA weights",
    )
    parser.add_argument("--num_images", default=64, type=int)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--guidance_scale", default=1.0, type=float)
    parser.add_argument("--eta", default=0.0, type=float)
    parser.add_argument("--batch_size", default=250, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fp16", action="store_true", help="Run the networks in fp16")
    parser.add_argument("--grid_out", default="samples.png", type=str)
    parser.add_argument(
        "--fid_dataset",
        default=None,
        choices=DATASETS,
        help="Score against this dataset",
    )
    parser.add_argument("--fid_path_to_dataset", default=None, type=str)
    parser.add_argument("--work_dir", default="fid_work", type=str)
    parser.add_argument("--result_file", default="ldm_eval.json", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LDMConfig.from_yaml(args.model_config)
    scale_file = args.scale_factor_file or os.path.join(
        os.path.dirname(os.path.normpath(args.ldm_checkpoint)), "scale_factor.json"
    )
    with open(scale_file) as f:
        scale_factor = json.load(f)["scale_factor"]

    vae = load_vae(config, args.vae_checkpoint).to(device)
    vae.scale_factor = scale_factor
    model = load_ldm(config, args.ldm_checkpoint, use_ema=not args.no_ema).to(device)
    scheduler = NoiseScheduler.from_config(config, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_shape = (
        config.latent_channels,
        config.latent_resolution,
        config.latent_resolution,
    )

    labels = None
    if config.class_conditioning:
        labels = torch.arange(args.num_images, device=device) % config.num_classes

    start = time.time()
    with torch.autocast(device.type, dtype=torch.float16, enabled=args.fp16):
        images = generate_images(
            vae,
            model,
            scheduler,
            args.num_images,
            latent_shape,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            class_labels=labels,
            generator=generator,
        )
    seconds = time.time() - start
    save_image((images[:64] + 1) / 2, args.grid_out, nrow=8)

    result = {
        "num_images": args.num_images,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "ema": not args.no_ema,
        "sampling_seconds": round(seconds, 1),
        "ldm_parameters": count_num_params(model),
        "vae_parameters": count_num_params(vae),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    if args.fid_dataset is not None:
        generated_dir = os.path.join(args.work_dir, "generated")
        real_dir = os.path.join(args.work_dir, "real")
        save_image_folder(images, generated_dir)
        real = load_real_images(
            args.fid_dataset,
            args.fid_path_to_dataset,
            config.img_size,
            args.num_images,
            args.seed,
        )
        save_image_folder(real, real_dir)
        result["fid"] = compute_fid(generated_dir, real_dir, device.type == "cuda")

    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
