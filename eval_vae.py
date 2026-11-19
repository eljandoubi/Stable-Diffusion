"""Reconstruction quality of a trained VAE: PSNR, LPIPS and the FID between reconstructions and originals."""

import argparse
import json
import os

import lpips
import torch

from helpers import (
    DATASETS,
    compute_fid,
    count_num_params,
    load_real_images,
    load_vae,
    save_image_folder,
)
from modules import LDMConfig


@torch.no_grad()
def reconstruct(vae, images: torch.Tensor, batch_size: int, device) -> torch.Tensor:
    """Decode the posterior mean of every image, so the result is deterministic."""
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        mu = vae.encode(batch, return_stats=True)["mu"]
        outputs.append(vae.forward_dec(mu).float().clamp(-1, 1).cpu())
    return torch.cat(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE reconstruction metrics")
    parser.add_argument("--model_config", required=True, type=str)
    parser.add_argument("--vae_checkpoint", required=True, type=str)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--path_to_dataset", required=True, type=str)
    parser.add_argument("--num_images", default=5000, type=int)
    parser.add_argument("--batch_size", default=250, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--work_dir", default="vae_eval_work", type=str)
    parser.add_argument("--result_file", default="vae_eval.json", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LDMConfig.from_yaml(args.model_config)
    vae = load_vae(config, args.vae_checkpoint).to(device)

    originals = load_real_images(
        args.dataset, args.path_to_dataset, config.img_size, args.num_images, args.seed
    )
    reconstructions = reconstruct(vae, originals, args.batch_size, device)

    mse = ((originals - reconstructions) / 2).pow(2).flatten(1).mean(1)
    psnr = (10 * torch.log10(1 / mse)).mean().item()

    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(originals), args.batch_size):
            a = originals[start : start + args.batch_size].to(device)
            b = reconstructions[start : start + args.batch_size].to(device)
            scores.append(lpips_fn(a, b).flatten().cpu())
    lpips_score = torch.cat(scores).mean().item()

    original_dir = os.path.join(args.work_dir, "original")
    reconstruction_dir = os.path.join(args.work_dir, "reconstruction")
    save_image_folder(originals, original_dir)
    save_image_folder(reconstructions, reconstruction_dir)

    result = {
        "num_images": args.num_images,
        "psnr_db": round(psnr, 3),
        "lpips_vgg": round(lpips_score, 4),
        "rfid": compute_fid(reconstruction_dir, original_dir, device.type == "cuda"),
        "latent_shape": [
            config.latent_channels,
            config.latent_resolution,
            config.latent_resolution,
        ],
        "vae_parameters": count_num_params(vae),
    }
    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
