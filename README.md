# Stable-Diffusion

Latent diffusion from scratch in PyTorch: a KL autoencoder trained with an L1 + LPIPS loss, a UNet with cross-attention blocks, the DDPM objective in latent space, a DDIM sampler with classifier-free guidance, EMA weights, and PSNR / LPIPS / FID evaluation. An LPIPS network reproduced against the original is included as well.

## Contents

- [What is in the repository](#what-is-in-the-repository)
- [Installation](#installation)
- [Usage](#usage)
- [Status](#status)
- [LPIPS evaluation](#lpips-evaluation-on-the-validation-sets)
- [Development](#development)

## What is in the repository

```
modules/
├── vae.py            # KL autoencoder: residual blocks, attention, latent projections
├── lpips.py          # LPIPS network, reproduced and trainable (see the LPIPS section)
├── discriminator.py  # PatchGAN (off in the short CelebA run)
├── unet.py           # UNet2DModel: Down, Mid and Up blocks, optional attention
├── transformer.py    # self- and cross-attention transformer blocks
├── diffusion.py      # LatentDiffusion, NoiseScheduler, sample_latents (DDIM + guidance)
└── config.py         # LDMConfig, read from a YAML file
helpers/              # datasets, shared command-line options, sampling and FID helpers
configs/              # celeba64*.yaml (the CelebA setting below), ldm.yaml and vae_train.yaml (Conceptual Captions)
vae_trainer.py        # train the autoencoder
ldm_trainer.py        # train the latent diffusion model (EMA, guidance dropout, scale factor)
eval_vae.py           # PSNR, LPIPS-VGG and rFID of reconstructions
ldm_sample.py         # generate images and score them with FID
inference_vae.py      # reconstruct images with a trained autoencoder
lpips_trainer.py      # train the LPIPS linear heads
tests/                # model shapes, sampler maths, an end-to-end CPU training and sampling run
```

- **Two stages.** The autoencoder maps a 64 x 64 image to a 4 x 16 x 16 latent (12 times fewer values) and the diffusion model works there. After training the autoencoder, `ldm_trainer.py` measures the standard deviation of the latents on 50 batches and stores `scale_factor.json`, so the diffusion model sees unit-variance latents.
- **Sizes.** In the CelebA configuration the autoencoder has 13.9M parameters and the diffusion UNet 32.9M.
- **Sampling.** `sample_latents` runs a strided DDIM sequence and, when a model has class conditioning, classifier-free guidance. Class and text conditioning (an extra null embedding for guidance) are implemented and unit-tested, but the CelebA setting is unconditional.
- **Weights.** Evaluation uses the EMA weights (decay 0.999) saved in `ema.safetensors`.

## Installation

```bash
git clone https://github.com/eljandoubi/Stable-Diffusion.git
cd Stable-Diffusion
uv sync --extra eval
```

## Usage

The CelebA 64 x 64 setting, in three steps. On a T4 add `ACCELERATE_MIXED_PRECISION=fp16` (the default config asks for bf16).

```bash
# 1. autoencoder, 10000 steps at batch size 32
uv run accelerate launch --config_file accelerate_config.yaml vae_trainer.py \
  --experiment_name vae_celeba64 --wandb_run_name vae_celeba64 --no-log_wandb \
  --working_directory work_dir --training_config configs/celeba64_vae_train.yaml \
  --model_config configs/celeba64.yaml --dataset hf_images \
  --path_to_dataset nielsr/CelebA-faces --path_to_save_gens gens/vae_celeba64

# 2. latent diffusion, 15000 steps at batch size 64
uv run accelerate launch --config_file accelerate_config.yaml ldm_trainer.py \
  --experiment_name ldm_celeba64 --wandb_run_name ldm_celeba64 --no-log_wandb \
  --working_directory work_dir --training_config configs/celeba64_ldm_train.yaml \
  --model_config configs/celeba64.yaml --dataset hf_images \
  --path_to_dataset nielsr/CelebA-faces --path_to_save_gens gens/ldm_celeba64

# 3. evaluate: reconstructions, then 5000 samples with 50 DDIM steps
uv run eval_vae.py --model_config configs/celeba64.yaml \
  --vae_checkpoint work_dir/vae_celeba64/checkpoint_10000 \
  --dataset hf_images --path_to_dataset nielsr/CelebA-faces
uv run ldm_sample.py --model_config configs/celeba64.yaml \
  --vae_checkpoint work_dir/vae_celeba64/checkpoint_10000 \
  --ldm_checkpoint work_dir/ldm_celeba64/checkpoint_15000 \
  --num_images 5000 --num_steps 50 --fp16 \
  --fid_dataset hf_images --fid_path_to_dataset nielsr/CelebA-faces
```

`eval_vae.py` writes `vae_eval.json` (PSNR, LPIPS-VGG, rFID of the reconstructions) and `ldm_sample.py` writes `ldm_eval.json` (FID, sampling time) and a grid of 64 samples.

## Status

The pipeline runs end to end in the tests on CPU, and every stage (autoencoder, diffusion model, sampling, PSNR / LPIPS / FID scripts) ran for a 30-step smoke test on a Colab T4. The full CelebA run above has not been completed, so this README reports no PSNR or FID for trained models yet. The only measured result is the LPIPS reproduction below.

## LPIPS evaluation on the validation sets

Accuracy of the reproduced LPIPS against the original on the validation sets of each distortion type.

| Dataset        | LPIPS Reproduction | Original LPIPS |
|---------------|-------------------:|---------------:|
| CNN           | 0.826              | 0.821          |
| TRADITIONAL   | 0.769              | 0.732          |
| COLOR         | 0.623              | 0.615          |
| DEBLUR        | 0.594              | 0.594          |
| FRAMEINTERP   | 0.629              | 0.622          |
| SUPERRES      | 0.694              | 0.695          |

- The reproduced LPIPS closely matches the original implementation across all datasets.
- Differences are minor and within expected variance.
- Exact parity is observed on the **DEBLUR** dataset.

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

CI runs the same three steps on every push and pull request.
