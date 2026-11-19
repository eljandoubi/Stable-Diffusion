"""Train the latent diffusion model on the latents of a trained VAE."""

import argparse
import copy
import json
import os
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers.optimization import get_scheduler

from helpers import (
    add_common_arguments,
    count_num_params,
    generate_images,
    get_dataset,
    load_vae,
)
from modules import LatentDiffusion, LDMConfig, NoiseScheduler, VAE

LOSSES = {"mse": F.mse_loss, "l1": F.l1_loss, "huber": F.huber_loss}


class EMA:
    """Exponential moving average of the model weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1 - self.decay
                )
            else:
                self.shadow[name].copy_(value)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = state


@torch.no_grad()
def estimate_scale_factor(
    vae: VAE, dataloader: DataLoader, num_batches: int, device
) -> float:
    """1 / std of the latents, so the diffusion model sees unit variance."""
    total, total_sq, count = 0.0, 0.0, 0
    for i, batch in enumerate(dataloader):
        if i == num_batches:
            break
        z = vae.encode(batch["images"].to(device), scale_factor=1.0)[
            "posterior"
        ].float()
        total += z.sum().item()
        total_sq += z.pow(2).sum().item()
        count += z.numel()
    mean = total / count
    return 1.0 / (total_sq / count - mean**2) ** 0.5


def drop_conditions(batch: dict[str, Any], config: LDMConfig):
    """Return the class labels and text context of a batch, replacing a random
    share of them by the empty condition so the model also learns the unconditional case."""
    labels = batch.get("class_conditioning")
    context = batch.get("text_conditioning")
    mask = batch.get("text_attention_mask")
    ref = labels if labels is not None else context
    if ref is None:
        return None, None, None

    drop = torch.rand(ref.shape[0], device=ref.device) < config.cfg_dropout_prob
    if labels is not None:
        labels = torch.where(drop, torch.full_like(labels, config.num_classes), labels)
    if context is not None:
        context = context.clone()
        mask = mask.clone()
        context[drop] = 0
        mask[drop] = False
        mask[drop, 0] = True
    return labels, context, mask


def save_checkpoint(accelerator: Accelerator, ema: EMA, path: str) -> None:
    accelerator.save_state(output_dir=path)
    if accelerator.is_main_process:
        weights = {k: v.contiguous() for k, v in ema.state_dict().items()}
        save_file(weights, os.path.join(path, "ema.safetensors"))


def save_sample_grid(accelerator, vae, model, ema, config, path_to_folder, step, args):
    ema_model = copy.deepcopy(accelerator.unwrap_model(model))
    ema.copy_to(ema_model)
    ema_model.eval()
    scheduler = NoiseScheduler.from_config(config, accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(0)
    labels = None
    if config.class_conditioning:
        labels = (
            torch.arange(args["num_samples"], device=accelerator.device)
            % config.num_classes
        )
    with accelerator.autocast():
        images = generate_images(
            vae,
            ema_model,
            scheduler,
            args["num_samples"],
            (
                config.latent_channels,
                config.latent_resolution,
                config.latent_resolution,
            ),
            num_steps=args["sample_steps"],
            guidance_scale=args["guidance_scale"],
            class_labels=labels,
            generator=generator,
        )
    os.makedirs(path_to_folder, exist_ok=True)
    save_image(
        (images + 1) / 2, os.path.join(path_to_folder, f"iteration_{step}.png"), nrow=8
    )


def train(args: argparse.Namespace) -> None:
    with open(args.training_config) as f:
        training_config: dict[str, Any] = yaml.safe_load(f)["training_args"]
    config = LDMConfig.from_yaml(args.model_config)

    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        gradient_accumulation_steps=training_config["gradient_accumulations_steps"],
        log_with="wandb" if args.log_wandb else None,
    )
    if args.log_wandb:
        accelerator.init_trackers(
            args.experiment_name, init_kwargs={"wandb": {"name": args.wandb_run_name}}
        )

    vae = load_vae(config, training_config["vae_checkpoint"]).to(accelerator.device)

    dataset, collate_fn = get_dataset(
        dataset=args.dataset,
        path_to_data=args.path_to_dataset,
        num_channels=config.in_channels,
        img_size=config.img_size,
        random_resize=training_config["random_resize"],
        interpolation=training_config["interpolation"],
        random_flip_p=training_config["random_flip_p"],
        return_caption=config.text_conditioning,
        return_classes=config.class_conditioning,
        pre_encoded_text=config.pre_encoded_text,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["per_gpu_batch_size"]
        // training_config["gradient_accumulations_steps"],
        collate_fn=collate_fn,
        pin_memory=training_config["pin_memory"],
        num_workers=training_config["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    accelerator.print("Number of Training Samples:", len(dataset))

    ### The scale factor is measured once and stored next to the checkpoints ###
    scale_path = os.path.join(path_to_experiment, "scale_factor.json")
    if config.vae_scale_factor is not None:
        scale_factor = config.vae_scale_factor
    else:
        if accelerator.is_main_process and not os.path.exists(scale_path):
            os.makedirs(path_to_experiment, exist_ok=True)
            scale = estimate_scale_factor(
                vae,
                dataloader,
                training_config["scale_factor_batches"],
                accelerator.device,
            )
            with open(scale_path, "w") as f:
                json.dump({"scale_factor": scale}, f)
        accelerator.wait_for_everyone()
        with open(scale_path) as f:
            scale_factor = json.load(f)["scale_factor"]
    accelerator.print(f"VAE scale factor: {scale_factor:.4f}")

    model = LatentDiffusion(config)
    accelerator.print(f"NUMBER OF PARAMETERS: {count_num_params(model)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        betas=(training_config["optimizer_beta1"], training_config["optimizer_beta2"]),
        weight_decay=training_config["optimizer_weight_decay"],
    )
    lr_scheduler = get_scheduler(
        training_config["lr_scheduler"],
        optimizer=optimizer,
        num_training_steps=training_config["total_training_iterations"],
        num_warmup_steps=training_config["lr_warmup_steps"],
    )
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader
    )
    ema = EMA(accelerator.unwrap_model(model), training_config["ema_decay"])
    accelerator.register_for_checkpointing(ema)
    scheduler = NoiseScheduler.from_config(config, accelerator.device)
    loss_fn = LOSSES[config.diffusion_loss_fn]

    global_step = 0
    if args.resume_from_checkpoint is not None:
        accelerator.load_state(
            os.path.join(path_to_experiment, args.resume_from_checkpoint)
        )
        global_step = int(args.resume_from_checkpoint.split("_")[-1])

    total_steps = training_config["total_training_iterations"]
    progress_bar = tqdm(
        range(total_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    while global_step < total_steps:
        model.train()
        for batch in dataloader:
            with torch.no_grad(), accelerator.autocast():
                latents = vae.encode(batch["images"], scale_factor=scale_factor)[
                    "posterior"
                ]
            latents = latents.float()
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                config.num_diffusion_timesteps,
                (latents.shape[0],),
                device=latents.device,
            )
            noisy = scheduler.add_noise(latents, noise, timesteps)
            labels, context, mask = drop_conditions(batch, config)

            with accelerator.accumulate(model):
                noise_pred = model(noisy, timesteps, context, mask, labels)
                loss = loss_fn(noise_pred.float(), noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), training_config["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            ema.update(accelerator.unwrap_model(model))
            global_step += 1
            progress_bar.update(1)
            logs = {
                "loss": accelerator.gather_for_metrics(loss.detach()).mean().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(loss=round(logs["loss"], 4))

            if (
                global_step % training_config["sample_freq"] == 0
                and accelerator.is_main_process
            ):
                save_sample_grid(
                    accelerator,
                    vae,
                    model,
                    ema,
                    config,
                    args.path_to_save_gens,
                    global_step,
                    training_config,
                )
            if (
                global_step % training_config["checkpoint_iterations"] == 0
                or global_step == total_steps
            ):
                save_checkpoint(
                    accelerator,
                    ema,
                    os.path.join(path_to_experiment, f"checkpoint_{global_step}"),
                )
            if global_step >= total_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent diffusion training")
    add_common_arguments(parser)
    train(parser.parse_args())
