import math
from typing import Optional

import torch
import torch.nn as nn

from .config import LDMConfig
from .unet import UNet2DModel


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep features followed by a two layer MLP."""

    def __init__(self, start_dim: int, proj_dim: int, requires_grad: bool = True):
        super().__init__()
        half = start_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.freqs = nn.Parameter(freqs, requires_grad=requires_grad)
        self.mlp = nn.Sequential(
            nn.Linear(2 * half, proj_dim), nn.SiLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        angles = timesteps.float()[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))


class LatentDiffusion(nn.Module):
    """The denoiser of the latent diffusion model: timestep and class embeddings feed the UNet."""

    def __init__(self, config: LDMConfig):
        super().__init__()
        self.config = config
        self.time_embedding = TimeEmbedding(
            config.time_embed_start_dim,
            config.time_embed_proj_dim,
            config.time_embed_requires_grad,
        )
        self.class_embedding = None
        if config.class_conditioning:
            ### One extra row is the "no class" label used by classifier-free guidance ###
            self.class_embedding = nn.Embedding(
                config.num_classes + 1, config.class_embed_dim
            )
        self.unet = UNet2DModel(config)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_conditioning: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        class_embed = None
        if class_labels is not None:
            class_embed = self.class_embedding(class_labels)
        return self.unet(
            latents,
            self.time_embedding(timesteps),
            text_conditioning,
            text_attention_mask,
            class_embed,
        )


class NoiseScheduler:
    """Linear beta schedule with the forward noising step and the DDIM update."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    @classmethod
    def from_config(cls, config: LDMConfig, device: torch.device | str = "cpu"):
        return cls(
            config.num_diffusion_timesteps, config.beta_start, config.beta_end, device
        )

    def add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * latents + (1 - alpha).sqrt() * noise

    def ddim_step(
        self,
        latents: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        ### prev_timestep = -1 means the clean latent ###
        alpha_t = self.alphas_cumprod[timestep]
        if prev_timestep >= 0:
            alpha_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_prev = torch.ones_like(alpha_t)

        x0_estimate = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        sigma = (
            eta
            * ((1 - alpha_prev) / (1 - alpha_t)).sqrt()
            * (1 - alpha_t / alpha_prev).sqrt()
        )
        direction = (1 - alpha_prev - sigma**2).clamp(min=0).sqrt() * noise_pred
        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=generator,
        )
        return alpha_prev.sqrt() * x0_estimate + direction + sigma * noise


@torch.no_grad()
def sample_latents(
    model: LatentDiffusion,
    scheduler: NoiseScheduler,
    shape: tuple[int, int, int, int],
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    eta: float = 0.0,
    class_labels: Optional[torch.Tensor] = None,
    text_conditioning: Optional[torch.Tensor] = None,
    text_attention_mask: Optional[torch.Tensor] = None,
    null_text_conditioning: Optional[torch.Tensor] = None,
    null_text_attention_mask: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Strided DDIM sampling in latent space, with classifier-free guidance when
    a condition is given and `guidance_scale` is above 1."""
    device = scheduler.alphas_cumprod.device
    latents = torch.randn(shape, device=device, generator=generator)
    steps = torch.linspace(scheduler.num_timesteps - 1, 0, num_steps).round().long()
    steps = steps.tolist()

    conditioned = class_labels is not None or text_conditioning is not None
    null_labels = None
    if class_labels is not None:
        null_labels = torch.full_like(class_labels, model.config.num_classes)

    for i, t in enumerate(steps):
        timesteps = torch.full((shape[0],), t, device=device)
        noise_pred = model(
            latents, timesteps, text_conditioning, text_attention_mask, class_labels
        )
        if conditioned and guidance_scale != 1.0:
            uncond = model(
                latents,
                timesteps,
                null_text_conditioning,
                null_text_attention_mask,
                null_labels,
            )
            noise_pred = uncond + guidance_scale * (noise_pred - uncond)
        prev = steps[i + 1] if i + 1 < len(steps) else -1
        latents = scheduler.ddim_step(latents, t, prev, noise_pred, eta, generator)
    return latents
