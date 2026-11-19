import pytest
import torch

from modules import (
    VAE,
    LatentDiffusion,
    LDMConfig,
    NoiseScheduler,
    TimeEmbedding,
    sample_latents,
)


def tiny_config(**overrides) -> LDMConfig:
    settings = dict(
        img_size=32,
        latent_channels=4,
        vae_channels_per_block=(16, 32),
        down_block_types=("AttnDown", "Down"),
        mid_block_types="AttnMid",
        up_block_types=("Up", "AttnUp"),
        unet_channels_per_block=(32, 64),
        time_embed_start_dim=16,
        time_embed_proj_dim=64,
        groupnorm_groups=4,
        attention_head_dim=8,
        num_diffusion_timesteps=20,
    )
    settings.update(overrides)
    return LDMConfig(**settings)


def test_time_embedding_shape_and_frozen_frequencies():
    frozen = TimeEmbedding(16, 64, requires_grad=False)
    assert frozen(torch.tensor([0, 5, 10])).shape == (3, 64)
    assert not frozen.freqs.requires_grad
    assert TimeEmbedding(16, 64).freqs.requires_grad


def test_different_timesteps_give_different_embeddings():
    embedding = TimeEmbedding(16, 64)
    out = embedding(torch.tensor([1, 900]))
    assert not torch.allclose(out[0], out[1])


def test_vae_reconstructs_the_input_shape_and_scales_the_latent():
    torch.manual_seed(0)
    vae = VAE(tiny_config()).eval()
    images = torch.randn(2, 3, 32, 32)
    output = vae(images)
    assert output["reconstruction"].shape == images.shape
    assert output["posterior"].shape == (2, 4, 16, 16)
    assert output["kl_loss"].shape == (2,)

    stats = vae.encode(images, return_stats=True)
    doubled = vae.encode(images, return_stats=True, scale_factor=2.0)
    assert torch.allclose(stats["mu"], doubled["mu"])
    assert vae.decode(torch.zeros(1, 4, 16, 16)).shape == (1, 3, 32, 32)


def test_unconditional_denoiser_keeps_the_latent_shape_and_trains():
    model = LatentDiffusion(tiny_config())
    latents = torch.randn(2, 4, 16, 16)
    out = model(latents, torch.tensor([3, 12]))
    assert out.shape == latents.shape
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in model.unet.parameters())


def test_class_conditioning_uses_an_extra_null_label():
    model = LatentDiffusion(
        tiny_config(class_conditioning=True, num_classes=5, class_embed_dim=16)
    )
    assert model.class_embedding.num_embeddings == 6
    latents = torch.randn(2, 4, 16, 16)
    timesteps = torch.tensor([3, 12])
    labelled = model(latents, timesteps, class_labels=torch.tensor([0, 4]))
    null = model(latents, timesteps, class_labels=torch.tensor([5, 5]))
    assert labelled.shape == null.shape == latents.shape
    assert not torch.allclose(labelled, null)


def test_text_conditioning_needs_a_context_and_respects_the_mask():
    model = LatentDiffusion(tiny_config(text_conditioning=True, text_embed_dim=24))
    latents = torch.randn(2, 4, 16, 16)
    timesteps = torch.tensor([3, 12])
    with pytest.raises(Exception, match="Context"):
        model(latents, timesteps)

    context = torch.randn(2, 5, 24)
    full = torch.ones(2, 5, dtype=torch.bool)
    partial = full.clone()
    partial[:, 3:] = False
    out_full = model(latents, timesteps, context, full)
    out_partial = model(latents, timesteps, context, partial)
    assert out_full.shape == latents.shape
    assert not torch.allclose(out_full, out_partial)


def test_noise_scheduler_follows_the_closed_form():
    scheduler = NoiseScheduler(num_timesteps=20)
    latents = torch.randn(3, 4, 4, 4)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([0, 7, 19])
    noisy = scheduler.add_noise(latents, noise, timesteps)
    alpha = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    assert torch.allclose(noisy, alpha.sqrt() * latents + (1 - alpha).sqrt() * noise)


def test_ddim_step_with_the_true_noise_follows_the_forward_process():
    scheduler = NoiseScheduler(num_timesteps=50)
    latents = torch.randn(2, 4, 4, 4)
    noise = torch.randn_like(latents)
    t, prev = 40, 15
    noisy = scheduler.add_noise(latents, noise, torch.tensor([t, t]))
    stepped = scheduler.ddim_step(noisy, t, prev, noise)
    expected = scheduler.add_noise(latents, noise, torch.tensor([prev, prev]))
    assert torch.allclose(stepped, expected, atol=1e-4)
    last = scheduler.ddim_step(noisy, t, -1, noise)
    assert torch.allclose(last, latents, atol=1e-4)


def test_sampling_is_reproducible_and_guidance_changes_the_result():
    torch.manual_seed(0)
    config = tiny_config(class_conditioning=True, num_classes=5, class_embed_dim=16)
    model = LatentDiffusion(config).eval()
    scheduler = NoiseScheduler.from_config(config)
    labels = torch.tensor([1, 2])

    def run(scale):
        generator = torch.Generator().manual_seed(7)
        return sample_latents(
            model,
            scheduler,
            (2, 4, 16, 16),
            num_steps=3,
            guidance_scale=scale,
            class_labels=labels,
            generator=generator,
        )

    plain = run(1.0)
    assert plain.shape == (2, 4, 16, 16)
    assert torch.equal(plain, run(1.0))
    assert not torch.allclose(plain, run(3.0))
