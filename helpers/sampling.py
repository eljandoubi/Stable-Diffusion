import os
from typing import Optional

import torch
from safetensors.torch import load_file
from torchvision.utils import save_image

from modules import VAE, LatentDiffusion, LDMConfig, NoiseScheduler, sample_latents

from .dataset import get_dataset


def load_vae(config: LDMConfig, path_to_checkpoint: str) -> VAE:
    vae = VAE(config)
    vae.load_state_dict(
        load_file(os.path.join(path_to_checkpoint, "model.safetensors"))
    )
    return vae.eval().requires_grad_(False)


@torch.no_grad()
def generate_images(
    vae: VAE,
    model: LatentDiffusion,
    scheduler: NoiseScheduler,
    num_images: int,
    latent_shape: tuple[int, int, int],
    batch_size: int = 64,
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    eta: float = 0.0,
    class_labels: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample latents, decode them with the VAE and return images in [-1, 1] on the CPU."""
    batches = []
    for start in range(0, num_images, batch_size):
        n = min(batch_size, num_images - start)
        labels = None
        if class_labels is not None:
            labels = class_labels[start : start + n]
        latents = sample_latents(
            model,
            scheduler,
            (n, *latent_shape),
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            class_labels=labels,
            generator=generator,
        )
        images = vae.decode(latents)
        batches.append(images.float().clamp(-1, 1).cpu())
    return torch.cat(batches)


def save_image_folder(images: torch.Tensor, directory: str) -> None:
    """Write images in [-1, 1] as numbered PNG files, the layout FID tools read."""
    os.makedirs(directory, exist_ok=True)
    for idx, image in enumerate(images):
        save_image((image + 1) / 2, os.path.join(directory, f"{idx:05d}.png"))


def load_real_images(
    dataset: str, path_to_dataset: str, img_size: int, num_images: int, seed: int = 0
) -> torch.Tensor:
    """A random subset of a dataset with the evaluation transform, in [-1, 1]."""
    data, _ = get_dataset(
        dataset=dataset,
        path_to_data=path_to_dataset,
        img_size=img_size,
        random_resize=False,
        random_flip_p=0.0,
        train=False,
        return_caption=False,
        return_classes=False,
    )
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))
    return torch.stack([data[i]["images"] for i in order[:num_images].tolist()])


def compute_fid(generated_dir: str, real_dir: str, cuda: bool) -> float:
    import torch_fidelity

    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir, input2=real_dir, cuda=cuda, fid=True, verbose=False
    )
    return metrics["frechet_inception_distance"]
