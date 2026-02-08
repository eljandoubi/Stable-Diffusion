import os
from typing import Literal
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer
from datasets import load_from_disk

from datasets import disable_caching

disable_caching()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def image_transforms(
    num_channels: int = 3,
    img_size: int = 256,
    random_resize: bool = True,
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
    random_flip_p: float = 0,
    train: bool = True,
):

    assert interpolation in ["nearest", "bilinear", "bicubic"]

    interpolation_dict = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }

    if random_resize and train:
        resize = transforms.RandomResizedCrop(
            img_size, scale=(0.6, 1.0), interpolation=interpolation_dict[interpolation]
        )
    else:
        resize = transforms.Resize((img_size, img_size))

    if not train:
        random_flip_p = 0

    image2tensor = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: img.convert("RGB") if num_channels == 3 else img
            ),
            resize,
            transforms.RandomHorizontalFlip(p=random_flip_p),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(num_channels)], [0.5 for _ in range(num_channels)]
            ),
        ]
    )

    return image2tensor
