import os
import argparse
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from dotenv import load_dotenv
from modules.lpips import LPIPSForTraining

load_dotenv()


def download_data(path: str = "./data"):

    import kaggle

    kaggle.api.dataset_download_files(
        "chaitanyakohli678/berkeley-adobe-perceptual-patch-similarity-bapps",
        path=path,
        unzip=True,
        quiet=False,
    )


class BAPPSDataset(Dataset):
    """
    Perceptual Dataset Loader

    Args:
        - path_to_root: Path to BAPPS Dataset Root
        - train: Training Split vs Validation Splits
        - dirs: Which directories do you want to load images?
            train: ["cnn", "mix", "traditional"]
            val: ["cnn", "color", "deblur", "frameinterp", "suprres", "traditional"]
        -img_size: What image size do you want to train on?

    We will be training on 64x64 images, as that is what the LPIPS paper does here
    https://github.com/richzhang/PerceptualSimilarity/blob/master/data/dataset/twoafc_dataset.py

    As far as I can tell, we can inference on any resolution we want later (its a convolution after all)
    and the model seems to be robust to resolution differences. So lets go with this for now!

    """

    def __init__(
        self,
        path_to_root: str,
        train: bool = True,
        dirs: Optional[list[str]] = None,
        img_size: int = 64,
    ):

        if not os.path.exists(path_to_root):
            download_data()

        if train:
            split = "train"
            if dirs is None:
                dirs = ["cnn", "mix", "traditional"]

        else:
            split = "val"
            if dirs is None:
                dirs = [
                    "cnn",
                    "color",
                    "deblur",
                    "frameinterp",
                    "superres",
                    "traditional",
                ]

        if isinstance(dirs, str):
            dirs = [dirs]

        path_to_dirs = [os.path.join(path_to_root, split, dir) for dir in dirs]

        self._generate_dataset(path_to_dirs)

        self.transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                ),  # scales [0,1] -> [-1,1]
            ]
        )

    def _generate_dataset(self, path_to_dirs: list[str]) -> None:

        samples = []
        for dir in path_to_dirs:
            path_to_p0 = os.path.join(dir, "p0")
            path_to_p1 = os.path.join(dir, "p1")
            path_to_ref = os.path.join(dir, "ref")
            path_to_target = os.path.join(dir, "judge")

            file_idxs = [
                file.split(".")[0]
                for file in tqdm(
                    os.listdir(path_to_p0), desc=f"load dataset paths from {dir}"
                )
            ]

            for idx in file_idxs:
                p0 = os.path.join(path_to_p0, f"{idx}.png")
                p1 = os.path.join(path_to_p1, f"{idx}.png")
                ref = os.path.join(path_to_ref, f"{idx}.png")
                target = os.path.join(path_to_target, f"{idx}.npy")

                samples.append((p0, p1, ref, target))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.float32]:

        img1, img2, ref, target = self.samples[idx]

        ### Load Perturbed Images and Original Reference ###
        img1 = self.transforms(Image.open(img1).convert("RGB"))
        img2 = self.transforms(Image.open(img2).convert("RGB"))
        ref = self.transforms(Image.open(ref).convert("RGB"))

        ### Load Labels ###
        target = np.load(target)[0]

        return img1, img2, ref, target


class LRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        total_iterations: int,
        decay_iterations: float,
        min_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_iterations = total_iterations
        self.decay_iterations = decay_iterations
        self.constant_iterations = total_iterations - decay_iterations
        self.min_lr = min_lr if min_lr is not None else 0
        self.current_step = 0

    def step(self):

        if self.current_step < self.constant_iterations:
            lr = self.initial_lr
        else:
            decay_ratio = (
                self.current_step - self.constant_iterations
            ) / self.decay_iterations
            lr = max(self.min_lr, self.initial_lr * (1 - decay_ratio))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1


def compute_accuracy(diff1, diff2, target):

    preds = (diff2 < diff1).flatten().int()
    target = target.flatten()
    accuracy = torch.mean(preds * target + (1 - preds) * (1 - target))

    return accuracy


if __name__ == "__main__":
    ds = BAPPSDataset("data/dataset/2afc")
    print(len(ds))
    im1, im2, ref, tr = ds[0]
    print(im1.shape)
    print(im2.shape)
    print(ref.shape)
    print(type(tr).__name__, tr)
