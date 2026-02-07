import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from accelerate import Accelerator
from dotenv import load_dotenv
from modules.lpips import LPIPS, DiffToLogits

load_dotenv()


def download_data(path: str = "./data"):

    import kaggle

    kaggle.api.dataset_download_files(
        "chaitanyakohli678/berkeley-adobe-perceptual-patch-similarity-bapps",
        path=path,
        unzip=True,
        quiet=False,
    )


if __name__ == "__main__":
    download_data()
