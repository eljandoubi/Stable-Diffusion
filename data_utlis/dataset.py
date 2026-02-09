import os
from typing import Literal, Callable, Optional
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


class GenericImageDataset(Dataset):
    """
    Generic Image Dataset

    Args:
        - path_to_data: Points to a folder full of images of faces
        - nested: Does that path_to_data contain folders, in which there are images
        - return_classes: Do you want to return the class label (only available when nested=True)

    """

    def __init__(
        self,
        path_to_data: str,
        nested: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        return_classes=True,
    ):

        self.transforms = transform
        self.return_classes = return_classes

        if not nested:
            if return_classes:
                raise Exception("Dataset is not nested, there are no class labels!")
            self.path_to_files = [
                os.path.join(path_to_data, file) for file in os.listdir(path_to_data)
            ]
            self.classes = None
        else:
            self.path_to_files = []
            classes = os.listdir(path_to_data)
            self.classes = {cls: idx for (idx, cls) in enumerate(classes)}

            for dir in classes:
                path_to_dir = os.path.join(path_to_data, dir)
                self.path_to_files.extend(
                    [
                        os.path.join(path_to_dir, file)
                        for file in os.listdir(path_to_dir)
                    ]
                )

    def __len__(self):
        return len(self.path_to_files)

    def __getitem__(self, idx: int):
        img_path = self.path_to_files[idx]

        if self.classes is not None:
            class_label = self.classes[img_path.split("/")[-2]]

        img = Image.open(img_path)
        img = self.transforms(img)

        if self.return_classes:
            return {"images": img, "class_conditioning": class_label}
        else:
            return {"images": img}


def conceptual_captions(
    path_to_data: str, transforms: Callable[[Image.Image], torch.Tensor]
):

    dataset = load_from_disk(path_to_data)

    def sample_transforms(batch):

        transformed_images = [transforms(image) for image in batch["image"]]

        batch["images"] = transformed_images

        batch.pop("image")

        return batch

    dataset.set_transform(sample_transforms)

    return dataset["train"]


def ConceptualCaptionsCollateFunction(
    model_name: str = "openai/clip-vit-large-patch14",
    pre_encoded_text: bool = True,
    return_captions: bool = True,
):

    if not pre_encoded_text:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _collate_fn(batch):

        images = torch.stack([b["images"] for b in batch])

        if return_captions:
            if pre_encoded_text:
                if "encoded_text" not in batch[0].keys():
                    raise Exception(
                        "Conceptual Captions is not pre-encoded, use pre_encoded_text=False"
                    )

                seq_lens = [len(b["encoded_text"]) for b in batch]
                text_conditioning = [torch.tensor(b["encoded_text"]) for b in batch]
                text_conditioning = torch.nn.utils.rnn.pad_sequence(
                    text_conditioning, batch_first=True
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    [torch.ones(s) for s in seq_lens], batch_first=True, padding_value=0
                ).bool()

            else:
                if "caption" not in batch[0].keys():
                    raise Exception(
                        "Conceptual Captions is pre-encoded, use pre_encoded_text=True"
                    )
                output = tokenizer(
                    [b["caption"] for b in batch], return_tensors="pt", padding=True
                )
                text_conditioning = output["input_ids"]
                attention_mask = output["attention_mask"].bool()

            return {
                "images": images,
                "text_conditioning": text_conditioning,
                "text_attention_mask": attention_mask,
            }

        else:
            return {"images": images}

    return _collate_fn


def get_dataset(
    dataset: Literal["celebahq", "imagenet", "conceptual_captions"],
    path_to_data: str,
    num_channels: int = 3,
    img_size: int = 256,
    random_resize: bool = True,
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
    random_flip_p: int = 0.5,
    train: bool = True,
    return_caption: bool = True,
    return_classes: bool = True,
    text_encoder_model: str = "openai/clip-vit-large-patch14",
    pre_encoded_text: bool = True,
):

    img_transform = image_transforms(
        num_channels=num_channels,
        img_size=img_size,
        random_resize=random_resize,
        interpolation=interpolation,
        random_flip_p=random_flip_p,
        train=train,
    )

    if dataset == "celebahq":
        if return_caption:
            raise Exception("CelebAHQ Has No Captions!")
        if return_classes:
            raise Exception("celeba Has no Classes!")

        trainset = GenericImageDataset(
            path_to_data=path_to_data,
            transform=img_transform,
            nested=False,
            return_classes=False,
        )

        collate_fn = None

    elif dataset == "imagenet":
        if return_caption:
            raise Exception("Imagenet Has No Captions!")

        trainset = GenericImageDataset(
            path_to_data=path_to_data,
            transform=img_transform,
            nested=True,
            return_classes=return_classes,
        )

        collate_fn = None

    elif dataset == "conceptual_captions":
        trainset = conceptual_captions(path_to_data, img_transform)

        collate_fn = ConceptualCaptionsCollateFunction(
            model_name=text_encoder_model, pre_encoded_text=pre_encoded_text
        )
    else:
        raise ValueError(f"{dataset} is not Supported")

    return trainset, collate_fn


if __name__ == "__main__":
    path_to_conceptual = "data/GCC/hf_train_encoded"

    dataset, collate_fn = get_dataset(
        dataset="conceptual_captions",
        path_to_data=path_to_conceptual,
        pre_encoded_text=True,
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for sample in loader:
        print(sample)
        break
