from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import warnings

warnings.filterwarnings("ignore")


class LPIPS(nn.Module):
    """
    VGG16 LPIPS Perceptual Loss as proposed in https://github.com/richzhang/PerceptualSimilarity

    Normally we do an L2 Reconstruction loss between our generated images and the real ground truth
    but the problem can be it leads to blurry pictures. This is why, instead of just minimizing
    the reconstruction loss, we also want to minimize between VGG features extracted from our
    real and fake images
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        train_backbone: bool = False,
        use_dropout: bool = True,
        img_range: Literal["zero_to_one", "minus_one_to_one"] = "minus_one_to_one",
        pretrained_weights: Optional[str] = None,
    ):

        super(LPIPS, self).__init__()

        self.pretrained_backbone = pretrained_backbone
        self.train_backbone = train_backbone
        self.use_dropout = use_dropout
        self.img_range = img_range

        ### Load a Pretrained VGG BackBone and its Channel Pattern###
        vgg_model = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        ).features
        self.channels = [64, 128, 256, 512, 512]
        self.layer_groups = [(0, 3), (4, 8), (9, 15), (16, 22), (23, 29)]

        ### Turn of Gradients on Backbone ###
        if not train_backbone:
            for param in vgg_model.parameters():
                param.requires_grad_(False)

        ### Compute the Norm Constants ###
        self.scale_constants(img_range)

        ### Slices of the Model ###
        slices = {}
        for i, (start, end) in enumerate(self.layer_groups):
            layers = []
            for j in range(start, end + 1):
                layers.append(vgg_model[j])
            slices[f"slice{i + 1}_layers"] = nn.Sequential(*layers)

        self.slices = nn.ModuleDict(slices)

        ### Now that VGG16 is Sliced and Stored, Delete Original Model ###
        del vgg_model

        ### Projections of Patches (B x C x H x W) -> (B x 1 x H x W) ###
        proj = {}
        for i, in_channels in enumerate(self.channels):
            layers = (
                [
                    nn.Dropout(),
                ]
                if (use_dropout)
                else []
            )
            layers += [
                nn.Conv2d(
                    in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False
                )
            ]
            proj[f"slice{i + 1}_conv_proj"] = nn.Sequential(*layers)

        self.proj = nn.ModuleDict(proj)

        ### Spatial Pooling ###
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        ### Load Checkpoint ###
        if pretrained_weights is not None:
            print("Loading LPIPS Checkpoint:", pretrained_weights)
            self.load_checkpoint(pretrained_weights)

    def scale_constants(
        self, range: Literal["zero_to_one", "minus_one_to_one"] = "minus_one_to_one"
    ) -> None:

        if range not in ["zero_to_one", "minus_one_to_one"]:
            raise ValueError(
                "Indicate if images are zero_to_one [0,1] or minus_one_to_one [-1,1]"
            )

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])

        ### Imagenet Mean assumed [0,1] images, if [-1,1] we have to rescale ###
        if range == "minus_one_to_one":
            ### If we double the range from [0,1] to [-1,1] we double the std ###
            imagenet_std = imagenet_std * 2

            ### If we double the range from [0,1] to [-1,1] we shift the mean ###
            imagenet_mean = (imagenet_mean * 2) - 1

        ### Add extra dimensions to broadcast over (B,3,H,W)
        imagenet_mean = imagenet_mean.reshape(1, 3, 1, 1)
        imagenet_std = imagenet_std.reshape(1, 3, 1, 1)

        self.register_buffer("mean", imagenet_mean)
        self.register_buffer("std", imagenet_std)

    def load_checkpoint(self, path_to_checkpoint: str):
        self.load_state_dict(torch.load(path_to_checkpoint, weights_only=True))

    def forward_vgg(self, x: torch.Tensor):
        return_outputs: dict[str, torch.Tensor] = {}
        slice_out = x
        for i in range(len(self.layer_groups)):
            slice_out = self.slices[f"slice{i + 1}_layers"](slice_out)
            return_outputs[f"slice{i + 1}"] = slice_out
        return return_outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md = LPIPS().to(device)
    tr = torch.randn(3, 256, 256, device=device)
    out = md.forward_vgg(tr)
    for key, v in out.items():
        print(key, v.shape)
