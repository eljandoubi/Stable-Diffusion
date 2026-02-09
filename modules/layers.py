import torch
import torch.nn as nn


class UpSampleBlock2D(nn.Module):
    """
    Upsampling Block that takes

    (B x C x H x W) -> (B x C x H*2 x W*2)

    Args:
        - in_channels: Input channels of images (no change in channels)
        - kernel_size: Kernel size in learnable convolution
        - upsample_factor: By what factor do you want to upsample image by?
    """

    def __init__(
        self, in_channels: int, kernel_size: int = 3, upsample_factor: int = 2
    ):

        super(UpSampleBlock2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.factor = upsample_factor

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsample_factor, mode="nearest"),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x: torch.Tensor):

        *_, height, width = x.shape

        upsampled = self.upsample(x)

        assert upsampled.shape[2:] == (height * self.factor, width * self.factor)

        return upsampled


class DownSampleBlock2D(nn.Module):
    """
    Downsampling Block that takes

    (B x C x H x W) -> (B x C x H/2 x W/2)

    Args:
        - in_channels: Input channels of images (no change in channels)
        - kernel_size: Kernel size in learnable convolution
        - kernel_size: What stride do you want to use to downsample?

    """

    def __init__(
        self, in_channels: int, kernel_size: int = 3, downsample_factor: int = 2
    ):

        super(DownSampleBlock2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.factor = downsample_factor

        self.downsample_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=downsample_factor,
            padding=1,
        )

    def forward(self, x: torch.Tensor):

        *_, height, width = x.shape

        downsampled = self.downsample_conv(x)

        assert downsampled.shape[2:] == (height // self.factor, width // self.factor)

        return downsampled
