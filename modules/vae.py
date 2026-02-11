from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResidualBlock2D, UpSampleBlock2D, DownSampleBlock2D
from .transformer import Attention


class EncoderBlock2D(nn.Module):
    """
    The Encoder block is a stack of Residual Blocks with an optional
    downsampling layer to reduce the image size

    Args:
        - in_channels: The number of input channels to the Encoder
        - out_channels: Number of output channels of the Encoder
        - dropout_p: The dropout probability in the Residual Blocks
        - norm_eps: Groupnorm eps
        - num_residual_blocks: Number of Residual Blocks in the Encoder
        - time_embed_proj: Do you want to enable time embeddings?
        - time_embed_dim: Time embedding dimension
        - add_downsample: Do you want to downsample the image?
        - downsample_factor: By what factor do you want to downsample
        - downsample_kernel_size: Kernel size for downsampling convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        norm_eps: float = 1e-6,
        groupnorm_groups: int = 32,
        num_residual_blocks: int = 2,
        add_downsample: bool = True,
        downsample_factor: int = 2,
        downsample_kernel_size: int = 3,
    ):

        super(EncoderBlock2D, self).__init__()

        self.blocks = nn.ModuleList()

        for i in range(num_residual_blocks):
            conv_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResidualBlock2D(
                    in_channels=conv_in_channels,
                    out_channels=out_channels,
                    groupnorm_groups=groupnorm_groups,
                    dropout_p=dropout_p,
                    time_embed_proj=False,
                    class_embed_proj=False,
                    norm_eps=norm_eps,
                )
            )

        self.downsample = nn.Identity()
        if add_downsample:
            self.downsample = DownSampleBlock2D(
                in_channels=out_channels,
                downsample_factor=downsample_factor,
                kernel_size=downsample_kernel_size,
            )

    def forward(self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None):

        for block in self.blocks:
            x = block(x, time_embed)

        x = self.downsample(x)

        return x


class DecoderBlock2D(nn.Module):
    """
    The Decoder block is a stack of Residual Blocks with an optional
    upsampling layer to reduce the image size

    Args:
        - in_channels: The number of input channels to the Encoder
        - out_channels: Number of output channels of the Encoder
        - dropout_p: The dropout probability in the Residual Blocks
        - norm_eps: Groupnorm eps
        - num_residual_blocks: Number of Residual Blocks in the Encoder
        - time_embed_proj: Do you want to enable time embeddings?
        - time_embed_dim: Time embedding dimension
        - add_upsample: Do you want to upsample the image?
        - upsample_factor: By what factor do you want to upsample
        - upsample_kernel_size: Kernel size for upsampling convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels,
        dropout_p: float = 0.0,
        norm_eps: float = 1e-6,
        groupnorm_groups: int = 32,
        num_residual_blocks: int = 2,
        add_upsample: bool = True,
        upsample_factor: int = 2,
        upsample_kernel_size: int = 3,
    ):

        super(DecoderBlock2D, self).__init__()

        self.blocks = nn.ModuleList()

        for i in range(num_residual_blocks):
            conv_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                ResidualBlock2D(
                    in_channels=conv_in_channels,
                    out_channels=out_channels,
                    groupnorm_groups=groupnorm_groups,
                    dropout_p=dropout_p,
                    time_embed_proj=False,
                    class_embed_proj=False,
                    norm_eps=norm_eps,
                )
            )

        self.upsample = nn.Identity()
        if add_upsample:
            self.upsample = UpSampleBlock2D(
                in_channels=out_channels,
                upsample_factor=upsample_factor,
                kernel_size=upsample_kernel_size,
            )

    def forward(self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None):

        for block in self.blocks:
            x = block(x, time_embed)

        x = self.upsample(x)

        return x


class VAEAttentionResidualBlock(nn.Module):
    """
    In the implementation of autoencoder_kl (https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/autoencoders/autoencoder_kl.py)

    After all the ResidualBlocks of the Encoder, and at the start of the Decoder there is a ResidualBlock+Self-Attention that they call the UNetMidBlock2D.
    This class is exactly that, where each Block starts with 1 Residual Block, and then we toggle between Attention and Residual Blocks.

    Args:
        - in_channels: Number of input channels to our Block
        - dropout_p: What dropout probability do you want to use?
        - num_layers: How many iterations of Attention/ResidualBlocks do you want?
        - groupnorm_groups: How many groups in the GroupNormalization
        - norm_eps: eppassead_dim: Embed Dim for each head of attention
        - attention_residual_connections: Do you want a residual connection in Attention?

    """

    def __init__(
        self,
        in_channels: int,
        dropout_p: float = 0.0,
        num_layers: int = 1,
        groupnorm_groups: int = 32,
        norm_eps: float = 1e-6,
        attention_head_dim: int = 1,
        attention_residual_connection: bool = True,
    ):

        super(VAEAttentionResidualBlock, self).__init__()

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        ### There is Always One Residual Block ###
        self.resnets.append(
            ResidualBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout_p=dropout_p,
                groupnorm_groups=groupnorm_groups,
                norm_eps=norm_eps,
            )
        )

        ### For Every Layer, Create an Attention + Residual Block Stack ###
        for _ in range(num_layers):
            self.attentions.append(
                Attention(
                    embedding_dimension=in_channels,
                    head_dim=attention_head_dim,
                    attn_dropout=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    attention_residual_connection=attention_residual_connection,
                    return_shape="2D",
                )
            )

            self.resnets.append(
                ResidualBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_p=dropout_p,
                    groupnorm_groups=groupnorm_groups,
                    norm_eps=norm_eps,
                )
            )

    def forward(self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None):

        x = self.resnets[0](x, time_embed=time_embed)

        for attn, res in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = res(x, time_embed)

        return x
