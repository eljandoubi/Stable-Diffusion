from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def img2seq(x: torch.Tensor):
    """
    (B x C x H x W) -> (B x H*W x C)
    """
    batch, channels, height, width = x.shape

    x = x.reshape(batch, channels, height * width).transpose(-1, -2)

    seq_len = height * width

    return x, seq_len


def seq2img(x: torch.Tensor, img_dim: Optional[int] = None):
    """
    (B x H*W x C) -> (B x C x H x W)
    """
    batch, seq_len, channels = x.shape

    ### Assume Square Image if no img_dim is provided ###
    if img_dim is None:
        h = w = int(seq_len**0.5)

    else:
        h, w = img_dim

    x = x.transpose(-1, -2).reshape(batch, channels, h, w)

    return x


class Attention(nn.Module):
    """

    Implementation of Self and Cross Attention in One Module

    By default, self-attention will be computed on src (our images). If tgt is provided, then we are doing cross
    attention. In cross attention, an attention_mask can be used (padding mask for our embedded text), and
    src is our text and tgt is the images.

    Self-Attention:
        - Compute Self Attention on the src Tensor
            - One new step to include though is reshaping our src (image)
                from (B x C x H x W) -> (B x H*W x C) before doing attention

    Cross Attention
        - src: Our text Context (B x L x E)
        - tgt: What we want to weight against our src and output
            - One new step to include though is reshaping our tgt (image)
                from (B x C x H x W) -> (B x H*W x C) before doing attention
        - attention_mask: Padding mask for the text embeddings

    Args:
        - embedding_dimension: Number of channels in Image (the channels in BCHW act as our embedding)
        - cross_attn_dim: The embedding dimension of the text context (None for self-attention)
        - head_dim: What embedding dimension do you want to use per head?
        - attn_dropout: What dropout probability do you want to use in attention
        - groupnorm_groups: Number of groups in GroupNormalization (None if we dont need it)
        - attention_residual_connections: Do you want to add the input of attention to the output?

    """
