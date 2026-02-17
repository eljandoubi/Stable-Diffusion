from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


def img2seq(x: torch.Tensor):
    """
    (B x C x H x W) -> (B x H*W x C)
    """
    batch, channels, height, width = x.shape

    seq_len = height * width

    x = x.reshape(batch, channels, seq_len).transpose(-1, -2)

    return x, seq_len


def seq2img(x: torch.Tensor, img_dim: Optional[tuple[int, int]] = None):
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

    def __init__(
        self,
        embedding_dimension: int = 768,
        cross_attn_dim: Optional[int] = None,
        head_dim: int = 1,
        attn_dropout: float = 0.0,
        groupnorm_groups: Optional[int] = None,
        attention_residual_connection: bool = True,
        bias: bool = True,
        return_shape: Literal["1D", "2D"] = "1D",
    ):

        super(Attention, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.cross_attn_dim = cross_attn_dim
        self.attn_dropout = attn_dropout
        self.attn_residual = attention_residual_connection
        self.groupnorm_groups = groupnorm_groups

        if return_shape not in ["1D", "2D"]:
            raise Exception("Attention can output '1D' or '2D'")
        self.return_shape = return_shape

        ### Attention Head Dim ###
        self.head_dim = head_dim
        assert embedding_dimension % head_dim == 0
        self.num_heads = embedding_dimension // head_dim

        ### GroupNorm ###
        if self.groupnorm_groups is not None:
            self.groupnorm = nn.GroupNorm(
                num_channels=embedding_dimension, num_groups=groupnorm_groups, eps=1e-6
            )

        ### Attention Projections ###
        kv_input_dim = embedding_dimension if cross_attn_dim is None else cross_attn_dim
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.k_proj = nn.Linear(kv_input_dim, embedding_dimension, bias=bias)
        self.v_proj = nn.Linear(kv_input_dim, embedding_dimension, bias=bias)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)

    @staticmethod
    def _check_for_reshape(images: torch.Tensor):

        ### Reshape from Img Dim to Seq Dim if in shape (B,C,H,W) ###
        if len(images.shape) == 4:
            images, num_patches = img2seq(images)
        elif len(images.shape) == 3:
            num_patches = images.shape[1]
        else:
            raise ValueError("The images must be 3D or 4D tensors!")

        return images, num_patches

    def forward_self_attn(self, images: torch.Tensor) -> torch.Tensor:

        batch_size = images.shape[0]

        images, num_patches = self._check_for_reshape(images)

        ### QKV Projection ###
        q = (
            self.q_proj(images)
            .reshape(batch_size, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_proj(images)
            .reshape(batch_size, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(images)
            .reshape(batch_size, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        attention_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )

        ### Reshape back to (B, num_patches, head_dim) and Project with Linear ###
        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

    def forward_cross_attn(
        self,
        images: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        images, num_patches = self._check_for_reshape(images)

        ### Query Projection on Images ###
        q = (
            self.q_proj(images)
            .reshape(-1, num_patches, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        ### Key/Value Projections on Text ###
        batch, context_len, _ = context.shape
        k = (
            self.k_proj(context)
            .reshape(batch, context_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(context)
            .reshape(batch, context_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        ### This is our text attention mask ###
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = (
                attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, num_patches, 1)
            )

        attention_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        ### Reshape back to (B, num_patches, head_dim) and Project with Linear ###
        attention_out = attention_out.transpose(1, 2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        ### x can be 1D or 2D ###
        residual = x

        if self.groupnorm_groups is not None:
            x = self.groupnorm(x)

        if context is None:
            attention_out = self.forward_self_attn(x)
        else:
            attention_out = self.forward_cross_attn(x, context, attention_mask)

        ### attention_out is always 1d ###
        if self.return_shape == "2D":
            attention_out = seq2img(attention_out)

        if self.attn_residual:
            ### If residual shape doesnt match attention_out
            ### then the residuals must be (B,C,H,W), so we can
            ### reshape based on the output shape we want
            if len(attention_out.shape) != len(residual.shape):
                ### If we want a 1D output, flatten the residual before adding
                if self.return_shape == "1D":
                    residual, _ = img2seq(residual)
                else:
                    residual = seq2img(residual, attention_out.shape[-2:])

            attention_out = attention_out + residual

        return attention_out


class GEGLU(nn.Module):
    """
    GEGLU Activation as outlined here:
    https://paperswithcode.com/method/geglu
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool):

        super(GEGLU, self).__init__()

        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor):

        hidden_states: torch.Tensor = self.proj(x)

        hidden_states, gate = hidden_states.chunk(2, dim=-1)

        return hidden_states * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Standard MLP Module found in Transformers w/ GeGLU Activation
    """

    def __init__(
        self, dim: int, dim_mult: int, mlp_dropout: float = 0.0, bias: bool = True
    ):

        super(FeedForward, self).__init__()

        hidden_dim = dim * dim_mult
        self.fc1_geglu = GEGLU(dim, hidden_dim, bias=bias)
        self.drop = nn.Dropout(mlp_dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor):

        x = self.fc1_geglu(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class BasicTransformerBlock1D(nn.Module):
    """
    Standard Transformer Block as Described in Attention is All You Need

    This applies to 1D signals in the shape (B,L,E)

    Args:
        - embed_dim: Embedding dimension of images (number of channels)
        - cross_attn_dim: Embedding dimension in for Text (for cross attention)
        - dim_mult: FeedForward hidden layer projection
        - attention_head_dim: Embedding dimension for each head of attention
        - norm_eps: Layer Norm eps
        - dropout_p: Dropout probability on Attention and FeedForward
        - attn_bias: Do you want to use bias in QKV Projections
    """

    def __init__(
        self,
        embed_dim: int,
        cross_attn_dim: Optional[int] = None,
        dim_mult: int = 4,
        attention_head_dim: int = 1,
        norm_eps: float = 1e-6,
        dropout_p: float = 0.0,
        attn_bias: bool = True,
    ):

        super(BasicTransformerBlock1D, self).__init__()

        self.cross_attn_dim = cross_attn_dim

        ### Self-Attention ###
        self.norm1 = nn.LayerNorm(embed_dim, norm_eps)
        self.attn1 = Attention(
            embedding_dimension=embed_dim,
            cross_attn_dim=None,
            head_dim=attention_head_dim,
            attn_dropout=dropout_p,
            groupnorm_groups=None,
            attention_residual_connection=False,
            bias=attn_bias,
        )

        ### Cross Attention ###
        if cross_attn_dim is not None:
            self.norm2 = nn.LayerNorm(embed_dim, norm_eps)
            self.attn2 = Attention(
                embedding_dimension=embed_dim,
                cross_attn_dim=cross_attn_dim,
                head_dim=attention_head_dim,
                attn_dropout=dropout_p,
                groupnorm_groups=None,
                attention_residual_connection=False,
                bias=attn_bias,
            )

        ### FeedForward ###
        self.norm3 = nn.LayerNorm(embed_dim, norm_eps)
        self.feedforward = FeedForward(
            embed_dim, dim_mult=dim_mult, mlp_dropout=dropout_p
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        ### Normalize, Self-Attention, Residual Connection ###
        norm_hidden_states = self.norm1(x)
        attention_output = self.attn1(norm_hidden_states)
        x = x + attention_output

        ### Normalize, Cross-Attention, Residual Connection ###
        if self.cross_attn_dim is not None:
            if context is None:
                raise Exception("Not Passing in Context to a Block W Cross Attention")

            norm_hidden_states = self.norm2(x)
            attention_output = self.attn2(norm_hidden_states, context, attention_mask)
            x = x + attention_output

        ### Normalize and FeedForward ###
        norm_hidden_states = self.norm3(x)
        mlp_out = self.feedforward(norm_hidden_states)
        x = x + mlp_out

        return x


if __name__ == "__main__":
    module = Attention(5, return_shape="2D")
    x = torch.randn(7, 16, 5)
    res = module(x)
    print(res.shape)
