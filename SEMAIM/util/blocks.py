import math

import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

# causal mask

class Attention_SelfMask(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # if scale is not provided use the default scale of 1/ sqrt(head_dim)
        self.scale = qk_scale or head_dim ** -0.5

        # compute qkv in a single linear layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # dropouts for attention probabilities and for final projection output
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, return_attention=False):
        B, N, C = x.shape

        # Compute Q, K, V: shape after qkv is (B, N, 3 * C).
        # Reshape and permute to get (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Unpack Q, K, V (each has shape B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # Compute attention scores size (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale #shouldnt this be a division?

        # Apply the mask if it is provided for causal masking
        if mask is not None:
            attn += mask
        
        #softmax over last aka key dimension
        attn = attn.softmax(dim=-1)
        if return_attention:
            return attn # B H N N

        # apply attention dropout to attention probabilities
        attn = self.attn_drop(attn)

        # compute the weighted average of the values (v)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # project the output back to the original dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_SelfMask(nn.Module):
    """
        A transformer block with a self-attention mechanism with masking.
        Mask can implement causal attention.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # pre-attention normalization
        self.norm1 = norm_layer(dim)

        # self attention mechanism
        self.attn = Attention_SelfMask(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here done for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # pre_MLP normalization
        self.norm2 = norm_layer(dim)
        # feed forward network MLP with configurable ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attention=False):
        # if we only want attention weights compute and return them
        if return_attention:
            return self.attn(self.norm1(x), mask, return_attention)
        # apply self attention and residual connection
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        # apply MLP and residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention_SelfCrossMask(nn.Module):
    """
        This attention layer can function as either self-attention or cross-attention:
        - Self-attention: q, k, v are the same tensor.
        - Cross-attention: q is from a "query" sequence, while k and v might be from another "key/value" sequence.
    """	
        
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # linear layers are separate, allowing q to come from a different source than k, v
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None, return_attention=False):
        B, N, C = q.shape
        # B, N_k, C = k.shape
        # B, N_v, C = v.shape
        # project input sequences to q, k, v
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(k).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v(v).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        # compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # add the mask if it is provided
        if mask is not None:
            attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if return_attention:
            return attn

        # apply the weighted average to the values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # final projections
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_SelfCrossMask(nn.Module):
    """
        The universal attention block can be used as both self-attention and cross-attention.
        q,k,v can define separately.
        If we only assign a value to q, it's a self-attention block;
        if we assign values for q and k, it's a cross-attention block.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_SelfCrossMask(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, k=None, v=None, mask=None, return_attention=False):
        # if k or v are not specified, assume they are the same as q for self-attention
        if k is None:
            k = q
        if v is None:
            v = k
        #if requested return attention weights only
        if return_attention:
            return self.attn(self.norm1(q), self.norm1(k), self.norm1(v), mask, return_attention)
        
        # perform attention, residual and drop path
        x = q + self.drop_path(self.attn(self.norm1(q), self.norm1(k), self.norm1(v), mask))
        # perform mlp, residual and drop path
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class GaussianConv2d(nn.Module):
    """
        A 2D Gaussian convolution layer. This layer creates a fixed Gaussian kernel and 
        applies it as a convolution to the input. It can be used to blur or smooth images.

        Steps:
        - Construct a 2D Gaussian kernel with a given kernel size and standard deviation (sigma).
        - Normalize the kernel so that it sums to 1.
        - Apply the kernel as a convolution with the given number of channels (groups).

        This layer does not learn parameters; it uses a fixed Gaussian kernel.
    """
    def __init__(self, channels=3, kernel_size=9, sigma=1):
        super().__init__()
        # create a 2D grid of positions
        position = torch.stack(torch.meshgrid([torch.arange(kernel_size), torch.arange(kernel_size)]), dim=-1)
        # mean and standard deviation
        mean = torch.tensor([(kernel_size - 1) // 2, (kernel_size - 1) // 2])
        std = torch.tensor([sigma, sigma])
        # compute the Gaussian kernel using the formula  G(x,y) = 1/(2πσ²) * exp(-((x-μ_x)²+(y-μ_y)²)/(2σ²))
        kernel = 1 / (2 * math.pi * torch.prod(std, dim=-1)) * math.e ** (-((position - mean) ** 2 / std ** 2).sum(-1)/2)
        kernel = kernel / kernel.sum() # normalize the kernel so that it sums to 1

        # reshape the kernel to the right dimensions
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        # register the kernel as a buffer (not a learnable parameter)
        self.register_buffer('weight', kernel)

        # convolution groups equals number of channels to apply the same kernel to all channels
        self.groups = channels
        # padding to maintain the input size
        self.padding = kernel_size // 2

    def forward(self, input):
        # apply gaussian filter as 2D convolution
        return F.conv2d(input, weight=self.weight, groups=self.groups, padding=self.padding)