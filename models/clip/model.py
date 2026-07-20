from collections import OrderedDict
from typing import Optional, Tuple, Union
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import loratorch
import math


DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 8


def set_lora_config(rank: Optional[int] = None, alpha: Optional[int] = None):
    global DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA
    if rank is not None:
        DEFAULT_LORA_RANK = max(0, rank)
    if alpha is not None:
        DEFAULT_LORA_ALPHA = max(1, alpha)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor):
        return x * self.gamma


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(1024, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 1024)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        N, B, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[1:, :].reshape(B, 16, 16, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 16 * 16, self.dim)

        x_cls = x_down[:1, :].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        x_up = x_up.permute(1, 0, 2)

        return x_up

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        if DEFAULT_LORA_RANK > 0:
            self.attn = loratorch.MultiheadAttention(
                d_model,
                n_head,
                r=DEFAULT_LORA_RANK,
                lora_alpha=DEFAULT_LORA_ALPHA,
                enable_lora=['q', 'k', 'v'],  # 仅对 qkv 加 LoRA，关闭 out_proj LoRA
            )
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        hidden_dim = int(mlp_hidden_dim) if mlp_hidden_dim is not None else d_model * 4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, hidden_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_dim, d_model))
        ]))
        self.ls_1 = LayerScale(d_model, init_value=1.0)
        self.ls_2 = LayerScale(d_model, init_value=1.0)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # self.adapter_mlp = Convpass()
        self.s = 1.
        self.store_attn = False
        self.track_attn_grad = False
        self.attn_storage = []

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.store_attn:
            out, attn_weights = self.attn(
                x, x, x, need_weights=True, attn_mask=self.attn_mask, average_attn_weights=False
            )
            if self.track_attn_grad:
                attn_weights.retain_grad()
            self.attn_storage.append(attn_weights)
            return out
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.ls_1(self.attention(self.ln_1(x)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        # x = x + self.mlp(self.ln_2(x)) + (self.adapter_mlp(self.ln_2(x))) * self.s
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(layers)
            ]
        )
        # Load pre-saved initialization from NPR if available.
        # NOTE: removed custom resblocks init load to avoid shape mismatch; base CLIP weights are used as-is.

    def enable_attention_tracking(self, track_grad: bool = False):
        for block in self.resblocks:
            block.store_attn = True
            block.track_attn_grad = bool(track_grad)
            block.attn_storage = []

    def clear_attention_storage(self):
        for block in self.resblocks:
            block.attn_storage = []

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature
        return out, x

        # return self.resblocks(x)  # This is the original code


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.grid_size = input_resolution // patch_size

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))



    def _encode_tokens(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid_h, grid_w]
        grid_h, grid_w = x.shape[-2], x.shape[-1]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        class_embed = self.class_embedding.to(x.dtype)
        class_embed = class_embed + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_embed, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if grid_h != self.grid_size or grid_w != self.grid_size:
            patch_pos = self.positional_embedding[1:]
            patch_pos = patch_pos.reshape(1, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
            pos_embed = torch.cat([self.positional_embedding[:1], patch_pos], dim=0)
        else:
            pos_embed = self.positional_embedding

        x = x + pos_embed.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        out, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        out['before_projection'] = x
        if self.proj is not None:
            x = x @ self.proj
        out['after_projection'] = x

        cls = x[:, 0, :]
        return cls, x

    def forward(self, x: torch.Tensor):
        cls, _ = self._encode_tokens(x)
        return cls

    def forward_with_tokens(self, x: torch.Tensor):
        return self._encode_tokens(x)

        # Return both intermediate features and final clip feature
        # return out

        # This only returns CLIP features
        return x

    def enable_attention_tracking(self, track_grad: bool = False):
        self.transformer.enable_attention_tracking(track_grad=track_grad)

    def clear_attention_storage(self):
        self.transformer.clear_attention_storage()


class PerceptionVisionTransformer(nn.Module):
    """
    Vision-only transformer used by Perception Encoder checkpoints.
    Unlike OpenAI CLIP ViT, this backbone has no class token/projection heads.
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        num_pos_tokens: int,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_resolution = int(input_resolution)
        self.patch_size = int(patch_size)
        self.output_dim = int(width)
        self.grid_size = self.input_resolution // self.patch_size
        self.num_pos_tokens = int(num_pos_tokens)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_pos_tokens, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_hidden_dim=mlp_hidden_dim)

    def _resize_positional_embedding(self, grid_h: int, grid_w: int, dtype: torch.dtype) -> torch.Tensor:
        if self.positional_embedding.shape[0] == grid_h * grid_w:
            return self.positional_embedding.to(dtype)

        base_grid = round(self.positional_embedding.shape[0] ** 0.5)
        if base_grid * base_grid != self.positional_embedding.shape[0]:
            raise ValueError(
                f"Invalid positional embedding length {self.positional_embedding.shape[0]} "
                f"for PE backbone (expected perfect square)."
            )

        patch_pos = self.positional_embedding.reshape(base_grid, base_grid, -1).permute(2, 0, 1).unsqueeze(0)
        patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, -1)
        return patch_pos.to(dtype)

    def _encode_tokens(self, x: torch.Tensor):
        x = self.conv1(x)  # [B,C,H',W']
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B,N,C]

        pos_embed = self._resize_positional_embedding(grid_h, grid_w, x.dtype)
        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        _, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled = x.mean(dim=1)
        # Keep CLIP-compatible output shape expected by downstream wrapper:
        # prepend pooled token so callers can strip [:, 1:, :] as patch tokens.
        tokens_with_pooled = torch.cat([pooled.unsqueeze(1), x], dim=1)
        return pooled, tokens_with_pooled

    def forward(self, x: torch.Tensor):
        pooled, _ = self._encode_tokens(x)
        return pooled

    def forward_with_tokens(self, x: torch.Tensor):
        return self._encode_tokens(x)

    def enable_attention_tracking(self, track_grad: bool = False):
        self.transformer.enable_attention_tracking(track_grad=track_grad)

    def clear_attention_storage(self):
        self.transformer.clear_attention_storage()


class PerceptionCLIPWrapper(nn.Module):
    """
    Minimal CLIP-like wrapper around PE visual backbone.
    Exposes encode_image / encode_image_with_tokens used by CLIPModel.
    """

    def __init__(self, visual: PerceptionVisionTransformer):
        super().__init__()
        self.visual = visual

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor):
        return self.visual(image.type(self.dtype))

    def encode_image_with_tokens(self, image: torch.Tensor):
        return self.visual.forward_with_tokens(image.type(self.dtype))


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 文本侧不用 LoRA，临时将 LoRA 配置置零后再恢复，避免修改接口或影响视觉侧
        _old_rank, _old_alpha = DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA
        try:
            set_lora_config(rank=0, alpha=_old_alpha)
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )
        finally:
            set_lora_config(rank=_old_rank, alpha=_old_alpha)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_image_with_tokens(self, image):
        if not hasattr(self.visual, "forward_with_tokens"):
            raise RuntimeError("Visual backbone does not support token outputs.")
        return self.visual.forward_with_tokens(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_pe_model(state_dict: dict):
    required_keys = ["conv1.weight", "ln_pre.weight", "ln_pre.bias", "positional_embedding"]
    for key in required_keys:
        if key not in state_dict:
            raise KeyError(f"Missing key `{key}` in PE checkpoint")

    vision_width = state_dict["conv1.weight"].shape[0]
    vision_patch_size = state_dict["conv1.weight"].shape[-1]
    num_pos_tokens = state_dict["positional_embedding"].shape[0]
    grid_size = round(num_pos_tokens ** 0.5)
    if grid_size * grid_size != num_pos_tokens:
        raise ValueError(
            f"PE positional_embedding length {num_pos_tokens} is not a perfect square."
        )
    image_resolution = vision_patch_size * grid_size
    vision_layers = len(
        [
            k
            for k in state_dict.keys()
            if k.startswith("transformer.resblocks.") and k.endswith(".attn.in_proj_weight")
        ]
    )
    mlp_hidden_dim = int(state_dict["transformer.resblocks.0.mlp.c_fc.weight"].shape[0])
    vision_heads = max(1, vision_width // 64)

    visual = PerceptionVisionTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        num_pos_tokens=num_pos_tokens,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    missing, unexpected = visual.load_state_dict(state_dict, strict=False)
    missing_non_lora = [k for k in missing if "lora_" not in k]
    if unexpected:
        print(f"[PE] Unexpected checkpoint keys (showing up to 8): {unexpected[:8]}")
    if missing_non_lora:
        print(f"[PE] Missing checkpoint keys (showing up to 8): {missing_non_lora[:8]}")

    return PerceptionCLIPWrapper(visual).eval()


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)

    return model.eval()
