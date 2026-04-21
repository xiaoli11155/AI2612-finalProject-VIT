from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_blocks import TransformerBlock


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, channels: list[int]):
        super().__init__()
        blocks = []
        prev = in_channels
        for c in channels:
            blocks.extend(
                [
                    nn.Conv2d(prev, c, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
            )
            prev = c
        self.net = nn.Sequential(*blocks)
        self.out_channels = prev
        self.downsample = 2 ** len(channels)

    def forward(self, x):
        return self.net(x)


class HybridViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        stem_channels: list[int] | None = None,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        stem_channels = stem_channels or [64, 128]
        self.stem = ConvStem(in_channels=in_channels, channels=stem_channels)
        feat_size = image_size // self.stem.downsample
        num_patches = feat_size * feat_size

        self.proj = nn.Linear(self.stem.out_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b = x.shape[0]
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])
