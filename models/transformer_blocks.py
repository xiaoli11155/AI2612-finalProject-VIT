from __future__ import annotations

import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Thin wrapper around PyTorch's official TransformerEncoderLayer.
    This keeps the ViT block interface stable while avoiding manual attention implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        effective_dropout = max(dropout, drop_path)
        self.block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=effective_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # PyTorch does not expose separate attn dropout in this module.
        # Keep the argument for compatibility with existing configs.
        self.attn_dropout = attn_dropout

    def forward(self, x):
        return self.block(x)
