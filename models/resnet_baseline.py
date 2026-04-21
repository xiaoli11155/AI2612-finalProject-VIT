from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34


class ResNetBaseline(nn.Module):
    def __init__(self, variant: str = "resnet18", num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        variant = variant.lower()
        if variant == "resnet18":
            self.backbone = resnet18(weights=None)
        elif variant == "resnet34":
            self.backbone = resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
