from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    warmup_epochs = max(1, warmup_epochs)
    total_epochs = max(warmup_epochs + 1, total_epochs)

    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
