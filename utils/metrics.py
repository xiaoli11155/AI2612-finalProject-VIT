from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Iterable[int]) -> Dict[int, float]:
    # CPU topk does not support float16, so ensure a safe dtype.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    batch_size = targets.size(0)

    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[k] = (correct_k * (100.0 / batch_size)).item()
    return results


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = targets * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def precision_recall_f1_from_cm(cm: torch.Tensor):
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return precision.mean().item(), recall.mean().item(), f1.mean().item()


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    topk: List[int] | None = None,
):
    topk = topk or [1]
    metrics = {}
    metrics.update({f"top{k}": v for k, v in topk_accuracy(logits, targets, topk).items()})
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, targets, num_classes)
    p, r, f1 = precision_recall_f1_from_cm(cm)
    metrics["precision_macro"] = p * 100.0
    metrics["recall_macro"] = r * 100.0
    metrics["f1_macro"] = f1 * 100.0
    metrics["confusion_matrix"] = cm.cpu().tolist()
    return metrics
