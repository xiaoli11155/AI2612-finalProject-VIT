from __future__ import annotations

import argparse

import torch
from torch import nn

from datasets import build_dataloaders
from models import build_model
from utils import compute_classification_metrics, load_config, save_json, set_seed


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, topk):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_classification_metrics(logits, targets, num_classes=num_classes, topk=topk)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    loaders, num_classes = build_dataloaders(cfg["dataset"], seed=int(cfg.get("seed", 42)))

    model = build_model(cfg["model"], num_classes=num_classes).to(args.device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])

    criterion = nn.CrossEntropyLoss()
    topk = cfg.get("eval", {}).get("topk", [1])
    metrics = evaluate(
        model=model,
        loader=loaders[args.split],
        criterion=criterion,
        device=args.device,
        num_classes=num_classes,
        topk=topk,
    )
    print(f"Split: {args.split}")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    save_json(metrics, f"{cfg.get('output_dir', 'outputs')}/logs/{cfg.get('experiment_name', 'exp')}_{args.split}_metrics.json")


if __name__ == "__main__":
    main()
