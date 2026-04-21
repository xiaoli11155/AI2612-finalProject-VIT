from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import build_dataloaders
from models import build_model
from utils import (
    CSVLogger,
    build_warmup_cosine_scheduler,
    compute_classification_metrics,
    load_config,
    save_json,
    set_seed,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_mixup(x, y, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, scaler, device, num_classes, topk, use_amp, mixup_alpha, grad_clip
):
    model.train()
    total_loss = 0.0
    topk = sorted(topk)
    topk_correct = {k: 0.0 for k in topk}
    total_seen = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, y_a, y_b, lam = apply_mixup(images, targets, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            if mixup_alpha > 0:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)

        with torch.no_grad():
            maxk = max(topk)
            preds = logits.topk(maxk, dim=1, largest=True, sorted=True).indices
            total_seen += targets.size(0)
            if mixup_alpha > 0:
                # For mixup training, count weighted correctness for both source labels.
                for k in topk:
                    correct_a = preds[:, :k].eq(y_a.unsqueeze(1)).any(dim=1).float().sum().item()
                    correct_b = preds[:, :k].eq(y_b.unsqueeze(1)).any(dim=1).float().sum().item()
                    topk_correct[k] += lam * correct_a + (1 - lam) * correct_b
            else:
                for k in topk:
                    correct = preds[:, :k].eq(targets.unsqueeze(1)).any(dim=1).float().sum().item()
                    topk_correct[k] += correct

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": optimizer.param_groups[0]["lr"]})

    scheduler.step()

    metrics = {f"top{k}": (topk_correct[k] * 100.0 / max(1, total_seen)) for k in topk}
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, topk):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_classification_metrics(logits_cat, targets_cat, num_classes=num_classes, topk=topk)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT/CNN/Hybrid classifiers")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    tb_dir = output_dir / "tensorboard"
    fig_dir = output_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    exp_name = cfg.get("experiment_name", Path(args.config).stem)
    csv_logger = CSVLogger(str(log_dir / f"{exp_name}.csv"))
    tb_writer = SummaryWriter(log_dir=str(tb_dir / exp_name))
    save_json(cfg, str(log_dir / f"{exp_name}_config.json"))

    loaders, num_classes = build_dataloaders(cfg["dataset"], seed=seed)
    model = build_model(cfg["model"], num_classes=num_classes).to(args.device)

    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    topk = eval_cfg.get("topk", [1])
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
        total_epochs=int(train_cfg["epochs"]),
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.01)),
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and args.device.startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val_top1 = -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_val_top1 = state.get("best_val_top1", -1.0)

    print(f"Experiment: {exp_name}")
    print(f"Model: {cfg['model']['type']} | Params: {count_parameters(model):,}")
    print(f"Device: {args.device}")

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=args.device,
            num_classes=num_classes,
            topk=topk,
            use_amp=use_amp,
            mixup_alpha=float(train_cfg.get("mixup_alpha", 0.0)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=args.device,
            num_classes=num_classes,
            topk=topk,
        )
        test_metrics = evaluate(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            device=args.device,
            num_classes=num_classes,
            topk=topk,
        )
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_top1": train_metrics.get("top1", 0.0),
            "val_top1": val_metrics.get("top1", 0.0),
            "test_top1": test_metrics.get("top1", 0.0),
            "train_time_sec": epoch_time,
        }
        csv_logger.log(row)
        tb_writer.add_scalar("train/loss", train_metrics["loss"], epoch + 1)
        tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch + 1)
        tb_writer.add_scalar("test/loss", test_metrics["loss"], epoch + 1)
        tb_writer.add_scalar("train/top1", train_metrics.get("top1", 0.0), epoch + 1)
        tb_writer.add_scalar("val/top1", val_metrics.get("top1", 0.0), epoch + 1)
        tb_writer.add_scalar("test/top1", test_metrics.get("top1", 0.0), epoch + 1)
        tb_writer.add_scalar("train/top5", train_metrics.get("top5", 0.0), epoch + 1)
        tb_writer.add_scalar("val/top5", val_metrics.get("top5", 0.0), epoch + 1)
        tb_writer.add_scalar("test/top5", test_metrics.get("top5", 0.0), epoch + 1)
        tb_writer.add_scalar("train/f1_macro", train_metrics.get("f1_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("val/f1_macro", val_metrics.get("f1_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("test/f1_macro", test_metrics.get("f1_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("train/precision_macro", train_metrics.get("precision_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("val/precision_macro", val_metrics.get("precision_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("test/precision_macro", test_metrics.get("precision_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("train/recall_macro", train_metrics.get("recall_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("val/recall_macro", val_metrics.get("recall_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("test/recall_macro", test_metrics.get("recall_macro", 0.0), epoch + 1)
        tb_writer.add_scalar("train/lr", lr, epoch + 1)
        tb_writer.add_scalar("train/time_sec", epoch_time, epoch + 1)

        current_top1 = val_metrics.get("top1", 0.0)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_top1": best_val_top1,
            "config": cfg,
            "num_classes": num_classes,
        }
        if (epoch + 1) % int(train_cfg.get("save_every", 10)) == 0:
            torch.save(state, ckpt_dir / f"{exp_name}_epoch{epoch + 1}.pt")
        if current_top1 > best_val_top1:
            best_val_top1 = current_top1
            state["best_val_top1"] = best_val_top1
            torch.save(state, ckpt_dir / f"{exp_name}_best.pt")

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_top1={train_metrics.get('top1', 0.0):.2f} "
            f"val_top1={val_metrics.get('top1', 0.0):.2f} "
            f"test_top1={test_metrics.get('top1', 0.0):.2f} "
            f"lr={lr:.6f} time={epoch_time:.1f}s"
        )

    final_summary = {
        "best_val_top1": best_val_top1,
        "experiment_name": exp_name,
        "model_type": cfg["model"]["type"],
        "num_parameters": count_parameters(model),
    }
    save_json(final_summary, str(log_dir / f"{exp_name}_summary.json"))
    tb_writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
