from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_training_log(csv_path: str):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in log file: {csv_path}")
    return rows


def _to_series(rows, key: str):
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def plot_training_curves(csv_path: str, figure_dir: str, experiment_name: str):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    rows = _read_training_log(csv_path)
    epochs = _to_series(rows, "epoch")

    train_loss = _to_series(rows, "train_loss")
    val_loss = _to_series(rows, "val_loss")
    train_top1 = _to_series(rows, "train_top1")
    val_top1 = _to_series(rows, "val_top1")
    test_top1 = _to_series(rows, "test_top1")

    figure_dir_path = Path(figure_dir)
    figure_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir_path / f"{experiment_name}_curves.png"

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_top1, label="Train Top-1", linewidth=2)
    plt.plot(epochs, val_top1, label="Val Top-1", linewidth=2)
    plt.plot(epochs, test_top1, label="Test Top-1", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 Accuracy Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(output_path)
