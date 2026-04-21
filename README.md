# AI2612 Final Project: Vision Transformer Reproduction

This repository provides a reproducible PyTorch pipeline for reproducing core ideas from:

- Dosovitskiy et al., "An Image is Worth 16x16 Words"
- Steiner et al., "How to train your ViT?"

It includes:

- ViT baseline (`models/vit.py`)
- ResNet baseline (`models/resnet_baseline.py`)
- Hybrid CNN + ViT (`models/hybrid_vit.py`)
- Full train/val/test loop with checkpointing and logging
- Config-driven experiments for comparison and ablations

## 1. Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Train

```bash
python train.py --config configs/vit.yaml
python train.py --config configs/resnet.yaml
python train.py --config configs/hybrid.yaml
```

## 3. Evaluate

```bash
python evaluate.py --config configs/vit.yaml --checkpoint outputs/checkpoints/vit_best.pt
```

## 4. Ablations

Edit config values and rerun, or use scripts in `scripts/` for quick starts:

- patch size ablation
- model scale ablation
- training tricks (mixup/label smoothing/randaugment)

## 5. Output layout

- `outputs/checkpoints/`: model checkpoints
- `outputs/logs/`: per-epoch csv logs + json summaries
- `outputs/figures/`: reserved for plots

## 6. Report

Use `report/report.md` as a structured template to fill in experiment results and conclusions.
