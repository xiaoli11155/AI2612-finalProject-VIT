#!/usr/bin/env bash
set -euo pipefail

# Example patch-size ablation for ViT.
python train.py --config configs/vit.yaml
