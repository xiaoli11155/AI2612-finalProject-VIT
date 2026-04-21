#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/vit_cifar100.yaml}"
python train.py --config "$CONFIG_PATH"
