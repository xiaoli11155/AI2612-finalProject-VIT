# Task: Reproduce Vision Transformer (ViT) for Image Classification

## 1. Objective

Implement and reproduce the core ideas of the Vision Transformer (ViT) paper:

- Dosovitskiy et al., **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**
- Steiner et al., **"How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers"**

The goal is **not** to fully reproduce the original large-scale pretraining setting (e.g. JFT-300M / ImageNet-21k), because that is computationally unrealistic for this project.  
Instead, the goal is to reproduce **as much of the original paper’s methodology, architecture design, training logic, and empirical conclusions as possible** on smaller public datasets.

This project should focus on:

1. Implementing a standard ViT for image classification.
2. Comparing ViT against a traditional CNN baseline such as ResNet.
3. Studying the effects of key ViT design choices:
   - patch size
   - model depth
   - number of attention heads
   - embedding dimension
   - data augmentation / regularization
4. Optionally implementing a hybrid model:
   - CNN/ResNet stem + Transformer encoder
5. Producing a clean experimental report with reproducible code, logs, plots, and conclusions.

---

## 2. Expected Reproduction Scope

### 2.1 What should be reproduced from the original ViT paper

Try to reproduce the following core aspects of the paper:

- Patch-based image tokenization
- Linear patch embedding
- Learnable class token
- Learnable positional embedding
- Transformer encoder stack
- MLP classification head
- Comparison between ViT and CNN baselines
- Study of patch size effects
- Study of model scale effects
- If possible, hybrid CNN + ViT structure
- If possible, regularization and augmentation improvements inspired by Steiner et al.

### 2.2 What does NOT need to be fully reproduced

The following are not required to be matched exactly due to resource limits:

- JFT-300M pretraining
- ImageNet-21k pretraining
- Huge ViT models such as ViT-L/16 or ViT-H/14 unless resources allow
- TPU-specific training setup
- Exact original accuracy numbers from the paper

Instead, the project should aim to reproduce the **trend and conclusions**, for example:

- ViT can perform competitively on image classification
- ViT performance is sensitive to data scale and regularization
- patch size matters
- hybrid models may help on smaller datasets
- CNN baselines can still be stronger in low-data regimes

---

## 3. Dataset Requirements

Use at least one public image classification dataset. Prefer the following order:

### Option A (recommended for feasibility)
- **CIFAR-10**
- **CIFAR-100**

### Option B (better for stronger experiments if compute allows)
- **Tiny ImageNet**

### Optional extension
- Try more than one dataset if resources allow

### Dataset split
For each dataset:
- training set
- validation set
- test set

If the dataset does not provide an official validation split, create one from the training set.

### Preprocessing
- resize images to a unified resolution
- normalize with dataset mean/std
- convert to tensor
- keep preprocessing reproducible

Suggested image sizes:
- CIFAR-10 / CIFAR-100: 32x32 directly, or upscale to 64x64 / 128x128 if needed for ViT experiments
- Tiny ImageNet: 64x64

---

## 4. Models to Implement

## 4.1 Baseline 1: Standard Vision Transformer (mandatory)

Implement a standard ViT classifier with the following components:

- image split into non-overlapping patches
- flatten each patch
- linear projection to embedding dimension
- prepend a learnable `[CLS]` token
- add learnable positional embeddings
- pass sequence through Transformer encoder layers
- use `[CLS]` token output for classification
- MLP head for final logits

### Minimum configurable hyperparameters
- image size
- patch size
- embedding dimension
- number of Transformer layers
- number of attention heads
- MLP hidden ratio
- dropout rate
- number of classes

---

## 4.2 Baseline 2: CNN model for comparison (mandatory)

Implement or use a standard ResNet baseline.

Recommended choices:
- ResNet-18
- ResNet-34

This model serves as the traditional CNN baseline for comparison.

---

## 4.3 Hybrid model (bonus but strongly encouraged)

Implement a hybrid model inspired by the ViT paper:

- use a CNN backbone or ResNet stem to extract local feature maps
- convert feature maps into token sequences
- feed them into Transformer encoder
- classify with a Transformer-based head

This is meant to test whether local inductive bias helps when training on smaller datasets.

---

## 5. Training Requirements

## 5.1 Core training pipeline

Build a full training pipeline including:

- training loop
- validation loop
- test loop
- checkpoint saving
- best model selection by validation accuracy
- logging of metrics
- random seed control for reproducibility

### Log at least:
- training loss
- validation loss
- training accuracy
- validation accuracy
- test accuracy
- learning rate
- training time per epoch

---

## 5.2 Evaluation metrics

At minimum report:
- Top-1 Accuracy

If feasible, also report:
- Top-5 Accuracy
- Precision
- Recall
- F1-score
- confusion matrix

For CIFAR-10, Top-5 is less informative but can still be included for completeness.

---

## 5.3 Suggested optimizer / scheduler

Start with:
- AdamW optimizer
- cosine learning rate schedule
- warmup epochs

These choices are closer to ViT-style training than SGD alone.

Also make weight decay configurable.

---

## 6. Required Experiments

The experiments should be organized clearly and reproducibly.

## 6.1 Experiment A: ViT vs CNN

Compare:
- ViT
- ResNet baseline

Report:
- accuracy
- training stability
- convergence speed
- parameter count
- approximate training cost

Goal:
- analyze performance differences between Transformer-based and CNN-based image classifiers

---

## 6.2 Experiment B: Patch size ablation

Train ViT with different patch sizes, for example:
- patch size = 4
- patch size = 8
- patch size = 16

Analyze:
- effect on accuracy
- effect on token sequence length
- effect on computational cost
- effect on overfitting / underfitting

---

## 6.3 Experiment C: Model scale ablation

Vary model size, for example:
- shallow vs deep Transformer
- small vs larger embedding dimension
- fewer vs more attention heads

Example settings:
- ViT-Tiny
- ViT-Small
- ViT-Base-like simplified version

Analyze:
- whether larger models help on the chosen dataset
- whether performance saturates or degrades in low-data settings

---

## 6.4 Experiment D: Training tricks from “How to train your ViT?” (bonus but recommended)

Try one or more of the following:
- MixUp
- CutMix
- RandAugment
- label smoothing
- dropout
- stochastic depth
- repeated augmentation

Analyze which methods help most.

---

## 6.5 Experiment E: Hybrid model (bonus)

Compare:
- standard ViT
- hybrid CNN + ViT
- ResNet baseline

Analyze:
- whether local feature extraction helps on smaller datasets
- whether hybrid structure improves optimization or generalization

---

## 7. Implementation Requirements

## 7.1 Code quality

The codebase should be modular and clean.

Suggested project structure:

```text
project/
├─ configs/
│  ├─ vit.yaml
│  ├─ resnet.yaml
│  ├─ hybrid.yaml
├─ datasets/
├─ models/
│  ├─ vit.py
│  ├─ resnet_baseline.py
│  ├─ hybrid_vit.py
│  ├─ transformer_blocks.py
├─ train.py
├─ evaluate.py
├─ utils/
│  ├─ metrics.py
│  ├─ logger.py
│  ├─ seed.py
│  ├─ scheduler.py
├─ scripts/
│  ├─ run_vit.sh
│  ├─ run_resnet.sh
│  ├─ run_ablation.sh
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ figures/
├─ report/
│  ├─ report.md
│  ├─ report.pdf
└─ README.md