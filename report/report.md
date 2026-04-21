# Vision Transformer Reproduction Report

## 1. Project Goal

Reproduce core ideas from ViT on compute-feasible datasets:

- Standard ViT pipeline
- Comparison against CNN baseline
- Patch-size and model-scale ablations
- Optional hybrid CNN + ViT
- Optional training tricks (MixUp / label smoothing / RandAugment)

## 2. Environment and Reproducibility

- Framework: PyTorch
- Seed: (fill in)
- Hardware: (fill in)
- Dataset(s): (fill in)
- Config files used: (fill in)

## 3. Models

### 3.1 ViT

- image size:
- patch size:
- embed dim:
- depth:
- heads:
- params:

### 3.2 ResNet Baseline

- variant:
- params:

### 3.3 Hybrid (Optional)

- CNN stem:
- transformer depth:
- params:

## 4. Training Setup

- optimizer: AdamW
- schedule: warmup + cosine decay
- epochs:
- batch size:
- weight decay:
- label smoothing:
- mixup:

## 5. Experiments

### 5.1 A: ViT vs ResNet

| Model | Top-1 | Train Time / Epoch | Params | Notes |
|---|---:|---:|---:|---|
| ViT |  |  |  |  |
| ResNet |  |  |  |  |

Analysis:

### 5.2 B: Patch Size Ablation

| Patch Size | Seq Length | Top-1 | Compute Cost | Notes |
|---:|---:|---:|---:|---|
| 4 |  |  |  |  |
| 8 |  |  |  |  |
| 16 |  |  |  |  |

Analysis:

### 5.3 C: Model Scale Ablation

| Setting | Depth | Embed Dim | Heads | Top-1 | Notes |
|---|---:|---:|---:|---:|---|
| Tiny |  |  |  |  |  |
| Small |  |  |  |  |  |
| Base-like |  |  |  |  |  |

Analysis:

### 5.4 D: Training Tricks (Optional)

| Trick | Top-1 Gain | Notes |
|---|---:|---|
| MixUp |  |  |
| Label smoothing |  |  |
| RandAugment |  |  |

Analysis:

### 5.5 E: Hybrid Model (Optional)

| Model | Top-1 | Notes |
|---|---:|---|
| ViT |  |  |
| Hybrid |  |  |
| ResNet |  |  |

Analysis:

## 6. Main Findings

1. 
2. 
3. 

## 7. Limitations and Future Work

1. 
2. 

## 8. Appendix

- Key configs
- Extra plots
- Confusion matrices
