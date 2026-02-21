# BraTS 2021 Brain Tumor Segmentation — Project Walkthrough

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data](#3-data)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Data Loading & Transforms](#5-data-loading--transforms)
6. [Model Architecture](#6-model-architecture)
7. [Training](#7-training)
8. [Inference](#8-inference)
9. [Evaluation](#9-evaluation)
10. [Results](#10-results)
11. [Figures & Visualization](#11-figures--visualization)
12. [How to Reproduce](#12-how-to-reproduce)

---

## 1. Project Overview

This project implements a **3D U-Net** for brain tumor segmentation on the **BraTS 2021** challenge dataset, built entirely with the [MONAI](https://monai.io/) framework. It covers the full pipeline: data preprocessing, model training with mixed-precision and cosine annealing, sliding-window inference, and comprehensive evaluation with per-region metrics.

**Key results:**

| Region | Dice | IoU | HD95 (mm) | Sensitivity |
|--------|------|-----|-----------|-------------|
| WT (Whole Tumor) | 0.9250 | 0.8674 | 6.80 | 0.9212 |
| TC (Tumor Core) | 0.9213 | 0.8667 | 5.31 | 0.9347 |
| ET (Enhancing Tumor) | 0.8913 | 0.8148 | 3.73 | 0.9085 |
| **Mean (all)** | **0.9126** | **0.8496** | **5.28** | **0.9215** |

**Best validation mean Dice during training: 0.9215** (epoch 277 of 289).

---

## 2. Directory Structure

```
brats-biohub/
├── README.md                          # Brief project description
├── pyproject.toml                     # Python package metadata & dependencies
├── train.py                           # Root-level entry point (forwards to src.train)
│
├── configs/
│   ├── train_3d_unet.yaml             # Training hyperparameters
│   ├── infer_sliding_window.yaml      # Inference configuration
│   └── preprocess.yaml                # Preprocessing pipeline settings
│
├── src/
│   ├── __init__.py
│   ├── train.py                       # Training loop (AMP, early stopping, wandb)
│   ├── infer.py                       # Sliding-window inference script
│   ├── eval.py                        # Evaluation metrics (Dice, IoU, HD95, Sensitivity)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── brats_dataset.py           # Dataset scanning & DataLoader construction
│   │   └── transforms.py             # MONAI transform pipelines (train & val)
│   └── preprocess/
│       ├── __init__.py
│       ├── bias_field.py              # N4ITK bias field correction (SimpleITK)
│       ├── normalize.py               # Z-score & min-max intensity normalization
│       └── resample.py                # Isotropic resampling (SimpleITK)
│
├── notebooks/
│   └── 00_preprocessing_demo.ipynb    # Interactive demo: N4, denoising, registration
│
├── checkpoints/
│   ├── best_model.pth                 # Best model (epoch 277, Dice=0.9215) — 221 MB
│   ├── latest.pth                     # Last epoch checkpoint (crash recovery)
│   └── epoch_NNNN.pth                 # Periodic snapshots every 10 epochs (0010–0280)
│
├── logs/
│   ├── train.log                      # First training run log (epochs 1–6, offline wandb)
│   └── train_run2.log                 # Resumed run (epochs 7–289, online wandb)
│
├── outputs/
│   ├── predictions/                   # 147 NIfTI prediction files (validation set)
│   │   └── BraTS2021_*_pred.nii.gz
│   ├── bias_field/                    # N4 corrected T1ce volumes + comparison figure
│   ├── denoised/                      # Curvature-flow denoised FLAIR + comparison figure
│   └── registration/                  # Rigid T1-to-T1ce registration demo
│
└── reports/
    ├── results.csv                    # Per-case evaluation metrics (147 cases x 3 regions)
    ├── training_history.json          # Full epoch-by-epoch metrics from wandb (287 entries)
    ├── generate_figures.py            # Script to produce all report figures
    └── figures/
        ├── training_curves.png        # Loss + Dice vs epoch
        ├── segmentation_comparison.png # FLAIR | GT | Pred for 4 cases
        ├── metrics_boxplot.png        # Dice/IoU/HD95 boxplots by region
        └── multi_view_prediction.png  # Axial + Coronal + Sagittal for best case
```

---

## 3. Data

**Dataset:** BraTS 2021 Training Data
**Location:** `/workspace/DataChallenge/data/BraTS2021_Training_Data/`
**Total subjects:** 731 directories (each named `BraTS2021_NNNNN`)

Each subject folder contains 5 co-registered NIfTI volumes (240 x 240 x 155, 1mm isotropic):

| File | Description |
|------|-------------|
| `*_flair.nii.gz` | Fluid Attenuated Inversion Recovery |
| `*_t1.nii.gz` | Native T1-weighted |
| `*_t1ce.nii.gz` | Contrast-enhanced T1-weighted |
| `*_t2.nii.gz` | T2-weighted |
| `*_seg.nii.gz` | Segmentation mask |

**BraTS label convention:**

| Label | Structure | Abbreviation |
|-------|-----------|--------------|
| 0 | Background | BG |
| 1 | Necrotic / Non-enhancing Tumor Core | NCR/NET |
| 2 | Peritumoral Edema | ED |
| 4 | GD-Enhancing Tumor | ET |

**Evaluation regions (derived from labels):**

| Region | Definition | Clinical meaning |
|--------|------------|------------------|
| Whole Tumor (WT) | Labels 1 + 2 + 4 | Complete extent of the lesion |
| Tumor Core (TC) | Labels 1 + 4 | Solid tumor without edema |
| Enhancing Tumor (ET) | Label 4 | Active tumor requiring treatment |

**Train/Val split:** 80/20 stratified by `sklearn.model_selection.train_test_split` with `seed=42`. This yields ~584 training cases and ~147 validation cases.

---

## 4. Preprocessing Pipeline

The project includes standalone preprocessing utilities in `src/preprocess/` and an interactive demo notebook. These are exploratory — the main training pipeline uses MONAI's on-the-fly transforms instead.

### 4.1 N4ITK Bias Field Correction (`src/preprocess/bias_field.py`)

Corrects low-frequency intensity inhomogeneity caused by MRI hardware:
- Uses SimpleITK's `N4BiasFieldCorrectionImageFilter`
- Shrink factor = 4 for speed; 4 fitting levels with 50 iterations each
- Otsu threshold generates a brain mask automatically
- Estimates log-bias field on shrunken image, applies to full resolution

**Config** (`configs/preprocess.yaml`): orientation RAS, spacing 1mm iso, z-score normalization on nonzero voxels with 0.5/99.5 percentile clipping.

### 4.2 Intensity Normalization (`src/preprocess/normalize.py`)

Two methods available via `normalize_volume(volume, method)`:
- **Z-score** (`z_score`): Clip to [0.5th, 99.5th] percentile, then `(x - mean) / std` on nonzero voxels
- **Min-max** (`min_max`): Scale to [0, 1] range after percentile clipping

### 4.3 Resampling (`src/preprocess/resample.py`)

- Resamples to isotropic 1mm spacing using SimpleITK
- B-spline interpolation for images, nearest-neighbor for labels
- Preserves physical extent by computing output size automatically

### 4.4 Preprocessing Demo Notebook (`notebooks/00_preprocessing_demo.ipynb`)

Demonstrates three techniques on 3 demo cases (BraTS2021_00000, 00002, 00003):

1. **N4ITK Bias Field Correction** — Applied to T1ce; visualizes original, estimated bias field, and corrected output
2. **Curvature Flow Denoising** — Applied to FLAIR via `sitk.CurvatureFlow` (timestep=0.0625, 5 iterations); shows original, denoised, and difference maps
3. **Rigid Registration** — Registers T1 to T1ce using Mattes mutual information + gradient descent + multi-resolution (shrink factors 4/2/1); visualized with checkerboard before/after and color overlay

---

## 5. Data Loading & Transforms

### 5.1 Dataset Scanning (`src/data/brats_dataset.py`)

`get_brats_cases(cfg)` scans the data root for `BraTS2021_*` directories and returns a list of dicts:
```python
{"flair": "/path/to/*_flair.nii.gz", "t1": "...", "t1ce": "...", "t2": "...", "seg": "..."}
```

`get_data_loaders(cfg)` builds train/val splits using `train_test_split` and wraps them in MONAI `CacheDataset` + `DataLoader`. Cache rate defaults to 0.3 (30% of data kept in RAM).

### 5.2 Label Conversion (`src/data/transforms.py`)

`ConvertToMultiChannelBasedOnBratsClassesd` is a custom MONAI `MapTransform` that converts the single-channel seg map (values 0/1/2/4) into 3 binary channels:

| Channel | Region | Formula |
|---------|--------|---------|
| 0 | TC (Tumor Core) | label == 1 OR label == 4 |
| 1 | WT (Whole Tumor) | label == 1 OR label == 2 OR label == 4 |
| 2 | ET (Enhancing Tumor) | label == 4 |

This means the model outputs 3 sigmoid channels, not 4-class softmax.

### 5.3 Training Transforms

Full augmentation pipeline (`get_train_transforms`), applied on-the-fly:

1. `LoadImaged` — Read NIfTI files
2. `EnsureChannelFirstd` — Add channel dimension
3. `EnsureTyped` — Cast images to float32, labels to uint8
4. `Orientationd` — Reorient to RAS
5. `ConvertToMultiChannelBasedOnBratsClassesd` — 3-channel binary labels
6. `Spacingd` — Resample to 1mm isotropic (bilinear for images, nearest for labels)
7. `CropForegroundd` — Remove air background (FLAIR as reference, margin=10)
8. `SpatialPadd` — Ensure minimum 128x128x128
9. `RandSpatialCropd` — Random crop to 128x128x128
10. `RandFlipd` — Random flips along all 3 axes (prob=0.5 each)
11. `RandAffined` — Random rotation (15 deg) + scaling (0.9–1.1), prob=0.5
12. `NormalizeIntensityd` — Per-channel z-score on nonzero voxels
13. `RandScaleIntensityd` — Multiplicative jitter (factor=0.1, prob=0.5)
14. `RandShiftIntensityd` — Additive jitter (offset=0.1, prob=0.5)

### 5.4 Validation Transforms

Deterministic subset of training transforms (`get_val_transforms`): Load, channel-first, orient, label conversion, spacing, foreground crop, normalize. No random augmentation, no random crop — uses full volume.

---

## 6. Model Architecture

**Architecture:** MONAI 3D U-Net (`monai.networks.nets.UNet`)

| Parameter | Value |
|-----------|-------|
| Spatial dimensions | 3 |
| Input channels | 4 (FLAIR, T1, T1ce, T2 concatenated) |
| Output channels | 3 (TC, WT, ET — sigmoid activation) |
| Encoder channels | [32, 64, 128, 256, 512] |
| Strides | [2, 2, 2, 2] |
| Residual units | 2 per block |
| Normalization | Instance normalization |
| **Total parameters** | **19,223,978** (~19.2M) |

The 4 modalities are concatenated along the channel dimension before being fed into the network. The model outputs 3 channels with sigmoid activation (multi-label, not mutually exclusive), matching the 3 overlapping evaluation regions (TC, WT, ET).

---

## 7. Training

### 7.1 Configuration (`configs/train_3d_unet.yaml`)

| Hyperparameter | Value |
|----------------|-------|
| Max epochs | 300 |
| Batch size | 2 |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR (T_max=300, eta_min=1e-7) |
| Loss function | DiceLoss (sigmoid=True, batch=True) |
| Early stopping | patience=50, monitors val mean Dice |
| ROI size | 128 x 128 x 128 |
| Cache rate | 0.3 (30% of volumes kept in RAM) |

### 7.2 Training Script (`src/train.py`)

Key features:
- **AMP mixed precision** — `torch.cuda.amp.GradScaler` + `autocast("cuda")` for memory efficiency and speed
- **Sliding-window validation** — Uses `monai.inferers.sliding_window_inference` with 50% overlap and Gaussian weighting during validation
- **CosineAnnealingLR** — Decays LR from 1e-4 to 1e-7 over 300 epochs
- **Early stopping** — Patience=50 epochs monitoring validation mean Dice; counter persisted in checkpoints
- **Checkpoint strategy:**
  - `checkpoints/latest.pth` — Overwritten every epoch (crash protection)
  - `checkpoints/epoch_NNNN.pth` — Saved every 10 epochs
  - `checkpoints/best_model.pth` — Updated when val mean Dice improves
- **Resume support** — `--resume checkpoints/latest.pth` restores model, optimizer, scheduler, scaler, epoch, best Dice, early-stopping state, and wandb run ID
- **wandb integration** — Logs train/val loss, per-class Dice, mean Dice, LR, and epoch time; resume reconnects to same run

### 7.3 Training History

The model was trained in two sessions:
1. **Run 1** (`logs/train.log`): Epochs 1–6, wandb offline mode, initial exploration
2. **Run 2** (`logs/train_run2.log`): Resumed from epoch 7, ran through epoch 289 where early stopping triggered

**Training timeline:**
- Epoch 2: Mean Dice = 0.291, Train Loss = 0.890
- Epoch ~50: Mean Dice ~0.90, rapid convergence
- Epoch 150+: Mean Dice plateaus around 0.92
- **Epoch 277 (best):** Mean Dice = **0.9215** (WT=0.9324, TC=0.9316, ET=0.9006)
- Epoch 289: Early stopping triggered (50 epochs without improvement)

**Total training time:** ~21.4 hours (77,063 seconds), ~267–339 seconds per epoch on GPU.

wandb project: `qianxu0517-vitra-labs/brats-biohub` (run: `rosy-haze-1`)

---

## 8. Inference

### 8.1 Configuration (`configs/infer_sliding_window.yaml`)

| Parameter | Value |
|-----------|-------|
| Checkpoint | `checkpoints/best_model.pth` |
| ROI size | 128 x 128 x 128 |
| Sliding window batch size | 4 |
| Overlap | 0.5 (50%) |
| Blending mode | Gaussian |
| Post-processing | Sigmoid > 0.5 threshold |

### 8.2 Inference Script (`src/infer.py`)

Inference pipeline:
1. Load best checkpoint and set model to eval mode
2. Re-derive the validation split (same seed=42) to ensure identical cases
3. For each validation case:
   - Apply inference transforms (Load, Orient to RAS, Spacing 1mm, Z-score normalize)
   - Run sliding-window inference with Gaussian blending
   - Sigmoid + threshold at 0.5 to get binary predictions for TC/WT/ET
   - Convert 3-channel predictions back to BraTS label format: WT→2 (ED), TC→1 (NCR/NET), ET→4 (ET)
   - Reorient from RAS back to original image orientation
   - Resample to original image dimensions if needed (nearest-neighbor)
   - Save as NIfTI with original affine matrix

**Output:** 147 prediction files in `outputs/predictions/` (one per validation case).

---

## 9. Evaluation

### 9.1 Metrics (`src/eval.py`)

Four metrics computed per-case, per-region:

| Metric | Description |
|--------|-------------|
| **Dice** | `2 * |P ∩ G| / (|P| + |G|)` — Overlap coefficient |
| **IoU** | `|P ∩ G| / |P ∪ G|` — Intersection over Union |
| **HD95** | 95th percentile Hausdorff distance — Surface-to-surface error in mm |
| **Sensitivity** | `TP / (TP + FN)` — Recall / True Positive Rate |

The evaluation script:
1. Scans `outputs/predictions/` for `*_pred.nii.gz` files
2. Loads corresponding ground truth from the BraTS data directory
3. Extracts WT/TC/ET binary masks from both prediction and ground truth
4. Computes all 4 metrics for each region of each case
5. Writes per-case results + mean summary rows to `reports/results.csv`

### 9.2 Region Extraction Logic

Predictions are converted from BraTS label format back to binary region masks:
- **TC:** label == 1 OR label == 4
- **WT:** label == 1 OR label == 2 OR label == 4
- **ET:** label == 4

---

## 10. Results

### 10.1 Summary Metrics (147 validation cases)

| Region | Dice | IoU | HD95 (mm) | Sensitivity |
|--------|------|-----|-----------|-------------|
| WT | 0.9250 | 0.8674 | 6.80 | 0.9212 |
| TC | 0.9213 | 0.8667 | 5.31 | 0.9347 |
| ET | 0.8913 | 0.8148 | 3.73 | 0.9085 |
| **Overall** | **0.9126** | **0.8496** | **5.28** | **0.9215** |

### 10.2 Per-Region Analysis

- **Whole Tumor (WT)** achieves the highest Dice (0.925) — the largest region is easiest to delineate
- **Tumor Core (TC)** is close behind (0.921) with the best sensitivity (0.935)
- **Enhancing Tumor (ET)** is the most challenging (0.891 Dice) — smallest structures with ambiguous boundaries
- **HD95** is lowest for ET (3.73mm) because ET regions are compact; WT has higher HD95 (6.80mm) due to a few cases with scattered false positives

### 10.3 Notable Cases

- **Best case:** BraTS2021_00816 — Mean Dice = 0.983 (WT=0.984, TC=0.989, ET=0.977)
- **Challenging cases:**
  - BraTS2021_00116 — ET Dice = 0.397 (very small enhancing region)
  - BraTS2021_00432 — TC Dice = 0.399 (unusual tumor morphology)
  - BraTS2021_01091 — WT Dice = 0.544 (diffuse tumor boundary)

---

## 11. Figures & Visualization

All figures are in `reports/figures/` and can be regenerated with:
```bash
python reports/generate_figures.py
```

### 11.1 Training Curves (`training_curves.png`)

Two-panel figure showing:
- **Left:** Train/Val loss (Dice loss) vs epoch — both curves converge around epoch 50; train loss reaches ~0.06, val loss ~0.08
- **Right:** Per-region Dice on validation set — WT and TC converge to ~0.93, ET to ~0.90; Mean Dice stabilizes around 0.92

### 11.2 Segmentation Comparison (`segmentation_comparison.png`)

Grid showing FLAIR input | Ground Truth overlay | Prediction overlay for 4 cases at different performance levels:
- Best (Dice=0.983), 75th percentile (0.961), median (0.939), 25th percentile (0.887)
- Color coding: Blue = NCR/NET (label 1), Green = ED (label 2), Red = ET (label 4)

### 11.3 Metrics Boxplot (`metrics_boxplot.png`)

Three side-by-side boxplots showing Dice, IoU, and HD95 distributions for WT/TC/ET:
- Most cases cluster at high Dice (>0.90) with a few outliers
- HD95 shows a long tail with some cases >50mm (anatomically complex tumors)

### 11.4 Multi-View Prediction (`multi_view_prediction.png`)

Best-performing case (BraTS2021_00816) shown in Axial, Coronal, and Sagittal views:
- Top row: Ground truth overlay
- Bottom row: Model prediction overlay
- Demonstrates accurate 3D segmentation across all views

### 11.5 Preprocessing Outputs

Located in `outputs/`:
- `bias_field/bias_field_comparison.png` — N4 bias field correction on 3 T1ce cases
- `denoised/denoising_comparison.png` — Curvature flow denoising on 3 FLAIR cases
- `registration/registration_demo.png` — Rigid T1-to-T1ce registration with checkerboard visualization

---

## 12. How to Reproduce

### 12.1 Environment Setup

```bash
pip install monai nibabel SimpleITK scikit-learn scikit-survival shap xgboost \
            seaborn plotly scipy kaggle tqdm wandb pandas numpy pyyaml matplotlib
```

Requires Python >= 3.10, PyTorch with CUDA, and a GPU with >= 16GB VRAM (for batch_size=2 with 128^3 volumes).

### 12.2 Training

```bash
# Full training from scratch (~21 hours on single GPU)
python -m src.train --config configs/train_3d_unet.yaml

# Resume from checkpoint
python -m src.train --config configs/train_3d_unet.yaml --resume checkpoints/latest.pth

# Quick dry-run (4 cases, 1 epoch)
python -m src.train --config configs/train_3d_unet.yaml --epochs 1

# Without wandb logging
python -m src.train --config configs/train_3d_unet.yaml --no-wandb
```

### 12.3 Inference

```bash
python -m src.infer --config configs/infer_sliding_window.yaml
```

Outputs 147 NIfTI predictions to `outputs/predictions/`.

### 12.4 Evaluation

```bash
python -m src.eval --pred-dir outputs/predictions \
                   --gt-dir /workspace/DataChallenge/data/BraTS2021_Training_Data \
                   --output-csv reports/results.csv
```

### 12.5 Generate Report Figures

```bash
python reports/generate_figures.py
```

### 12.6 Preprocessing Demo

```bash
jupyter notebook notebooks/00_preprocessing_demo.ipynb
```

---

## Appendix: Key Design Decisions

1. **Sigmoid multi-label vs softmax multi-class:** The model uses 3 sigmoid outputs for overlapping regions (TC, WT, ET) rather than 4-class softmax. This matches the BraTS evaluation protocol where regions overlap (e.g., ET is a subset of TC, which is a subset of WT).

2. **DiceLoss only (no cross-entropy):** Despite the config mentioning `DiceCELoss`, the actual implementation uses pure `DiceLoss` with sigmoid activation — this directly optimizes the evaluation metric.

3. **Sliding-window inference with Gaussian blending:** Handles the full 3D volume at test time (no cropping) by sweeping a 128^3 window with 50% overlap and Gaussian-weighted blending to avoid boundary artifacts.

4. **CacheDataset with 30% cache rate:** Balances RAM usage with I/O speed — 30% of training volumes are cached in memory, the rest are loaded from disk each epoch.

5. **On-the-fly transforms vs offline preprocessing:** The standalone preprocessing utilities (`src/preprocess/`) were used for exploration in the notebook. The actual training pipeline applies all spatial and intensity transforms on-the-fly through MONAI, which is more flexible and avoids storing preprocessed copies.
