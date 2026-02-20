"""
Generate publication-quality figures for BraTS 2021 evaluation report.

Figures:
  1. training_curves.png       — Loss + Dice vs epoch
  2. segmentation_comparison.png — FLAIR | GT | Pred for 4 cases
  3. metrics_boxplot.png       — Dice/IoU/HD95 boxplots by region
  4. multi_view_prediction.png — Axial + Coronal + Sagittal for 1 case

Usage:
    python reports/generate_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
PREDICTIONS = ROOT / "outputs" / "predictions"
DATA_ROOT = Path("/workspace/DataChallenge/data/BraTS2021_Training_Data")
HISTORY_JSON = REPORTS / "training_history.json"
RESULTS_CSV = REPORTS / "results.csv"

FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# BraTS label colors: 0=bg, 1=NCR/NET (blue), 2=ED (green), 4=ET (red)
SEG_CMAP = ListedColormap(["none", "#2196F3", "#4CAF50", "none", "#F44336"])
SEG_ALPHA = 0.55

REGION_COLORS = {"WT": "#4CAF50", "TC": "#2196F3", "ET": "#F44336"}


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Training Curves
# ═══════════════════════════════════════════════════════════════════════════
def plot_training_curves():
    print("Generating training_curves.png …")
    with open(HISTORY_JSON) as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history if r.get("epoch") is not None]
    train_loss = [r.get("train/loss") for r in history]
    val_loss = [r.get("val/loss") for r in history]
    dice_wt = [r.get("val/dice_WT") for r in history]
    dice_tc = [r.get("val/dice_TC") for r in history]
    dice_et = [r.get("val/dice_ET") for r in history]
    dice_mean = [r.get("val/dice_mean") for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1565C0", linewidth=1.2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#E65100", linewidth=1.2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Dice-CE)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(frameon=True, fancybox=False, edgecolor="#ccc")
    ax1.set_xlim(0, max(epochs))
    ax1.grid(True, alpha=0.3)

    # Dice curves
    ax2.plot(epochs, dice_wt, label="WT Dice", color=REGION_COLORS["WT"], linewidth=1.2)
    ax2.plot(epochs, dice_tc, label="TC Dice", color=REGION_COLORS["TC"], linewidth=1.2)
    ax2.plot(epochs, dice_et, label="ET Dice", color=REGION_COLORS["ET"], linewidth=1.2)
    ax2.plot(epochs, dice_mean, label="Mean Dice", color="#333", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice by Region")
    ax2.legend(frameon=True, fancybox=False, edgecolor="#ccc")
    ax2.set_xlim(0, max(epochs))
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png")
    plt.close(fig)
    print("  -> saved training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Segmentation Comparison
# ═══════════════════════════════════════════════════════════════════════════
def _find_best_slice(seg: np.ndarray, axis: int = 2) -> int:
    """Find axial slice with the most tumor voxels."""
    tumor_counts = np.sum(seg > 0, axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(tumor_counts))


def _overlay_seg(ax, img_slice, seg_slice, title):
    """Show grayscale image with segmentation overlay."""
    ax.imshow(img_slice.T, cmap="gray", origin="lower", aspect="equal")
    masked = np.ma.masked_where(seg_slice == 0, seg_slice)
    ax.imshow(masked.T, cmap=SEG_CMAP, vmin=0, vmax=4, alpha=SEG_ALPHA, origin="lower", aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_segmentation_comparison():
    print("Generating segmentation_comparison.png …")

    # Read CSV to find 4 diverse cases (pick by spread in Dice)
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["case_id"] != "MEAN"]

    # Compute mean Dice per case
    case_dice: dict[str, list[float]] = {}
    for r in rows:
        case_dice.setdefault(r["case_id"], []).append(float(r["dice"]))
    case_mean = {c: np.mean(v) for c, v in case_dice.items()}

    # Pick 4 cases: best, ~75th pct, ~median, ~25th pct
    sorted_cases = sorted(case_mean.items(), key=lambda x: x[1])
    n = len(sorted_cases)
    indices = [n - 1, int(0.75 * n), int(0.5 * n), int(0.25 * n)]
    selected = [sorted_cases[i][0] for i in indices]

    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)))

    legend_patches = [
        mpatches.Patch(color="#2196F3", alpha=SEG_ALPHA, label="NCR/NET (1)"),
        mpatches.Patch(color="#4CAF50", alpha=SEG_ALPHA, label="ED (2)"),
        mpatches.Patch(color="#F44336", alpha=SEG_ALPHA, label="ET (4)"),
    ]

    for row_idx, case_id in enumerate(selected):
        flair_path = DATA_ROOT / case_id / f"{case_id}_flair.nii.gz"
        gt_path = DATA_ROOT / case_id / f"{case_id}_seg.nii.gz"
        pred_path = PREDICTIONS / f"{case_id}_pred.nii.gz"

        flair = nib.load(str(flair_path)).get_fdata()
        gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)
        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)

        sl = _find_best_slice(gt, axis=2)
        dice_val = case_mean[case_id]

        ax_flair = axes[row_idx, 0]
        ax_gt = axes[row_idx, 1]
        ax_pred = axes[row_idx, 2]

        ax_flair.imshow(flair[:, :, sl].T, cmap="gray", origin="lower", aspect="equal")
        ax_flair.set_title(f"FLAIR — {case_id}", fontsize=9)
        ax_flair.axis("off")

        _overlay_seg(ax_gt, flair[:, :, sl], gt[:, :, sl], "Ground Truth")
        _overlay_seg(ax_pred, flair[:, :, sl], pred[:, :, sl],
                     f"Prediction (Dice={dice_val:.3f})")

    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               frameon=True, fancybox=False, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Segmentation Comparison (Axial View)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "segmentation_comparison.png")
    plt.close(fig)
    print("  -> saved segmentation_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Metrics Boxplot
# ═══════════════════════════════════════════════════════════════════════════
def plot_metrics_boxplot():
    print("Generating metrics_boxplot.png …")
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["case_id"] != "MEAN"]

    regions = ["WT", "TC", "ET"]
    metrics = ["dice", "iou", "hd95"]
    metric_labels = {"dice": "Dice Score", "iou": "IoU", "hd95": "HD95 (mm)"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        colors_list = []
        for region in regions:
            vals = [float(r[metric]) for r in rows if r["region"] == region]
            # Filter inf for HD95
            vals = [v for v in vals if np.isfinite(v)]
            data.append(vals)
            labels.append(region)
            colors_list.append(REGION_COLORS[region])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))

        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(metric_labels[metric], fontsize=11)
        ax.set_xlabel("Region")
        ax.set_ylabel(metric_labels[metric])
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Evaluation Metrics Distribution by Region", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "metrics_boxplot.png")
    plt.close(fig)
    print("  -> saved metrics_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Multi-View Prediction
# ═══════════════════════════════════════════════════════════════════════════
def plot_multi_view():
    print("Generating multi_view_prediction.png …")

    # Pick case with highest mean Dice for a clean visualization
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["case_id"] != "MEAN"]
    case_dice: dict[str, list[float]] = {}
    for r in rows:
        case_dice.setdefault(r["case_id"], []).append(float(r["dice"]))
    case_mean = {c: np.mean(v) for c, v in case_dice.items()}
    best_case = max(case_mean, key=case_mean.get)

    flair_path = DATA_ROOT / best_case / f"{best_case}_flair.nii.gz"
    pred_path = PREDICTIONS / f"{best_case}_pred.nii.gz"
    gt_path = DATA_ROOT / best_case / f"{best_case}_seg.nii.gz"

    flair = nib.load(str(flair_path)).get_fdata()
    pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
    gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

    # Find best slices for each view
    ax_sl = _find_best_slice(gt, axis=2)  # axial
    cor_sl = _find_best_slice(gt, axis=1)  # coronal
    sag_sl = _find_best_slice(gt, axis=0)  # sagittal

    view_specs = [
        ("Axial", flair[:, :, ax_sl], pred[:, :, ax_sl], gt[:, :, ax_sl]),
        ("Coronal", flair[:, cor_sl, :], pred[:, cor_sl, :], gt[:, cor_sl, :]),
        ("Sagittal", flair[sag_sl, :, :], pred[sag_sl, :, :], gt[sag_sl, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    legend_patches = [
        mpatches.Patch(color="#2196F3", alpha=SEG_ALPHA, label="NCR/NET (1)"),
        mpatches.Patch(color="#4CAF50", alpha=SEG_ALPHA, label="ED (2)"),
        mpatches.Patch(color="#F44336", alpha=SEG_ALPHA, label="ET (4)"),
    ]

    for col, (view_name, img_sl, pred_sl, gt_sl) in enumerate(view_specs):
        # Top row: GT overlay
        ax_top = axes[0, col]
        ax_top.imshow(img_sl.T, cmap="gray", origin="lower", aspect="equal")
        masked_gt = np.ma.masked_where(gt_sl == 0, gt_sl)
        ax_top.imshow(masked_gt.T, cmap=SEG_CMAP, vmin=0, vmax=4, alpha=SEG_ALPHA,
                      origin="lower", aspect="equal")
        ax_top.set_title(f"{view_name} — Ground Truth", fontsize=10)
        ax_top.axis("off")

        # Bottom row: Prediction overlay
        ax_bot = axes[1, col]
        ax_bot.imshow(img_sl.T, cmap="gray", origin="lower", aspect="equal")
        masked_pred = np.ma.masked_where(pred_sl == 0, pred_sl)
        ax_bot.imshow(masked_pred.T, cmap=SEG_CMAP, vmin=0, vmax=4, alpha=SEG_ALPHA,
                      origin="lower", aspect="equal")
        ax_bot.set_title(f"{view_name} — Prediction", fontsize=10)
        ax_bot.axis("off")

    fig.suptitle(f"Multi-View: {best_case}  (Mean Dice = {case_mean[best_case]:.3f})",
                 fontsize=13)
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               frameon=True, fancybox=False, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    fig.savefig(FIGURES / "multi_view_prediction.png")
    plt.close(fig)
    print("  -> saved multi_view_prediction.png")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_training_curves()
    plot_segmentation_comparison()
    plot_metrics_boxplot()
    plot_multi_view()
    print("\nAll figures saved to", FIGURES)
