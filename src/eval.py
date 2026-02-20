"""
src/eval.py — BraTS 2021 evaluation metrics.

Computes per-class Dice, IoU, HD95, and Sensitivity for regions (TC, WT, ET)
on predictions vs ground truth. Saves per-case and summary results to CSV.

Usage
-----
    python -m src.eval
    python -m src.eval --pred-dir outputs/predictions --gt-dir /path/to/BraTS2021_Training_Data
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# BraTS class mapping
CLASS_NAMES = ["WT", "TC", "ET"]


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(pred & gt)
    if pred.sum() + gt.sum() == 0:
        return 1.0  # both empty -> perfect
    return 2.0 * intersection / (pred.sum() + gt.sum())


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    if union == 0:
        return 1.0  # both empty -> perfect
    return float(intersection / union)


def sensitivity_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Sensitivity (Recall / True Positive Rate)."""
    tp = np.sum(pred & gt)
    fn = np.sum(~pred.astype(bool) & gt.astype(bool))
    if tp + fn == 0:
        return 1.0  # no ground truth -> perfect
    return float(tp / (tp + fn))


def hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute 95th-percentile Hausdorff distance between two binary masks."""
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    # Distance from pred surface to gt
    gt_dist = distance_transform_edt(~gt.astype(bool))
    pred_surface = pred.astype(bool) & ~_erode(pred.astype(bool))
    d_pred_to_gt = gt_dist[pred_surface] if pred_surface.any() else gt_dist[pred.astype(bool)]

    # Distance from gt surface to pred
    pred_dist = distance_transform_edt(~pred.astype(bool))
    gt_surface = gt.astype(bool) & ~_erode(gt.astype(bool))
    d_gt_to_pred = pred_dist[gt_surface] if gt_surface.any() else pred_dist[gt.astype(bool)]

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion (1-voxel, 6-connectivity)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


def extract_regions(label_map: np.ndarray) -> dict[str, np.ndarray]:
    """Convert BraTS single-label map to binary region masks."""
    return {
        "TC": ((label_map == 1) | (label_map == 4)).astype(np.uint8),
        "WT": ((label_map == 1) | (label_map == 2) | (label_map == 4)).astype(np.uint8),
        "ET": (label_map == 4).astype(np.uint8),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2021 evaluation")
    p.add_argument(
        "--pred-dir",
        default="/workspace/brats-biohub/outputs/predictions",
        help="Directory containing prediction NIfTI files (*_pred.nii.gz)",
    )
    p.add_argument(
        "--gt-dir",
        default="/workspace/DataChallenge/data/BraTS2021_Training_Data",
        help="BraTS training data root with ground truth segmentations",
    )
    p.add_argument(
        "--output-csv",
        default="/workspace/brats-biohub/reports/results.csv",
        help="Path to output CSV file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*_pred.nii.gz"))
    if not pred_files:
        log.error("No prediction files found in %s", pred_dir)
        return

    log.info("Evaluating %d predictions …", len(pred_files))

    METRICS = ["dice", "iou", "hd95", "sensitivity"]
    rows: list[dict] = []

    for pred_path in tqdm(pred_files, desc="Evaluating", unit="case"):
        case_id = pred_path.name.replace("_pred.nii.gz", "")
        gt_path = gt_dir / case_id / f"{case_id}_seg.nii.gz"

        if not gt_path.exists():
            log.warning("Ground truth not found for %s, skipping", case_id)
            continue

        pred_map = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        gt_map = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

        pred_regions = extract_regions(pred_map)
        gt_regions = extract_regions(gt_map)

        for region in CLASS_NAMES:
            p_r, g_r = pred_regions[region], gt_regions[region]
            rows.append({
                "case_id": case_id,
                "region": region,
                "dice": dice_score(p_r, g_r),
                "iou": iou_score(p_r, g_r),
                "hd95": hausdorff_95(p_r, g_r),
                "sensitivity": sensitivity_score(p_r, g_r),
            })

    # ── Write per-case CSV ────────────────────────────────────────────────
    fieldnames = ["case_id", "region", "dice", "iou", "hd95", "sensitivity"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        # Append mean rows
        for region in CLASS_NAMES:
            region_rows = [r for r in rows if r["region"] == region]
            mean_row = {"case_id": "MEAN", "region": region}
            for m in METRICS:
                vals = np.array([r[m] for r in region_rows])
                finite_vals = vals[np.isfinite(vals)]
                mean_row[m] = float(finite_vals.mean()) if len(finite_vals) > 0 else np.nan
            writer.writerow(mean_row)

        # Overall mean across all regions
        overall = {"case_id": "MEAN", "region": "ALL"}
        for m in METRICS:
            vals = np.array([r[m] for r in rows])
            finite_vals = vals[np.isfinite(vals)]
            overall[m] = float(finite_vals.mean()) if len(finite_vals) > 0 else np.nan
        writer.writerow(overall)

    log.info("Per-case results saved to %s  (%d rows)", output_csv, len(rows))

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BraTS 2021 Evaluation Results")
    print("=" * 80)
    header = f"{'Region':<8} {'Dice':>10} {'IoU':>10} {'HD95':>10} {'Sensitivity':>12}"
    print(header)
    print("-" * 80)

    for region in CLASS_NAMES:
        region_rows = [r for r in rows if r["region"] == region]
        dice_arr = np.array([r["dice"] for r in region_rows])
        iou_arr = np.array([r["iou"] for r in region_rows])
        hd95_arr = np.array([r["hd95"] for r in region_rows])
        hd95_finite = hd95_arr[np.isfinite(hd95_arr)]
        sens_arr = np.array([r["sensitivity"] for r in region_rows])
        print(
            f"{region:<8} {dice_arr.mean():>10.4f} {iou_arr.mean():>10.4f} "
            f"{hd95_finite.mean():>10.2f} {sens_arr.mean():>12.4f}"
        )

    # Overall
    all_dice = np.array([r["dice"] for r in rows])
    all_iou = np.array([r["iou"] for r in rows])
    all_hd95 = np.array([r["hd95"] for r in rows])
    all_hd95_f = all_hd95[np.isfinite(all_hd95)]
    all_sens = np.array([r["sensitivity"] for r in rows])
    print("-" * 80)
    print(
        f"{'Mean':<8} {all_dice.mean():>10.4f} {all_iou.mean():>10.4f} "
        f"{all_hd95_f.mean():>10.2f} {all_sens.mean():>12.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
