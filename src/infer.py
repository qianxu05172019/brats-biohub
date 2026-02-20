"""
src/infer.py — BraTS 2021 sliding-window inference.

Loads the best checkpoint, runs sliding-window inference on the validation
split, and saves predictions as NIfTI files in the original image space.

Usage
-----
    python -m src.infer --config configs/infer_sliding_window.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from tqdm import tqdm

from src.data import get_brats_cases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NUM_SEG_CHANNELS = 3  # TC / WT / ET


def _get_infer_transforms(modalities: list[str]) -> Compose:
    """Image-only transforms for inference — no crop, no label conversion."""
    return Compose([
        LoadImaged(keys=modalities),
        EnsureChannelFirstd(keys=modalities),
        EnsureTyped(keys=modalities, dtype=torch.float32),
        Orientationd(keys=modalities, axcodes="RAS"),
        Spacingd(keys=modalities, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys=modalities, nonzero=True, channel_wise=True),
    ])


def _reorient_ras_to_orig(data: np.ndarray, orig_affine: np.ndarray) -> np.ndarray:
    """Reorient a 3D array from RAS back to the original image orientation."""
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    orig_ornt = io_orientation(orig_affine)
    transform = ornt_transform(ras_ornt, orig_ornt)
    return nib.orientations.apply_orientation(data, transform)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2021 sliding-window inference")
    p.add_argument(
        "--config",
        default="configs/infer_sliding_window.yaml",
        help="Path to inference YAML config",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path from config",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]
    sw_cfg = infer_cfg["sliding_window"]
    output_cfg = cfg["output"]

    checkpoint_path = args.checkpoint or model_cfg["checkpoint"]
    save_dir = Path(output_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Model ────────────────────────────────────────────────────────────
    model = UNet(
        spatial_dims=model_cfg["spatial_dims"],
        in_channels=model_cfg["in_channels"],
        out_channels=NUM_SEG_CHANNELS,
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        num_res_units=model_cfg.get("num_res_units", 2),
        norm=model_cfg.get("norm", "instance"),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded checkpoint: %s (epoch %d)", checkpoint_path, ckpt.get("epoch", -1))

    # ── Data ─────────────────────────────────────────────────────────────
    data_cfg = {
        "data": {
            "root_dir": cfg["data"]["root_dir"],
            "modalities": cfg["data"]["modalities"],
            "label_key": "seg",
        },
    }
    modalities = cfg["data"]["modalities"]
    all_cases = get_brats_cases(data_cfg)
    val_ratio = cfg["data"].get("val_ratio", 0.2)
    seed = cfg["data"].get("seed", 42)
    _, val_cases = train_test_split(
        all_cases, test_size=val_ratio, random_state=seed, shuffle=True,
    )

    infer_transforms = _get_infer_transforms(modalities)

    roi_size = sw_cfg["roi_size"]
    sw_batch_size = sw_cfg.get("sw_batch_size", 4)
    overlap = sw_cfg.get("overlap", 0.5)

    log.info("Running inference on %d cases …", len(val_cases))

    # ── Inference loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for case in tqdm(val_cases, desc="Inference", unit="case"):
            # Load original image to get reference shape and affine
            ref_img = nib.load(case[modalities[0]])
            orig_shape = ref_img.shape  # (H, W, D) in original space
            orig_affine = ref_img.affine

            # Apply inference transforms (Orientationd -> RAS, Spacingd -> 1mm)
            infer_case = {mod: case[mod] for mod in modalities}
            sample = infer_transforms(infer_case)

            inputs = torch.cat(
                [sample[mod].unsqueeze(0).to(device) for mod in modalities], dim=1
            )

            with autocast("cuda"):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )

            # Post-process: sigmoid -> threshold -> convert to BraTS labels
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()[0]  # (3, H, W, D)

            tc, wt, et = preds[0], preds[1], preds[2]
            label_map = np.zeros_like(tc, dtype=np.uint8)
            label_map[wt == 1] = 2        # ED
            label_map[tc == 1] = 1        # NCR/NET
            label_map[et == 1] = 4        # ET

            # Reorient prediction from RAS back to original orientation
            label_map = _reorient_ras_to_orig(label_map, orig_affine)

            # Resample prediction back to original image dimensions if needed
            if label_map.shape != orig_shape:
                scale = np.array(orig_shape) / np.array(label_map.shape)
                label_map = zoom(label_map, scale, order=0)  # nearest-neighbor

            # Save as NIfTI in original space
            case_id = Path(case[modalities[0]]).parent.name
            out_path = save_dir / f"{case_id}_pred.nii.gz"
            nii = nib.Nifti1Image(label_map, affine=orig_affine)
            nib.save(nii, out_path)

    log.info("Predictions saved to %s  (%d files)", save_dir, len(val_cases))


if __name__ == "__main__":
    main()
