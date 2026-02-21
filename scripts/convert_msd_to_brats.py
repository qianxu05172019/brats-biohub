"""
Convert MSD Task01_BrainTumour to BraTS2021 directory format.

MSD format:
  imagesTr/BRATS_XXX.nii.gz  (4D: FLAIR, T1w, T1gd, T2w)
  labelsTr/BRATS_XXX.nii.gz  (0=bg, 1=edema, 2=non-enhancing, 3=enhancing)

BraTS2021 format:
  BraTS2021_XXXXX/BraTS2021_XXXXX_flair.nii.gz
  BraTS2021_XXXXX/BraTS2021_XXXXX_t1.nii.gz
  BraTS2021_XXXXX/BraTS2021_XXXXX_t1ce.nii.gz
  BraTS2021_XXXXX/BraTS2021_XXXXX_t2.nii.gz
  BraTS2021_XXXXX/BraTS2021_XXXXX_seg.nii.gz  (0=bg, 1=NCR, 2=ED, 4=ET)
"""
import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

MSD_DIR = Path("/workspace/DataChallenge/data/Task01_BrainTumour")
OUT_DIR = Path("/workspace/DataChallenge/data/BraTS2021_Training_Data")
# MSD channel order
MOD_MAP = {0: "flair", 1: "t1", 2: "t1ce", 3: "t2"}
# MSD label -> BraTS label
LABEL_MAP = {0: 0, 1: 2, 2: 1, 3: 4}  # edema->2, non-enh->1(NCR), enh->4(ET)

N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 30  # default 30 cases


def convert():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MSD_DIR / "dataset.json") as f:
        meta = json.load(f)

    training = meta["training"][:N_CASES]
    print(f"Converting {len(training)} MSD cases to BraTS2021 format...")

    for i, entry in enumerate(tqdm(training)):
        img_path = MSD_DIR / entry["image"].lstrip("./")
        lbl_path = MSD_DIR / entry["label"].lstrip("./")

        # Generate BraTS2021 case ID
        case_id = f"BraTS2021_{i:05d}"
        case_dir = OUT_DIR / case_id
        case_dir.mkdir(exist_ok=True)

        # Load 4D image
        img_nii = nib.load(str(img_path))
        img_data = img_nii.get_fdata()  # (H, W, D, 4)
        affine = img_nii.affine
        header = img_nii.header

        # Split channels and save
        for ch_idx, mod_name in MOD_MAP.items():
            ch_data = img_data[:, :, :, ch_idx].astype(np.float32)
            out_nii = nib.Nifti1Image(ch_data, affine)
            nib.save(out_nii, str(case_dir / f"{case_id}_{mod_name}.nii.gz"))

        # Convert labels
        lbl_nii = nib.load(str(lbl_path))
        lbl_data = lbl_nii.get_fdata().astype(np.int16)
        brats_lbl = np.zeros_like(lbl_data, dtype=np.int16)
        for msd_val, brats_val in LABEL_MAP.items():
            brats_lbl[lbl_data == msd_val] = brats_val
        out_lbl = nib.Nifti1Image(brats_lbl, lbl_nii.affine)
        nib.save(out_lbl, str(case_dir / f"{case_id}_seg.nii.gz"))

    print(f"Done! Converted {len(training)} cases to {OUT_DIR}")

    # Generate synthetic survival info
    np.random.seed(42)
    rows = []
    for i in range(len(training)):
        case_id = f"BraTS2021_{i:05d}"
        age = np.random.normal(55, 12)
        age = max(20, min(85, age))
        resection = np.random.choice(["GTR", "STR"], p=[0.7, 0.3])
        # Survival correlated with age (younger = longer survival)
        base_surv = np.random.exponential(400)
        age_effect = (70 - age) * 5
        surv_days = max(30, base_surv + age_effect + np.random.normal(0, 50))
        rows.append(f"{case_id},{age:.1f},{surv_days:.0f},{resection}")

    csv_path = OUT_DIR / "survival_info.csv"
    with open(csv_path, "w") as f:
        f.write("BraTS21ID,Age,Survival_days,Extent_of_Resection\n")
        for row in rows:
            f.write(row + "\n")
    print(f"Survival info saved to {csv_path}")


if __name__ == "__main__":
    convert()
