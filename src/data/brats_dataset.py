"""
BraTS 2021 dataset utilities.

Provides:
  get_brats_cases(cfg)    -> list of dicts with paths for each modality + seg
  get_data_loaders(cfg)   -> (train_loader, val_loader) using MONAI CacheDataset
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from monai.data import CacheDataset, DataLoader
from sklearn.model_selection import train_test_split

from .transforms import get_train_transforms, get_val_transforms


def get_brats_cases(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """
    Scan the BraTS 2021 root directory and return a list of case dicts.

    Each dict has keys: "flair", "t1", "t1ce", "t2", "seg"
    pointing to the corresponding NIfTI file paths.

    Args:
        cfg: The loaded YAML config (expects cfg["data"]["root_dir"] and
             cfg["data"]["modalities"]).

    Returns:
        Sorted list of case dicts.
    """
    root = Path(cfg["data"]["root_dir"])
    modalities: list[str] = cfg["data"]["modalities"]      # e.g. [flair, t1, t1ce, t2]
    label_key: str = cfg["data"].get("label_key", "seg")

    cases: list[dict[str, str]] = []

    subject_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("BraTS2021_")
    )

    for subject_dir in subject_dirs:
        case: dict[str, str] = {}
        valid = True

        # modalities
        for mod in modalities:
            nii_path = subject_dir / f"{subject_dir.name}_{mod}.nii.gz"
            if not nii_path.exists():
                valid = False
                break
            case[mod] = str(nii_path)

        # segmentation label
        seg_path = subject_dir / f"{subject_dir.name}_{label_key}.nii.gz"
        if not seg_path.exists():
            valid = False

        if valid:
            case[label_key] = str(seg_path)
            cases.append(case)

    if not cases:
        raise RuntimeError(f"No valid BraTS cases found under {root}")

    return cases


def get_data_loaders(
    cfg: dict[str, Any],
    cache_rate: float = 1.0,
    seed: int = 42,
    max_cases: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build MONAI CacheDataset-backed DataLoaders for train and validation splits.

    Args:
        cfg:        Full YAML config dict.
        cache_rate: Fraction of data to cache in RAM (0.0–1.0).
                    Set lower if RAM is limited.
        seed:       Random seed for the train/val split.
        max_cases:  If set, use only the first N cases (useful for dry-run /
                    debugging without waiting for the full dataset).

    Returns:
        (train_loader, val_loader)
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # ---- gather all cases ------------------------------------------------
    all_cases = get_brats_cases(cfg)
    if max_cases is not None:
        all_cases = all_cases[:max_cases]

    # ---- train / val split -----------------------------------------------
    val_ratio: float = data_cfg.get("val_ratio", 0.2)
    train_cases, val_cases = train_test_split(
        all_cases,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    # ---- transforms ------------------------------------------------------
    train_tf = get_train_transforms(cfg)
    val_tf = get_val_transforms(cfg)

    # ---- datasets --------------------------------------------------------
    num_workers: int = data_cfg.get("num_workers", 4)

    train_ds = CacheDataset(
        data=train_cases,
        transform=train_tf,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = CacheDataset(
        data=val_cases,
        transform=val_tf,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    # ---- loaders ---------------------------------------------------------
    batch_size: int = train_cfg.get("batch_size", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # validate one volume at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
