"""
MONAI transform pipelines for BraTS 2021.

Custom transforms
-----------------
ConvertToMultiChannelBasedOnBratsClassesd
    Converts a single-channel segmentation map (values 0/1/2/4) into three
    binary channels:
        CH 0 – TC  (Tumor Core)       = label 1 OR label 4
        CH 1 – WT  (Whole Tumor)      = label 1 OR label 2 OR label 4
        CH 2 – ET  (Enhancing Tumor)  = label 4

    Implemented with torch.where for clarity and GPU compatibility.
    Must be placed *after* Orientationd so the tensor is already in
    canonical orientation before the channel split.

Public helpers
--------------
get_train_transforms(cfg) -> Compose
get_val_transforms(cfg)   -> Compose
"""

from __future__ import annotations

from typing import Any, Hashable, Mapping

import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    SpatialPadd,
)


# ---------------------------------------------------------------------------
# Custom transform
# ---------------------------------------------------------------------------

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert a BraTS segmentation label map to three binary channel masks.

    Expects the label tensor to have shape (1, H, W, D) with integer values
    in {0, 1, 2, 4}.  Produces a float tensor of shape (3, H, W, D):
        channel 0 → TC  (label ∈ {1, 4})
        channel 1 → WT  (label ∈ {1, 2, 4})
        channel 2 → ET  (label == 4)

    All comparisons use torch.where for GPU compatibility.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: torch.Tensor) -> torch.Tensor:
        # img: (1, H, W, D) or (H, W, D)
        zero = torch.zeros_like(img, dtype=torch.float32)
        one = torch.ones_like(img, dtype=torch.float32)

        # TC = NCR/NET (1) + ET (4)
        tc = torch.where((img == 1) | (img == 4), one, zero)
        # WT = NCR/NET (1) + ED (2) + ET (4)
        wt = torch.where((img == 1) | (img == 2) | (img == 4), one, zero)
        # ET = Enhancing Tumor (4)
        et = torch.where(img == 4, one, zero)

        # stack along channel dim → (3, H, W, D)
        return torch.cat([tc, wt, et], dim=0)

    def __call__(
        self, data: Mapping[Hashable, Any]
    ) -> dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _image_keys(cfg: dict) -> list[str]:
    return list(cfg["data"]["modalities"])  # e.g. ["flair", "t1", "t1ce", "t2"]


def _label_key(cfg: dict) -> str:
    return cfg["data"].get("label_key", "seg")


def get_train_transforms(cfg: dict[str, Any]) -> Compose:
    """
    Full augmentation pipeline for training.

    Order:
      1. LoadImaged          – read NIfTI files
      2. EnsureChannelFirstd – add channel dim if missing
      3. EnsureTyped         – cast to float32 / int64
      4. Orientationd        – reorient to RAS
      5. ConvertToMultiChannelBasedOnBratsClassesd  ← after orientation
      6. Spacingd            – resample to 1 mm isotropic
      7. CropForegroundd     – remove air background
      8. SpatialPadd         – ensure minimum roi_size
      9. RandSpatialCropd    – random crop to roi_size
     10. RandFlipd           – random horizontal flips
     11. RandAffined         – random affine (rotate + scale)
     12. NormalizeIntensityd – per-channel z-score (nonzero mask)
     13. RandScaleIntensityd – multiplicative jitter
     14. RandShiftIntensityd – additive jitter
    """
    img_keys = _image_keys(cfg)
    lbl_key = _label_key(cfg)
    all_keys = img_keys + [lbl_key]

    t_cfg = cfg.get("transforms", {})
    roi_size: list[int] = t_cfg.get("roi_size", [128, 128, 128])
    flip_prob: float = t_cfg.get("rand_flip_prob", 0.5)
    rotate_range: float = t_cfg.get("rand_rotate_range", 0.2618)
    scale_range: list[float] = t_cfg.get("rand_scale_range", [0.9, 1.1])
    intensity_shift: float = t_cfg.get("rand_intensity_shift", 0.1)
    intensity_scale: float = t_cfg.get("rand_intensity_scale", 0.1)

    return Compose(
        [
            # ── I/O ──────────────────────────────────────────────────────
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            EnsureTyped(
                keys=img_keys, dtype=torch.float32
            ),
            EnsureTyped(keys=[lbl_key], dtype=torch.uint8),
            # ── Spatial orientation ───────────────────────────────────────
            Orientationd(keys=all_keys, axcodes="RAS"),
            # ── Label conversion (placed after Orientationd) ──────────────
            ConvertToMultiChannelBasedOnBratsClassesd(keys=[lbl_key]),
            # ── Resampling ────────────────────────────────────────────────
            Spacingd(
                keys=img_keys,
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            Spacingd(
                keys=[lbl_key],
                pixdim=(1.0, 1.0, 1.0),
                mode="nearest",
            ),
            # ── Cropping ──────────────────────────────────────────────────
            CropForegroundd(
                keys=all_keys,
                source_key=img_keys[0],   # use FLAIR as reference
                margin=10,
            ),
            SpatialPadd(keys=all_keys, spatial_size=roi_size),
            RandSpatialCropd(
                keys=all_keys,
                roi_size=roi_size,
                random_size=False,
            ),
            # ── Augmentation ──────────────────────────────────────────────
            RandFlipd(keys=all_keys, prob=flip_prob, spatial_axis=0),
            RandFlipd(keys=all_keys, prob=flip_prob, spatial_axis=1),
            RandFlipd(keys=all_keys, prob=flip_prob, spatial_axis=2),
            RandAffined(
                keys=all_keys,
                mode=["bilinear"] * len(img_keys) + ["nearest"],
                prob=0.5,
                rotate_range=(rotate_range,) * 3,
                scale_range=(
                    (scale_range[0] - 1.0, scale_range[1] - 1.0),
                ) * 3,
                padding_mode="border",
            ),
            # ── Intensity ─────────────────────────────────────────────────
            NormalizeIntensityd(
                keys=img_keys,
                nonzero=True,
                channel_wise=True,
            ),
            RandScaleIntensityd(
                keys=img_keys,
                factors=intensity_scale,
                prob=0.5,
            ),
            RandShiftIntensityd(
                keys=img_keys,
                offsets=intensity_shift,
                prob=0.5,
            ),
        ]
    )


def get_val_transforms(cfg: dict[str, Any]) -> Compose:
    """
    Deterministic pipeline for validation / inference.

    No random augmentation; mirrors the spatial/intensity pre-processing of
    get_train_transforms but uses the full volume (no random crop).
    """
    img_keys = _image_keys(cfg)
    lbl_key = _label_key(cfg)
    all_keys = img_keys + [lbl_key]

    return Compose(
        [
            # ── I/O ──────────────────────────────────────────────────────
            LoadImaged(keys=all_keys),
            EnsureChannelFirstd(keys=all_keys),
            EnsureTyped(keys=img_keys, dtype=torch.float32),
            EnsureTyped(keys=[lbl_key], dtype=torch.uint8),
            # ── Spatial orientation ───────────────────────────────────────
            Orientationd(keys=all_keys, axcodes="RAS"),
            # ── Label conversion (placed after Orientationd) ──────────────
            ConvertToMultiChannelBasedOnBratsClassesd(keys=[lbl_key]),
            # ── Resampling ────────────────────────────────────────────────
            Spacingd(
                keys=img_keys,
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            Spacingd(
                keys=[lbl_key],
                pixdim=(1.0, 1.0, 1.0),
                mode="nearest",
            ),
            # ── Cropping (deterministic) ───────────────────────────────────
            CropForegroundd(
                keys=all_keys,
                source_key=img_keys[0],
                margin=10,
            ),
            # ── Intensity ─────────────────────────────────────────────────
            NormalizeIntensityd(
                keys=img_keys,
                nonzero=True,
                channel_wise=True,
            ),
        ]
    )
