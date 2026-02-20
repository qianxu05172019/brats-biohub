"""
Volume resampling utilities using SimpleITK.

Functions
---------
resample_to_spacing(image, new_spacing, interpolator, default_value)
    Resample a SimpleITK image to the requested voxel spacing.

resample_subject(subject_dir, modalities, output_dir, ...)
    Resample all modalities (and optionally the label) for one BraTS subject.
"""

from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)


def resample_to_spacing(
    image: sitk.Image,
    new_spacing: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkBSpline,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample a SimpleITK image to *new_spacing*.

    The output size is computed automatically to preserve the physical extent
    of the volume.

    Args:
        image:         Input SimpleITK image (any dimension).
        new_spacing:   Target voxel spacing in mm, e.g. (1.0, 1.0, 1.0).
        interpolator:  SimpleITK interpolator constant.
                       Use sitk.sitkNearestNeighbor for label maps.
        default_value: Pad value for voxels outside the original FOV.

    Returns:
        Resampled SimpleITK image.
    """
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    new_spacing_arr = np.array(new_spacing, dtype=float)

    # Compute output size so the physical extent is unchanged
    new_size = np.round(
        original_size * original_spacing / new_spacing_arr
    ).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing_arr.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def resample_subject(
    subject_dir: str | Path,
    modalities: list[str],
    output_dir: str | Path,
    *,
    label_key: str | None = "seg",
    new_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    suffix: str = "resampled",
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Resample all modalities (and optionally the segmentation label) for a
    single BraTS subject to isotropic *new_spacing*.

    Modalities are resampled with B-spline interpolation; the label map uses
    nearest-neighbour interpolation to preserve integer class indices.

    Args:
        subject_dir:  Path to the subject folder (e.g. .../BraTS2021_00000/).
        modalities:   List of modality names, e.g. ["flair", "t1", "t1ce", "t2"].
        output_dir:   Directory where resampled files will be saved.
        label_key:    Key for the segmentation file ("seg").
                      Pass None to skip label resampling.
        new_spacing:  Target voxel spacing in mm.
        suffix:       Suffix added to the output filename before the extension.
        overwrite:    Skip existing files when False.

    Returns:
        Dict mapping key → output Path (includes label_key if processed).
    """
    subject_dir = Path(subject_dir)
    output_dir = Path(output_dir)
    subject_id = subject_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    keys_to_process = list(modalities)
    if label_key is not None:
        keys_to_process.append(label_key)

    for key in keys_to_process:
        input_path = subject_dir / f"{subject_id}_{key}.nii.gz"
        if not input_path.exists():
            logger.warning("Missing file: %s", input_path)
            continue

        output_path = output_dir / f"{subject_id}_{key}_{suffix}.nii.gz"

        if not overwrite and output_path.exists():
            logger.info("Skipping (exists): %s", output_path)
            results[key] = output_path
            continue

        logger.info(
            "Resampling %s → spacing %s", input_path.name, new_spacing
        )

        is_label = key == label_key
        interpolator = (
            sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
        )
        pixel_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32

        image = sitk.ReadImage(str(input_path), pixel_type)
        resampled = resample_to_spacing(
            image,
            new_spacing=new_spacing,
            interpolator=interpolator,
            default_value=0,
        )
        sitk.WriteImage(resampled, str(output_path))
        logger.info("Saved: %s", output_path)
        results[key] = output_path

    return results
