"""
N4ITK Bias Field Correction using SimpleITK.

Functions
---------
n4_bias_field_correction(input_path, output_path, ...)
    Correct a single NIfTI volume and write to disk.

correct_subject(subject_dir, modalities, output_dir, ...)
    Correct all requested modalities for one BraTS subject.
"""

from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def n4_bias_field_correction(
    input_path: str | Path,
    output_path: str | Path,
    *,
    shrink_factor: int = 4,
    convergence_threshold: float = 0.001,
    max_iterations: list[int] | None = None,
    fitting_levels: int = 4,
    overwrite: bool = False,
) -> Path:
    """
    Apply N4ITK bias field correction to a single NIfTI volume.

    Args:
        input_path:            Path to the input .nii / .nii.gz file.
        output_path:           Destination path for the corrected volume.
        shrink_factor:         Down-sampling factor applied before estimation
                               (speeds up computation, typically 2–4).
        convergence_threshold: Convergence tolerance for the N4 optimiser.
        max_iterations:        Maximum iterations per fitting level.
                               Defaults to [50, 50, 50, 50].
        fitting_levels:        Number of B-spline fitting levels.
        overwrite:             If False and output_path already exists, skip.

    Returns:
        Path to the corrected output file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not overwrite and output_path.exists():
        logger.info("Skipping (exists): %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_iterations is None:
        max_iterations = [50] * fitting_levels

    logger.info("N4 correction: %s → %s", input_path.name, output_path.name)

    # Read
    image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)

    # Build brain mask (nonzero voxels)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    # Shrink for speed
    shrunken_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    shrunken_mask = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())

    # Estimate bias field on shrunken image
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(
        max_iterations * (fitting_levels // len(max_iterations))
        if len(max_iterations) != fitting_levels
        else max_iterations
    )
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrector.SetNumberOfControlPoints([fitting_levels] * image.GetDimension())

    corrector.Execute(shrunken_image, shrunken_mask)

    # Apply log-bias field to the full-resolution image
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image = image / sitk.Exp(log_bias_field)

    # Copy metadata
    corrected_image.CopyInformation(image)

    # Write
    sitk.WriteImage(corrected_image, str(output_path))
    logger.info("Saved: %s", output_path)

    return output_path


def correct_subject(
    subject_dir: str | Path,
    modalities: list[str],
    output_dir: str | Path,
    *,
    suffix: str = "n4",
    overwrite: bool = False,
    **n4_kwargs,
) -> dict[str, Path]:
    """
    Apply N4ITK bias field correction to all specified modalities of one BraTS
    subject.

    Args:
        subject_dir:  Path to the subject folder (e.g. .../BraTS2021_00000/).
        modalities:   List of modality names to correct (e.g. ["t1", "t1ce"]).
        output_dir:   Directory where corrected files will be saved.
        suffix:       Suffix appended before the file extension
                      (e.g. "n4" → BraTS2021_00000_t1ce_n4.nii.gz).
        overwrite:    Passed to n4_bias_field_correction.
        **n4_kwargs:  Extra keyword arguments forwarded to
                      n4_bias_field_correction.

    Returns:
        Dict mapping modality name → corrected output path.
    """
    subject_dir = Path(subject_dir)
    output_dir = Path(output_dir)
    subject_id = subject_dir.name

    results: dict[str, Path] = {}

    for mod in modalities:
        input_path = subject_dir / f"{subject_id}_{mod}.nii.gz"
        if not input_path.exists():
            logger.warning("Missing modality file: %s", input_path)
            continue

        output_path = output_dir / f"{subject_id}_{mod}_{suffix}.nii.gz"
        results[mod] = n4_bias_field_correction(
            input_path,
            output_path,
            overwrite=overwrite,
            **n4_kwargs,
        )

    return results
