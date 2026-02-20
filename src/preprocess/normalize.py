"""
Intensity normalisation utilities.

Functions
---------
zscore_normalize(volume, nonzero_only, clip_percentiles)
    Z-score normalise a single 3-D numpy array.

minmax_normalize(volume, nonzero_only, out_range)
    Min-max normalise a single 3-D numpy array.

normalize_volume(volume, method, **kwargs)
    Dispatcher that calls the appropriate normaliser.
"""

from __future__ import annotations

import numpy as np


def zscore_normalize(
    volume: np.ndarray,
    *,
    nonzero_only: bool = True,
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
) -> np.ndarray:
    """
    Z-score normalise a volumetric array.

    Args:
        volume:           3-D (or N-D) float array.
        nonzero_only:     Compute mean/std only over nonzero voxels (brain
                          tissue), then normalise the full volume.
        clip_percentiles: If given, clip values to (low%, high%) percentiles
                          *before* computing mean/std.  Pass None to skip.

    Returns:
        Normalised array of the same shape and dtype=float32.
    """
    volume = volume.astype(np.float32)

    if nonzero_only:
        mask = volume != 0
    else:
        mask = np.ones(volume.shape, dtype=bool)

    roi = volume[mask]

    if clip_percentiles is not None:
        lo, hi = np.percentile(roi, clip_percentiles)
        volume = np.clip(volume, lo, hi)
        roi = volume[mask]

    mean = roi.mean()
    std = roi.std()

    if std < 1e-8:
        # Constant region – return zeros
        return np.zeros_like(volume, dtype=np.float32)

    out = (volume - mean) / std

    if nonzero_only:
        out[~mask] = 0.0

    return out.astype(np.float32)


def minmax_normalize(
    volume: np.ndarray,
    *,
    nonzero_only: bool = True,
    out_range: tuple[float, float] = (0.0, 1.0),
    clip_percentiles: tuple[float, float] | None = (0.5, 99.5),
) -> np.ndarray:
    """
    Min-max normalise a volumetric array to *out_range*.

    Args:
        volume:           3-D (or N-D) float array.
        nonzero_only:     Compute min/max only from nonzero voxels.
        out_range:        Target (min, max) of the output.
        clip_percentiles: Clip to percentile range before computing min/max.

    Returns:
        Normalised array of the same shape and dtype=float32.
    """
    volume = volume.astype(np.float32)

    if nonzero_only:
        mask = volume != 0
    else:
        mask = np.ones(volume.shape, dtype=bool)

    roi = volume[mask]

    if clip_percentiles is not None:
        lo_pct, hi_pct = np.percentile(roi, clip_percentiles)
        volume = np.clip(volume, lo_pct, hi_pct)
        roi = volume[mask]

    v_min = roi.min()
    v_max = roi.max()

    if (v_max - v_min) < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    out_min, out_max = out_range
    out = (volume - v_min) / (v_max - v_min) * (out_max - out_min) + out_min

    if nonzero_only:
        out[~mask] = 0.0

    return out.astype(np.float32)


def normalize_volume(
    volume: np.ndarray,
    method: str = "z_score",
    **kwargs,
) -> np.ndarray:
    """
    Dispatch normalisation based on *method*.

    Args:
        volume: Input 3-D float array.
        method: "z_score" or "min_max".
        **kwargs: Forwarded to the underlying normaliser.

    Returns:
        Normalised float32 array.

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method in ("z_score", "zscore"):
        return zscore_normalize(volume, **kwargs)
    elif method in ("min_max", "minmax"):
        return minmax_normalize(volume, **kwargs)
    else:
        raise ValueError(
            f"Unknown normalisation method '{method}'. "
            "Choose 'z_score' or 'min_max'."
        )
