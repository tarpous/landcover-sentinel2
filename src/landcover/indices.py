"""Sentinel-2 spectral indices and per-patch feature extraction.

These are the hand-built features the Random Forest classifier stands on — the
2024 "classical" method, kept as the baseline the deep models must beat. Band
order follows Sentinel-2 / EuroSAT convention (13 bands, 0-indexed):

    0:B01 1:B02(blue) 2:B03(green) 3:B04(red) 4:B05 5:B06 6:B07
    7:B08(NIR) 8:B08A 9:B09 10:B10 11:B11(SWIR1) 12:B12(SWIR2)

Indices are the standard separators: NDVI (vegetation), NDWI (water), NDBI
(built-up), plus brightness. All are computed with a safe divide so a zero
denominator yields 0 rather than a NaN that would poison the forest.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Named band indices (Sentinel-2, 13-band stack).
B_BLUE, B_GREEN, B_RED = 1, 2, 3
B_NIR, B_SWIR1, B_SWIR2 = 7, 11, 12

#: The features the RF consumes, in fixed order (used for column names too).
FEATURE_NAMES: tuple[str, ...] = (
    "mean_blue",
    "mean_green",
    "mean_red",
    "mean_nir",
    "mean_swir1",
    "mean_swir2",
    "ndvi",
    "ndwi",
    "ndbi",
    "brightness",
    "ndvi_std",
)


def safe_normalized_difference(a: FloatArray, b: FloatArray) -> FloatArray:
    """(a - b) / (a + b) with a zero denominator mapped to 0."""
    numerator = a - b
    denominator = a + b
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def ndvi(stack: FloatArray) -> FloatArray:
    """Normalized Difference Vegetation Index (NIR, Red), per pixel."""
    return safe_normalized_difference(stack[B_NIR], stack[B_RED])


def ndwi(stack: FloatArray) -> FloatArray:
    """Normalized Difference Water Index (Green, NIR), per pixel."""
    return safe_normalized_difference(stack[B_GREEN], stack[B_NIR])


def ndbi(stack: FloatArray) -> FloatArray:
    """Normalized Difference Built-up Index (SWIR1, NIR), per pixel."""
    return safe_normalized_difference(stack[B_SWIR1], stack[B_NIR])


def patch_features(stack: FloatArray) -> FloatArray:
    """Reduce one (bands, H, W) patch to the fixed feature vector.

    Reflectance means give the RF spectral level; index means give it class
    separability; ``ndvi_std`` adds within-patch texture (a smooth field vs a
    mixed urban patch differ in NDVI variance even at equal means).
    """
    if stack.ndim != 3:
        raise ValueError(f"expected (bands, H, W), got shape {stack.shape}")
    stack = stack.astype(np.float64)
    vegetation = ndvi(stack)
    water = ndwi(stack)
    built = ndbi(stack)
    brightness = np.sqrt(np.mean(stack[[B_RED, B_GREEN, B_BLUE]] ** 2, axis=0))
    return np.array(
        [
            float(stack[B_BLUE].mean()),
            float(stack[B_GREEN].mean()),
            float(stack[B_RED].mean()),
            float(stack[B_NIR].mean()),
            float(stack[B_SWIR1].mean()),
            float(stack[B_SWIR2].mean()),
            float(vegetation.mean()),
            float(water.mean()),
            float(built.mean()),
            float(brightness.mean()),
            float(vegetation.std()),
        ],
        dtype=np.float64,
    )


def pixel_features(stack: FloatArray) -> FloatArray:
    """Per-pixel feature matrix (H*W, n_features) for the segmentation RF.

    Same spectral + index features as :func:`patch_features` but kept per
    pixel; texture (``ndvi_std``) is filled with 0 since it is a patch-level
    statistic — the column stays so the two feature spaces line up.
    """
    if stack.ndim != 3:
        raise ValueError(f"expected (bands, H, W), got shape {stack.shape}")
    stack = stack.astype(np.float64)
    _, height, width = stack.shape
    n_pixels = height * width
    columns = [
        stack[B_BLUE],
        stack[B_GREEN],
        stack[B_RED],
        stack[B_NIR],
        stack[B_SWIR1],
        stack[B_SWIR2],
        ndvi(stack),
        ndwi(stack),
        ndbi(stack),
        np.sqrt(np.mean(stack[[B_RED, B_GREEN, B_BLUE]] ** 2, axis=0)),
        np.zeros((height, width), dtype=np.float64),
    ]
    return np.stack([column.reshape(n_pixels) for column in columns], axis=1)
