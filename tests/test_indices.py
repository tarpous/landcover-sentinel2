"""Spectral-index and feature-extraction tests with synthetic band stacks."""

import numpy as np
import pytest

from landcover.indices import (
    B_GREEN,
    B_NIR,
    B_RED,
    B_SWIR1,
    FEATURE_NAMES,
    ndbi,
    ndvi,
    ndwi,
    patch_features,
    pixel_features,
    safe_normalized_difference,
)


def make_stack(values: dict[int, float], size: int = 4) -> np.ndarray:
    """A (13, size, size) stack with given per-band constant values."""
    stack = np.zeros((13, size, size), dtype=np.float64)
    for band, value in values.items():
        stack[band] = value
    return stack


class TestSafeNormalizedDifference:
    def test_zero_denominator_is_zero_not_nan(self) -> None:
        a = np.zeros(3)
        b = np.zeros(3)
        result = safe_normalized_difference(a, b)
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))

    def test_known_value(self) -> None:
        result = safe_normalized_difference(np.array([3.0]), np.array([1.0]))
        assert result[0] == pytest.approx(0.5)  # (3-1)/(3+1)


class TestIndices:
    def test_vegetation_has_high_ndvi(self) -> None:
        stack = make_stack({B_NIR: 0.4, B_RED: 0.05})
        assert ndvi(stack).mean() > 0.5

    def test_water_has_positive_ndwi(self) -> None:
        stack = make_stack({B_GREEN: 0.2, B_NIR: 0.02})
        assert ndwi(stack).mean() > 0.5

    def test_builtup_has_positive_ndbi(self) -> None:
        stack = make_stack({B_SWIR1: 0.3, B_NIR: 0.1})
        assert ndbi(stack).mean() > 0.3


class TestPatchFeatures:
    def test_vector_length_matches_names(self) -> None:
        stack = make_stack({B_NIR: 0.4, B_RED: 0.1})
        features = patch_features(stack)
        assert features.shape == (len(FEATURE_NAMES),)
        assert not np.any(np.isnan(features))

    def test_uniform_patch_has_zero_ndvi_std(self) -> None:
        stack = make_stack({B_NIR: 0.4, B_RED: 0.1})
        features = patch_features(stack)
        assert features[FEATURE_NAMES.index("ndvi_std")] == pytest.approx(0.0)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="bands, H, W"):
            patch_features(np.zeros((13, 4)))


class TestPixelFeatures:
    def test_matrix_shape_is_pixels_by_features(self) -> None:
        stack = make_stack({B_NIR: 0.4, B_RED: 0.1}, size=8)
        matrix = pixel_features(stack)
        assert matrix.shape == (64, len(FEATURE_NAMES))

    def test_pixel_ndvi_column_matches_direct_index(self) -> None:
        rng = np.random.default_rng(0)
        stack = rng.random((13, 5, 5))
        matrix = pixel_features(stack)
        direct = ndvi(stack).reshape(25)
        assert np.allclose(matrix[:, FEATURE_NAMES.index("ndvi")], direct)
