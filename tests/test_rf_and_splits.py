"""RF classifier and spatial-split tests on synthetic but separable data."""

import numpy as np
import pytest

from landcover.classes import LandCover
from landcover.metrics import compute_metrics
from landcover.rf import PatchRandomForest, PixelRandomForest
from landcover.splits import ChipRef, assert_block_disjoint, spatial_block_split


def synthetic_patch(class_id: int, rng: np.random.Generator, size: int = 8) -> np.ndarray:
    """A 13-band patch whose spectra separate the five classes."""
    stack = rng.normal(0.05, 0.01, (13, size, size))
    if class_id == LandCover.FOREST:
        stack[7] += 0.4  # high NIR
        stack[3] += 0.02  # low red → high NDVI
    elif class_id == LandCover.WATER:
        stack[2] += 0.2  # green
        stack[7] = 0.01  # very low NIR
    elif class_id == LandCover.URBAN:
        stack[11] += 0.3  # SWIR1 → NDBI
        stack[7] += 0.1
    elif class_id == LandCover.AGRICULTURE:
        stack[7] += 0.25
        stack[3] += 0.1
    else:  # BARREN
        stack[:] += 0.3  # bright, flat spectrum
    return np.clip(stack, 0, 1)


class TestPatchRandomForest:
    def test_learns_separable_classes(self) -> None:
        rng = np.random.default_rng(0)
        classes = list(LandCover)
        train_patches, train_labels = [], []
        for class_id in classes:
            for _ in range(20):
                train_patches.append(synthetic_patch(class_id, rng))
                train_labels.append(int(class_id))

        model = PatchRandomForest(n_estimators=100).fit(
            train_patches, np.array(train_labels, dtype=np.int64)
        )

        test_patches, test_labels = [], []
        for class_id in classes:
            for _ in range(10):
                test_patches.append(synthetic_patch(class_id, rng))
                test_labels.append(int(class_id))
        predictions = model.predict(test_patches)
        metrics = compute_metrics(np.array(test_labels), predictions)
        assert metrics.overall_accuracy > 0.9

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            PatchRandomForest().predict([])

    def test_feature_importances_sum_to_one(self) -> None:
        rng = np.random.default_rng(1)
        patches = [synthetic_patch(i % 5, rng) for i in range(30)]
        labels = np.array([i % 5 for i in range(30)], dtype=np.int64)
        model = PatchRandomForest(n_estimators=50).fit(patches, labels)
        assert model.feature_importances().sum() == pytest.approx(1.0)


class TestPixelRandomForest:
    def test_segments_a_two_region_chip(self) -> None:
        rng = np.random.default_rng(2)
        # Build a chip: left half forest, right half water.
        forest = synthetic_patch(LandCover.FOREST, rng, size=16)
        water = synthetic_patch(LandCover.WATER, rng, size=16)
        chip = np.concatenate([forest[:, :, :8], water[:, :, :8]], axis=2)
        label = np.concatenate(
            [np.full((16, 8), int(LandCover.FOREST)), np.full((16, 8), int(LandCover.WATER))],
            axis=1,
        ).astype(np.int64)

        model = PixelRandomForest(n_estimators=50).fit([chip], [label])
        prediction = model.predict(chip)
        assert prediction.shape == (16, 16)
        assert compute_metrics(label, prediction).overall_accuracy > 0.9


class TestSpatialSplit:
    def _grid_chips(self, n: int = 6) -> list[ChipRef]:
        # n*n chips of 512 px laid on a grid; block_size 1024 → (n/2)^2 blocks.
        return [
            ChipRef(name=f"chip_{c}_{r}", col=c * 512, row=r * 512)
            for c in range(n)
            for r in range(n)
        ]

    def test_folds_are_block_disjoint(self) -> None:
        chips = self._grid_chips()
        split = spatial_block_split(chips, block_size=1024, val_fraction=0.2, test_fraction=0.2)
        assert_block_disjoint(split, chips, block_size=1024)  # must not raise
        total = len(split.train) + len(split.val) + len(split.test)
        assert total == len(chips)
        assert split.train
        assert split.val
        assert split.test

    def test_determinism(self) -> None:
        chips = self._grid_chips()
        a = spatial_block_split(chips, block_size=1024, seed=3)
        b = spatial_block_split(chips, block_size=1024, seed=3)
        assert a == b

    def test_leakage_guard_catches_shared_block(self) -> None:
        chips = [ChipRef("a", 0, 0), ChipRef("b", 100, 100)]  # same 1024 block
        from landcover.splits import SpatialSplit

        leaky = SpatialSplit(train=("a",), val=("b",), test=())
        with pytest.raises(ValueError, match="leakage"):
            assert_block_disjoint(leaky, chips, block_size=1024)

    def test_too_few_blocks_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            spatial_block_split([ChipRef("a", 0, 0)], block_size=1024)
