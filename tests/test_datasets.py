"""Dataset-loader tests on the committed synthetic sample fixtures."""

from pathlib import Path

import numpy as np

from landcover.classes import LandCover
from landcover.datasets import (
    class_distribution,
    label_counts_ok,
    load_chips,
    load_eurosat,
    stack_to_rgb,
)

SAMPLE = Path("data/sample")


class TestEuroSat:
    def test_loads_patches_with_remapped_labels(self) -> None:
        samples = load_eurosat(SAMPLE / "eurosat")
        assert len(samples) == 30  # 10 EuroSAT classes * 3 patches
        labels = {sample.label for sample in samples}
        assert labels <= {int(member) for member in LandCover}
        assert all(sample.bands.shape[0] == 13 for sample in samples)

    def test_sealake_and_river_collapse_to_water(self) -> None:
        samples = load_eurosat(SAMPLE / "eurosat")
        water = [s for s in samples if s.label == int(LandCover.WATER)]
        names = {s.name.split("_")[0] for s in water}
        assert names == {"River", "SeaLake"}


class TestChips:
    def test_loads_image_label_pairs(self) -> None:
        chips = load_chips(SAMPLE / "chips")
        assert len(chips) == 2
        for chip in chips:
            assert chip.bands.shape == (13, 64, 64)
            assert chip.labels.shape == (64, 64)

    def test_labels_are_valid_classes(self) -> None:
        chips = load_chips(SAMPLE / "chips")
        assert label_counts_ok(chips)

    def test_chip_has_all_four_quadrant_classes(self) -> None:
        chip = load_chips(SAMPLE / "chips")[0]
        distribution = class_distribution(chip.labels)
        assert set(distribution) == {"forest", "water", "urban", "agriculture"}


class TestVisualization:
    def test_stack_to_rgb_shape_and_range(self) -> None:
        chip = load_chips(SAMPLE / "chips")[0]
        rgb = stack_to_rgb(chip.bands)
        assert rgb.shape == (64, 64, 3)
        assert rgb.min() >= 0
        assert rgb.max() <= 255
        assert not np.any(np.isnan(rgb))
