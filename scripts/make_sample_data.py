"""Generate small **synthetic** sample fixtures for offline tests and smoke runs.

These are NOT real Sentinel-2 data — they are deterministic, class-separable
stand-ins (13-band patches + one chip/label pair) that let the whole pipeline
run offline in CI and in the training ``--smoke`` path. Real EuroSAT patches and
STAC/WorldCover chips are fetched by ``scripts/fetch_data.py`` for the actual
runs whose numbers appear in the README. Regenerate with:

    uv run python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from landcover.classes import EUROSAT_TO_LANDCOVER, LandCover
from landcover.indices import B_GREEN, B_NIR, B_RED, B_SWIR1

SAMPLE = Path("data/sample")
RNG = np.random.default_rng(0)


def class_spectrum(land_class: LandCover, size: int) -> np.ndarray:
    """A (13, size, size) patch whose bands separate the five target classes."""
    stack = RNG.normal(0.08, 0.01, (13, size, size)).astype(np.float32)
    if land_class == LandCover.FOREST:
        stack[B_NIR] += 0.4
        stack[B_RED] = 0.03
    elif land_class == LandCover.WATER:
        stack[B_GREEN] += 0.15
        stack[B_NIR] = 0.01
    elif land_class == LandCover.URBAN:
        stack[B_SWIR1] += 0.3
        stack[B_NIR] += 0.12
    elif land_class == LandCover.AGRICULTURE:
        stack[B_NIR] += 0.22
        stack[B_RED] += 0.08
    else:  # BARREN
        stack += 0.28
    return np.clip(stack, 0, 1)


def make_eurosat_patches() -> None:
    """A few synthetic EuroSAT patches per original class, as .npy under class dirs."""
    for eurosat_class, land_class in EUROSAT_TO_LANDCOVER.items():
        class_dir = SAMPLE / "eurosat" / eurosat_class
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            patch = class_spectrum(land_class, size=16)
            np.save(class_dir / f"{eurosat_class}_{index}.npy", patch)


def make_chip_pair() -> None:
    """One 64x64 synthetic segmentation chip: four class quadrants + label map."""
    chips_dir = SAMPLE / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    size = 64
    half = size // 2
    layout = [
        (LandCover.FOREST, slice(0, half), slice(0, half)),
        (LandCover.WATER, slice(0, half), slice(half, size)),
        (LandCover.URBAN, slice(half, size), slice(0, half)),
        (LandCover.AGRICULTURE, slice(half, size), slice(half, size)),
    ]
    image = np.zeros((13, size, size), dtype=np.float32)
    label = np.zeros((size, size), dtype=np.int64)
    for land_class, rows, cols in layout:
        block = class_spectrum(land_class, size)
        image[:, rows, cols] = block[:, rows, cols]
        label[rows, cols] = int(land_class)
    np.save(chips_dir / "aoi_00_image.npy", image)
    np.save(chips_dir / "aoi_00_label.npy", label)
    # A second chip so a spatial split has ≥2 blocks.
    np.save(chips_dir / "aoi_01_image.npy", image[:, ::-1, :].copy())
    np.save(chips_dir / "aoi_01_label.npy", label[::-1, :].copy())


def main() -> None:
    make_eurosat_patches()
    make_chip_pair()
    total = sum(f.stat().st_size for f in SAMPLE.rglob("*.npy"))
    print(f"wrote synthetic sample fixtures under {SAMPLE} ({total / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
