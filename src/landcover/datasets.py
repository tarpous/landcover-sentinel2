"""Dataset loaders shared by the classical and deep tracks.

EuroSAT (Track 1) is a folder of per-class ``.tif`` patches; the segmentation
track (Track 2) is (chip, label) ``.npy`` pairs produced by the STAC/WorldCover
fetch script. Both loaders are dependency-light (numpy + the class scheme) and
run offline on the committed samples, so the deep-learning scripts and the RF
baseline consume exactly the same arrays — the comparison is fair by loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from landcover.classes import EUROSAT_TO_LANDCOVER, LandCover

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PatchSample:
    """One EuroSAT patch: (bands, H, W) reflectance and its 5-class label."""

    name: str
    bands: FloatArray
    label: int


def _read_patch(path: Path) -> FloatArray:
    """Read a EuroSAT ``.tif`` / ``.npy`` patch as (bands, H, W) float64."""
    if path.suffix == ".npy":
        array = np.load(path)
    else:
        import rasterio

        with rasterio.open(path) as raster:
            array = raster.read()
    return np.asarray(array, dtype=np.float64)


def load_eurosat(root: Path) -> list[PatchSample]:
    """Load EuroSAT patches from ``root/<EuroSATClass>/<file>`` into 5-class labels.

    EuroSAT's directory names are its 10 original classes; each maps to one of
    the five target classes via ``EUROSAT_TO_LANDCOVER``.
    """
    samples: list[PatchSample] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        eurosat_class = class_dir.name
        if eurosat_class not in EUROSAT_TO_LANDCOVER:
            continue
        label = int(EUROSAT_TO_LANDCOVER[eurosat_class])
        for patch_path in sorted(class_dir.glob("*.tif")) + sorted(class_dir.glob("*.npy")):
            samples.append(PatchSample(patch_path.name, _read_patch(patch_path), label))
    return samples


@dataclass(frozen=True, slots=True)
class ChipSample:
    """One segmentation chip: (bands, H, W) reflectance + (H, W) label map."""

    name: str
    bands: FloatArray
    labels: IntArray


def load_chips(root: Path) -> list[ChipSample]:
    """Load (``*_image.npy``, ``*_label.npy``) chip pairs from ``root``."""
    samples: list[ChipSample] = []
    for image_path in sorted(root.glob("*_image.npy")):
        label_path = image_path.with_name(image_path.name.replace("_image.npy", "_label.npy"))
        if not label_path.exists():
            continue
        samples.append(
            ChipSample(
                name=image_path.stem.replace("_image", ""),
                bands=np.asarray(np.load(image_path), dtype=np.float64),
                labels=np.asarray(np.load(label_path), dtype=np.int64),
            )
        )
    return samples


def class_distribution(labels: IntArray) -> dict[str, int]:
    """Pixel/patch count per class name — a sanity check for label remaps."""
    from landcover.classes import CLASS_NAMES

    values, counts = np.unique(labels, return_counts=True)
    return {CLASS_NAMES.get(int(v), str(v)): int(c) for v, c in zip(values, counts, strict=True)}


def stack_to_rgb(bands: FloatArray) -> FloatArray:
    """(bands,H,W) → (H,W,3) uint8-scaled RGB for visualization (B04,B03,B02)."""
    from landcover.indices import B_BLUE, B_GREEN, B_RED

    rgb = np.stack([bands[B_RED], bands[B_GREEN], bands[B_BLUE]], axis=-1)
    lo, hi = np.percentile(rgb, [2, 98])
    scaled = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    result: FloatArray = (scaled * 255).astype(np.float64)
    return result


def label_counts_ok(samples: list[ChipSample], *, n_classes: int = len(LandCover)) -> bool:
    """True if every label value across chips is a valid class id."""
    return all(
        int(chip.labels.min()) >= 0 and int(chip.labels.max()) < n_classes for chip in samples
    )
