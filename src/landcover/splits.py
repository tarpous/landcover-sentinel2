"""Spatially blocked train/val/test splitting for the segmentation track.

Adjacent Sentinel-2 chips are spatially autocorrelated; a random chip split
leaks neighbours across folds and inflates accuracy — the same mistake this
project's sibling repos guard against. Chips are assigned to a coarse spatial
grid and whole grid blocks go to one fold, so training and test areas are
geographically separated. The disjointness is asserted, and that assertion is
part of the test suite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChipRef:
    """A chip located by its top-left pixel offset in the source mosaic."""

    name: str
    col: int
    row: int


@dataclass(frozen=True, slots=True)
class SpatialSplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def block_of(chip: ChipRef, *, block_size: int) -> tuple[int, int]:
    """The spatial block a chip belongs to (integer grid over pixel offsets)."""
    return (chip.col // block_size, chip.row // block_size)


def spatial_block_split(
    chips: list[ChipRef],
    *,
    block_size: int = 2048,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> SpatialSplit:
    """Assign whole spatial blocks to train/val/test (block-disjoint folds)."""
    if not chips:
        raise ValueError("no chips to split")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val + test fractions must leave room for training")

    blocks: dict[tuple[int, int], list[str]] = {}
    for chip in chips:
        blocks.setdefault(block_of(chip, block_size=block_size), []).append(chip.name)
    if len(blocks) < 3:
        raise ValueError(f"need at least 3 spatial blocks, got {len(blocks)} (reduce block_size)")

    ordered = sorted(blocks)
    random.Random(seed).shuffle(ordered)
    n_total = len(chips)

    val: list[str] = []
    test: list[str] = []
    remaining = list(ordered)
    for target, bucket in ((test_fraction, test), (val_fraction, val)):
        while remaining and len(bucket) < target * n_total:
            bucket.extend(blocks[remaining.pop()])
    train = [name for block in remaining for name in blocks[block]]

    split = SpatialSplit(train=tuple(train), val=tuple(val), test=tuple(test))
    assert_block_disjoint(split, chips, block_size=block_size)
    return split


def assert_block_disjoint(split: SpatialSplit, chips: list[ChipRef], *, block_size: int) -> None:
    """Raise if any spatial block is shared across folds — the leakage guard."""
    by_name = {chip.name: chip for chip in chips}
    fold_blocks = {}
    for fold_name, names in (("train", split.train), ("val", split.val), ("test", split.test)):
        fold_blocks[fold_name] = {block_of(by_name[name], block_size=block_size) for name in names}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        shared = fold_blocks[a] & fold_blocks[b]
        if shared:
            raise ValueError(f"spatial-block leakage between {a} and {b}: {sorted(shared)}")
