"""The five land-cover classes and the ESA WorldCover → 5-class remap.

The project keeps the original 2024 five-class scheme (urban / water / forest /
agriculture / barren) as the common target across both tracks. WorldCover 2021
(10 m, 11 classes) supervises the segmentation track, so its codes are remapped
here; the remap is the weak-supervision seam and is unit-tested, because a wrong
code mapping silently corrupts every downstream metric.
"""

from __future__ import annotations

from enum import IntEnum


class LandCover(IntEnum):
    """The five target classes (contiguous ids for model outputs)."""

    URBAN = 0
    WATER = 1
    FOREST = 2
    AGRICULTURE = 3
    BARREN = 4


CLASS_NAMES: dict[int, str] = {member.value: member.name.lower() for member in LandCover}

#: ESA WorldCover 2021 class codes → the five-class scheme. WorldCover codes:
#: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 70 snow, 80 water,
#: 90 herbaceous wetland, 95 mangrove, 100 moss/lichen.
WORLDCOVER_TO_LANDCOVER: dict[int, LandCover] = {
    10: LandCover.FOREST,
    20: LandCover.FOREST,  # shrubland → forest (woody vegetation)
    30: LandCover.AGRICULTURE,  # grassland → agriculture (managed/open vegetation)
    40: LandCover.AGRICULTURE,
    50: LandCover.URBAN,
    60: LandCover.BARREN,
    70: LandCover.BARREN,  # snow/ice → barren (non-vegetated; rare in the AOI)
    80: LandCover.WATER,
    90: LandCover.WATER,  # herbaceous wetland → water
    95: LandCover.FOREST,  # mangrove → forest
    100: LandCover.BARREN,
}

#: EuroSAT's 10 classes → the five-class scheme, for the patch-classification track.
EUROSAT_TO_LANDCOVER: dict[str, LandCover] = {
    "AnnualCrop": LandCover.AGRICULTURE,
    "PermanentCrop": LandCover.AGRICULTURE,
    "Pasture": LandCover.AGRICULTURE,
    "HerbaceousVegetation": LandCover.AGRICULTURE,
    "Forest": LandCover.FOREST,
    "River": LandCover.WATER,
    "SeaLake": LandCover.WATER,
    "Residential": LandCover.URBAN,
    "Industrial": LandCover.URBAN,
    "Highway": LandCover.BARREN,  # bare/paved linear features
}


def remap_worldcover(code: int) -> int:
    """Map one WorldCover code to a five-class id; unknown codes → BARREN."""
    return int(WORLDCOVER_TO_LANDCOVER.get(code, LandCover.BARREN))
