"""Class-scheme and remap tests — the weak-supervision correctness guard."""

import pytest

from landcover.classes import (
    CLASS_NAMES,
    EUROSAT_TO_LANDCOVER,
    WORLDCOVER_TO_LANDCOVER,
    LandCover,
    remap_worldcover,
)


def test_five_contiguous_classes() -> None:
    assert [member.value for member in LandCover] == [0, 1, 2, 3, 4]
    assert CLASS_NAMES == {0: "urban", 1: "water", 2: "forest", 3: "agriculture", 4: "barren"}


class TestWorldCoverRemap:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (10, LandCover.FOREST),
            (50, LandCover.URBAN),
            (80, LandCover.WATER),
            (40, LandCover.AGRICULTURE),
            (60, LandCover.BARREN),
        ],
    )
    def test_known_codes(self, code: int, expected: LandCover) -> None:
        assert remap_worldcover(code) == expected

    def test_unknown_code_falls_back_to_barren(self) -> None:
        assert remap_worldcover(255) == LandCover.BARREN

    def test_every_mapped_code_is_a_valid_class(self) -> None:
        assert all(value in LandCover for value in WORLDCOVER_TO_LANDCOVER.values())


class TestEuroSatMap:
    def test_all_ten_eurosat_classes_mapped(self) -> None:
        assert len(EUROSAT_TO_LANDCOVER) == 10
        assert all(value in LandCover for value in EUROSAT_TO_LANDCOVER.values())

    def test_sealake_and_river_are_water(self) -> None:
        assert EUROSAT_TO_LANDCOVER["SeaLake"] == LandCover.WATER
        assert EUROSAT_TO_LANDCOVER["River"] == LandCover.WATER
