"""CLI tests on the synthetic sample chips (offline, no GPU)."""

import json
from pathlib import Path

import rasterio
from rasterio.transform import Affine
from typer.testing import CliRunner

from landcover.cli import app
from landcover.datasets import load_chips

SAMPLE = Path("data/sample")
runner = CliRunner()


class TestRfEval:
    def test_reports_metrics_on_sample_chips(self) -> None:
        result = runner.invoke(app, ["rf-eval", "--chips", str(SAMPLE / "chips")])
        assert result.exit_code == 0
        assert "OA=" in result.output
        assert "mIoU=" in result.output

    def test_missing_chips_errors(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        result = runner.invoke(app, ["rf-eval", "--chips", str(tmp_path / "empty")])
        assert result.exit_code == 1


class TestPredict:
    def test_writes_classified_geotiff_and_areas(self, tmp_path: Path) -> None:
        # Wrap a sample chip as a georeferenced 13-band GeoTIFF.
        chip = load_chips(SAMPLE / "chips")[0]
        raster_path = tmp_path / "scene.tif"
        transform = Affine.translation(500000, 4500000) * Affine.scale(10, -10)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=chip.bands.shape[1],
            width=chip.bands.shape[2],
            count=13,
            dtype="float32",
            crs="EPSG:32634",
            transform=transform,
        ) as dst:
            dst.write(chip.bands.astype("float32"))

        out = tmp_path / "classified.tif"
        result = runner.invoke(
            app,
            [
                "predict",
                "--raster",
                str(raster_path),
                "--chips",
                str(SAMPLE / "chips"),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

        with rasterio.open(out) as classified:
            assert classified.count == 1
            assert classified.crs.to_epsg() == 32634
            data = classified.read(1)
            assert data.min() >= 0
            assert data.max() < 5  # five valid classes

        areas = json.loads((out.with_suffix(".areas.json")).read_text(encoding="utf-8"))
        assert sum(areas.values()) > 0
        # each 10 m pixel = 0.01 ha; a 64x64 chip = 4096 px = 40.96 ha total
        assert abs(sum(areas.values()) - 40.96) < 0.5
