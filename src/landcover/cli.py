"""``landcover`` — CPU-side command line for the classical baseline and outputs.

Training lives in the GPU scripts; this covers the always-available pieces:
train + evaluate the Random Forest on stored chips, and produce the classified
GeoTIFF + area statistics that Plan 05's geoagent later wraps as a tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Sentinel-2 land cover: RF baseline, evaluation, GeoTIFF output.", no_args_is_help=True
)


@app.command("rf-eval")
def rf_eval(
    chips: Annotated[
        Path, typer.Option(exists=True, help="Directory of *_image/_label .npy chips.")
    ],
    block_size: Annotated[int, typer.Option(help="Spatial-block size in pixels.")] = 2048,
    seed: Annotated[int, typer.Option(help="Split seed.")] = 0,
) -> None:
    """Train + evaluate the per-pixel Random Forest on a spatial-block split."""
    import numpy as np

    from landcover.datasets import load_chips
    from landcover.metrics import compute_metrics
    from landcover.rf import PixelRandomForest
    from landcover.splits import ChipRef, spatial_block_split

    samples = load_chips(chips)
    if len(samples) < 2:
        typer.echo(f"need ≥2 chips under {chips}", err=True)
        raise typer.Exit(code=1)

    refs = [ChipRef(s.name, col * 512, 0) for col, s in enumerate(samples)]
    if len({r.col for r in refs}) >= 3:
        split = spatial_block_split(refs, block_size=block_size, seed=seed)
        by_name = {s.name: s for s in samples}
        train = [by_name[n] for n in split.train]
        test = [by_name[n] for n in split.val + split.test]
    else:
        cut = max(1, len(samples) // 2)
        train, test = samples[:cut], samples[cut:]

    model = PixelRandomForest().fit([c.bands for c in train], [c.labels for c in train])
    truth = np.concatenate([c.labels.reshape(-1) for c in test])
    prediction = np.concatenate([model.predict(c.bands).reshape(-1) for c in test])
    metrics = compute_metrics(truth, prediction)
    typer.echo(
        f"RF: OA={metrics.overall_accuracy:.3f} macroF1={metrics.macro_f1:.3f} "
        f"mIoU={metrics.mean_iou:.3f}"
    )


@app.command()
def predict(
    raster: Annotated[Path, typer.Option(exists=True, help="Sentinel-2 13-band GeoTIFF.")],
    chips: Annotated[Path, typer.Option(exists=True, help="Labeled chips to train the RF on.")],
    out: Annotated[Path, typer.Option(help="Output classified GeoTIFF.")] = Path("landcover.tif"),
) -> None:
    """Classify a Sentinel-2 GeoTIFF into the 5 classes; write a GeoTIFF + area stats.

    This is the artifact Plan 05's geoagent wraps: input imagery → class map with
    a preserved CRS/transform, plus per-class area in hectares.
    """
    import numpy as np
    import rasterio

    from landcover.classes import CLASS_NAMES
    from landcover.datasets import load_chips
    from landcover.rf import PixelRandomForest

    samples = load_chips(chips)
    if not samples:
        typer.echo(f"no training chips under {chips}", err=True)
        raise typer.Exit(code=1)
    model = PixelRandomForest().fit([c.bands for c in samples], [c.labels for c in samples])

    with rasterio.open(raster) as source:
        stack = source.read().astype(np.float64)
        profile = source.profile
        pixel_area_ha = abs(source.transform.a * source.transform.e) / 10_000.0

    labels = model.predict(stack)
    profile.update(count=1, dtype="uint8", nodata=255)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dest:
        dest.write(labels.astype("uint8"), 1)

    values, counts = np.unique(labels, return_counts=True)
    areas = {
        CLASS_NAMES.get(int(v), str(v)): round(int(c) * pixel_area_ha, 2)
        for v, c in zip(values, counts, strict=True)
    }
    (out.with_suffix(".areas.json")).write_text(
        json.dumps(areas, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"wrote {out} and {out.with_suffix('.areas.json')}: {areas}")


if __name__ == "__main__":
    app()
