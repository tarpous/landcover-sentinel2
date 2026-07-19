"""Download the real datasets the result tables are computed from.

Sources, pinned 2026-07-19:

- **EuroSAT MS** (Track 1): Zenodo record 7711810, ``EuroSAT_MS.zip`` (2.07 GB,
  27 000 13-band Sentinel-2 patches, 10 classes), MD5-verified. Extracts to one
  directory per EuroSAT class, which ``landcover.datasets.load_eurosat`` reads
  and remaps to the five target classes.
- **Sentinel-2 L2A composite + ESA WorldCover** (Track 2): a cloud-masked median
  composite over an AOI, fetched from the Earth Search STAC API, with WorldCover
  2021 labels remapped to the five classes. See ``--aoi``.

The committed ``data/sample`` fixtures are synthetic and only exercise wiring;
the numbers in the README come from the real data this script fetches.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import requests

RAW = Path("data/raw")
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1"
EUROSAT_MD5 = "091174add3c8e680a49244acf185b9f0"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    if dest.exists() and (expected_md5 is None or md5_of(dest) == expected_md5):
        print(f"{dest} already present")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        part = dest.with_suffix(dest.suffix + ".part")
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        part.replace(dest)
    if expected_md5 is not None and md5_of(dest) != expected_md5:
        dest.unlink()
        raise RuntimeError(f"{dest}: md5 mismatch")
    return dest


def fetch_eurosat() -> None:
    """Download and extract EuroSAT MS into ``data/raw/eurosat/<Class>/``."""
    zip_path = download(EUROSAT_URL, RAW / "EuroSAT_MS.zip", EUROSAT_MD5)
    target = RAW / "eurosat"
    if target.exists() and any(target.iterdir()):
        print(f"{target} already extracted")
        return
    print(f"extracting {zip_path} -> {target}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW / "_eurosat_tmp")
    # The zip nests the class dirs under EuroSAT_MS/ or similar; flatten to
    # data/raw/eurosat/<Class>/.
    class_dirs = [
        p for p in (RAW / "_eurosat_tmp").rglob("*") if p.is_dir() and any(p.glob("*.tif"))
    ]
    target.mkdir(parents=True, exist_ok=True)
    for class_dir in class_dirs:
        class_dir.rename(target / class_dir.name)
    print(f"extracted {len(list(target.iterdir()))} class directories")


def fetch_aoi(bbox: tuple[float, float, float, float], date_range: str, chip: int) -> None:
    """Fetch a Sentinel-2 L2A median composite + WorldCover labels as chips.

    Requires the STAC extras (pystac-client, stackstac). Chips are written as
    ``data/raw/chips/aoi_<c>_<r>_image.npy`` / ``_label.npy`` — the shape
    ``landcover.datasets.load_chips`` expects.
    """
    import numpy as np
    import pystac_client
    import stackstac
    from rasterio.enums import Resampling

    from landcover.classes import remap_worldcover

    chips_dir = RAW / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)

    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    s2 = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}},
    )
    bands = [
        "coastal",
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir08",
        "nir09",
        "swir16",
        "swir22",
    ]
    stack = stackstac.stack(
        list(s2.items()), assets=bands, epsg=4326, resolution=0.0001, bounds_latlon=bbox
    )
    composite = stack.median(dim="time").compute().to_numpy()  # (bands, H, W)

    wc = catalog.search(collections=["esa-worldcover"], bbox=list(bbox))
    wc_stack = stackstac.stack(
        list(wc.items())[:1],
        assets=["map"],
        epsg=4326,
        resolution=0.0001,
        bounds_latlon=bbox,
        resampling=Resampling.nearest,
    )
    labels_raw = wc_stack.isel(time=0, band=0).compute().to_numpy()
    remap = np.vectorize(remap_worldcover)
    labels = remap(labels_raw.astype(int))

    _, height, width = composite.shape
    n_rows, n_cols = height // chip, width // chip
    count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            img = composite[:, r * chip : (r + 1) * chip, c * chip : (c + 1) * chip]
            lab = labels[r * chip : (r + 1) * chip, c * chip : (c + 1) * chip]
            if img.shape[1:] != (chip, chip):
                continue
            np.save(chips_dir / f"aoi_{c}_{r}_image.npy", img.astype(np.float32))
            np.save(chips_dir / f"aoi_{c}_{r}_label.npy", lab.astype(np.int64))
            count += 1
    print(f"wrote {count} chips to {chips_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eurosat", action="store_true", help="EuroSAT MS (2 GB).")
    parser.add_argument(
        "--aoi",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="Sentinel-2 + WorldCover chips for a lon/lat bbox.",
    )
    parser.add_argument("--date-range", default="2021-06-01/2021-09-01")
    parser.add_argument("--chip", type=int, default=256)
    args = parser.parse_args()

    if not args.eurosat and not args.aoi:
        parser.error("pass --eurosat and/or --aoi W S E N")
    if args.eurosat:
        fetch_eurosat()
    if args.aoi:
        fetch_aoi(tuple(args.aoi), args.date_range, args.chip)


if __name__ == "__main__":
    main()
