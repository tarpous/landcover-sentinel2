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


WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)


def worldcover_tile(west: float, south: float) -> str:
    """The 3-degree WorldCover tile name covering a bbox's SW corner (e.g. N39E021)."""
    lat3 = int(south // 3 * 3)
    lon3 = int(west // 3 * 3)
    ns = f"N{lat3:02d}" if lat3 >= 0 else f"S{abs(lat3):02d}"
    ew = f"E{lon3:03d}" if lon3 >= 0 else f"W{abs(lon3):03d}"
    return f"{ns}{ew}"


def fetch_aoi(bbox: tuple[float, float, float, float], date_range: str, chip: int) -> None:
    """Fetch a Sentinel-2 L2A median composite + WorldCover labels as chips.

    Sentinel-2 comes from the Earth Search STAC (stackstac median composite);
    ESA WorldCover 2021 is read directly from its public S3 COG (it is not on
    Earth Search) and resampled onto the composite grid. Chips are written as
    ``data/raw/chips/aoi_<c>_<r>_image.npy`` / ``_label.npy``.
    """
    import numpy as np
    import pystac_client
    import rasterio
    import stackstac
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    from landcover.classes import remap_worldcover

    chips_dir = RAW / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox

    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    s2 = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}},
    )
    # Earth Search S2 L2A has 12 bands (no B10 cirrus — that is L1C only). Order
    # them to the EuroSAT/indices.py 13-band convention and splice a zero B10 at
    # index 10 so B11(SWIR1)=11 and B12(SWIR2)=12 line up with the feature code.
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
    items = list(s2.items())
    if not items:
        raise SystemExit("no Sentinel-2 items for that bbox/date range")
    print(f"compositing {len(items)} Sentinel-2 scenes ...")
    stack = stackstac.stack(items, assets=bands, epsg=4326, resolution=0.0001, bounds_latlon=bbox)
    twelve = stack.median(dim="time").compute().to_numpy()  # (12, H, W)
    _, height, width = twelve.shape
    composite = np.insert(twelve, 10, 0.0, axis=0)  # -> (13, H, W), zero B10

    tile = worldcover_tile(west, south)
    print(f"reading WorldCover tile {tile} ...")
    with rasterio.open(WORLDCOVER_URL.format(tile=tile)) as wc:
        window = from_bounds(west, south, east, north, wc.transform)
        labels_raw = wc.read(
            1, window=window, out_shape=(height, width), resampling=Resampling.nearest
        )
    labels = np.vectorize(remap_worldcover)(labels_raw.astype(int))

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
