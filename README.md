# landcover-sentinel2

[![CI](https://github.com/tarpous/landcover-sentinel2/actions/workflows/ci.yml/badge.svg)](https://github.com/tarpous/landcover-sentinel2/actions/workflows/ci.yml)

> Sentinel-2 land cover as a **study, not just a model**: the original Random-Forest classifier kept as a baseline and put up against a U-Net and a LoRA-fine-tuned geospatial foundation model on identical, spatially blocked splits — plus a DINOv3 linear probe on EuroSAT and label-efficiency curves.

🚧 **Modernization in progress (July 2026).** The classical core is built and CI-tested (spectral indices, the 5-class scheme with a unit-tested ESA-WorldCover remap, per-pixel and per-patch Random Forests, shared metrics, spatially blocked splits). The deep and foundation-model tracks (ResNet-18, DINOv3 probe, U-Net, TerraMind via TerraTorch) land next as T4 notebooks whose metrics feed the results tables.

## Why this shape

The 2024 version was "an sklearn Random Forest on Sentinel-2." The upgrade keeps that RF as an honest baseline and asks the question that actually separates candidates: **when does the classical method win, and when does the extra model capacity earn its cost?** Two tracks answer it —

- **Track 1 — EuroSAT patch classification:** RF on spectral indices vs a fine-tuned ResNet-18 vs a DINOv3 satellite linear probe, against published EuroSAT numbers.
- **Track 2 — Sentinel-2 segmentation:** per-pixel RF vs a U-Net vs a LoRA-fine-tuned geospatial foundation model, on ESA-WorldCover-supervised chips over a Thessaloniki AOI, with per-class F1/IoU and a **label-efficiency curve** (performance at 10 / 25 / 100 % of training labels — the argument foundation-model papers make).

## Status

The classical, CPU-only foundation is complete and tested (34 tests, CI green). Data fetching (Earth Search STAC composites + WorldCover labels), the deep-learning notebooks, the analysis section, and the GeoTIFF-producing CLI are the remaining milestones. This README fills in with generated result tables as those land.

## Provenance

Originally built Fall 2024 as an independent project (Random Forest on Sentinel-2, validated against a held-out set). Published and extended in 2026 into a classical-vs-deep-vs-foundation-model study. No backdated history — the commit log is the build log.

## License

[MIT](LICENSE)
