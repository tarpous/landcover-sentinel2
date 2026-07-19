# AGENTS.md

## Setup

- Python 3.12, uv-managed. Create the environment once: `uv venv .venv --python 3.12`, then activate it
  (`source .venv/Scripts/activate` on Windows, `source .venv/bin/activate` on Linux/macOS) and run `uv sync`.
- Every `python`/`pip` command runs inside the activated `.venv`. Add dependencies with `uv add <pkg>`
  (or `uv add --dev` for tooling) so `pyproject.toml` and `uv.lock` stay in sync.

## Commands

- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- Types: `uv run mypy` (strict, scoped to `src/`)
- Tests: `uv run pytest` (offline, runs from `data/sample/` + `tests/fixtures/`)
- CLI: `uv run landcover --help`
- CI (`.github/workflows/ci.yml`) runs the same gates on ubuntu-latest / Python 3.12.
- When chaining gates in a shell, use `set -o pipefail` — piping through `tail` otherwise masks failures.

## Layout & tracks

- **Track 1 — EuroSAT patch classification:** Random Forest on spectral indices (CPU) vs
  ResNet-18 fine-tune and a DINOv3 satellite linear probe (T4 notebooks).
- **Track 2 — Sentinel-2 segmentation:** per-pixel RF vs U-Net vs a LoRA-fine-tuned geospatial
  foundation model (TerraMind via TerraTorch), on ESA WorldCover-supervised chips.
- `src/landcover/` — indices, features, RF models, metrics, splits, CLI (all typed, CPU)
- `notebooks/` — parameterized T4 fine-tunes (ruff-excluded); metrics → `results/*.json`
- `data/sample/` — a few EuroSAT patches + one composite/label chip pair (≤5 MB, offline tests)
- `results/` — generated metrics; the only source of README numbers

## Conventions

- Conventional-commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`). No co-author trailers.
- GPU work never runs locally — deep/foundation models live in `notebooks/` for Colab/Kaggle T4.
- Splits are spatially blocked (Track 2) / label-remap unit-tested (WorldCover → 5 classes).
- README numbers come only from `results/*.json`. No personal details of the repo owner in committed content.
