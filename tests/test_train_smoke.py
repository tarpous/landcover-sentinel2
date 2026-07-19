"""Training-wiring smoke tests (marked `smoke`).

Drive the EuroSAT and segmentation training scripts' ``--smoke`` path end-to-end
on CPU over the synthetic sample fixtures: load → RF + deep model (1 epoch) →
score. Proves the GPU scripts stay wired to the tested package without a GPU.
Skipped when the `train` group (torch/timm/smp) is absent, so default CPU CI
stays lean; a separate CI job installs `--group train` and runs `pytest -m smoke`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip("torch", reason="train group not installed")
pytest.importorskip("segmentation_models_pytorch", reason="train group not installed")


def run_script(*script_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *script_args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


def test_train_eurosat_smoke_wiring() -> None:
    result = run_script("scripts/train_eurosat.py", "--smoke")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "smoke OK" in result.stdout


def test_train_segmentation_smoke_wiring() -> None:
    result = run_script("scripts/train_segmentation.py", "--smoke")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "smoke OK" in result.stdout
