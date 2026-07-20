"""Render the README figures for the segmentation track from the real chips.

Two artifacts, committed to ``docs/img``:

- ``segmentation_map.png`` — a held-out chip shown three ways: true-colour RGB,
  the per-pixel Random-Forest prediction, and the ESA-WorldCover ground truth.
- ``confusion_matrix.png`` — the row-normalized confusion matrix of the RF over
  the held-out pixels, which is where the "which classes get confused" story
  lives.

Run: uv run python scripts/make_figures.py --root data/raw/chips
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from landcover.classes import CLASS_NAMES, LandCover
from landcover.datasets import load_chips, stack_to_rgb
from landcover.metrics import confusion_matrix
from landcover.rf import PixelRandomForest

IMG = Path("docs/img")
# One colour per target class (urban, water, forest, agriculture, barren).
CLASS_COLORS = ["#b0490f", "#2a6fdb", "#1b7a3d", "#e0c341", "#8a7355"]
CMAP = ListedColormap(CLASS_COLORS)


def split_chips(chips: list, *, cut_frac: float = 0.5) -> tuple[list, list]:
    cut = max(1, int(len(chips) * cut_frac))
    return chips[:cut], chips[cut:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chips")
    args = parser.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    chips = load_chips(Path(args.root))
    if len(chips) < 2:
        raise SystemExit(f"need >=2 chips under {args.root} — run fetch_data.py --aoi ...")
    train, test = split_chips(chips)

    model = PixelRandomForest().fit([c.bands for c in train], [c.labels for c in train])

    # --- Figure 1: RGB | RF prediction | WorldCover truth on one held-out chip ---
    chip = test[0]
    prediction = model.predict(chip.bands)
    rgb = stack_to_rgb(chip.bands).astype(np.uint8)
    n_classes = len(LandCover)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true colour", fontsize=10)
    axes[1].imshow(prediction, cmap=CMAP, vmin=0, vmax=n_classes - 1)
    axes[1].set_title("Random Forest prediction", fontsize=10)
    axes[2].imshow(chip.labels, cmap=CMAP, vmin=0, vmax=n_classes - 1)
    axes[2].set_title("ESA WorldCover (ground truth)", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i]) for i in range(n_classes)]
    fig.legend(
        handles,
        [CLASS_NAMES[i] for i in range(n_classes)],
        loc="lower center",
        ncol=n_classes,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(IMG / "segmentation_map.png", dpi=150, facecolor="white")
    plt.close(fig)

    # --- Figure 2: row-normalized confusion matrix over held-out pixels ---
    y_true = np.concatenate([c.labels.reshape(-1) for c in test])
    y_pred = np.concatenate([model.predict(c.bands).reshape(-1) for c in test])
    matrix = confusion_matrix(y_true, y_pred).astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normed = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    names = [CLASS_NAMES[i] for i in range(n_classes)]
    ax.set_xticks(range(n_classes), names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_classes), names, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j,
                i,
                f"{normed[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if normed[i, j] > 0.5 else "black",
                fontsize=8,
            )
    ax.set_title("Random Forest confusion matrix (row-normalized)", fontsize=10)
    fig.tight_layout()
    fig.savefig(IMG / "confusion_matrix.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {IMG / 'segmentation_map.png'} and {IMG / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
