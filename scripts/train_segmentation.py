"""Track 2 — semantic segmentation: per-pixel RF vs U-Net (label-efficiency curve).

Per-pixel Random Forest (CPU baseline) vs a U-Net (segmentation-models-pytorch,
ResNet-18 encoder) on the local GPU, on identical spatially blocked chips. The
U-Net is trained at 10 / 25 / 100 % of the training labels to trace the
label-efficiency curve — the argument foundation-model papers make (the
TerraMind LoRA variant runs from the Docker CUDA container and appends its own
rows). Results → ``results/segmentation.json`` → README table + curve.

Run (GPU):   uv run python scripts/train_segmentation.py --root data/raw/chips --epochs 40
Smoke (CPU): uv run python scripts/train_segmentation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 consoles

# Keep every model/weights cache inside the repo (gitignored), not the machine.
_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("HF_HOME", _CACHE + "/huggingface")
os.environ.setdefault("TORCH_HOME", _CACHE + "/torch")

from landcover.classes import LandCover
from landcover.datasets import ChipSample, load_chips
from landcover.metrics import Metrics, compute_metrics
from landcover.rf import PixelRandomForest
from landcover.splits import ChipRef, spatial_block_split

RESULTS = Path("results")
N_CLASSES = len(LandCover)


def split_chips(
    chips: list[ChipSample], *, block_size: int, seed: int
) -> tuple[list[ChipSample], list[ChipSample]]:
    """Spatial-block split; chip grid position parsed from the ``aoi_<c>_<r>`` name."""
    refs = []
    for index, chip in enumerate(chips):
        parts = chip.name.split("_")
        col = int(parts[-1]) if parts[-1].isdigit() else index
        refs.append(ChipRef(name=chip.name, col=col * 512, row=0))
    if len({r.col for r in refs}) < 3:  # too few blocks for a 3-way split
        cut = max(1, len(chips) // 2)
        return chips[:cut], chips[cut:]
    split = spatial_block_split(refs, block_size=block_size, seed=seed)
    by_name = {chip.name: chip for chip in chips}
    train = [by_name[n] for n in split.train]
    test = [by_name[n] for n in (split.val + split.test)]
    return train, test


def rf_metrics(train: list[ChipSample], test: list[ChipSample]) -> Metrics:
    model = PixelRandomForest(n_estimators=200).fit(
        [c.bands for c in train], [c.labels for c in train]
    )
    truth = np.concatenate([c.labels.reshape(-1) for c in test])
    predictions = np.concatenate([model.predict(c.bands).reshape(-1) for c in test])
    return compute_metrics(truth, predictions)


def unet_metrics(
    train: list[ChipSample], test: list[ChipSample], *, epochs: int, device: str, fraction: float
) -> Metrics:
    """Train a U-Net on ``fraction`` of the training chips; return test metrics."""
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn

    n_keep = max(1, round(len(train) * fraction))
    subset = train[:n_keep]

    def to_tensors(chips: list[ChipSample]) -> tuple[torch.Tensor, torch.Tensor]:
        images = np.stack([c.bands for c in chips]).astype(np.float32)
        labels = np.stack([c.labels for c in chips]).astype(np.int64)
        return torch.from_numpy(images), torch.from_numpy(labels)

    x_train, y_train = to_tensors(subset)
    x_test, y_test = to_tensors(test)

    model = smp.Unet(
        encoder_name="resnet18", encoder_weights=None, in_channels=13, classes=N_CLASSES
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train.to(device)), y_train.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x_test.to(device)).argmax(1).cpu().numpy().astype(np.int64)
    return compute_metrics(y_test.numpy(), predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chips")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = Path("data/sample/chips") if args.smoke else Path(args.root)
    epochs = 1 if args.smoke else args.epochs
    device = "cpu" if args.smoke else args.device
    fractions = [1.0] if args.smoke else [0.1, 0.25, 1.0]

    chips = load_chips(root)
    if len(chips) < 2:
        raise SystemExit(f"need ≥2 chips under {root} — run fetch_data.py --chips")
    train, test = split_chips(chips, block_size=args.block_size, seed=args.seed)

    results: dict[str, object] = {"random_forest": rf_metrics(train, test).to_dict()}
    curve = {}
    for fraction in fractions:
        metrics = unet_metrics(train, test, epochs=epochs, device=device, fraction=fraction)
        curve[f"{int(fraction * 100)}pct"] = metrics.to_dict()
        pct = int(fraction * 100)
        print(f"U-Net @{pct}%: mIoU={metrics.mean_iou:.3f} OA={metrics.overall_accuracy:.3f}")
    results["unet_label_efficiency"] = curve

    if args.smoke:
        print("smoke OK: load → split → RF + U-Net → score wiring intact")
        return
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "segmentation.json").write_text(
        json.dumps({"track": "segmentation", "models": results}, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote results/segmentation.json")


if __name__ == "__main__":
    main()
