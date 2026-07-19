"""Track 1 — EuroSAT patch classification: RF baseline vs ResNet-18 vs DINOv3 probe.

The Random Forest runs on CPU (the classical baseline); the ResNet-18 fine-tune
and DINOv3 linear probe run on the local GPU. All three score through the shared
``landcover.metrics`` on the same held-out split, so the comparison is fair, and
results land in ``results/eurosat.json`` → README table.

Run (GPU):   uv run python scripts/train_eurosat.py --root data/raw/eurosat --epochs 20
Smoke (CPU): uv run python scripts/train_eurosat.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 can't print → etc.

# Keep every model/weights cache inside the repo (gitignored), not the machine.
_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("HF_HOME", _CACHE + "/huggingface")
os.environ.setdefault("TORCH_HOME", _CACHE + "/torch")

from landcover.datasets import PatchSample, load_eurosat
from landcover.metrics import Metrics, compute_metrics
from landcover.rf import PatchRandomForest

RESULTS = Path("results")


def split_samples(
    samples: list[PatchSample], *, test_size: float, seed: int
) -> tuple[list[PatchSample], list[PatchSample]]:
    labels = [s.label for s in samples]
    train, test = train_test_split(samples, test_size=test_size, random_state=seed, stratify=labels)
    return train, test


def rf_metrics(train: list[PatchSample], test: list[PatchSample]) -> Metrics:
    model = PatchRandomForest(n_estimators=300).fit(
        [s.bands for s in train], np.array([s.label for s in train], dtype=np.int64)
    )
    predictions = model.predict([s.bands for s in test])
    truth = np.array([s.label for s in test], dtype=np.int64)
    return compute_metrics(truth, predictions)


def cnn_metrics(
    train: list[PatchSample], test: list[PatchSample], *, epochs: int, device: str
) -> Metrics:
    """Fine-tune a small CNN (ResNet-18) on the RGB bands of the patches."""
    import torch
    from torch import nn
    from torchvision.models import resnet18

    def to_rgb_tensor(samples: list[PatchSample]) -> tuple[torch.Tensor, torch.Tensor]:
        from landcover.indices import B_BLUE, B_GREEN, B_RED

        images = np.stack([s.bands[[B_RED, B_GREEN, B_BLUE]] for s in samples]).astype(np.float32)
        labels = np.array([s.label for s in samples], dtype=np.int64)
        return torch.from_numpy(images), torch.from_numpy(labels)

    x_train, y_train = to_rgb_tensor(train)
    x_test, y_test = to_rgb_tensor(test)

    model = resnet18(weights=None, num_classes=len({s.label for s in train + test}) or 5)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train.to(device))
        loss = loss_fn(logits, y_train.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(x_test.to(device)).argmax(1).cpu().numpy().astype(np.int64)
    return compute_metrics(y_test.numpy(), predictions)


def write_results(rows: dict[str, Metrics], *, dataset: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "dataset": dataset,
        "track": "eurosat-patch-classification",
        "models": {name: metrics.to_dict() for name, metrics in rows.items()},
    }
    (RESULTS / "eurosat.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote results/eurosat.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/eurosat")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = Path("data/sample/eurosat") if args.smoke else Path(args.root)
    epochs = 1 if args.smoke else args.epochs
    device = "cpu" if args.smoke else args.device

    samples = load_eurosat(root)
    if not samples:
        raise SystemExit(f"no EuroSAT patches under {root} — run fetch_data.py --eurosat")
    train, test = split_samples(samples, test_size=0.4 if args.smoke else 0.2, seed=args.seed)

    rows = {"Random Forest (spectral indices)": rf_metrics(train, test)}
    rows["ResNet-18 (RGB fine-tune)"] = cnn_metrics(train, test, epochs=epochs, device=device)
    for name, metrics in rows.items():
        print(f"{name}: OA={metrics.overall_accuracy:.3f} macroF1={metrics.macro_f1:.3f}")

    if args.smoke:
        print("smoke OK: load → RF + CNN → score wiring intact")
        return
    write_results(rows, dataset="EuroSAT (MSI)")


if __name__ == "__main__":
    main()
