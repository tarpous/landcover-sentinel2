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


# ImageNet normalization for the pretrained ResNet-18 stem.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# Sentinel-2 L2A surface reflectance is scaled by 10000; divide to reach [0, 1].
_S2_SCALE = 10000.0
_N_CLASSES = 5


def _rgb_batch(samples: list[PatchSample]) -> tuple[object, object]:
    """(N, 3, H, W) reflectance→[0,1] RGB tensor + label tensor, ImageNet-normed."""
    import torch

    from landcover.indices import B_BLUE, B_GREEN, B_RED

    images = np.stack([s.bands[[B_RED, B_GREEN, B_BLUE]] for s in samples]).astype(np.float32)
    images = np.clip(images / _S2_SCALE, 0.0, 1.0)
    tensor = torch.from_numpy(images)
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    labels = torch.from_numpy(np.array([s.label for s in samples], dtype=np.int64))
    return tensor, labels


def cnn_metrics(
    train: list[PatchSample], test: list[PatchSample], *, epochs: int, device: str, batch: int = 128
) -> Metrics:
    """Fine-tune an ImageNet-pretrained ResNet-18 on the RGB bands, mini-batched."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import ResNet18_Weights, resnet18

    weights = None if len(train) < 50 else ResNet18_Weights.IMAGENET1K_V1  # smoke: from scratch
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, _N_CLASSES)
    model = model.to(device)

    x_train, y_train = _rgb_batch(train)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    x_test, y_test = _rgb_batch(test)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch):
            logits = model(x_test[start : start + batch].to(device))
            predictions.append(logits.argmax(1).cpu().numpy())
    return compute_metrics(y_test.numpy(), np.concatenate(predictions).astype(np.int64))


DINOV3_SAT = "facebook/dinov3-vitl16-pretrain-sat493m"


def dinov3_probe_metrics(
    train: list[PatchSample], test: list[PatchSample], *, device: str, batch: int = 64
) -> Metrics:
    """Linear probe on frozen DINOv3 satellite ViT features (RGB, 224 px).

    The backbone is never updated — only a logistic-regression head is fit on
    the pooled embeddings, which is the standard cheap way to read out a
    self-supervised representation. DINOv3-sat is EO-pretrained, so it should be
    a strong, near-training-free classifier.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(DINOV3_SAT).to(device).eval()

    def embed(samples: list[PatchSample]) -> np.ndarray:
        from landcover.indices import B_BLUE, B_GREEN, B_RED

        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        features = []
        with torch.no_grad():
            for start in range(0, len(samples), batch):
                chunk = samples[start : start + batch]
                rgb = np.stack([s.bands[[B_RED, B_GREEN, B_BLUE]] for s in chunk]).astype(
                    np.float32
                )
                x = torch.from_numpy(np.clip(rgb / _S2_SCALE, 0.0, 1.0)).to(device)
                x = torch.nn.functional.interpolate(
                    x, size=224, mode="bilinear", align_corners=False
                )
                x = (x - mean) / std
                out = backbone(pixel_values=x)
                pooled = getattr(out, "pooler_output", None)
                emb = pooled if pooled is not None else out.last_hidden_state[:, 0]
                features.append(emb.cpu().numpy())
        return np.concatenate(features)

    x_train, x_test = embed(train), embed(test)
    y_train = np.array([s.label for s in train], dtype=np.int64)
    y_test = np.array([s.label for s in test], dtype=np.int64)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_train, y_train)
    return compute_metrics(y_test, clf.predict(x_test).astype(np.int64))


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
    parser.add_argument(
        "--dinov3",
        action="store_true",
        help="Add a DINOv3-sat linear probe (needs `pip install transformers` + gated HF access).",
    )
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
    if args.dinov3:
        rows["DINOv3-sat linear probe"] = dinov3_probe_metrics(train, test, device=device)
    for name, metrics in rows.items():
        print(f"{name}: OA={metrics.overall_accuracy:.3f} macroF1={metrics.macro_f1:.3f}")

    if args.smoke:
        print("smoke OK: load → RF + CNN → score wiring intact")
        return
    write_results(rows, dataset="EuroSAT (MSI)")


if __name__ == "__main__":
    main()
