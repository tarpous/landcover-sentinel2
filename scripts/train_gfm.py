"""Track 2 extension — LoRA-fine-tune a geospatial foundation model (Prithvi-EO-2.0).

Builds a Prithvi-EO-v2-300 semantic-segmentation model via TerraTorch's
``EncoderDecoderFactory`` (ViT encoder + UNet decoder), applies **LoRA** to the
frozen backbone through the factory's ``peft_config``, and fine-tunes on the
same Sentinel-2 / WorldCover chips as the RF and U-Net, scored through the same
``landcover.metrics``. Prithvi consumes the 6 HLS bands
(B02, B03, B04, B8A, B11, B12), which are sliced from our 13-band chips.

Heavy, Linux-first stack: install with ``uv pip install terratorch``. Adds a row
to ``results/segmentation.json``.

Run: uv run python scripts/train_gfm.py --root data/raw/chips --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("HF_HOME", _CACHE + "/huggingface")
os.environ.setdefault("TORCH_HOME", _CACHE + "/torch")

from landcover.datasets import ChipSample, load_chips
from landcover.metrics import Metrics, compute_metrics

RESULTS = Path("results")
NAME = "Prithvi-EO-2.0 (LoRA)"
# HLS bands Prithvi expects, as indices into our 13-band EuroSAT/S2 stack.
PRITHVI_BANDS = [1, 2, 3, 8, 11, 12]  # B02 B03 B04 B8A B11 B12
_S2_SCALE = 10000.0


def split_chips(chips: list[ChipSample]) -> tuple[list[ChipSample], list[ChipSample]]:
    cut = max(1, len(chips) // 2)
    return chips[:cut], chips[cut:]


def build_model(n_classes: int):
    from terratorch.models import EncoderDecoderFactory

    factory = EncoderDecoderFactory()
    return factory.build_model(
        task="segmentation",
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=True,
        backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        backbone_num_frames=1,
        necks=[
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage", "effective_time_dim": 1},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        head_dropout=0.1,
        num_classes=n_classes,
        peft_config={
            # replace_qkv splits Prithvi's fused qkv Linear into q/k/v linears;
            # LoRA then adapts the query and value projections (the standard pair).
            "method": "LORA",
            "replace_qkv": "qkv",
            "peft_config_kwargs": {
                "target_modules": ["q_linear", "v_linear"],
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
            },
        },
    )


def to_tensor(chips: list[ChipSample], mean, std, device: str):
    import torch

    x = np.stack([c.bands[PRITHVI_BANDS] for c in chips]).astype(np.float32)
    x = np.clip(x / _S2_SCALE, 0.0, 1.0)
    tensor = (torch.from_numpy(x) - mean) / std  # per-band standardization
    y = np.stack([c.labels for c in chips]).astype(np.int64)
    return tensor.to(device), torch.from_numpy(y).to(device)


def gfm_metrics(
    train: list[ChipSample], test: list[ChipSample], *, epochs: int, device: str, batch: int = 2
) -> Metrics:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Standardize each of the 6 bands over the training chips — Prithvi expects
    # standardized input, not raw [0, 1] reflectance, or its features are junk.
    raw = np.stack([c.bands[PRITHVI_BANDS] for c in train]).astype(np.float32) / _S2_SCALE
    mean = torch.tensor(raw.mean(axis=(0, 2, 3))).view(1, -1, 1, 1)
    std = torch.tensor(raw.std(axis=(0, 2, 3)) + 1e-6).view(1, -1, 1, 1)

    model = build_model(n_classes=5).to(device)
    x_train, y_train = to_tensor(train, mean, std, device)
    x_test, y_test = to_tensor(test, mean, std, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA + decoder trainable params: {n_train:,} / {n_total:,} "
        f"({100 * n_train / n_total:.1f}%)"
    )

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch):
            out = model(x_test[start : start + batch]).output
            predictions.append(out.argmax(1).cpu().numpy())
    return compute_metrics(y_test.cpu().numpy(), np.concatenate(predictions).astype(np.int64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/chips")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    chips = load_chips(Path(args.root))
    if len(chips) < 2:
        raise SystemExit(f"need >=2 chips under {args.root}")
    train, test = split_chips(chips)
    metrics = gfm_metrics(train, test, epochs=args.epochs, device=args.device)
    print(
        f"{NAME}: OA={metrics.overall_accuracy:.3f} mIoU={metrics.mean_iou:.3f} "
        f"macroF1={metrics.macro_f1:.3f}"
    )

    path = RESULTS / "segmentation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["prithvi_lora"] = metrics.to_dict()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote results/segmentation.json")


if __name__ == "__main__":
    main()
