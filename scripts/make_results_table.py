"""Render the landcover README result tables from ``results/*.json``.

Track 1 numbers come from ``results/eurosat.json`` (written by
``train_eurosat.py``); Track 2 from ``results/segmentation.json`` (written by
``train_segmentation.py``). Whatever is present is rendered; missing tracks show
a "no run recorded yet" line. Numbers are never hand-typed — this is the only
path from measured metrics to the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

RESULTS = Path("results")


def cell(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


def track1() -> list[str]:
    path = RESULTS / "eurosat.json"
    lines = ["### Track 1 — EuroSAT patch classification", ""]
    if not path.exists():
        lines.append("_No run recorded yet — see `scripts/train_eurosat.py`._")
        return lines
    data = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"**Dataset:** {data.get('dataset', 'EuroSAT')} · five target classes")
    lines.append("")
    lines.append("| Model | Overall accuracy | Macro-F1 | Test patches |")
    lines.append("|---|---:|---:|---:|")
    for name, m in data["models"].items():
        lines.append(
            f"| {name} | {cell(m['overall_accuracy'])} | {cell(m['macro_f1'])} "
            f"| {m.get('n_samples', '—')} |"
        )
    return lines


def track2() -> list[str]:
    path = RESULTS / "segmentation.json"
    lines = ["### Track 2 — Sentinel-2 segmentation", ""]
    if not path.exists():
        lines.append("_No run recorded yet — see `scripts/train_segmentation.py`._")
        return lines
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data["models"]
    lines.append("| Model | Overall accuracy | mean IoU | Macro-F1 |")
    lines.append("|---|---:|---:|---:|")
    rf = models.get("random_forest")
    if rf:
        lines.append(
            f"| Random Forest (per-pixel) | {cell(rf['overall_accuracy'])} "
            f"| {cell(rf['mean_iou'])} | {cell(rf['macro_f1'])} |"
        )
    curve = models.get("unet_label_efficiency", {})
    if "100pct" in curve:
        m = curve["100pct"]
        lines.append(
            f"| U-Net (100% labels) | {cell(m['overall_accuracy'])} "
            f"| {cell(m['mean_iou'])} | {cell(m['macro_f1'])} |"
        )
    gfm = models.get("prithvi_lora")
    if gfm:
        lines.append(
            f"| Prithvi-EO-2.0 (LoRA) | {cell(gfm['overall_accuracy'])} "
            f"| {cell(gfm['mean_iou'])} | {cell(gfm['macro_f1'])} |"
        )
    if curve:
        lines.append("")
        lines.append("**Label-efficiency curve (U-Net mean IoU):**")
        lines.append("")
        lines.append("| Training labels | 10% | 25% | 100% |")
        lines.append("|---|---:|---:|---:|")
        row = " | ".join(cell(curve.get(f"{p}pct", {}).get("mean_iou")) for p in (10, 25, 100))
        lines.append(f"| mean IoU | {row} |")
    return lines


def render() -> str:
    return "\n".join([*track1(), "", *track2(), ""]) + "\n"


def inject(readme: Path = Path("README.md")) -> None:
    begin, end = "<!-- results:begin -->", "<!-- results:end -->"
    content = readme.read_text(encoding="utf-8")
    block = render()
    if begin in content and end in content:
        head, rest = content.split(begin, 1)
        _, tail = rest.split(end, 1)
        readme.write_text(f"{head}{begin}\n{block}{end}{tail}", encoding="utf-8")
    else:
        print(f"markers not found in {readme}; printing only")
    print(block)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    inject()


if __name__ == "__main__":
    main()
