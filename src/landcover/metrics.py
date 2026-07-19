"""Classification/segmentation metrics shared by every model in the study.

One implementation, used for the RF, the U-Net and the foundation model alike,
so the comparison is fair by construction: overall accuracy, macro-F1,
per-class F1 and IoU, and the confusion matrix. Kept dependency-light (numpy +
the class scheme) and cross-checked against tiny hand-computed fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from landcover.classes import CLASS_NAMES

IntArray = npt.NDArray[np.int64]
N_CLASSES = len(CLASS_NAMES)


def confusion_matrix(y_true: IntArray, y_pred: IntArray, *, n_classes: int = N_CLASSES) -> IntArray:
    """Rows = true class, columns = predicted class."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    np.add.at(matrix, (flat_true, flat_pred), 1)
    return matrix


@dataclass(frozen=True, slots=True)
class Metrics:
    """A model's scores on one dataset — the row of every README table."""

    overall_accuracy: float
    macro_f1: float
    mean_iou: float
    per_class_f1: dict[str, float]
    per_class_iou: dict[str, float]
    confusion: list[list[int]] = field(default_factory=list)
    n_samples: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_accuracy": round(self.overall_accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "per_class_f1": {name: round(value, 4) for name, value in self.per_class_f1.items()},
            "per_class_iou": {name: round(value, 4) for name, value in self.per_class_iou.items()},
            "confusion": self.confusion,
            "n_samples": self.n_samples,
        }


def compute_metrics(y_true: IntArray, y_pred: IntArray, *, n_classes: int = N_CLASSES) -> Metrics:
    """All headline metrics from labels + predictions (class ids)."""
    matrix = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall_accuracy = correct / total if total else 0.0

    per_class_f1: dict[str, float] = {}
    per_class_iou: dict[str, float] = {}
    f1_values, iou_values = [], []
    for class_id in range(n_classes):
        name = CLASS_NAMES.get(class_id, str(class_id))
        true_positive = int(matrix[class_id, class_id])
        false_positive = int(matrix[:, class_id].sum()) - true_positive
        false_negative = int(matrix[class_id, :].sum()) - true_positive
        support = true_positive + false_negative

        if support == 0:  # class absent from the ground truth → excluded from macro means
            per_class_f1[name] = float("nan")
            per_class_iou[name] = float("nan")
            continue
        denominator_f1 = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator_f1 if denominator_f1 else 0.0
        union = true_positive + false_positive + false_negative
        iou = true_positive / union if union else 0.0
        per_class_f1[name] = f1
        per_class_iou[name] = iou
        f1_values.append(f1)
        iou_values.append(iou)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
    return Metrics(
        overall_accuracy=overall_accuracy,
        macro_f1=macro_f1,
        mean_iou=mean_iou,
        per_class_f1=per_class_f1,
        per_class_iou=per_class_iou,
        confusion=matrix.tolist(),
        n_samples=total,
    )
