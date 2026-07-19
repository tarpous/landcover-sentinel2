"""Metric tests against hand-computed confusion matrices."""

import math

import numpy as np

from landcover.metrics import compute_metrics, confusion_matrix


class TestConfusionMatrix:
    def test_perfect_predictions_are_diagonal(self) -> None:
        y = np.array([0, 1, 2, 3, 4])
        matrix = confusion_matrix(y, y)
        assert np.array_equal(matrix, np.eye(5, dtype=np.int64))

    def test_off_diagonal_counts(self) -> None:
        y_true = np.array([0, 0, 1])
        y_pred = np.array([0, 1, 1])
        matrix = confusion_matrix(y_true, y_pred)
        assert matrix[0, 0] == 1
        assert matrix[0, 1] == 1
        assert matrix[1, 1] == 1


class TestComputeMetrics:
    def test_perfect_prediction_scores_one(self) -> None:
        y = np.array([0, 1, 2, 3, 4, 0, 1])
        metrics = compute_metrics(y, y)
        assert metrics.overall_accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.mean_iou == 1.0

    def test_two_class_hand_computed(self) -> None:
        # 3 urban (0), 3 water (1). Predict urban perfectly; one water→urban.
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.overall_accuracy == 5 / 6
        # urban: TP=3 FP=1 FN=0 → F1 = 2*3/(6+1+0)=6/7; IoU=3/4
        assert metrics.per_class_f1["urban"] == 6 / 7
        assert metrics.per_class_iou["urban"] == 0.75
        # water: TP=2 FP=0 FN=1 → F1 = 4/5; IoU=2/3
        assert metrics.per_class_f1["water"] == 0.8
        assert metrics.per_class_iou["water"] == 2 / 3

    def test_absent_class_excluded_from_macro(self) -> None:
        # Only classes 0 and 1 present in truth; 2,3,4 absent → NaN, excluded.
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics.macro_f1 == 1.0  # averaged over present classes only
        assert math.isnan(metrics.per_class_f1["forest"])

    def test_serialization_rounds(self) -> None:
        y = np.array([0, 1, 2, 3, 4])
        payload = compute_metrics(y, y).to_dict()
        assert payload["overall_accuracy"] == 1.0
        assert isinstance(payload["confusion"], list)
        assert payload["n_samples"] == 5
