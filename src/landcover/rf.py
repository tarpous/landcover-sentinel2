"""Random Forest classifiers — the classical baseline for both tracks.

Track 1 (patch): one feature vector per EuroSAT patch → RandomForest.
Track 2 (segmentation): per-pixel features over a chip → RandomForest → label
map. Both wrap scikit-learn behind a tiny typed surface so the notebooks and
the CLI call the exact same code, and both expose ``predict`` returning class
ids ready for :mod:`landcover.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier

from landcover.indices import patch_features, pixel_features

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class PatchRandomForest:
    """RF over per-patch spectral-index features (Track 1)."""

    n_estimators: int = 300
    max_depth: int | None = None
    random_state: int = 0
    _model: RandomForestClassifier | None = None

    def features(self, patches: list[FloatArray]) -> FloatArray:
        return np.stack([patch_features(patch) for patch in patches])

    def fit(self, patches: list[FloatArray], labels: IntArray) -> PatchRandomForest:
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._model.fit(self.features(patches), labels)
        return self

    def predict(self, patches: list[FloatArray]) -> IntArray:
        return np.asarray(self._require_model().predict(self.features(patches)), dtype=np.int64)

    def feature_importances(self) -> FloatArray:
        return np.asarray(self._require_model().feature_importances_, dtype=np.float64)

    def _require_model(self) -> RandomForestClassifier:
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._model


@dataclass
class PixelRandomForest:
    """Per-pixel RF for semantic segmentation (Track 2).

    Trained on the stacked pixel features of the training chips; ``predict``
    returns a full (H, W) label map so the segmentation metrics apply directly.
    """

    n_estimators: int = 200
    max_depth: int | None = 20
    random_state: int = 0
    _model: RandomForestClassifier | None = None

    def fit(self, chips: list[FloatArray], labels: list[IntArray]) -> PixelRandomForest:
        feature_blocks = [pixel_features(chip) for chip in chips]
        label_blocks = [label.reshape(-1) for label in labels]
        x = np.concatenate(feature_blocks, axis=0)
        y = np.concatenate(label_blocks, axis=0)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self._model.fit(x, y)
        return self

    def predict(self, chip: FloatArray) -> IntArray:
        _, height, width = chip.shape
        flat = self._require_model().predict(pixel_features(chip))
        return np.asarray(flat, dtype=np.int64).reshape(height, width)

    def _require_model(self) -> RandomForestClassifier:
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
        return self._model
