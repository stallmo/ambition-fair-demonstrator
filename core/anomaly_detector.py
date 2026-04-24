"""GMM-based anomaly detector with online scoring."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

_TRAIN_AFTER = 100          # samples before first training
_GMM_COMPONENTS = 1
_THRESHOLD_PERCENTILE = 2   # 5th percentile of training log-likelihoods


@dataclass
class DetectionResult:
    timestamp: float
    value: float
    score: float            # raw log-likelihood (higher = more normal)
    confidence: float       # 0–1, higher = more anomalous
    is_anomaly: bool
    is_injected: bool


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class AnomalyDetector(QObject):
    """Receives samples, trains a GMM, and emits detection results.

    Signals
    -------
    detection_ready(DetectionResult)
    model_trained()
    training_progress_changed(float)   # 0.0 – 1.0
    """

    detection_ready = pyqtSignal(object)   # DetectionResult
    model_trained = pyqtSignal()
    training_progress_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer: list[float] = []
        self._gmm: Optional[GaussianMixture] = None
        self._scaler: Optional[StandardScaler] = None
        self._threshold: float = 0.0
        self._trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, timestamp: float, value: float, is_injected: bool) -> None:
        self._buffer.append(value)
        n = len(self._buffer)

        # Emit training progress
        if not self._trained:
            progress = min(n / _TRAIN_AFTER, 1.0)
            self.training_progress_changed.emit(progress)

        # Auto-train once we have enough samples
        if not self._trained and n >= _TRAIN_AFTER:
            self._train()

        # Score (only if model is ready)
        if self._trained:
            result = self._score(timestamp, value, is_injected)
            self.detection_ready.emit(result)
        else:
            # Emit unscored result so the chart still updates
            result = DetectionResult(
                timestamp=timestamp,
                value=value,
                score=0.0,
                confidence=0.0,
                is_anomaly=False,
                is_injected=is_injected,
            )
            self.detection_ready.emit(result)

    def get_model_params(self) -> dict | None:
        if not self._trained:
            return None
        scale = self._scaler.scale_[0]
        mean_train = self._scaler.mean_[0]
        means, stds, weights = [], [], []
        for i in range(_GMM_COMPONENTS):
            m_scaled = float(self._gmm.means_[i, 0])
            var_scaled = float(self._gmm.covariances_[i, 0, 0])
            means.append(m_scaled * scale + mean_train)
            stds.append(math.sqrt(abs(var_scaled)) * scale)
            weights.append(float(self._gmm.weights_[i]))
        lo = mean_train - 6 * scale
        hi = mean_train + 6 * scale
        xs_grid = np.linspace(lo, hi, 2000).reshape(-1, 1)
        scores_grid = self._gmm.score_samples(self._scaler.transform(xs_grid))
        above = scores_grid >= self._threshold
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]
        boundary_values = [float((xs_grid[i, 0] + xs_grid[i + 1, 0]) / 2) for i in crossings]

        return {"means": means, "stds": stds, "weights": weights,
                "boundary_values": boundary_values}

    def reset(self) -> None:
        self._buffer.clear()
        self._gmm = None
        self._scaler = None
        self._trained = False
        self.training_progress_changed.emit(0.0)

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    def retrain(self, start: int, end: int) -> None:
        subset = self._buffer[start:end]
        if len(subset) < 2:
            return
        self._train_on(subset)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train(self) -> None:
        self._train_on(self._buffer)

    def _train_on(self, data: list[float]) -> None:
        X = np.array(data).reshape(-1, 1)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        self._gmm = GaussianMixture(
            n_components=_GMM_COMPONENTS,
            covariance_type="full",
            random_state=42,
        )
        self._gmm.fit(Xs)

        scores = self._gmm.score_samples(Xs)
        self._threshold = float(np.percentile(scores, _THRESHOLD_PERCENTILE))

        self._trained = True
        self.model_trained.emit()
        self.training_progress_changed.emit(1.0)

    def _score(self, timestamp: float, value: float, is_injected: bool) -> DetectionResult:
        assert self._gmm is not None and self._scaler is not None
        X = np.array([[value]])
        Xs = self._scaler.transform(X)
        raw_score = float(self._gmm.score_samples(Xs)[0])

        # Confidence: higher value → more anomalous
        confidence = float(_sigmoid(self._threshold - raw_score))
        is_anomaly = raw_score < self._threshold

        return DetectionResult(
            timestamp=timestamp,
            value=value,
            score=raw_score,
            confidence=confidence,
            is_anomaly=is_anomaly,
            is_injected=is_injected,
        )
