"""GMM-based anomaly detector with online scoring.

Qt-free port of ``core/anomaly_detector.py`` from the desktop app.
The original emitted ``detection_ready``/``model_trained``/
``training_progress_changed`` Qt signals; here :meth:`AnomalyDetector.feed`
returns a :class:`DetectionResult` directly, and training progress /
trained state are exposed as plain properties instead of signals. The
detection algorithm (GMM training, thresholding, scoring) is
byte-for-byte identical to the original.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_TRAIN_AFTER = 50          # samples before first training
_GMM_COMPONENTS = 1
_THRESHOLD_PERCENTILE = 2   # 2nd percentile of training log-likelihoods


@dataclass
class DetectionResult:
    """Result of scoring a single sample against the trained GMM.

    :param timestamp: Wall-clock timestamp of the sample.
    :type timestamp: float
    :param value: The sensor reading that was scored.
    :type value: float
    :param score: Raw log-likelihood under the GMM (higher = more normal).
    :type score: float
    :param confidence: Value in ``[0, 1]``; higher means more anomalous.
    :type confidence: float
    :param is_anomaly: Whether ``score`` fell below the trained threshold.
    :type is_anomaly: bool
    :param is_injected: Whether the sample was a synthetically injected anomaly.
    :type is_injected: bool
    """

    timestamp: float
    value: float
    score: float
    confidence: float
    is_anomaly: bool
    is_injected: bool


def _sigmoid(x: float) -> float:
    """Numerically simple logistic sigmoid.

    :param x: Input value.
    :type x: float
    :returns: ``1 / (1 + exp(-x))``.
    :rtype: float
    """
    return 1.0 / (1.0 + math.exp(-x))


class AnomalyDetector:
    """Buffers samples, trains a 1-component GMM, and scores new samples.

    Mirrors the desktop app's ``AnomalyDetector`` but without Qt
    signals: :meth:`feed` returns a :class:`DetectionResult` directly,
    and callers can poll :attr:`training_progress` / :attr:`is_trained`
    instead of connecting to ``training_progress_changed`` /
    ``model_trained``.
    """

    def __init__(self) -> None:
        self._buffer: List[float] = []
        self._gmm: Optional[GaussianMixture] = None
        self._scaler: Optional[StandardScaler] = None
        self._threshold: float = 0.0
        self._trained = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, timestamp: float, value: float, is_injected: bool) -> DetectionResult:
        """Feed one sample into the detector, training and/or scoring it.

        Auto-trains the GMM the first time the internal buffer reaches
        ``_TRAIN_AFTER`` (100) samples. Before training completes, an
        unscored result (``score=0.0``, ``confidence=0.0``,
        ``is_anomaly=False``) is returned so callers can still update
        charts while the model warms up.

        :param timestamp: Wall-clock timestamp of the sample.
        :type timestamp: float
        :param value: The sensor reading to buffer/score.
        :type value: float
        :param is_injected: Whether the sample was a synthetically
            injected anomaly (pass-through, not used for detection).
        :type is_injected: bool
        :returns: The detection result for this sample.
        :rtype: DetectionResult
        """
        self._buffer.append(value)
        n = len(self._buffer)

        # Auto-train once we have enough samples.
        if not self._trained and n >= _TRAIN_AFTER:
            self._train()

        if self._trained:
            return self._score(timestamp, value, is_injected)

        # Emit unscored result so the chart still updates during warm-up.
        return DetectionResult(
            timestamp=timestamp,
            value=value,
            score=0.0,
            confidence=0.0,
            is_anomaly=False,
            is_injected=is_injected,
        )

    @property
    def training_progress(self) -> float:
        """Fraction of samples collected towards the auto-train threshold.

        :returns: Value in ``[0.0, 1.0]``; ``1.0`` once trained.
        :rtype: float
        """
        if self._trained:
            return 1.0
        return min(len(self._buffer) / _TRAIN_AFTER, 1.0)

    @property
    def is_trained(self) -> bool:
        """Whether the GMM has been trained at least once.

        :returns: True once :meth:`feed` or :meth:`retrain` has trained a model.
        :rtype: bool
        """
        return self._trained

    def get_model_params(self) -> Optional[dict]:
        """Compute display-friendly parameters of the trained GMM.

        :returns: ``None`` if not yet trained, otherwise a dict with
            keys ``means``, ``stds``, ``weights`` (one entry per GMM
            component, in original data units) and ``boundary_values``
            (x-values where the score curve crosses the anomaly
            threshold), or ``None`` if untrained.
        :rtype: dict | None
        """
        if not self._trained:
            return None
        assert self._gmm is not None and self._scaler is not None
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

        return {
            "means": means,
            "stds": stds,
            "weights": weights,
            "boundary_values": boundary_values,
        }

    def reset(self) -> None:
        """Clear the sample buffer and discard the trained model.

        :returns: None
        :rtype: None
        """
        self._buffer.clear()
        self._gmm = None
        self._scaler = None
        self._trained = False
        logger.debug("AnomalyDetector reset; buffer and model cleared")

    @property
    def sample_count(self) -> int:
        """Number of samples currently buffered.

        :returns: Count of buffered samples (including those used to train).
        :rtype: int
        """
        return len(self._buffer)

    def retrain(self, start: int, end: int) -> None:
        """Retrain the GMM on a slice of the buffered samples.

        :param start: Start index (inclusive) into the internal buffer.
        :type start: int
        :param end: End index (exclusive) into the internal buffer.
        :type end: int
        :returns: None
        :rtype: None
        """
        subset = self._buffer[start:end]
        if len(subset) < 2:
            # Defensive: GMM fitting on <2 points is meaningless/unstable.
            logger.warning(
                "retrain() called with insufficient samples (%d); skipping", len(subset)
            )
            return
        self._train_on(subset)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """Train on the full current buffer."""
        self._train_on(self._buffer)

    def _train_on(self, data: List[float]) -> None:
        """Fit a scaler + GMM on ``data`` and compute the anomaly threshold.

        :param data: Sequence of raw sensor values to train on.
        :type data: list[float]
        :returns: None
        :rtype: None
        """
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
        logger.info(
            "AnomalyDetector trained on %d samples; threshold=%.4f", len(data), self._threshold
        )

    def _score(self, timestamp: float, value: float, is_injected: bool) -> DetectionResult:
        """Score a single value against the trained GMM.

        :param timestamp: Wall-clock timestamp of the sample.
        :type timestamp: float
        :param value: The sensor reading to score.
        :type value: float
        :param is_injected: Pass-through flag for whether this was a synthetic anomaly.
        :type is_injected: bool
        :returns: The scored detection result.
        :rtype: DetectionResult
        """
        assert self._gmm is not None and self._scaler is not None
        X = np.array([[value]])
        Xs = self._scaler.transform(X)
        raw_score = float(self._gmm.score_samples(Xs)[0])

        # Confidence: higher value -> more anomalous.
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
