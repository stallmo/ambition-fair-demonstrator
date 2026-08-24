"""Tests for the Qt-free :mod:`core.anomaly_detector` port."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from core.anomaly_detector import AnomalyDetector, DetectionResult


def _feed_normal_samples(det: AnomalyDetector, n: int, seed: int = 42) -> np.ndarray:
    """Feed ``n`` Gaussian samples into ``det`` and return the values fed.

    :param det: Detector instance to feed.
    :type det: AnomalyDetector
    :param n: Number of samples to generate/feed.
    :type n: int
    :param seed: RNG seed for reproducibility.
    :type seed: int
    :returns: The array of values that were fed, in order.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    values = rng.normal(70.0, 2.0, size=n)
    for i, v in enumerate(values):
        det.feed(float(i), float(v), False)
    return values


def test_feed_returns_detection_result() -> None:
    """feed() must return a DetectionResult, not emit a signal."""
    det = AnomalyDetector()
    result = det.feed(0.0, 70.0, False)
    assert isinstance(result, DetectionResult)
    assert result.value == 70.0
    assert result.is_injected is False


def test_untrained_feed_returns_unscored_result() -> None:
    """Before the training threshold, results must be unscored placeholders."""
    det = AnomalyDetector()
    result = det.feed(0.0, 70.0, False)
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.is_anomaly is False
    assert det.is_trained is False


def test_trains_after_exactly_100_samples() -> None:
    """The model must remain untrained through sample 99 and train on sample 100."""
    det = AnomalyDetector()
    rng = np.random.default_rng(0)
    for i in range(99):
        det.feed(float(i), float(rng.normal(70, 2)), False)
        assert det.is_trained is False
    det.feed(99.0, float(rng.normal(70, 2)), False)
    assert det.is_trained is True
    assert det.sample_count == 100


def test_training_progress_reports_fraction_until_trained() -> None:
    """training_progress must climb from 0 to 1.0 as samples accumulate, then hold at 1.0."""
    det = AnomalyDetector()
    assert det.training_progress == 0.0
    rng = np.random.default_rng(0)
    for i in range(50):
        det.feed(float(i), float(rng.normal(70, 2)), False)
    assert det.training_progress == pytest.approx(0.5)
    for i in range(50, 100):
        det.feed(float(i), float(rng.normal(70, 2)), False)
    assert det.training_progress == 1.0
    # Feeding more samples post-training should not change progress.
    det.feed(100.0, 70.0, False)
    assert det.training_progress == 1.0


def test_uses_specified_gmm_and_scaler_configuration() -> None:
    """The trained GMM must be a single-component, full-covariance model (random_state=42)
    fit on StandardScaler-normalized data, matching the desktop app's configuration.
    """
    det = AnomalyDetector()
    _feed_normal_samples(det, 100)

    assert det._gmm is not None
    assert isinstance(det._gmm, GaussianMixture)
    assert det._gmm.n_components == 1
    assert det._gmm.covariance_type == "full"
    assert det._gmm.random_state == 42
    assert isinstance(det._scaler, StandardScaler)


def test_threshold_is_2nd_percentile_of_training_scores() -> None:
    """Threshold must equal the 2nd percentile of the training log-likelihoods."""
    det = AnomalyDetector()
    values = _feed_normal_samples(det, 100)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(values.reshape(-1, 1))
    gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
    gmm.fit(Xs)
    scores = gmm.score_samples(Xs)
    expected_threshold = float(np.percentile(scores, 2))

    assert det._threshold == pytest.approx(expected_threshold, abs=1e-9)


def test_confidence_is_sigmoid_of_threshold_minus_raw_score() -> None:
    """confidence must equal sigmoid(threshold - raw_score) for a scored sample."""
    det = AnomalyDetector()
    _feed_normal_samples(det, 100)

    result = det.feed(101.0, 70.0, False)
    Xs = det._scaler.transform(np.array([[70.0]]))
    raw_score = float(det._gmm.score_samples(Xs)[0])
    expected_confidence = 1.0 / (1.0 + np.exp(-(det._threshold - raw_score)))

    assert result.score == pytest.approx(raw_score, abs=1e-9)
    assert result.confidence == pytest.approx(expected_confidence, abs=1e-9)
    assert result.is_anomaly == (raw_score < det._threshold)


def test_get_model_params_structure() -> None:
    """get_model_params() must return means/stds/weights/boundary_values before-and-after training."""
    det = AnomalyDetector()
    assert det.get_model_params() is None  # untrained

    _feed_normal_samples(det, 100)
    params = det.get_model_params()
    assert params is not None
    assert set(params.keys()) == {"means", "stds", "weights", "boundary_values"}
    assert len(params["means"]) == 1
    assert len(params["stds"]) == 1
    assert len(params["weights"]) == 1
    # A single-component GMM should have weight 1.0.
    assert params["weights"][0] == pytest.approx(1.0)
    assert isinstance(params["boundary_values"], list)


def test_retrain_updates_model_on_subset() -> None:
    """retrain(start, end) must refit the GMM/scaler on the given buffer slice only."""
    det = AnomalyDetector()
    values = _feed_normal_samples(det, 150)
    params_before = det.get_model_params()

    det.retrain(0, 50)
    params_after = det.get_model_params()

    # Retraining on a different (smaller) subset should generally change
    # the fitted mean, since the subset statistics differ.
    assert params_before["means"][0] != pytest.approx(params_after["means"][0], abs=1e-9)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(values[:50].reshape(-1, 1))
    gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
    gmm.fit(Xs)
    expected_mean = float(gmm.means_[0, 0]) * scaler.scale_[0] + scaler.mean_[0]
    assert params_after["means"][0] == pytest.approx(expected_mean, abs=1e-9)


def test_retrain_with_fewer_than_2_samples_is_a_defensive_noop() -> None:
    """retrain() on a too-small slice must not raise and must leave the model untouched."""
    det = AnomalyDetector()
    _feed_normal_samples(det, 100)
    params_before = det.get_model_params()

    det.retrain(0, 1)  # only 1 sample: below the GaussianMixture minimum
    params_after = det.get_model_params()

    assert params_before == params_after


def test_reset_clears_buffer_and_model() -> None:
    """reset() must clear the buffer, discard the model, and reset training_progress."""
    det = AnomalyDetector()
    _feed_normal_samples(det, 100)
    assert det.is_trained is True

    det.reset()
    assert det.is_trained is False
    assert det.sample_count == 0
    assert det.training_progress == 0.0
    assert det.get_model_params() is None
