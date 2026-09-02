"""Tests for the Qt-free :mod:`core.data_generator` port."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.data_generator import TEMPLATES, DataGenerator, StreamConfig


def test_templates_match_original_desktop_values() -> None:
    """TEMPLATES constants must match the desktop app exactly (byte-for-byte)."""
    assert TEMPLATES["Temperature"] == {
        "mean": 70.0, "std": 2.0, "noise": 0.5, "unit": "°C", "anomaly_magnitude": 15.0,
    }
    assert TEMPLATES["Vibration"] == {
        "mean": 0.5, "std": 0.1, "noise": 0.05, "unit": "g", "anomaly_magnitude": 3.0,
    }
    assert TEMPLATES["Pressure"] == {
        "mean": 100.0, "std": 3.0, "noise": 1.0, "unit": "kPa", "anomaly_magnitude": 30.0,
    }


def test_tick_returns_timestamp_value_is_injected_tuple() -> None:
    """tick() must return a (timestamp, value, is_injected) tuple, not emit a signal."""
    gen = DataGenerator(rng=np.random.default_rng(0))
    result = gen.tick()
    assert isinstance(result, tuple)
    assert len(result) == 3
    ts, value, is_injected = result
    assert isinstance(ts, float)
    assert isinstance(value, float)
    assert isinstance(is_injected, bool)


def test_tick_reproduces_drift_noise_formula_exactly() -> None:
    """tick()'s math must exactly match drift = std*sin(2*pi*t/30) + noise, 0.1s step.

    Uses a manually-seeded RNG and replays the *identical* sequence of
    RNG calls (normal, then random, then optionally choice) that the
    implementation is documented to make, to independently verify the
    formula rather than just re-deriving it from the implementation.
    """
    cfg = StreamConfig(
        mean=70.0, std=2.0, noise_amplitude=0.5,
        anomaly_probability=0.3, anomaly_magnitude=15.0, unit="°C",
    )
    seed = 7
    gen = DataGenerator(rng=np.random.default_rng(seed))
    gen.configure(cfg)

    # Independent RNG stream, consumed in the same order/arity the
    # implementation is specified to use, to compute expected values.
    expected_rng = np.random.default_rng(seed)

    t = 0.0
    for _ in range(50):
        _, value, is_injected = gen.tick()

        drift = cfg.std * math.sin(2 * math.pi * t / 30.0)
        noise = expected_rng.normal(0.0, cfg.noise_amplitude)
        expected_value = cfg.mean + drift + noise
        expected_injected = expected_rng.random() < cfg.anomaly_probability
        if expected_injected:
            sign = expected_rng.choice([-1, 1])
            expected_value += sign * cfg.anomaly_magnitude

        assert is_injected == expected_injected
        assert value == pytest.approx(expected_value, abs=1e-12)

        t += 0.1


def test_tick_advances_elapsed_time_by_0_1s_per_call() -> None:
    """Internal elapsed-time step must be exactly 0.1s to preserve the drift period."""
    gen = DataGenerator(rng=np.random.default_rng(1))
    assert gen._t == 0.0
    gen.tick()
    assert gen._t == pytest.approx(0.1)
    gen.tick()
    assert gen._t == pytest.approx(0.2)


def test_reset_zeroes_elapsed_time() -> None:
    """reset() must zero the internal elapsed-time counter."""
    gen = DataGenerator(rng=np.random.default_rng(2))
    for _ in range(5):
        gen.tick()
    assert gen._t > 0.0
    gen.reset()
    assert gen._t == 0.0


def test_default_rng_used_when_none_provided() -> None:
    """Defensive default: constructing without an rng must not raise and must still tick."""
    gen = DataGenerator()
    ts, value, is_injected = gen.tick()
    assert isinstance(value, float)
