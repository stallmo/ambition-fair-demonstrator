"""Synthetic sensor data generator with optional anomaly injection.

Qt-free port of ``core/data_generator.py`` from the desktop app. The
original fired on a repeating 100 ms Qt timer that emitted a
``new_sample`` Qt signal; here :meth:`DataGenerator.tick` is called
directly (e.g. from a Streamlit polling loop) and returns the sample as
a plain tuple instead. The generation math itself -- drift, noise,
anomaly injection -- is byte-for-byte identical to the original.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Template defaults for each sensor type. Values match the desktop app
# exactly so demos behave identically across both front ends.
TEMPLATES: dict[str, dict] = {
    "Temperature": {
        "mean": 70.0,
        "std": 2.0,
        "noise": 0.5,
        "unit": "°C",
        "anomaly_magnitude": 15.0,
    },
    "Product Dimension": {
        "mean": 0.5,
        "std": 0.1,
        "noise": 0.05,
        "unit": "g",
        "anomaly_magnitude": 3.0,
    },
    "Acoustic Emission": {
        "mean": 100.0,
        "std": 3.0,
        "noise": 1.0,
        "unit": "kPa",
        "anomaly_magnitude": 30.0,
    },
}


@dataclass
class StreamConfig:
    """Configuration for a synthetic sensor stream.

    :param mean: Baseline signal mean.
    :type mean: float
    :param std: Standard deviation driving both the slow sinusoidal
        drift amplitude and (indirectly) typical spread of readings.
    :type std: float
    :param noise_amplitude: Standard deviation of the Gaussian noise
        added on top of the drift.
    :type noise_amplitude: float
    :param anomaly_probability: Per-tick probability of injecting a
        spike anomaly, in ``[0, 1]``.
    :type anomaly_probability: float
    :param anomaly_magnitude: Absolute magnitude added to (or
        subtracted from) the value when an anomaly is injected.
    :type anomaly_magnitude: float
    :param unit: Display unit for the stream (e.g. ``"°C"``).
    :type unit: str
    """

    mean: float = 70.0
    std: float = 2.0
    noise_amplitude: float = 0.5
    anomaly_probability: float = 0.02
    anomaly_magnitude: float = 15.0
    unit: str = "°C"


class DataGenerator:
    """Generates synthetic sensor readings, one sample per call to :meth:`tick`.

    This mirrors the desktop app's ``DataGenerator`` but without the
    Qt timer/signal machinery: callers are responsible for
    invoking :meth:`tick` at the desired cadence (nominally every
    100 ms, matching the original) and consuming the returned tuple
    directly instead of connecting to a signal.

    :param rng: Optional numpy random generator to inject for
        deterministic testing. Defaults to a fresh
        ``np.random.default_rng()`` instance.
    :type rng: numpy.random.Generator | None
    """

    #: Simulated tick duration in seconds; matches the desktop app's
    #: 100 ms timer interval so the drift/noise math is identical.
    _TICK_S = 0.1

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self._config = StreamConfig()
        self._t = 0.0  # elapsed simulated seconds
        # Defensive default: always have a usable RNG even if caller passes None.
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, config: StreamConfig) -> None:
        """Replace the active stream configuration.

        :param config: New stream configuration to use for subsequent ticks.
        :type config: StreamConfig
        :returns: None
        :rtype: None
        """
        self._config = config

    def reset(self) -> None:
        """Reset the elapsed simulated time back to zero.

        :returns: None
        :rtype: None
        """
        self._t = 0.0
        logger.debug("DataGenerator reset; elapsed time set to 0.0")

    def tick(self) -> Tuple[float, float, bool]:
        """Generate the next sample.

        Reproduces the original ``_generate_tick`` math exactly:
        a slow sinusoidal drift (period ~30 s) plus Gaussian noise,
        with a chance of a randomly-signed spike ("anomaly") added on
        top.

        :returns: Tuple of ``(timestamp, value, is_injected)`` where
            ``timestamp`` is a wall-clock ``time.time()`` value,
            ``value`` is the generated reading, and ``is_injected``
            indicates whether a synthetic anomaly spike was added.
        :rtype: tuple[float, float, bool]
        """
        cfg = self._config
        ts = time.time()

        # Slow sinusoidal drift (period ~ 30 s) + Gaussian noise.
        drift = cfg.std * math.sin(2 * math.pi * self._t / 30.0)
        noise = self._rng.normal(0.0, cfg.noise_amplitude)
        value = cfg.mean + drift + noise

        # Spike injection.
        is_injected = self._rng.random() < cfg.anomaly_probability
        if is_injected:
            sign = self._rng.choice([-1, 1])
            value += sign * cfg.anomaly_magnitude

        self._t += self._TICK_S
        return ts, float(value), bool(is_injected)
