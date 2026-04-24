"""Synthetic sensor data generator with optional anomaly injection."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

# Template defaults for each sensor type
TEMPLATES: dict[str, dict] = {
    "Temperature": {
        "mean": 70.0,
        "std": 2.0,
        "noise": 0.5,
        "unit": "°C",
        "anomaly_magnitude": 15.0,
    },
    "Vibration": {
        "mean": 0.5,
        "std": 0.1,
        "noise": 0.05,
        "unit": "g",
        "anomaly_magnitude": 3.0,
    },
    "Pressure": {
        "mean": 100.0,
        "std": 3.0,
        "noise": 1.0,
        "unit": "kPa",
        "anomaly_magnitude": 30.0,
    },
}


@dataclass
class StreamConfig:
    mean: float = 70.0
    std: float = 2.0
    noise_amplitude: float = 0.5
    anomaly_probability: float = 0.02
    anomaly_magnitude: float = 15.0
    unit: str = "°C"


class DataGenerator(QObject):
    """Generates synthetic sensor readings at 100 ms intervals.

    Signals
    -------
    new_sample(timestamp: float, value: float, is_injected: bool)
    """

    new_sample = pyqtSignal(float, float, bool)

    _TICK_MS = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config = StreamConfig()
        self._timer = QTimer(self)
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._generate_tick)
        self._t = 0.0          # elapsed seconds
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(self, config: StreamConfig) -> None:
        self._config = config

    def start(self) -> None:
        self._t = 0.0
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def reset(self) -> None:
        self._timer.stop()
        self._t = 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_tick(self) -> None:
        cfg = self._config
        ts = time.time()

        # Slow sinusoidal drift (period ≈ 30 s) + Gaussian noise
        drift = cfg.std * math.sin(2 * math.pi * self._t / 30.0)
        noise = self._rng.normal(0.0, cfg.noise_amplitude)
        value = cfg.mean + drift + noise

        # Spike injection
        is_injected = self._rng.random() < cfg.anomaly_probability
        if is_injected:
            sign = self._rng.choice([-1, 1])
            value += sign * cfg.anomaly_magnitude

        self._t += self._TICK_MS / 1000.0
        self.new_sample.emit(ts, float(value), bool(is_injected))
