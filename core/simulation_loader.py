"""Simulation scenario loader and player."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

_ASSETS_DIR = Path(__file__).parent.parent / "assets" / "simulations"

# ------------------------------------------------------------------
# CSV generation (deterministic, only runs if files are missing)
# ------------------------------------------------------------------

def _generate_temperature_scenario(path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    t = 0.0
    anomaly_times = {30, 31, 60, 61, 90, 91, 32, 62, 92, 33, 63, 93}
    for i in range(1200):  # 120 s at 100 ms
        value = 70.0 + 2.0 * np.sin(2 * np.pi * t / 60.0) + rng.normal(0, 0.5)
        is_anomaly = False
        if round(t) in anomaly_times:
            value += rng.choice([-1, 1]) * rng.uniform(12, 18)
            is_anomaly = True
        rows.append({"timestamp": round(t, 3), "value": round(value, 4), "is_anomaly": int(is_anomaly)})
        t += 0.1
    pd.DataFrame(rows).to_csv(path, index=False)


def _generate_vibration_scenario(path: Path) -> None:
    rng = np.random.default_rng(1)
    rows = []
    t = 0.0
    anomaly_start = 45.0
    for i in range(900):  # 90 s
        base = 0.5
        if t >= anomaly_start:
            ramp = min((t - anomaly_start) / 20.0, 1.0)
            base += ramp * 2.5
        value = base + 0.1 * np.sin(2 * np.pi * t / 10.0) + rng.normal(0, 0.05)
        is_anomaly = t >= anomaly_start and rng.random() < 0.4
        rows.append({"timestamp": round(t, 3), "value": round(value, 4), "is_anomaly": int(is_anomaly)})
        t += 0.1
    pd.DataFrame(rows).to_csv(path, index=False)


def _generate_pressure_scenario(path: Path) -> None:
    rng = np.random.default_rng(2)
    rows = []
    t = 0.0
    drop_times = {20, 20.1, 20.2, 40, 40.1, 40.2}
    for i in range(600):  # 60 s
        value = 100.0 + 3.0 * np.sin(2 * np.pi * t / 30.0) + rng.normal(0, 1.0)
        is_anomaly = False
        for dt in drop_times:
            if abs(t - dt) < 0.05:
                value -= rng.uniform(25, 35)
                is_anomaly = True
                break
        rows.append({"timestamp": round(t, 3), "value": round(value, 4), "is_anomaly": int(is_anomaly)})
        t += 0.1
    pd.DataFrame(rows).to_csv(path, index=False)


def generate_scenarios() -> None:
    """Generate all simulation CSVs if they are missing."""
    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("scenario_temperature.csv", _generate_temperature_scenario),
        ("scenario_vibration.csv", _generate_vibration_scenario),
        ("scenario_pressure.csv", _generate_pressure_scenario),
    ]
    for filename, generator in specs:
        path = _ASSETS_DIR / filename
        if not path.exists():
            generator(path)


# ------------------------------------------------------------------
# Public classes
# ------------------------------------------------------------------

@dataclass
class SimulationScenario:
    name: str
    path: Path
    data: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path) -> "SimulationScenario":
        df = pd.read_csv(path)
        required = {"timestamp", "value", "is_anomaly"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV {path} missing columns: {required - set(df.columns)}")
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        return cls(name=path.stem.replace("scenario_", "").capitalize(), path=path, data=df)


class SimulationPlayer(QObject):
    """Replays a SimulationScenario, emitting same signal as DataGenerator.

    Signals
    -------
    new_sample(timestamp: float, value: float, is_injected: bool)
    finished()
    """

    new_sample = pyqtSignal(float, float, bool)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scenario: SimulationScenario | None = None
        self._index = 0
        self._speed = 1.0

    def load(self, scenario: SimulationScenario) -> None:
        self._scenario = scenario
        self._index = 0

    def set_speed(self, multiplier: float) -> None:
        self._speed = max(0.1, multiplier)

    def play(self) -> None:
        if self._scenario is None:
            return
        self._index = 0
        self._schedule_next()

    def stop(self) -> None:
        # Timer is one-shot; stopping is handled by not scheduling further ticks
        self._index = len(self._scenario.data) if self._scenario else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        if self._scenario is None:
            return
        df = self._scenario.data
        if self._index >= len(df):
            self.finished.emit()
            return

        row = df.iloc[self._index]
        # Compute delay from row timestamp deltas
        if self._index == 0:
            delay_ms = 0
        else:
            prev_row = df.iloc[self._index - 1]
            delta_s = float(row["timestamp"]) - float(prev_row["timestamp"])
            delay_ms = max(0, int(delta_s * 1000 / self._speed))

        QTimer.singleShot(delay_ms, self._emit_current)

    def _emit_current(self) -> None:
        if self._scenario is None:
            return
        df = self._scenario.data
        if self._index >= len(df):
            self.finished.emit()
            return
        row = df.iloc[self._index]
        ts = time.time()
        self.new_sample.emit(ts, float(row["value"]), bool(row["is_anomaly"]))
        self._index += 1
        self._schedule_next()
