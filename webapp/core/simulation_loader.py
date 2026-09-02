"""Simulation scenario loader and player.

Qt-free port of ``core/simulation_loader.py`` from the desktop app.
The CSV-generation functions and :class:`SimulationScenario` are ported
essentially verbatim (byte-for-byte identical algorithms/constants).

:class:`SimulationPlayer`, however, is a genuine *rewrite* rather than a
straight port: the original scheduled rows one at a time via
chained one-shot Qt timers, which has no equivalent without a Qt
event loop. Streamlit has no persistent event loop of its own -- it
re-runs the script on each interaction/timer tick -- so the player is
redesigned around a "due rows" polling model: :meth:`SimulationPlayer.due_rows`
is called with the caller's current time and returns *every* row whose
scheduled deadline has passed, in original order, none skipped. See
that method's docstring for the scheduling design.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mirrors the desktop app's ``assets/simulations`` layout, relative to
# this package's own location (i.e. ``webapp/assets/simulations``).
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "simulations"

# ------------------------------------------------------------------
# CSV generation (deterministic, only runs if files are missing)
# ------------------------------------------------------------------
# NOTE: these three functions and their constants are byte-for-byte
# identical to the desktop app's ``core/simulation_loader.py`` so that
# regenerated CSVs are pixel-for-pixel identical demo scenarios.


def _generate_temperature_scenario(path: Path) -> None:
    """Generate the deterministic temperature demo scenario CSV.

    :param path: Destination CSV path.
    :type path: pathlib.Path
    :returns: None
    :rtype: None
    """
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
    """Generate the deterministic vibration demo scenario CSV.

    :param path: Destination CSV path.
    :type path: pathlib.Path
    :returns: None
    :rtype: None
    """
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
    """Generate the deterministic pressure demo scenario CSV.

    :param path: Destination CSV path.
    :type path: pathlib.Path
    :returns: None
    :rtype: None
    """
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
    """Generate all simulation CSVs under :data:`_ASSETS_DIR` if missing.

    :returns: None
    :rtype: None
    """
    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("scenario_temperature.csv", _generate_temperature_scenario),
        ("scenario_vibration.csv", _generate_vibration_scenario),
        ("scenario_pressure.csv", _generate_pressure_scenario),
    ]
    for filename, generator in specs:
        path = _ASSETS_DIR / filename
        if not path.exists():
            logger.info("Generating missing simulation scenario CSV: %s", path)
            generator(path)


# ------------------------------------------------------------------
# Public classes
# ------------------------------------------------------------------

@dataclass
class SimulationScenario:
    """A loaded simulation scenario CSV.

    :param name: Human-readable scenario name (derived from filename).
    :type name: str
    :param path: Source CSV path.
    :type path: pathlib.Path
    :param data: Parsed scenario rows (``timestamp``, ``value``, ``is_anomaly``).
    :type data: pandas.DataFrame
    """

    name: str
    path: Path
    data: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path) -> "SimulationScenario":
        """Load a scenario from a CSV file.

        :param path: Path to a scenario CSV with ``timestamp``, ``value``,
            ``is_anomaly`` columns.
        :type path: pathlib.Path
        :raises ValueError: If required columns are missing from the CSV.
        :returns: The loaded scenario.
        :rtype: SimulationScenario
        """
        df = pd.read_csv(path)
        required = {"timestamp", "value", "is_anomaly"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV {path} missing columns: {required - set(df.columns)}")
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        return cls(name=path.stem.replace("scenario_", "").capitalize(), path=path, data=df)


class SimulationPlayer:
    """Replays a :class:`SimulationScenario` using a "due rows" polling model.

    Unlike the desktop app's chained one-shot-timer player, this
    version has no background thread or timer of its own. Instead, the
    caller (e.g. a Streamlit fragment re-run on a short interval) calls
    :meth:`due_rows` with its current time on every poll, and receives
    back every scenario row that has "come due" since the last poll.

    Scheduling design (see :meth:`due_rows` for the algorithm): each
    row's deadline is computed as the *previous* row's deadline plus
    the original inter-row time delta scaled by ``1 / speed``. This
    makes the schedule self-correcting -- it never resets itself from
    the caller's wall-clock ``now`` -- so infrequent or bursty polling
    (e.g. Streamlit reruns arriving late, or many reruns queued up at
    high playback speed) does not cause drift: every due row is always
    returned, in order, exactly once, regardless of how many poll
    intervals it took to notice they were due.
    """

    def __init__(self) -> None:
        self._scenario: Optional[SimulationScenario] = None
        self._index = 0
        self._speed = 1.0
        # Deadline (in the same time domain as the `now` passed to
        # due_rows, e.g. time.monotonic()) for the row at self._index.
        self._pending_deadline: Optional[float] = None
        self._finished_flag = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, scenario: SimulationScenario) -> None:
        """Load a scenario for playback, resetting any prior playback state.

        :param scenario: The scenario to play.
        :type scenario: SimulationScenario
        :returns: None
        :rtype: None
        """
        self._scenario = scenario
        self._index = 0
        self._pending_deadline = None
        self._finished_flag = False

    def set_speed(self, multiplier: float) -> None:
        """Set the playback speed multiplier.

        Takes effect for rows scheduled *after* the call (mirrors the
        original, which recomputed each row's delay from the current
        speed at the moment it was scheduled). Clamped to a sane
        minimum to avoid runaway/zero-delay playback.

        :param multiplier: Desired speed multiplier (1.0 = real-time).
        :type multiplier: float
        :returns: None
        :rtype: None
        """
        self._speed = max(0.1, multiplier)

    def start(self, now: Optional[float] = None) -> None:
        """Begin playback, anchoring the schedule to ``now``.

        :param now: Time to anchor the schedule to, in the same domain
            the caller will later pass to :meth:`due_rows` (e.g.
            ``time.monotonic()``). Defaults to ``time.monotonic()`` if
            not provided; overriding is mainly useful for deterministic
            tests.
        :type now: float | None
        :returns: None
        :rtype: None
        """
        if self._scenario is None:
            logger.warning("SimulationPlayer.start() called with no scenario loaded")
            return
        self._index = 0
        self._finished_flag = False
        anchor = now if now is not None else time.monotonic()
        # The first row is always immediately due, matching the
        # original's delay_ms=0 for index 0.
        self._pending_deadline = anchor

    def stop(self) -> None:
        """Stop playback; subsequent :meth:`due_rows` calls return nothing.

        :returns: None
        :rtype: None
        """
        if self._scenario is not None:
            self._index = len(self._scenario.data)
        self._finished_flag = True

    def due_rows(self, now: float) -> List[Tuple[float, float, bool]]:
        """Return every scenario row whose deadline is ``<= now``.

        Deadlines are computed cumulatively: the deadline for row
        ``i + 1`` is ``deadline(i) + (timestamp[i+1] - timestamp[i]) / speed``.
        Because each new deadline is derived from the *previous
        deadline* rather than from the current ``now``, the schedule is
        self-correcting and does not drift even if this method is
        called infrequently or ``now`` jumps forward a lot between
        calls (e.g. Streamlit rerunning late): the loop below simply
        drains every row that has become due since the last call, in
        original CSV order, none skipped.

        :param now: Caller's current time, in the same domain passed to
            :meth:`start` (e.g. ``time.monotonic()``).
        :type now: float
        :returns: List of ``(timestamp, value, is_anomaly)`` tuples for
            every row that is now due, in original order. Empty if none
            are due yet, playback has not started, or playback finished.
        :rtype: list[tuple[float, float, bool]]
        """
        if self._scenario is None or self._pending_deadline is None:
            return []

        df = self._scenario.data
        n = len(df)
        delivered: List[Tuple[float, float, bool]] = []

        while self._index < n and self._pending_deadline <= now:
            row = df.iloc[self._index]
            delivered.append((float(row["timestamp"]), float(row["value"]), bool(row["is_anomaly"])))

            next_index = self._index + 1
            if next_index < n:
                next_row = df.iloc[next_index]
                delta_s = float(next_row["timestamp"]) - float(row["timestamp"])
                # Defensive: guard against out-of-order/duplicate timestamps
                # producing a negative delta, which would otherwise move
                # the schedule backwards.
                self._pending_deadline = self._pending_deadline + max(0.0, delta_s) / self._speed
            self._index = next_index

        if self._index >= n:
            self._finished_flag = True

        return delivered

    @property
    def finished(self) -> bool:
        """Whether playback has delivered all rows (or was stopped).

        :returns: True once every row has been returned by :meth:`due_rows`.
        :rtype: bool
        """
        return self._finished_flag
