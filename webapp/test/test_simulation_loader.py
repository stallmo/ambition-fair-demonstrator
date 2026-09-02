"""Tests for :mod:`core.simulation_loader`, including the rewritten
``SimulationPlayer.due_rows()`` polling scheduler.
"""
from __future__ import annotations

import filecmp
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from core.simulation_loader import (
    SimulationPlayer,
    SimulationScenario,
    _generate_pressure_scenario,
    _generate_temperature_scenario,
    _generate_vibration_scenario,
    generate_scenarios,
)

# Read-only reference to the desktop app's already-generated CSVs, used
# only to diff against (never written to).
_ORIGINAL_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "simulations"


def _make_scenario(rows: List[tuple]) -> SimulationScenario:
    """Build an in-memory SimulationScenario from (timestamp, value, is_anomaly) rows.

    :param rows: Sequence of ``(timestamp, value, is_anomaly)`` tuples.
    :type rows: list[tuple]
    :returns: A scenario wrapping the rows as a DataFrame.
    :rtype: SimulationScenario
    """
    df = pd.DataFrame(rows, columns=["timestamp", "value", "is_anomaly"])
    return SimulationScenario(name="test", path=Path("unused.csv"), data=df)


# ------------------------------------------------------------------
# CSV generation determinism / parity with the desktop app
# ------------------------------------------------------------------

@pytest.mark.skipif(
    not _ORIGINAL_ASSETS_DIR.exists(), reason="Original desktop app's generated CSVs not present"
)
@pytest.mark.parametrize(
    ("filename", "generator"),
    [
        ("scenario_temperature.csv", _generate_temperature_scenario),
        ("scenario_vibration.csv", _generate_vibration_scenario),
        ("scenario_pressure.csv", _generate_pressure_scenario),
    ],
)
def test_generated_csv_matches_original_desktop_csv_byte_for_byte(
    tmp_path: Path, filename: str, generator
) -> None:
    """Regenerating a scenario CSV must exactly match the desktop app's checked-in CSV.

    This is a strong, direct proof that the CSV-generation algorithms
    (RNG seeds, formulas, anomaly windows) were ported byte-for-byte,
    without needing to invoke the original PyQt5-based code at all.
    """
    original_path = _ORIGINAL_ASSETS_DIR / filename
    new_path = tmp_path / filename
    generator(new_path)
    assert filecmp.cmp(original_path, new_path, shallow=False), (
        f"regenerated {filename} differs from the original desktop app's CSV"
    )


def test_generate_scenarios_creates_missing_files_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_scenarios() must create missing CSVs and skip existing ones."""
    import core.simulation_loader as sim_loader

    monkeypatch.setattr(sim_loader, "_ASSETS_DIR", tmp_path)
    generate_scenarios()

    expected = {"scenario_temperature.csv", "scenario_vibration.csv", "scenario_pressure.csv"}
    assert expected.issubset({p.name for p in tmp_path.iterdir()})

    # Sentinel: overwrite one file, then confirm re-running does not touch it.
    sentinel_path = tmp_path / "scenario_temperature.csv"
    sentinel_path.write_text("sentinel")
    generate_scenarios()
    assert sentinel_path.read_text() == "sentinel"


# ------------------------------------------------------------------
# SimulationScenario.from_csv
# ------------------------------------------------------------------

def test_scenario_from_csv_requires_expected_columns(tmp_path: Path) -> None:
    """from_csv() must raise ValueError if required columns are missing."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("timestamp,value\n0.0,1.0\n")
    with pytest.raises(ValueError):
        SimulationScenario.from_csv(bad_csv)


def test_scenario_from_csv_derives_name_from_filename(tmp_path: Path) -> None:
    """Scenario name must be derived from the filename, matching the desktop app's convention."""
    csv_path = tmp_path / "scenario_temperature.csv"
    pd.DataFrame(
        {"timestamp": [0.0, 0.1], "value": [70.0, 70.1], "is_anomaly": [0, 0]}
    ).to_csv(csv_path, index=False)
    scenario = SimulationScenario.from_csv(csv_path)
    assert scenario.name == "Temperature"
    assert scenario.data["is_anomaly"].dtype == bool


# ------------------------------------------------------------------
# SimulationPlayer.due_rows() polling scheduler (the genuine rewrite)
# ------------------------------------------------------------------

def test_due_rows_returns_nothing_before_start() -> None:
    """Calling due_rows() before start() must return no rows (defensive no-op)."""
    player = SimulationPlayer()
    player.load(_make_scenario([(0.0, 1.0, False), (0.1, 2.0, False)]))
    assert player.due_rows(now=1000.0) == []


def test_due_rows_first_row_immediately_due_at_start() -> None:
    """Row 0 must always be due at (or after) the start() anchor time."""
    player = SimulationPlayer()
    player.load(_make_scenario([(0.0, 1.0, False), (0.1, 2.0, False)]))
    player.start(now=100.0)
    assert player.due_rows(now=100.0) == [(0.0, 1.0, False)]


def test_due_rows_delivers_rows_in_order_none_skipped_on_burst_poll() -> None:
    """A single poll long after start() must return *every* elapsed row, in order.

    This is the core guarantee of the due-rows design: infrequent /
    bursty polling must not silently skip rows, unlike a naive "is a
    new row due right now" check.
    """
    rows = [(round(i * 0.1, 3), float(i), False) for i in range(10)]
    player = SimulationPlayer()
    player.load(_make_scenario(rows))
    player.start(now=0.0)

    # Poll once, far in the future: every row with cumulative deadline
    # <= 5.0 should come back, in original order, none skipped.
    delivered = player.due_rows(now=5.0)
    assert delivered == rows  # all 10 rows have deadlines <= 0.9 << 5.0
    assert player.finished is True


def test_due_rows_respects_cumulative_deltas_between_polls() -> None:
    """Rows must only be returned once their cumulative scheduled deadline has passed."""
    rows = [(0.0, 0.0, False), (0.1, 1.0, False), (0.3, 2.0, False)]  # deltas: 0.1, 0.2
    player = SimulationPlayer()
    player.load(_make_scenario(rows))
    player.start(now=0.0)

    assert player.due_rows(now=0.0) == [rows[0]]
    assert player.due_rows(now=0.05) == []  # row 1's deadline (0.1) not yet reached
    assert player.due_rows(now=0.1) == [rows[1]]
    assert player.due_rows(now=0.29) == []  # row 2's deadline (0.1+0.2=0.3) not yet reached
    assert player.due_rows(now=0.3) == [rows[2]]
    assert player.finished is True


def test_due_rows_no_row_returned_twice_across_many_small_polls() -> None:
    """Polling many times must never re-deliver a row already returned."""
    rows = [(round(i * 0.1, 3), float(i), False) for i in range(20)]
    player = SimulationPlayer()
    player.load(_make_scenario(rows))
    player.start(now=0.0)

    seen: List[tuple] = []
    for step in range(1, 40):
        seen.extend(player.due_rows(now=step * 0.05))
    assert seen == rows


def test_due_rows_speed_10_with_150ms_poll_cadence_bursts_without_drift() -> None:
    """At speed=10, polled every 150ms (a representative Streamlit fragment
    poll cadence -- note: the live fragment's actual configured cadence is
    :data:`ui_web.live_tick.LIVE_TICK_INTERVAL`, ``"300ms"`` as of story
    P2; this test's 150ms poll interval is simply a conservative,
    tighter-than-production stress value that still exercises the same
    bursting behavior), due_rows() must return multiple rows per call
    (bursting, since 0.1s / 10 = 0.01s between deadlines is far below the
    0.15s poll interval) and the cumulative schedule must not drift from
    the sum of the original inter-row deltas scaled by 1/speed.

    This directly exercises the self-correcting design described in
    :meth:`SimulationPlayer.due_rows`'s docstring: each new deadline is
    derived from the *previous* deadline, not reset off wall-clock
    ``now``, so repeated polls at high speed never lag behind or
    accumulate error relative to the "ideal" total playback duration,
    regardless of the exact poll cadence used.
    """
    n = 300
    delta = 0.1  # matches the real scenario CSVs' 100ms sampling interval
    rows = [(round(i * delta, 3), float(i), False) for i in range(n)]
    player = SimulationPlayer()
    player.load(_make_scenario(rows))
    player.set_speed(10.0)

    start_time = 1_000.0
    player.start(now=start_time)

    poll_interval = 0.15
    now = start_time
    delivered: List[tuple] = []
    max_batch = 0
    polls = 0
    # Safety cap avoids an infinite loop if a regression ever broke
    # termination (e.g. finished never flipping).
    while not player.finished and polls < 10_000:
        now += poll_interval
        batch = player.due_rows(now=now)
        max_batch = max(max_batch, len(batch))
        delivered.extend(batch)
        polls += 1

    # Every row must have been delivered, in order, none skipped or
    # duplicated -- even though most polls arrive well after several
    # rows' deadlines have passed.
    assert delivered == rows
    assert player.finished is True

    # Bursting: at 0.01s between deadlines and a 0.15s poll cadence,
    # each poll (once playback is underway) should deliver ~15 rows.
    assert max_batch >= 10, "expected due_rows() to burst multiple rows per call under load"

    # No-drift: total simulated elapsed time (sum of original deltas
    # scaled by 1/speed) must match wall-clock elapsed time within one
    # poll interval -- not exhibit compounding drift across ~20 polls.
    expected_total = (rows[-1][0] - rows[0][0]) / 10.0  # sum(deltas) / speed
    actual_total = now - start_time
    assert actual_total == pytest.approx(expected_total, abs=poll_interval)


def test_set_speed_scales_deadlines() -> None:
    """A 2x speed must halve the wall-clock delay between successive row deadlines."""
    rows = [(0.0, 0.0, False), (1.0, 1.0, False)]  # 1.0s delta at normal speed
    player = SimulationPlayer()
    player.load(_make_scenario(rows))
    player.set_speed(2.0)
    player.start(now=0.0)

    assert player.due_rows(now=0.0) == [rows[0]]
    assert player.due_rows(now=0.49) == []  # 1.0 / 2.0 = 0.5s expected delay
    assert player.due_rows(now=0.5) == [rows[1]]


def test_set_speed_clamped_to_minimum() -> None:
    """Speed must be clamped to a sane minimum, preventing zero/negative delays."""
    player = SimulationPlayer()
    player.set_speed(0.0)
    assert player._speed == pytest.approx(0.1)
    player.set_speed(-5.0)
    assert player._speed == pytest.approx(0.1)


def test_stop_marks_finished_and_suppresses_further_rows() -> None:
    """stop() must prevent any further rows from being returned."""
    rows = [(0.0, 0.0, False), (0.1, 1.0, False), (0.2, 2.0, False)]
    player = SimulationPlayer()
    player.load(_make_scenario(rows))
    player.start(now=0.0)
    assert player.due_rows(now=0.0) == [rows[0]]

    player.stop()
    assert player.finished is True
    assert player.due_rows(now=1000.0) == []


def test_no_scenario_loaded_is_defensive_noop() -> None:
    """Calling start()/due_rows() with nothing loaded must not raise."""
    player = SimulationPlayer()
    player.start(now=0.0)  # no scenario loaded
    assert player.due_rows(now=100.0) == []
    assert player.finished is False
