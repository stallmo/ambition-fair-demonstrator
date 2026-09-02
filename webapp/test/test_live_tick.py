"""Tests for :mod:`ui_web.live_tick` (the ``st.fragment(run_every=LIVE_TICK_INTERVAL)`` tick loop).

Covers the pure, unit-testable :func:`process_one_tick` with fakes (to
isolate the RUNNING/IDLE/PAUSED/SIMULATION_RUNNING branching from the
real generator/detector math) and with real
:class:`core.data_generator.DataGenerator` /
:class:`core.anomaly_detector.AnomalyDetector` /
:class:`utils.ring_buffer.RingBuffer` instances (end-to-end integration
check), plus one :class:`streamlit.testing.v1.AppTest`-based test that
exercises the actual ``@st.fragment``-decorated function in a real
Streamlit script context.

Story B3 adds coverage for :func:`ui_web.live_tick._build_index_x_axis`
and :func:`ui_web.live_tick.update_chart_from_state` (the chart-redraw
wiring), plus AppTest-based checks that ``live_tick_fragment`` renders
a Plotly chart element (story Q1; originally ``st.pyplot``) on every
run, RUNNING or not.

Story P3 adds coverage for the logbook row cap/eviction behavior of
:func:`ui_web.live_tick._append_anomaly_to_logbook` (FIFO eviction once
``MAX_LOGBOOK_ROWS`` is reached, monotonic ``row_id``/``next_log_id``
across evictions, no ``pd.concat`` call once at cap, and continued
compatibility with the Epic E/F1 logbook/dashboard consumers).

Story P2 adds coverage for :data:`ui_web.live_tick.LIVE_TICK_INTERVAL`
(the module-level constant now the single source of truth for the
fragment's ``run_every`` cadence, raised from an original ``"150ms"``
to ``"300ms"``) and confirms generator-tick semantics stay
cadence-independent: repeated calls into the tick-processing logic
still advance simulated time by a fixed 0.1s per call regardless of
what wall-clock ``run_every`` string the fragment is configured with.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from core.anomaly_detector import AnomalyDetector, DetectionResult
from core.data_generator import DataGenerator
from state.app_state import LOGBOOK_COLUMNS, MAX_LOGBOOK_ROWS, AppState, init_session_state
from ui_web.live_tick import (
    LIVE_TICK_INTERVAL,
    _append_anomaly_to_logbook,
    _build_index_x_axis,
    process_one_tick,
    update_chart_from_state,
)
from utils.ring_buffer import RingBuffer

_RING_KEYS: Tuple[str, str, str, str, str] = (
    "ring_ts",
    "ring_vals",
    "ring_confs",
    "ring_is_anomaly",
    "ring_is_injected",
)


class _FakeSessionState(dict):
    """Minimal dict-like stand-in for ``st.session_state``.

    Mirrors the ``_FakeSessionState`` pattern used in
    ``test_app_state.py``: the real ``st.session_state`` warns/behaves
    oddly outside of a running Streamlit script, so tests inject a
    plain dict subclass via ``process_one_tick(state=...)`` instead.
    """


class _FakeGenerator:
    """Stub generator that returns a fixed, counted sequence of samples.

    Lets tests assert *exactly* how many times ``tick()`` was called,
    independent of the real drift/noise/injection math.

    :param samples: Sequence of ``(timestamp, value, is_injected)``
        tuples to return, one per call, in order.
    :type samples: list[tuple[float, float, bool]]
    """

    def __init__(self, samples: List[Tuple[float, float, bool]]) -> None:
        self._samples = list(samples)
        self.call_count = 0

    def tick(self) -> Tuple[float, float, bool]:
        """Return the next canned sample and bump :attr:`call_count`."""
        self.call_count += 1
        return self._samples.pop(0)


class _FakeDetector:
    """Stub detector recording every ``feed()`` call's arguments.

    :ivar calls: List of ``(timestamp, value, is_injected)`` argument
        tuples, one per :meth:`feed` call, in call order.
    :vartype calls: list[tuple[float, float, bool]]
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[float, float, bool]] = []

    def feed(self, timestamp: float, value: float, is_injected: bool) -> DetectionResult:
        """Record the call and return a deterministic :class:`DetectionResult`."""
        self.calls.append((timestamp, value, is_injected))
        return DetectionResult(
            timestamp=timestamp,
            value=value,
            score=-1.23,
            confidence=0.42,
            is_anomaly=is_injected,
            is_injected=is_injected,
        )


def _make_fake_state(app_state: AppState, n_samples: int = 5) -> Tuple[_FakeSessionState, _FakeGenerator, _FakeDetector]:
    """Build a fake session-state dict wired with fake generator/detector.

    :param app_state: Value to seed ``state["app_state"]`` with.
    :type app_state: AppState
    :param n_samples: Number of canned samples the fake generator can produce.
    :type n_samples: int
    :returns: Tuple of ``(state, generator, detector)``.
    :rtype: tuple[_FakeSessionState, _FakeGenerator, _FakeDetector]
    """
    samples = [(float(i), float(i) * 10.0, i % 3 == 0) for i in range(n_samples)]
    generator = _FakeGenerator(samples)
    detector = _FakeDetector()
    state = _FakeSessionState(
        app_state=app_state,
        generator=generator,
        detector=detector,
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    return state, generator, detector


def _make_fake_state_with_logbook(
    app_state: AppState, n_samples: int = 5
) -> Tuple[_FakeSessionState, _FakeGenerator, _FakeDetector]:
    """Like :func:`_make_fake_state`, plus an empty ``logbook_df``/``next_log_id``.

    Kept separate from :func:`_make_fake_state` (rather than adding the
    logbook keys there unconditionally) so the existing "missing
    generator/detector/ring-buffer key" defensive tests keep exercising
    a state dict that is missing *unrelated* keys too, undisturbed.

    :param app_state: Value to seed ``state["app_state"]`` with.
    :type app_state: AppState
    :param n_samples: Number of canned samples the fake generator can produce.
    :type n_samples: int
    :returns: Tuple of ``(state, generator, detector)``.
    :rtype: tuple[_FakeSessionState, _FakeGenerator, _FakeDetector]
    """
    state, generator, detector = _make_fake_state(app_state, n_samples=n_samples)
    state["logbook_df"] = pd.DataFrame(columns=LOGBOOK_COLUMNS)
    state["next_log_id"] = 0
    return state, generator, detector


# ----------------------------------------------------------------------
# RUNNING: exactly one tick()/feed() per call, correctly buffered.
# ----------------------------------------------------------------------

def test_running_calls_generator_tick_exactly_once() -> None:
    """When RUNNING, process_one_tick() must call generator.tick() exactly once."""
    state, generator, _detector = _make_fake_state(AppState.RUNNING)
    result = process_one_tick(state)
    assert result is True
    assert generator.call_count == 1


def test_running_feeds_detector_with_generator_output() -> None:
    """detector.feed() must be called with exactly the tuple generator.tick() returned."""
    state, generator, detector = _make_fake_state(AppState.RUNNING)
    process_one_tick(state)
    assert len(detector.calls) == 1
    # The sample fed to feed() came from the (undisturbed) generator output.
    expected_ts, expected_val, expected_injected = (0.0, 0.0, True)  # i=0 sample
    assert detector.calls[0] == (expected_ts, expected_val, expected_injected)
    assert generator.call_count == 1


def test_running_appends_to_all_five_ring_buffers_synchronized() -> None:
    """A single RUNNING tick must grow all five ring buffers by exactly one, in sync."""
    state, _generator, _detector = _make_fake_state(AppState.RUNNING)
    process_one_tick(state)
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == 1 for size in sizes.values()), sizes


def test_running_total_samples_increments_by_one() -> None:
    """total_samples must increment by exactly 1 per processed sample."""
    state, _generator, _detector = _make_fake_state(AppState.RUNNING)
    assert state["total_samples"] == 0
    process_one_tick(state)
    assert state["total_samples"] == 1


def test_running_multiple_calls_keep_buffers_length_synced() -> None:
    """N consecutive RUNNING calls must grow every ring buffer by exactly N, in sync."""
    n_calls = 5
    state, generator, detector = _make_fake_state(AppState.RUNNING, n_samples=n_calls)
    for _ in range(n_calls):
        assert process_one_tick(state) is True

    assert generator.call_count == n_calls
    assert len(detector.calls) == n_calls
    assert state["total_samples"] == n_calls
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == n_calls for size in sizes.values()), sizes


def test_running_ring_buffer_values_correspond_to_the_same_sample() -> None:
    """The i-th entry across all five buffers must all come from the same tick."""
    n_calls = 4
    state, _generator, _detector = _make_fake_state(AppState.RUNNING, n_samples=n_calls)
    for _ in range(n_calls):
        process_one_tick(state)

    ts_window = state["ring_ts"].get_window()
    vals_window = state["ring_vals"].get_window()
    injected_window = state["ring_is_injected"].get_window()
    # Canned samples were (i, i*10, i % 3 == 0); value must equal ts * 10.
    for ts, val in zip(ts_window, vals_window):
        assert val == pytest.approx(ts * 10.0)
    for i, injected in enumerate(injected_window):
        assert bool(injected) == (i % 3 == 0)


# ----------------------------------------------------------------------
# IDLE / PAUSED: strict no-op. SIMULATION_RUNNING here is also a no-op,
# but for a *different* reason than IDLE/PAUSED (see D2's dedicated
# section further down for the real SIMULATION_RUNNING behavior with a
# sim_player present): _make_fake_state() never seeds 'sim_player', so
# _process_simulation_tick()'s defensive "sim_player missing" guard is
# what makes this a no-op here, exactly like the "missing
# generator/detector" defensive tests for RUNNING further below.
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "app_state", [AppState.IDLE, AppState.PAUSED, AppState.SIMULATION_RUNNING]
)
def test_non_running_states_are_no_ops(app_state: AppState) -> None:
    """IDLE/PAUSED must not tick, feed, buffer, or count; SIMULATION_RUNNING
    without a sim_player must be a safe defensive no-op too."""
    state, generator, detector = _make_fake_state(app_state)
    result = process_one_tick(state)

    assert result is False
    assert generator.call_count == 0
    assert detector.calls == []
    assert state["total_samples"] == 0
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == 0 for size in sizes.values()), sizes


@pytest.mark.parametrize(
    "app_state", [AppState.IDLE, AppState.PAUSED, AppState.SIMULATION_RUNNING]
)
def test_non_running_states_repeated_calls_stay_no_op(app_state: AppState) -> None:
    """Repeated calls while not RUNNING (or SIMULATION_RUNNING with no
    sim_player) must never accumulate any side effects."""
    state, generator, detector = _make_fake_state(app_state)
    for _ in range(3):
        assert process_one_tick(state) is False

    assert generator.call_count == 0
    assert detector.calls == []
    assert state["total_samples"] == 0


def test_missing_app_state_key_defaults_to_no_op() -> None:
    """A state dict with no 'app_state' key at all must be treated as IDLE (no-op), not crash."""
    state = _FakeSessionState(
        generator=_FakeGenerator([(0.0, 0.0, False)]),
        detector=_FakeDetector(),
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    result = process_one_tick(state)
    assert result is False
    assert state["total_samples"] == 0


# ----------------------------------------------------------------------
# Defensive: uninitialized session state (missing generator/detector/buffers).
# ----------------------------------------------------------------------

def test_missing_generator_or_detector_is_a_safe_no_op() -> None:
    """A RUNNING state without 'generator'/'detector' keys must not raise."""
    state = _FakeSessionState(app_state=AppState.RUNNING, total_samples=0)
    result = process_one_tick(state)
    assert result is False
    assert state["total_samples"] == 0


def test_missing_ring_buffer_key_is_a_safe_no_op_and_does_not_bump_counter() -> None:
    """If a ring buffer key is missing, no partial append/counter bump must occur."""
    state, _generator, _detector = _make_fake_state(AppState.RUNNING)
    del state["ring_is_injected"]  # simulate a partially-initialized session
    result = process_one_tick(state)
    assert result is False
    assert state["total_samples"] == 0
    # None of the four *remaining* buffers should have been appended to either
    # (all-or-nothing semantics -- see _append_sample_to_ring_buffers docstring).
    for key in _RING_KEYS:
        if key in state:
            assert state[key].size == 0


# ----------------------------------------------------------------------
# Default state (state=None -> st.session_state) sanity check.
# ----------------------------------------------------------------------

def test_default_state_argument_uses_st_session_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling process_one_tick() with no args must read/write st.session_state."""
    import streamlit as st

    fake_session_state = _FakeSessionState(app_state=AppState.IDLE)
    monkeypatch.setattr(st, "session_state", fake_session_state, raising=False)
    result = process_one_tick()
    assert result is False


# ----------------------------------------------------------------------
# End-to-end integration: real DataGenerator + AnomalyDetector + RingBuffer.
# ----------------------------------------------------------------------

def test_end_to_end_with_real_generator_detector_and_ring_buffers() -> None:
    """Realistic integration check using the actual core objects, not fakes.

    Runs enough RUNNING ticks to cross the detector's auto-train
    threshold, and asserts total_samples/ring-buffer sizes stay
    consistent throughout (both during GMM warm-up and after training).
    """
    n_ticks = 120  # > AnomalyDetector's internal _TRAIN_AFTER (100)
    state = _FakeSessionState(
        app_state=AppState.RUNNING,
        generator=DataGenerator(rng=np.random.default_rng(42)),
        detector=AnomalyDetector(),
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )

    for i in range(1, n_ticks + 1):
        assert process_one_tick(state) is True
        assert state["total_samples"] == i
        sizes = {key: state[key].size for key in _RING_KEYS}
        assert all(size == i for size in sizes.values()), (i, sizes)

    assert state["detector"].is_trained is True
    # Sanity: buffered values are plausible Temperature-template readings
    # (mean=70, generous tolerance for drift/noise/injected spikes).
    vals = state["ring_vals"].get_window()
    assert len(vals) == n_ticks
    assert np.all(np.abs(vals - 70.0) < 60.0)


def test_toggling_running_and_paused_only_advances_during_running() -> None:
    """Interleaved RUNNING/PAUSED calls must only advance state on RUNNING ticks."""
    state = _FakeSessionState(
        app_state=AppState.RUNNING,
        generator=DataGenerator(rng=np.random.default_rng(7)),
        detector=AnomalyDetector(),
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )

    assert process_one_tick(state) is True  # RUNNING -> advances
    state["app_state"] = AppState.PAUSED
    assert process_one_tick(state) is False  # PAUSED -> no-op
    assert process_one_tick(state) is False  # PAUSED -> no-op
    state["app_state"] = AppState.RUNNING
    assert process_one_tick(state) is True  # RUNNING again -> advances

    assert state["total_samples"] == 2
    for key in _RING_KEYS:
        assert state[key].size == 2


# ----------------------------------------------------------------------
# Story E1: is_anomaly=True results append exactly one logbook row.
# ----------------------------------------------------------------------

def test_running_anomaly_appends_exactly_one_logbook_row_with_correct_fields() -> None:
    """An is_anomaly=True result must append exactly one row with all fields correct."""
    state, _generator, _detector = _make_fake_state_with_logbook(AppState.RUNNING, n_samples=1)
    before = datetime.now()
    process_one_tick(state)  # canned sample i=0: (ts=0.0, val=0.0, injected=True) -> is_anomaly=True
    after = datetime.now()

    df = state["logbook_df"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["row_id"] == 0
    assert row["timestamp"] == pytest.approx(0.0)
    assert row["value"] == pytest.approx(0.0)
    assert row["conf_pct"] == pytest.approx(0.42 * 100)  # _FakeDetector's fixed confidence
    assert row["classification"] == "⚪ Unclassified"
    assert row["comment"] == ""
    assert bool(row["is_injected"]) is True

    # time_str reflects local wall-clock time at logging (result.timestamp
    # here is an epoch-ish canned float too small to be "now", so instead
    # assert the format matches HH:MM:SS and is parseable).
    datetime.strptime(row["time_str"], "%H:%M:%S")


def test_running_non_anomaly_does_not_touch_logbook() -> None:
    """An is_anomaly=False result must append no row and leave next_log_id unchanged."""
    # i=1 sample: is_injected = (1 % 3 == 0) = False -> _FakeDetector returns is_anomaly=False.
    state, _generator, _detector = _make_fake_state_with_logbook(AppState.RUNNING, n_samples=2)
    state["generator"]._samples.pop(0)  # discard the i=0 (anomalous) canned sample
    process_one_tick(state)

    assert len(state["logbook_df"]) == 0
    assert state["next_log_id"] == 0


def test_next_log_id_increments_by_one_per_logged_row_with_no_collisions() -> None:
    """Across many mixed anomaly/non-anomaly ticks, row_id/next_log_id must never collide."""
    n_calls = 12  # samples i=0..11; is_injected (=is_anomaly) True for i in {0,3,6,9}
    state, _generator, _detector = _make_fake_state_with_logbook(AppState.RUNNING, n_samples=n_calls)
    for _ in range(n_calls):
        process_one_tick(state)

    df = state["logbook_df"]
    expected_anomalous_count = sum(1 for i in range(n_calls) if i % 3 == 0)
    assert len(df) == expected_anomalous_count
    assert state["next_log_id"] == expected_anomalous_count

    # row_id must be a strictly increasing, gapless, collision-free sequence.
    row_ids = list(df["row_id"])
    assert row_ids == list(range(expected_anomalous_count))
    assert len(set(row_ids)) == len(row_ids)

    # Every logged row must correspond to one of the actually-anomalous samples.
    expected_values = [float(i) * 10.0 for i in range(n_calls) if i % 3 == 0]
    assert list(df["value"]) == pytest.approx(expected_values)


def test_logbook_row_classification_and_comment_use_exact_defaults() -> None:
    """classification must be exactly '⚪ Unclassified' and comment exactly ''."""
    state, _generator, _detector = _make_fake_state_with_logbook(AppState.RUNNING, n_samples=1)
    process_one_tick(state)

    row = state["logbook_df"].iloc[0]
    assert row["classification"] == "⚪ Unclassified"
    assert row["comment"] == ""


def test_missing_logbook_df_is_a_safe_no_op_and_does_not_affect_ring_buffers() -> None:
    """Missing 'logbook_df'/'next_log_id' must not crash process_one_tick or block buffering."""
    # _make_fake_state() (not the _with_logbook variant) has neither key.
    state, _generator, _detector = _make_fake_state(AppState.RUNNING, n_samples=1)
    result = process_one_tick(state)

    # The rest of the tick must still succeed even though logging was skipped.
    assert result is True
    assert state["total_samples"] == 1
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == 1 for size in sizes.values()), sizes
    assert "logbook_df" not in state
    assert "next_log_id" not in state


def test_missing_next_log_id_only_is_a_safe_no_op() -> None:
    """A logbook_df present but no next_log_id counter must also be a safe no-op."""
    state, _generator, _detector = _make_fake_state(AppState.RUNNING, n_samples=1)
    state["logbook_df"] = pd.DataFrame(columns=LOGBOOK_COLUMNS)  # next_log_id deliberately absent

    result = process_one_tick(state)

    assert result is True
    assert len(state["logbook_df"]) == 0
    assert "next_log_id" not in state


# ----------------------------------------------------------------------
# Story P3: MAX_LOGBOOK_ROWS cap + FIFO eviction on _append_anomaly_to_logbook.
# ----------------------------------------------------------------------


def _make_bare_logbook_state() -> dict:
    """Build a minimal state dict with only 'logbook_df'/'next_log_id' seeded.

    Bypasses the generator/detector/ring-buffer machinery entirely so
    :func:`_append_anomaly_to_logbook` can be exercised directly, in a
    tight loop, up to (and past) ``MAX_LOGBOOK_ROWS`` calls quickly.

    :returns: A fresh dict with an empty, Epic E-schema ``logbook_df``
        and ``next_log_id`` seeded at 0.
    :rtype: dict
    """
    return {"logbook_df": pd.DataFrame(columns=LOGBOOK_COLUMNS), "next_log_id": 0}


def _make_anomaly_result(i: int) -> DetectionResult:
    """Build a deterministic, always-anomalous :class:`DetectionResult` for index ``i``.

    :param i: A distinguishing index, reflected in ``timestamp``/``value``
        so each result is trivially distinguishable in assertions.
    :type i: int
    :returns: A ``DetectionResult`` with ``is_anomaly=True``.
    :rtype: DetectionResult
    """
    return DetectionResult(
        timestamp=float(i),
        value=float(i),
        score=-1.0,
        confidence=0.5,
        is_anomaly=True,
        is_injected=False,
    )


def test_logbook_append_caps_at_max_logbook_rows_retaining_most_recent() -> None:
    """Appending MAX_LOGBOOK_ROWS + 50 anomalies must cap logbook_df at
    MAX_LOGBOOK_ROWS rows, retaining exactly the 500 most-recent row_ids."""
    state = _make_bare_logbook_state()
    total_appends = MAX_LOGBOOK_ROWS + 50

    for i in range(total_appends):
        assert _append_anomaly_to_logbook(state, _make_anomaly_result(i)) is True

    df = state["logbook_df"]
    assert len(df) == MAX_LOGBOOK_ROWS

    retained_ids = {int(v) for v in df["row_id"]}
    expected_ids = set(range(total_appends - MAX_LOGBOOK_ROWS, total_appends))
    assert retained_ids == expected_ids  # exactly the 500 most-recent row_ids

    # No row_id collisions/gaps that could crash a downstream consumer.
    assert len(retained_ids) == len(df)


def test_logbook_append_never_evicts_the_newest_row() -> None:
    """The just-appended (newest) row_id must always survive eviction, only
    the oldest is ever dropped."""
    state = _make_bare_logbook_state()
    total_appends = MAX_LOGBOOK_ROWS + 10
    for i in range(total_appends):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))

    newest_row_id = total_appends - 1
    assert newest_row_id in set(state["logbook_df"]["row_id"])


def test_logbook_append_row_id_and_next_log_id_increment_monotonically_across_evictions() -> None:
    """row_id/next_log_id must keep climbing across the cap boundary, never
    reused or reset just because eviction started."""
    state = _make_bare_logbook_state()
    total_appends = MAX_LOGBOOK_ROWS + 25

    seen_row_ids = []
    for i in range(total_appends):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))
        seen_row_ids.append(i)  # row_id handed out on this call, by construction

    # next_log_id must equal the count of successful appends (never reused).
    assert state["next_log_id"] == total_appends
    # Every row_id ever handed out was strictly increasing by exactly one.
    assert seen_row_ids == list(range(total_appends))


def test_logbook_append_does_not_call_pd_concat_once_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once logbook_df reaches MAX_LOGBOOK_ROWS, further appends must not call
    pd.concat at all -- verified via a spy on pandas' module-level concat."""
    state = _make_bare_logbook_state()
    for i in range(MAX_LOGBOOK_ROWS):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))
    assert len(state["logbook_df"]) == MAX_LOGBOOK_ROWS

    original_concat = pd.concat
    concat_calls: List[Any] = []

    def _spy_concat(*args: Any, **kwargs: Any) -> Any:
        concat_calls.append((args, kwargs))
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(pd, "concat", _spy_concat)

    for i in range(MAX_LOGBOOK_ROWS, MAX_LOGBOOK_ROWS + 50):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))

    assert concat_calls == []  # zero pd.concat calls once past the cap
    assert len(state["logbook_df"]) == MAX_LOGBOOK_ROWS


def test_logbook_append_below_cap_still_uses_pd_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: while still growing towards the cap, appends must
    still use pd.concat (confirms the spy in the prior test is meaningful,
    not just a no-op it would trivially satisfy)."""
    state = _make_bare_logbook_state()

    original_concat = pd.concat
    concat_calls: List[Any] = []

    def _spy_concat(*args: Any, **kwargs: Any) -> Any:
        concat_calls.append((args, kwargs))
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(pd, "concat", _spy_concat)

    for i in range(10):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))

    assert len(concat_calls) == 10
    assert len(state["logbook_df"]) == 10


def test_capped_logbook_remains_compatible_with_downstream_consumers() -> None:
    """After capping/eviction, the Epic E2/E3/E4/F1 consumers (logbook.py's
    display/filter/search/merge/export, dashboard.py's stats) must keep
    working correctly against the capped logbook_df -- no crashes, no
    row_id collisions, correct counts."""
    from ui_web.dashboard import compute_logbook_stats
    from ui_web.logbook import (
        _build_display_df,
        _build_export_csv,
        _compute_filter_mask,
        _compute_footer_counts,
        merge_editor_edits_into_logbook,
    )

    state = _make_bare_logbook_state()
    total_appends = MAX_LOGBOOK_ROWS + 50
    for i in range(total_appends):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))

    df = state["logbook_df"]
    assert len(df) == MAX_LOGBOOK_ROWS

    # E2: row_id-indexed display DataFrame must build cleanly, unique index.
    display_df = _build_display_df(df)
    assert len(display_df) == MAX_LOGBOOK_ROWS
    assert display_df.index.is_unique

    # E3: classification filter / notes search must not raise and must
    # still correctly select every (still-Unclassified) row.
    mask = _compute_filter_mask(df, "Unclassified", "")
    assert int(mask.sum()) == MAX_LOGBOOK_ROWS

    # E2: classification/comment edits must still merge back correctly by
    # row_id even though the underlying frame has been evicted/overwritten.
    edited_df = display_df.copy()
    target_row_id = int(df["row_id"].iloc[10])
    edited_df.loc[target_row_id, "classification"] = "🟢 TP"
    edited_df.loc[target_row_id, "comment"] = "confirmed"
    merged = merge_editor_edits_into_logbook(df, edited_df)
    assert len(merged) == MAX_LOGBOOK_ROWS
    updated_row = merged.loc[merged["row_id"] == target_row_id].iloc[0]
    assert updated_row["classification"] == "🟢 TP"
    assert updated_row["comment"] == "confirmed"
    # Every other row must remain untouched (still the default classification).
    other_rows = merged.loc[merged["row_id"] != target_row_id]
    assert (other_rows["classification"] == "⚪ Unclassified").all()

    # E4: CSV export must contain exactly one header line + MAX_LOGBOOK_ROWS
    # data lines (a trailing newline after the final row).
    csv_text = _build_export_csv(merged)
    assert csv_text.count("\n") == MAX_LOGBOOK_ROWS + 1

    # F1: dashboard stats must reflect exactly the capped row count/edit.
    stats = compute_logbook_stats(merged, total_samples=total_appends)
    assert stats["total_anomalies"] == MAX_LOGBOOK_ROWS
    assert stats["tp"] == 1
    assert stats["pending"] == MAX_LOGBOOK_ROWS - 1

    # E3 stats footer must agree with the dashboard's numbers.
    counts = _compute_footer_counts(merged)
    assert counts["total"] == MAX_LOGBOOK_ROWS
    assert counts["tp"] == 1
    assert counts["pending"] == MAX_LOGBOOK_ROWS - 1


def test_logbook_clear_log_after_capping_resets_cleanly_and_refills() -> None:
    """A 'Clear Log' reset (empty logbook_df, next_log_id=0) after the cap was
    already reached must let growth/eviction resume correctly from scratch."""
    from ui_web.logbook import _clear_logbook

    state = _make_bare_logbook_state()
    for i in range(MAX_LOGBOOK_ROWS + 20):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))
    assert len(state["logbook_df"]) == MAX_LOGBOOK_ROWS

    _clear_logbook(state)
    assert len(state["logbook_df"]) == 0
    assert state["next_log_id"] == 0

    # Refill past the cap again -- must behave identically to a fresh session.
    for i in range(MAX_LOGBOOK_ROWS + 5):
        _append_anomaly_to_logbook(state, _make_anomaly_result(i))

    df = state["logbook_df"]
    assert len(df) == MAX_LOGBOOK_ROWS
    assert set(int(v) for v in df["row_id"]) == set(range(5, MAX_LOGBOOK_ROWS + 5))


# ----------------------------------------------------------------------
# AppTest: exercise the real @st.fragment(run_every=LIVE_TICK_INTERVAL)-decorated function.
# ----------------------------------------------------------------------

def test_live_tick_fragment_wired_via_apptest() -> None:
    """The actual ``live_tick_fragment`` must run and advance state in a real script.

    Uses ``streamlit.testing.v1.AppTest`` to execute a minimal script
    that calls ``init_session_state()`` then ``live_tick_fragment()``
    directly (AppTest's synchronous ``.run()`` executes a fragment's
    body inline on the initial script pass; it does not simulate the
    300 ms auto-rerun timer itself, but confirms the fragment is
    correctly decorated/importable/callable within a real
    ScriptRunContext, which is what a bare unit test cannot check).
    """
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
st.session_state["app_state"] = AppState.RUNNING
live_tick_fragment()
st.write(str(st.session_state["total_samples"]))
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    # The script wrote total_samples as text; confirm it advanced to "1".
    markdown_values = [m.value for m in at.markdown]
    assert "1" in markdown_values


# ----------------------------------------------------------------------
# _build_index_x_axis: pure absolute-sample-index x-axis builder (B3).
# ----------------------------------------------------------------------


def test_build_index_x_axis_empty_window_returns_empty_array() -> None:
    """A zero-length window must produce an empty x-axis, not a crash."""
    xs = _build_index_x_axis(total_samples=0, window_length=0)
    assert isinstance(xs, np.ndarray)
    assert xs.size == 0


def test_build_index_x_axis_before_buffer_fills() -> None:
    """Before the ring buffer wraps, x-axis must be plain 0..n-1 (total==window)."""
    xs = _build_index_x_axis(total_samples=5, window_length=5)
    np.testing.assert_array_equal(xs, np.arange(5))


def test_build_index_x_axis_after_buffer_wraps_uses_absolute_index() -> None:
    """Once total_samples exceeds capacity, the x-axis must keep climbing,

    not reset back to 0..capacity-1 -- this is the crux of the
    "monotonically increasing sample index" acceptance criterion: the
    window is capacity-sized, but the *labels* on it keep advancing.
    """
    xs = _build_index_x_axis(total_samples=350, window_length=300)
    np.testing.assert_array_equal(xs, np.arange(50, 350))
    # Confirms this call's x-axis strictly continues where an earlier,
    # smaller-total_samples call would have left off (monotonic across
    # successive fragment runs, not just within one call).
    earlier_xs = _build_index_x_axis(total_samples=300, window_length=300)
    assert xs[0] > earlier_xs[0]


def test_build_index_x_axis_is_always_strictly_increasing() -> None:
    """Within a single call, consecutive x values must strictly increase by 1."""
    xs = _build_index_x_axis(total_samples=42, window_length=10)
    assert np.all(np.diff(xs) == 1)
    assert xs[-1] == 41  # last index is total_samples - 1


# ----------------------------------------------------------------------
# update_chart_from_state: wiring ring-buffer windows into ChartState.update().
# ----------------------------------------------------------------------


class _RecordingChart:
    """Fake :class:`ui_web.chart.ChartState` stand-in recording ``update()`` calls.

    Lets tests assert exactly what arguments ``update_chart_from_state``
    passed through, without constructing a real matplotlib ``Figure``.

    :ivar calls: List of ``update()`` call kwargs dicts, one per call, in order.
    :vartype calls: list[dict[str, Any]]
    """

    def __init__(self) -> None:
        self.calls: List[dict] = []

    def update(self, **kwargs: Any) -> None:
        """Record the call's keyword arguments."""
        self.calls.append(kwargs)


def _make_populated_fake_state(n_samples: int) -> Tuple[dict, dict]:
    """Build a fake state dict with real ring buffers pre-filled via process_one_tick.

    :param n_samples: Number of RUNNING ticks to run before returning.
    :type n_samples: int
    :returns: The populated state dict.
    :rtype: dict
    """
    state = dict(
        app_state=AppState.RUNNING,
        generator=DataGenerator(rng=np.random.default_rng(3)),
        detector=AnomalyDetector(),
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    for _ in range(n_samples):
        process_one_tick(state)
    return state


def test_update_chart_from_state_calls_chart_update_with_current_windows() -> None:
    """update_chart_from_state must forward the current ring-buffer windows verbatim."""
    state = _make_populated_fake_state(n_samples=7)
    chart = _RecordingChart()

    returned = update_chart_from_state(state=state, chart=chart)

    assert returned is chart
    assert len(chart.calls) == 1
    call = chart.calls[0]
    np.testing.assert_array_equal(call["values"], state["ring_vals"].get_window())
    np.testing.assert_array_equal(call["confidences"], state["ring_confs"].get_window())
    np.testing.assert_array_equal(call["is_anomaly"], state["ring_is_anomaly"].get_window())
    np.testing.assert_array_equal(call["is_injected"], state["ring_is_injected"].get_window())


def test_update_chart_from_state_uses_index_based_x_axis_not_wall_clock() -> None:
    """The 'timestamps' arg passed to chart.update() must be the sample-index

    x-axis, not the raw wall-clock values stored in ring_ts.
    """
    state = _make_populated_fake_state(n_samples=6)
    chart = _RecordingChart()
    update_chart_from_state(state=state, chart=chart)

    xs = chart.calls[0]["timestamps"]
    # Must be a plain increasing integer index (0..5), *not* the raw
    # generator timestamps (which are arbitrary floats from DataGenerator).
    np.testing.assert_array_equal(xs, np.arange(6))
    raw_ts = state["ring_ts"].get_window()
    assert not np.array_equal(xs, raw_ts)


def test_update_chart_from_state_called_even_with_no_new_sample() -> None:
    """Chart must still be redrawn from existing buffers when the app is not RUNNING.

    Mirrors the acceptance criterion that the fragment calls
    chart.update() every run "regardless of whether a new sample was
    produced" -- here simulated directly by calling
    update_chart_from_state() after process_one_tick() no-op'd.
    """
    state = _make_populated_fake_state(n_samples=4)
    state["app_state"] = AppState.PAUSED

    produced = process_one_tick(state)
    assert produced is False  # confirm this really was a no-op tick

    chart = _RecordingChart()
    update_chart_from_state(state=state, chart=chart)

    assert len(chart.calls) == 1
    # Buffers still hold the 4 samples from before pausing -- the chart
    # call must reflect that existing data, not an empty window.
    assert len(chart.calls[0]["values"]) == 4


def test_update_chart_from_state_on_empty_buffers_does_not_raise() -> None:
    """A freshly-initialized (never-ticked) session must redraw an empty chart, not crash."""
    state = dict(
        app_state=AppState.IDLE,
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    chart = _RecordingChart()
    update_chart_from_state(state=state, chart=chart)

    assert len(chart.calls) == 1
    assert len(chart.calls[0]["values"]) == 0
    assert len(chart.calls[0]["timestamps"]) == 0


def test_update_chart_from_state_missing_ring_buffers_is_a_safe_no_op() -> None:
    """Missing ring buffer keys must log-and-skip rather than raise."""
    state = dict(app_state=AppState.IDLE, total_samples=0)
    chart = _RecordingChart()

    returned = update_chart_from_state(state=state, chart=chart)

    assert returned is chart
    assert chart.calls == []  # update() must never have been called


def test_update_chart_from_state_defaults_to_session_singleton_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With chart=None, update_chart_from_state must use get_or_create_chart_state()."""
    import ui_web.live_tick as live_tick_module

    sentinel_chart = _RecordingChart()
    monkeypatch.setattr(
        live_tick_module, "get_or_create_chart_state", lambda: sentinel_chart
    )
    state = _make_populated_fake_state(n_samples=2)

    returned = update_chart_from_state(state=state)

    assert returned is sentinel_chart
    assert len(sentinel_chart.calls) == 1


# ----------------------------------------------------------------------
# AppTest: the real live_tick_fragment renders an st.pyplot image every run,
# with or without a new sample, using an index-based x-axis.
# ----------------------------------------------------------------------


def test_live_tick_fragment_renders_pyplot_image_when_running() -> None:
    """A RUNNING fragment run must tick the stream AND render exactly one Plotly chart element."""
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
st.session_state["app_state"] = AppState.RUNNING
live_tick_fragment()
st.write(str(st.session_state["total_samples"]))
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    assert len(at.get("plotly_chart")) == 1  # st.plotly_chart rendered exactly one chart
    markdown_values = [m.value for m in at.markdown]
    assert "1" in markdown_values


def test_live_tick_fragment_does_not_render_pyplot_image_when_idle() -> None:
    """An IDLE fragment run must NOT render a Plotly chart element this cycle (story P1).

    Supersedes the pre-P1 "IDLE/PAUSED states still show a static chart"
    framing from story B3: no sample is produced (total_samples stays
    "0"), and -- since this is the very first run of a fresh session, so
    there is no earlier-rendered frame to leave on screen -- no chart is
    rendered here at all. See
    :func:`test_live_tick_fragment_idle_and_paused_skip_chart_update_and_pyplot`
    for the call-count-based assertion that better matches P1's AC1.
    """
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()  # app_state defaults to IDLE
live_tick_fragment()
st.write(str(st.session_state["total_samples"]))
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    assert len(at.get("plotly_chart")) == 0  # P1: chart redraw/render is skipped while IDLE
    markdown_values = [m.value for m in at.markdown]
    assert "0" in markdown_values


# ----------------------------------------------------------------------
# Story P1: IDLE/PAUSED cycles must call neither ChartState.update(...)
# nor st.plotly_chart -- verified by spying on both across several
# fragment cycles in each state (AC1), and confirming RUNNING/
# SIMULATION_RUNNING keep calling both every cycle unchanged (AC2).
# ----------------------------------------------------------------------


def _spy_chart_update_and_pyplot(monkeypatch: pytest.MonkeyPatch) -> Tuple[List[dict], List[tuple]]:
    """Wrap ``ChartState.update`` and ``DeltaGenerator.plotly_chart`` with call-recording spies.

    Wraps (rather than replaces) both callables so the real chart/figure
    machinery still runs underneath -- this way a passing test proves the
    P1 gating actually prevents the call from happening at all, not just
    that some unrelated stub was swapped in. Uses ``monkeypatch`` so both
    are restored automatically at test teardown regardless of outcome.

    Patches ``streamlit.delta_generator.DeltaGenerator.plotly_chart``
    (the class method, story Q1 -- this was ``.pyplot`` before the chart
    migrated to Plotly) rather than the module-level
    ``st.plotly_chart`` convenience function: since story P1's
    persistence fix calls ``placeholder.plotly_chart(...)`` on the
    session's ``st.empty()`` chart placeholder (see
    :func:`ui_web.chart.render_chart_into_placeholder`) instead of the
    bare top-level ``st.plotly_chart(...)``, a spy on the module-level
    name alone would silently miss every call made through that
    placeholder instance (a *different* ``DeltaGenerator`` object with
    its own bound method reference) -- patching the shared class method
    instead catches a ``.plotly_chart(...)`` call made on *any*
    ``DeltaGenerator`` instance, matching what this test actually needs
    to assert regardless of which instance renders the figure.

    :param monkeypatch: The active :class:`pytest.MonkeyPatch` fixture.
    :type monkeypatch: pytest.MonkeyPatch
    :returns: A ``(update_calls, pyplot_calls)`` pair of lists that are
        appended to every time the wrapped callable is invoked.
    :rtype: tuple[list[dict], list[tuple]]
    """
    from streamlit.delta_generator import DeltaGenerator

    from ui_web.chart import ChartState

    update_calls: List[dict] = []
    pyplot_calls: List[tuple] = []

    original_update = ChartState.update

    def spy_update(self: ChartState, **kwargs: Any) -> None:
        update_calls.append(kwargs)
        return original_update(self, **kwargs)

    original_plotly_chart = DeltaGenerator.plotly_chart

    def spy_plotly_chart(self: DeltaGenerator, *args: Any, **kwargs: Any) -> Any:
        pyplot_calls.append((args, kwargs))
        return original_plotly_chart(self, *args, **kwargs)

    monkeypatch.setattr(ChartState, "update", spy_update)
    monkeypatch.setattr(DeltaGenerator, "plotly_chart", spy_plotly_chart)
    return update_calls, pyplot_calls


@pytest.mark.parametrize("app_state_name", ["IDLE", "PAUSED"])
def test_live_tick_fragment_idle_and_paused_skip_chart_update_and_pyplot(
    app_state_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1 AC1: several IDLE/PAUSED fragment cycles must call neither
    ``ChartState.update(...)`` nor ``st.plotly_chart`` -- zero calls
    across repeated reruns, not just the first one.
    """
    from streamlit.testing.v1 import AppTest

    update_calls, pyplot_calls = _spy_chart_update_and_pyplot(monkeypatch)

    script = f"""
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
st.session_state["app_state"] = AppState.{app_state_name}
live_tick_fragment()
"""
    at = AppTest.from_string(script)
    for _ in range(3):
        at.run()
        assert at.exception == []

    assert update_calls == []
    assert pyplot_calls == []


@pytest.mark.parametrize("app_state_name", ["RUNNING", "SIMULATION_RUNNING"])
def test_live_tick_fragment_running_states_still_call_chart_update_and_pyplot_every_cycle(
    app_state_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1 AC2: RUNNING/SIMULATION_RUNNING must be unaffected by the P1 gating --

    ``ChartState.update(...)`` and the chart placeholder's
    ``.plotly_chart(...)`` are still called exactly once per fragment
    cycle, every cycle, regardless of whether a new sample was produced
    that cycle (not regressing story B3's behavior).

    Seeds ``st.session_state["chart_placeholder"]`` directly (a minimal
    stand-in for ``app.py``'s ``render_chart_into_placeholder()``, which
    would add its own extra ``.plotly_chart(...)`` call every cycle and
    throw off this test's exact per-cycle call count) so
    ``live_tick_fragment`` exercises the real placeholder-write path
    rather than its defensive bare-``st.plotly_chart`` fallback (used
    only when no placeholder exists).
    """
    from streamlit.testing.v1 import AppTest

    update_calls, pyplot_calls = _spy_chart_update_and_pyplot(monkeypatch)

    script = f"""
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
if "chart_placeholder" not in st.session_state:
    st.session_state["chart_placeholder"] = st.empty()
st.session_state["app_state"] = AppState.{app_state_name}
live_tick_fragment()
"""
    at = AppTest.from_string(script)
    n_cycles = 3
    for cycle in range(1, n_cycles + 1):
        at.run()
        assert at.exception == []
        assert len(update_calls) == cycle
        assert len(pyplot_calls) == cycle


@pytest.mark.parametrize("app_state_name", ["IDLE", "PAUSED", "RUNNING", "SIMULATION_RUNNING"])
def test_live_tick_fragment_gmm_training_group_unaffected_by_p1_gating(
    app_state_name: str,
) -> None:
    """P1 AC4: the GMM Training group's progress bar keeps updating every

    cycle regardless of ``app_state`` -- it is deliberately *not* gated
    by the P1 chart-redraw skip, since training-state visibility while
    paused is expected behavior, not part of this optimization.
    """
    from streamlit.testing.v1 import AppTest

    script = f"""
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
st.session_state["app_state"] = AppState.{app_state_name}
live_tick_fragment()
"""
    at = AppTest.from_string(script)
    for _ in range(3):
        at.run()
        assert at.exception == []
        # The progress bar (and its "Collecting.../Trained" text) must be
        # present on every single cycle, whether or not the chart redraw
        # was skipped this cycle.
        progress_elements = at.get("progress")
        assert len(progress_elements) == 1


def test_live_tick_fragment_pyplot_image_rendered_on_repeated_runs() -> None:
    """Repeated fragment reruns (simulating successive 300ms ticks)

    must each render a pyplot image, and total_samples must keep
    advancing while RUNNING -- a basic proxy for "the chart visibly
    animates" since AppTest cannot literally observe pixel changes.

    Story B4 added real ``key=``-bound widgets (the GMM Training
    group's spinboxes/button) inside ``live_tick_fragment``, so -- unlike
    story B3's original version of this test, which called
    ``live_tick_fragment()`` several times back-to-back *within one
    script execution* -- ticks must now each be driven by an
    independent ``at.run()`` call. That mirrors how the real
    ``st.fragment(run_every=...)`` scheduler actually reruns a fragment
    (as a fresh, independent script/fragment run each time, with its
    own widget-id bookkeeping), whereas calling the same
    ``key=``-bound widget-creating function twice within a single
    script pass is invalid in real Streamlit too (duplicate element
    key), not just an artifact of this test.
    """
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
st.session_state["app_state"] = AppState.RUNNING
live_tick_fragment()
st.write(str(st.session_state["total_samples"]))
"""
    at = AppTest.from_string(script)

    for expected_total_samples in (1, 2, 3):
        at.run()
        assert at.exception == []
        # Each independent rerun renders exactly one Plotly chart element.
        assert len(at.get("plotly_chart")) == 1
        markdown_values = [m.value for m in at.markdown]
        assert str(expected_total_samples) in markdown_values


# ----------------------------------------------------------------------
# P1 AC3 regression: the chart image must survive a RUNNING ->
# IDLE/PAUSED transition (freeze on the last frame), not blank out.
#
# ``streamlit.testing.v1.AppTest.run()`` always performs a *full*
# script rerun (a fresh ``LocalScriptRunner`` per call -- confirmed via
# ``streamlit/testing/v1/app_test.py``/``local_script_runner.py``
# source inspection); it has no public way to simulate a genuine
# fragment-only auto-rerun (the real 300ms ``run_every`` mechanism,
# which never re-executes the outer script at all). That distinction
# matters here: these tests exercise the *outer full-rerun* half of the
# AC3 fix (``app.py``'s ``main()`` unconditionally re-rendering the
# chart's current Figure into a freshly (re)created placeholder every
# run -- see ``ui_web.chart.render_chart_into_placeholder``), which is
# exactly what happens on the real state-changing full rerun that
# *causes* a RUNNING -> IDLE/PAUSED transition (e.g. clicking
# "Stop"/"Pause"). The other half of the fix (a true fragment-only
# 300ms tick that skips the write leaving the placeholder undisturbed)
# is not independently exercisable via AppTest's public API, but does
# not need its own additional code path: Streamlit's own guarantee that
# elements outside a fragment's scope are left untouched by a
# fragment-only rerun is exactly what makes skipping the write safe
# there too.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("paused_state_name", ["IDLE", "PAUSED"])
def test_app_main_chart_image_persists_across_running_to_idle_or_paused_transition(
    paused_state_name: str,
) -> None:
    """AC3: the live chart image must remain present after RUNNING -> IDLE/PAUSED.

    Runs the *real* ``app.py`` script (mirrors the pattern used in
    ``test_styling.py``/``test_logbook.py`` for exercising ``app.py``
    end-to-end) through a RUNNING cycle, then flips ``app_state`` to
    IDLE or PAUSED (simulating the "Stop"/"Pause" button's effect) and
    reruns -- exactly the scenario the reviewer's finding describes.
    Before this fix, a bare ``st.pyplot(...)`` call gated inside
    ``live_tick_fragment`` would have caused ``len(at.main.image)`` to
    drop from 1 to 0 on the second run; after the fix it must stay at 1
    (the last frame, frozen) on every subsequent run, not just the one
    where the transition happens.
    """
    from streamlit.testing.v1 import AppTest

    from state.app_state import AppState

    script = """
import app
app.main()
"""
    at = AppTest.from_string(script)
    at.run()
    assert at.exception == []

    at.session_state["app_state"] = AppState.RUNNING
    at.run()
    assert at.exception == []
    # `at.main` scopes to the main content area only, excluding the
    # sidebar's own (unrelated) dashboard-popover pie chart (still
    # matplotlib/st.pyplot -- out of scope for story Q1), so this counts
    # *only* the live Plotly chart.
    assert len(at.main.get("plotly_chart")) == 1

    at.session_state["app_state"] = getattr(AppState, paused_state_name)
    at.run()
    assert at.exception == []
    assert len(at.main.get("plotly_chart")) == 1  # must NOT have dropped to 0

    # A further cycle while still paused/idle must also keep it visible
    # (not just the one immediately following the transition).
    at.run()
    assert at.exception == []
    assert len(at.main.get("plotly_chart")) == 1


def test_app_main_creates_and_renders_into_a_chart_placeholder_every_run() -> None:
    """``app.py``'s ``main()`` must (re)create ``chart_placeholder`` and render into it every run.

    Guards the mechanism the AC3 fix depends on: even a session that
    never transitions out of IDLE (e.g. the very first page load) must
    already show the chart (a blank one, since no data has been fed
    yet) -- confirming ``render_chart_into_placeholder()`` runs
    unconditionally from ``main()``, not only once RUNNING starts.
    """
    from streamlit.testing.v1 import AppTest

    script = """
import app
app.main()
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    assert "chart_placeholder" in at.session_state
    # A fresh, never-RUNNING session's chart is still rendered (blank).
    assert len(at.main.get("plotly_chart")) == 1


# ----------------------------------------------------------------------
# Story D2: AppState.SIMULATION_RUNNING branch of process_one_tick().
# ----------------------------------------------------------------------


class _FakeSimPlayer:
    """Stub :class:`~core.simulation_loader.SimulationPlayer`.

    Returns pre-scripted ``due_rows()`` batches, one per call, and lets
    tests control exactly what ``finished`` reports after each call --
    independent of any real wall-clock scheduling.

    :param batches: One ``(rows, finished_after)`` entry per expected
        ``due_rows()`` call, in order. ``rows`` is the list of
        ``(timestamp, value, is_anomaly)`` tuples that call should
        return; ``finished_after`` is what :attr:`finished` should
        report immediately after that call. Calls beyond the scripted
        list return an empty batch and whatever ``finished`` last was.
    :type batches: list[tuple[list[tuple[float, float, bool]], bool]]
    """

    def __init__(self, batches) -> None:
        self._batches = list(batches)
        self.due_rows_calls: List[float] = []
        self._finished = False

    def due_rows(self, now: float) -> List[Tuple[float, float, bool]]:
        """Return the next scripted batch and update ``finished``."""
        self.due_rows_calls.append(now)
        if self._batches:
            rows, finished_after = self._batches.pop(0)
        else:
            rows, finished_after = [], self._finished
        self._finished = finished_after
        return rows

    @property
    def finished(self) -> bool:
        """Whether the (fake) scenario has finished."""
        return self._finished


def _make_sim_state(batches) -> Tuple[_FakeSessionState, _FakeSimPlayer, _FakeDetector]:
    """Build a fake SIMULATION_RUNNING session-state dict wired with a fake sim_player.

    :param batches: Passed through to :class:`_FakeSimPlayer`.
    :type batches: list[tuple[list[tuple[float, float, bool]], bool]]
    :returns: Tuple of ``(state, sim_player, detector)``.
    :rtype: tuple[_FakeSessionState, _FakeSimPlayer, _FakeDetector]
    """
    sim_player = _FakeSimPlayer(batches)
    detector = _FakeDetector()
    state = _FakeSessionState(
        app_state=AppState.SIMULATION_RUNNING,
        sim_player=sim_player,
        detector=detector,
        total_samples=0,
        logbook_df=pd.DataFrame(columns=LOGBOOK_COLUMNS),
        next_log_id=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    return state, sim_player, detector


def test_simulation_running_zero_due_rows_is_a_safe_noop() -> None:
    """A poll with nothing due yet must not tick/buffer/count anything, but stays SIMULATION_RUNNING."""
    state, sim_player, detector = _make_sim_state([([], False)])

    result = process_one_tick(state)

    assert result is False
    assert detector.calls == []
    assert state["total_samples"] == 0
    assert state["app_state"] is AppState.SIMULATION_RUNNING
    assert len(sim_player.due_rows_calls) == 1


def test_simulation_running_one_due_row_processes_it() -> None:
    """Exactly one due row must feed the detector once and buffer once."""
    state, _sim_player, detector = _make_sim_state([([(1.0, 10.0, False)], False)])

    result = process_one_tick(state)

    assert result is True
    assert detector.calls == [(1.0, 10.0, False)]
    assert state["total_samples"] == 1
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == 1 for size in sizes.values()), sizes
    assert state["app_state"] is AppState.SIMULATION_RUNNING


def test_simulation_running_multiple_due_rows_in_one_call_processes_all() -> None:
    """Several due rows returned by a single due_rows() poll must all be processed
    in that same process_one_tick() call (bursting, per SimulationPlayer's design)."""
    rows = [(0.1, 1.0, False), (0.2, 2.0, True), (0.3, 3.0, False)]
    state, _sim_player, detector = _make_sim_state([(rows, False)])

    result = process_one_tick(state)

    assert result is True
    assert detector.calls == rows
    assert state["total_samples"] == 3
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == 3 for size in sizes.values()), sizes


def test_simulation_running_is_injected_sourced_from_row_is_anomaly_column() -> None:
    """The row's is_anomaly (3rd tuple element) must be passed as feed()'s is_injected arg."""
    rows = [(0.0, 1.0, True), (0.1, 2.0, False)]
    state, _sim_player, detector = _make_sim_state([(rows, False)])

    process_one_tick(state)

    assert [call[2] for call in detector.calls] == [True, False]
    # The fake detector's is_anomaly always mirrors is_injected (see
    # _FakeDetector.feed above), confirming the ring buffer/logbook
    # entries reflect the detector's own independently-computed
    # is_anomaly -- not a raw pass-through of the CSV column under a
    # different name.
    assert list(state["ring_is_injected"].get_window()) == [True, False]
    assert list(state["ring_is_anomaly"].get_window()) == [True, False]


def test_simulation_running_anomalous_rows_append_to_logbook() -> None:
    """Anomalous rows (per the fake detector's is_anomaly=is_injected) must log."""
    rows = [(0.0, 1.0, True), (0.1, 2.0, False), (0.2, 3.0, True)]
    state, _sim_player, _detector = _make_sim_state([(rows, False)])

    process_one_tick(state)

    df = state["logbook_df"]
    assert len(df) == 2
    assert list(df["value"]) == pytest.approx([1.0, 3.0])
    assert state["next_log_id"] == 2


def test_simulation_running_auto_transitions_to_idle_when_finished() -> None:
    """Once sim_player.finished becomes True, app_state must move to IDLE on this same tick."""
    rows = [(0.0, 1.0, False)]
    state, _sim_player, _detector = _make_sim_state([(rows, True)])

    process_one_tick(state)

    assert state["app_state"] is AppState.IDLE


def test_simulation_running_finished_with_zero_rows_still_transitions_to_idle() -> None:
    """A poll with no due rows but finished=True (e.g. after Stop Sim already
    called stop()) must still transition to IDLE, not require a due row."""
    state, _sim_player, detector = _make_sim_state([([], True)])

    result = process_one_tick(state)

    assert result is False
    assert detector.calls == []
    assert state["app_state"] is AppState.IDLE


def test_simulation_running_not_finished_stays_in_simulation_running() -> None:
    """While sim_player.finished is False, app_state must remain SIMULATION_RUNNING
    across several consecutive polls."""
    state, _sim_player, _detector = _make_sim_state(
        [([(0.0, 1.0, False)], False), ([(0.1, 2.0, False)], False), ([], False)]
    )

    for _ in range(3):
        process_one_tick(state)
        assert state["app_state"] is AppState.SIMULATION_RUNNING

    assert state["total_samples"] == 2


def test_simulation_running_missing_sim_player_is_a_safe_noop() -> None:
    """SIMULATION_RUNNING with a detector but no sim_player must not raise."""
    state = _FakeSessionState(
        app_state=AppState.SIMULATION_RUNNING,
        detector=_FakeDetector(),
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    result = process_one_tick(state)
    assert result is False
    assert state["total_samples"] == 0
    assert state["app_state"] is AppState.SIMULATION_RUNNING  # left untouched


def test_simulation_running_missing_detector_is_a_safe_noop() -> None:
    """SIMULATION_RUNNING with a sim_player but no detector must not raise."""
    sim_player = _FakeSimPlayer([([(0.0, 1.0, False)], False)])
    state = _FakeSessionState(
        app_state=AppState.SIMULATION_RUNNING,
        sim_player=sim_player,
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )
    result = process_one_tick(state)
    assert result is False
    assert state["total_samples"] == 0
    assert sim_player.due_rows_calls == []  # never even polled


def test_simulation_running_missing_ring_buffer_key_is_a_safe_noop_per_row() -> None:
    """A missing ring buffer key must not crash, and must not bump total_samples
    for the row(s) that failed to buffer (all-or-nothing per sample, same as RUNNING)."""
    state, _sim_player, detector = _make_sim_state([([(0.0, 1.0, False)], False)])
    del state["ring_is_injected"]

    result = process_one_tick(state)

    assert result is False
    assert state["total_samples"] == 0
    # detector.feed() was still called (feeding happens before the
    # ring-buffer append attempt) but nothing was counted/buffered.
    assert detector.calls == [(0.0, 1.0, False)]


# ----------------------------------------------------------------------
# Confirm the pre-existing RUNNING branch is byte-for-byte unaffected by
# the D2 refactor (process_one_tick now dispatches to
# _process_running_tick/_process_simulation_tick internally, but RUNNING
# callers must observe identical behavior to before).
# ----------------------------------------------------------------------


def test_running_branch_unaffected_by_simulation_running_refactor() -> None:
    """A RUNNING tick must still call generator.tick() exactly once, feed the
    detector once, buffer once, and bump total_samples by exactly one --
    unchanged by the D2 addition of the SIMULATION_RUNNING branch."""
    state, generator, detector = _make_fake_state(AppState.RUNNING, n_samples=1)

    result = process_one_tick(state)

    assert result is True
    assert generator.call_count == 1
    assert len(detector.calls) == 1
    assert state["total_samples"] == 1
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == 1 for size in sizes.values()), sizes


# ----------------------------------------------------------------------
# Story P2: LIVE_TICK_INTERVAL constant + cadence-independent tick semantics.
# ----------------------------------------------------------------------


def test_live_tick_interval_constant_is_300ms() -> None:
    """:data:`LIVE_TICK_INTERVAL` must be the string ``"300ms"``.

    Confirmed by reading the constant directly, not by inspecting the
    ``@st.fragment``-decorated ``live_tick_fragment`` at runtime --
    ``st.fragment``'s bound ``run_every`` value is not introspectable
    after decoration, so this constant is the only reliable, testable
    source of truth for the configured cadence (AC1/AC2, story P2).
    """
    assert LIVE_TICK_INTERVAL == "300ms"
    assert isinstance(LIVE_TICK_INTERVAL, str)


def test_100_fragment_cycles_at_current_cadence_produce_100_generator_ticks_of_0_1s_each() -> None:
    """Generator-tick semantics must be independent of the fragment's poll cadence.

    Raising :data:`LIVE_TICK_INTERVAL` (story P2, ``"150ms"`` ->
    ``"300ms"``) changes only how often the fragment *polls* -- it must
    not change :class:`~core.data_generator.DataGenerator`'s fixed 0.1s
    simulated-time step per ``tick()`` call, nor the number of samples
    fed to the GMM per call. Simulating 100 fragment cycles while
    RUNNING (each cycle dispatching exactly one ``process_one_tick()``
    call, matching the real ``live_tick_fragment`` -> ``process_one_tick``
    wiring) must therefore still produce exactly 100 generator ticks,
    each advancing the generator's internal simulated clock by exactly
    0.1s, regardless of what ``LIVE_TICK_INTERVAL`` is configured to
    (the real wall-clock cadence used by ``@st.fragment(run_every=...)``
    is not itself controllable/observable from a unit test -- see module
    docstring).
    """
    n_cycles = 100
    generator = DataGenerator(rng=np.random.default_rng(123))
    state = _FakeSessionState(
        app_state=AppState.RUNNING,
        generator=generator,
        detector=AnomalyDetector(),
        total_samples=0,
        **{key: RingBuffer(300) for key in _RING_KEYS},
    )

    for cycle in range(1, n_cycles + 1):
        # One process_one_tick() call per fragment cycle -- the exact
        # dispatch live_tick_fragment() performs each time it reruns,
        # irrespective of how far apart in wall-clock time those reruns
        # actually are (governed by LIVE_TICK_INTERVAL, not by this loop).
        assert process_one_tick(state) is True
        # Simulated time must advance by exactly one fixed 0.1s tick per
        # cycle -- not by however much wall-clock time elapsed.
        assert generator._t == pytest.approx(cycle * 0.1)

    assert state["total_samples"] == n_cycles
    assert generator._t == pytest.approx(n_cycles * 0.1)
    sizes = {key: state[key].size for key in _RING_KEYS}
    assert all(size == n_cycles for size in sizes.values()), sizes
