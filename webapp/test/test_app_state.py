"""Tests for :mod:`state.app_state` (AppState enum, controls matrix, session init)."""
from __future__ import annotations

import pandas as pd
import pytest

from core.anomaly_detector import AnomalyDetector
from core.data_generator import DataGenerator
from core.simulation_loader import SimulationPlayer
from state.app_state import MAX_LOGBOOK_ROWS, LOGBOOK_COLUMNS, AppState, controls_enabled, init_session_state
from utils.ring_buffer import RingBuffer


class _FakeSessionState(dict):
    """Minimal dict-like stand-in for ``st.session_state`` in tests.

    Streamlit's real ``session_state`` is inert/warns outside of a
    running script (no ``ScriptRunContext``), so tests inject this
    plain dict subclass instead via ``init_session_state(state=...)``.
    Attribute access is not needed by :func:`init_session_state`
    (it only uses ``in``/``__getitem__``/``__setitem__``), so a plain
    dict subclass is sufficient.
    """


# ----------------------------------------------------------------------
# AppState enum
# ----------------------------------------------------------------------

def test_app_state_has_exactly_four_members() -> None:
    """AppState must have exactly IDLE/RUNNING/PAUSED/SIMULATION_RUNNING, no TRAINING."""
    members = {m.name for m in AppState}
    assert members == {"IDLE", "RUNNING", "PAUSED", "SIMULATION_RUNNING"}
    assert "TRAINING" not in members


# ----------------------------------------------------------------------
# controls_enabled() matrix
# ----------------------------------------------------------------------

# Exact matrix from the story's acceptance criteria.
_EXPECTED_MATRIX = {
    AppState.IDLE: {
        "config_enabled": True,
        "start_enabled": True,
        "stop_enabled": False,
        "sim_play_enabled": True,
        "sim_stop_enabled": False,
        "reset_enabled": True,
        "retrain_enabled": True,
    },
    AppState.RUNNING: {
        "config_enabled": False,
        "start_enabled": False,
        "stop_enabled": True,
        "sim_play_enabled": True,
        "sim_stop_enabled": False,
        "reset_enabled": True,
        "retrain_enabled": True,
    },
    AppState.PAUSED: {
        "config_enabled": True,
        "start_enabled": True,
        "stop_enabled": False,
        "sim_play_enabled": True,
        "sim_stop_enabled": False,
        "reset_enabled": True,
        "retrain_enabled": True,
    },
    AppState.SIMULATION_RUNNING: {
        "config_enabled": True,
        "start_enabled": False,
        "stop_enabled": False,
        "sim_play_enabled": False,
        "sim_stop_enabled": True,
        "reset_enabled": True,
        "retrain_enabled": True,
    },
}


@pytest.mark.parametrize("state", list(AppState))
def test_controls_enabled_matches_matrix(state: AppState) -> None:
    """controls_enabled() output must match the story's exact matrix for every state."""
    result = controls_enabled(state)
    assert result == _EXPECTED_MATRIX[state]


def test_controls_enabled_returns_expected_keys_only() -> None:
    """The result dict must expose exactly the seven documented keys."""
    result = controls_enabled(AppState.IDLE)
    assert set(result.keys()) == {
        "config_enabled",
        "start_enabled",
        "stop_enabled",
        "sim_play_enabled",
        "sim_stop_enabled",
        "reset_enabled",
        "retrain_enabled",
    }


def test_controls_enabled_rejects_non_appstate_input() -> None:
    """Defensive: passing a non-AppState value must raise TypeError, not fail silently."""
    with pytest.raises(TypeError):
        controls_enabled("RUNNING")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# init_session_state()
# ----------------------------------------------------------------------

def test_init_session_state_creates_core_objects() -> None:
    """First call must instantiate generator/detector/sim_player of the right types."""
    state = _FakeSessionState()
    init_session_state(state)

    assert isinstance(state["generator"], DataGenerator)
    assert isinstance(state["detector"], AnomalyDetector)
    assert isinstance(state["sim_player"], SimulationPlayer)


def test_init_session_state_creates_five_independent_ring_buffers() -> None:
    """Each ring buffer key must be a distinct RingBuffer(300) instance."""
    state = _FakeSessionState()
    init_session_state(state)

    ring_keys = ["ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected"]
    buffers = [state[key] for key in ring_keys]
    for buf in buffers:
        assert isinstance(buf, RingBuffer)
        assert buf.size == 0

    # Must be independent objects, not the same buffer aliased under
    # multiple keys (mutating one must not affect the others).
    assert len({id(buf) for buf in buffers}) == len(buffers)
    buffers[0].append(1.0)
    assert buffers[1].size == 0


def test_init_session_state_sets_counters_and_state() -> None:
    """total_samples starts at 0 and app_state starts at IDLE."""
    state = _FakeSessionState()
    init_session_state(state)

    assert state["total_samples"] == 0
    assert state["app_state"] is AppState.IDLE


def test_init_session_state_creates_logbook_with_epic_e_schema() -> None:
    """logbook_df must be an empty DataFrame with the exact Epic E column schema."""
    state = _FakeSessionState()
    init_session_state(state)

    df = state["logbook_df"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == LOGBOOK_COLUMNS
    assert len(df) == 0


def test_init_session_state_seeds_next_log_id_at_zero() -> None:
    """next_log_id (Epic E1's row-id counter) must default to 0."""
    state = _FakeSessionState()
    init_session_state(state)

    assert state["next_log_id"] == 0


def test_init_session_state_sets_widget_defaults_from_temperature_template() -> None:
    """Widget-default keys must be seeded from the Temperature template exactly."""
    state = _FakeSessionState()
    init_session_state(state)

    assert state["cfg_template"] == "Temperature"
    assert state["cfg_mean"] == 70.0
    assert state["cfg_std"] == 2.0
    assert state["cfg_noise"] == 0.5
    assert state["cfg_anomaly_pct"] == 2
    assert state["cfg_retrain_start"] == 0
    assert state["cfg_retrain_end"] == 0
    assert state["cfg_end_tracking"] is True
    assert state["cfg_scenario"] == "Temperature"
    assert state["cfg_speed"] == 1


def test_init_session_state_is_idempotent_and_preserves_existing_data() -> None:
    """A second call must not reset total_samples, logbook_df, or ring buffers with data."""
    state = _FakeSessionState()
    init_session_state(state)

    # Simulate an in-progress session: bump the counter, log an
    # anomaly row, and push a sample into one ring buffer.
    state["total_samples"] = 42
    state["logbook_df"] = pd.concat(
        [
            state["logbook_df"],
            pd.DataFrame(
                [
                    {
                        "row_id": 1,
                        "timestamp": 123.0,
                        "time_str": "00:00:01",
                        "value": 71.2,
                        "conf_pct": 87.5,
                        "classification": "TP",
                        "comment": "",
                        "is_injected": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    state["ring_ts"].append(1.0)
    state["app_state"] = AppState.RUNNING
    state["next_log_id"] = 2  # matches the one row logged above
    original_generator = state["generator"]
    original_ring_ts = state["ring_ts"]

    # Second call must be a strict no-op for keys that already exist.
    init_session_state(state)

    assert state["total_samples"] == 42
    assert len(state["logbook_df"]) == 1
    assert state["next_log_id"] == 2
    assert state["ring_ts"].size == 1
    assert state["ring_ts"] is original_ring_ts
    assert state["generator"] is original_generator
    assert state["app_state"] is AppState.RUNNING


def test_init_session_state_returns_the_populated_state() -> None:
    """The function should return the same mapping object it was given."""
    state = _FakeSessionState()
    result = init_session_state(state)
    assert result is state


# ----------------------------------------------------------------------
# MAX_LOGBOOK_ROWS (story P3: cap logbook row count)
# ----------------------------------------------------------------------

def test_max_logbook_rows_constant_has_expected_default() -> None:
    """MAX_LOGBOOK_ROWS must be defined as the documented default of 500.

    Co-located with _RING_BUFFER_CAPACITY/LOGBOOK_COLUMNS per the story's
    acceptance criteria, and imported by ui_web.live_tick's append path.
    """
    assert MAX_LOGBOOK_ROWS == 500
    assert isinstance(MAX_LOGBOOK_ROWS, int)
