"""Tests for :mod:`ui_web.sidebar_config`'s C3 Start/Stop/Reset run controls.

Covers the pure, unit-testable callbacks (:func:`_on_start_clicked`,
:func:`_on_stop_clicked`, :func:`_on_reset_clicked`,
:func:`_reset_ring_buffers`) with fakes, plus
:class:`streamlit.testing.v1.AppTest`-based tests that drive the real
Start/Stop/Reset Session buttons (and the Template/Mean/Std/Noise
config inputs' ``disabled=`` state) in a real Streamlit script context,
per the story's acceptance criteria:

* Start is disabled unless ``controls_enabled(app_state)["start_enabled"]``;
  clicking transitions IDLE/PAUSED -> RUNNING.
* Stop is disabled unless RUNNING; clicking transitions RUNNING -> PAUSED.
* Reset Session is always enabled; clicking fully resets the session
  (ring buffers, detector, logbook, total_samples, app_state, and the
  GMM training group's widget-default keys).
* Template/Mean/Std/Noise inputs are disabled exactly when
  ``app_state == RUNNING``.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import pytest

from core.anomaly_detector import AnomalyDetector
from state.app_state import LOGBOOK_COLUMNS, AppState, controls_enabled
from ui_web.sidebar_config import (
    _on_reset_clicked,
    _on_start_clicked,
    _on_stop_clicked,
    _reset_ring_buffers,
)
from utils.ring_buffer import RingBuffer


class _FakeSessionState(dict):
    """Minimal dict-like stand-in for ``st.session_state``.

    Mirrors the ``_FakeSessionState`` pattern used elsewhere in this
    test suite (``test_app_state.py``, ``test_sidebar_config.py``,
    ``test_live_tick.py``).
    """


def _make_state(app_state: AppState) -> _FakeSessionState:
    """Build a fake session-state populated like a real, running session.

    :param app_state: The ``app_state`` to seed.
    :type app_state: AppState
    :returns: A populated fake session-state mapping.
    :rtype: _FakeSessionState
    """
    state = _FakeSessionState(app_state=app_state)
    for key in ("ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected"):
        buf = RingBuffer(300)
        buf.append(1.0)
        buf.append(2.0)
        state[key] = buf
    state["detector"] = AnomalyDetector()
    state["logbook_df"] = pd.DataFrame(
        [
            {
                "row_id": 1,
                "timestamp": 1.0,
                "time_str": "00:00:01",
                "value": 5.0,
                "conf_pct": 90.0,
                "classification": "TP",
                "comment": "",
                "is_injected": True,
            }
        ]
    )
    state["total_samples"] = 42
    state["cfg_retrain_start"] = 5
    state["cfg_retrain_end"] = 40
    state["cfg_end_tracking"] = False
    return state


# ----------------------------------------------------------------------
# _on_start_clicked
# ----------------------------------------------------------------------


@pytest.mark.parametrize("initial", [AppState.IDLE, AppState.PAUSED])
def test_start_clicked_transitions_idle_or_paused_to_running(initial: AppState) -> None:
    """Start must move IDLE or PAUSED to RUNNING."""
    state = _make_state(initial)

    _on_start_clicked(state)

    assert state["app_state"] is AppState.RUNNING


@pytest.mark.parametrize("initial", [AppState.RUNNING, AppState.SIMULATION_RUNNING])
def test_start_clicked_is_a_noop_from_other_states(initial: AppState) -> None:
    """Start must not change app_state when already RUNNING or SIMULATION_RUNNING."""
    state = _make_state(initial)

    _on_start_clicked(state)

    assert state["app_state"] is initial


# ----------------------------------------------------------------------
# _on_stop_clicked
# ----------------------------------------------------------------------


def test_stop_clicked_transitions_running_to_paused() -> None:
    """Stop must move RUNNING to PAUSED."""
    state = _make_state(AppState.RUNNING)

    _on_stop_clicked(state)

    assert state["app_state"] is AppState.PAUSED


@pytest.mark.parametrize("initial", [AppState.IDLE, AppState.PAUSED, AppState.SIMULATION_RUNNING])
def test_stop_clicked_is_a_noop_from_other_states(initial: AppState) -> None:
    """Stop must not change app_state unless currently RUNNING."""
    state = _make_state(initial)

    _on_stop_clicked(state)

    assert state["app_state"] is initial


# ----------------------------------------------------------------------
# _reset_ring_buffers
# ----------------------------------------------------------------------


def test_reset_ring_buffers_clears_all_five_in_place() -> None:
    """All five ring buffers must be emptied without replacing the objects."""
    state = _make_state(AppState.RUNNING)
    original_objects = {
        key: state[key]
        for key in ("ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected")
    }
    for buf in original_objects.values():
        assert buf.size == 2  # seeded by _make_state

    _reset_ring_buffers(state)

    for key, original in original_objects.items():
        assert state[key] is original  # same object, cleared in place
        assert state[key].size == 0


def test_reset_ring_buffers_missing_key_is_a_defensive_noop() -> None:
    """A missing ring buffer key must not raise; other buffers still clear."""
    state = _make_state(AppState.RUNNING)
    del state["ring_ts"]

    _reset_ring_buffers(state)  # must not raise

    assert "ring_ts" not in state
    assert state["ring_vals"].size == 0


# ----------------------------------------------------------------------
# _on_reset_clicked
# ----------------------------------------------------------------------


@pytest.mark.parametrize("initial", list(AppState))
def test_reset_clicked_clears_all_five_ring_buffers(initial: AppState) -> None:
    """Reset must clear all five ring buffers regardless of the current state."""
    state = _make_state(initial)

    _on_reset_clicked(state)

    for key in ("ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected"):
        assert state[key].size == 0


def test_reset_clicked_calls_detector_reset() -> None:
    """Reset must call detector.reset(), clearing its trained state/buffer."""
    state = _make_state(AppState.RUNNING)
    detector = state["detector"]
    # Feed a couple of samples so there is something for reset() to clear.
    detector.feed(0.0, 1.0, False)
    detector.feed(1.0, 2.0, False)
    assert detector.sample_count == 2

    _on_reset_clicked(state)

    assert state["detector"] is detector  # reset() mutates in place, not replaced
    assert detector.sample_count == 0
    assert detector.is_trained is False


def test_reset_clicked_replaces_detector_if_reset_method_missing() -> None:
    """Defensive fallback: a detector stub without reset() gets replaced, not crashed on."""

    class _NoResetDetector:
        pass

    state = _make_state(AppState.RUNNING)
    state["detector"] = _NoResetDetector()

    _on_reset_clicked(state)  # must not raise

    assert isinstance(state["detector"], AnomalyDetector)


def test_reset_clicked_clears_logbook_to_empty_schema_df() -> None:
    """Reset must replace logbook_df with an empty DataFrame using the Epic E schema."""
    state = _make_state(AppState.RUNNING)
    assert len(state["logbook_df"]) == 1  # seeded by _make_state

    _on_reset_clicked(state)

    df = state["logbook_df"]
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == LOGBOOK_COLUMNS
    assert len(df) == 0


def test_reset_clicked_zeroes_total_samples() -> None:
    """Reset must set total_samples back to 0."""
    state = _make_state(AppState.RUNNING)

    _on_reset_clicked(state)

    assert state["total_samples"] == 0


@pytest.mark.parametrize("initial", list(AppState))
def test_reset_clicked_sets_app_state_to_idle(initial: AppState) -> None:
    """Reset must set app_state to IDLE from any starting state."""
    state = _make_state(initial)

    _on_reset_clicked(state)

    assert state["app_state"] is AppState.IDLE


def test_reset_clicked_resets_gmm_training_widget_keys() -> None:
    """Reset must zero the retrain spinboxes and re-enable End auto-tracking."""
    state = _make_state(AppState.RUNNING)
    assert state["cfg_retrain_start"] == 5
    assert state["cfg_retrain_end"] == 40
    assert state["cfg_end_tracking"] is False

    _on_reset_clicked(state)

    assert state["cfg_retrain_start"] == 0
    assert state["cfg_retrain_end"] == 0
    assert state["cfg_end_tracking"] is True


# ----------------------------------------------------------------------
# AppTest: exercise the real Start/Stop/Reset buttons + config disabling.
# ----------------------------------------------------------------------

_APP_SCRIPT_TEMPLATE = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.sidebar_config import render_run_controls, render_template_and_stream_controls

init_session_state()
# Seed app_state only once (on the very first script run) -- Streamlit
# reruns this whole script top-to-bottom on every widget interaction
# (e.g. a button click), so unconditionally reassigning app_state here
# on *every* run would clobber whatever the button's on_click callback
# just set it to, before this test ever gets to observe the result.
if "_test_seeded" not in st.session_state:
    st.session_state["app_state"] = AppState.{state_name}
    st.session_state["_test_seeded"] = True
with st.sidebar:
    render_template_and_stream_controls()
    render_run_controls()
"""


def _run_app(state_name: str):
    """Build and run an AppTest script seeded with the given AppState name.

    :param state_name: An :class:`~state.app_state.AppState` member name
        (e.g. ``"IDLE"``).
    :type state_name: str
    :returns: The executed AppTest instance.
    :rtype: streamlit.testing.v1.AppTest
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT_TEMPLATE.format(state_name=state_name))
    at.run()
    return at


@pytest.mark.parametrize("state", list(AppState))
def test_start_button_disabled_matches_matrix(state: AppState) -> None:
    """Start button's disabled= must exactly match controls_enabled()["start_enabled"]."""
    at = _run_app(state.name)

    assert at.exception == []
    button = at.sidebar.button(key="btn_start")
    assert button.disabled == (not controls_enabled(state)["start_enabled"])


@pytest.mark.parametrize("state", list(AppState))
def test_stop_button_disabled_matches_matrix(state: AppState) -> None:
    """Stop button's disabled= must exactly match controls_enabled()["stop_enabled"]."""
    at = _run_app(state.name)

    assert at.exception == []
    button = at.sidebar.button(key="btn_stop")
    assert button.disabled == (not controls_enabled(state)["stop_enabled"])


@pytest.mark.parametrize("state", list(AppState))
def test_reset_button_always_enabled(state: AppState) -> None:
    """Reset Session button must never be disabled, in any AppState."""
    at = _run_app(state.name)

    assert at.exception == []
    button = at.sidebar.button(key="btn_reset")
    assert button.disabled is False


@pytest.mark.parametrize("state", list(AppState))
def test_config_inputs_disabled_matches_matrix(state: AppState) -> None:
    """Template/Mean/Std/Noise inputs must be disabled exactly when RUNNING."""
    at = _run_app(state.name)

    assert at.exception == []
    expected_disabled = state == AppState.RUNNING

    selectbox = at.sidebar.selectbox(key="cfg_template")
    mean_input = at.sidebar.number_input(key="cfg_mean")
    std_input = at.sidebar.number_input(key="cfg_std")
    noise_input = at.sidebar.number_input(key="cfg_noise")

    assert selectbox.disabled is expected_disabled
    assert mean_input.disabled is expected_disabled
    assert std_input.disabled is expected_disabled
    assert noise_input.disabled is expected_disabled


def test_clicking_start_from_idle_transitions_to_running() -> None:
    """Clicking Start from IDLE must set app_state to RUNNING."""
    at = _run_app("IDLE")

    at.sidebar.button(key="btn_start").click().run()

    assert at.exception == []
    assert at.session_state["app_state"] is AppState.RUNNING


def test_clicking_start_from_paused_transitions_to_running() -> None:
    """Clicking Start from PAUSED must set app_state to RUNNING."""
    at = _run_app("PAUSED")

    at.sidebar.button(key="btn_start").click().run()

    assert at.exception == []
    assert at.session_state["app_state"] is AppState.RUNNING


def test_clicking_stop_from_running_transitions_to_paused() -> None:
    """Clicking Stop from RUNNING must set app_state to PAUSED."""
    at = _run_app("RUNNING")

    at.sidebar.button(key="btn_stop").click().run()

    assert at.exception == []
    assert at.session_state["app_state"] is AppState.PAUSED


@pytest.mark.parametrize("state", list(AppState))
def test_clicking_reset_from_any_state_returns_to_idle_and_clears_everything(
    state: AppState,
) -> None:
    """Clicking Reset from any state must fully reset the session back to IDLE."""
    at = _run_app(state.name)

    # Simulate an in-progress session before resetting.
    at.session_state["total_samples"] = 17
    at.session_state["ring_ts"].append(1.0)
    at.session_state["ring_vals"].append(2.0)
    at.session_state["ring_confs"].append(3.0)
    at.session_state["ring_is_anomaly"].append(False)
    at.session_state["ring_is_injected"].append(False)
    at.session_state["cfg_retrain_start"] = 3
    at.session_state["cfg_retrain_end"] = 12
    at.session_state["cfg_end_tracking"] = False

    at.sidebar.button(key="btn_reset").click().run()

    assert at.exception == []
    assert at.session_state["app_state"] is AppState.IDLE
    assert at.session_state["total_samples"] == 0
    assert at.session_state["ring_ts"].size == 0
    assert at.session_state["ring_vals"].size == 0
    assert at.session_state["ring_confs"].size == 0
    assert at.session_state["ring_is_anomaly"].size == 0
    assert at.session_state["ring_is_injected"].size == 0
    assert len(at.session_state["logbook_df"]) == 0
    assert list(at.session_state["logbook_df"].columns) == LOGBOOK_COLUMNS
    assert at.session_state["cfg_retrain_start"] == 0
    assert at.session_state["cfg_retrain_end"] == 0
    assert at.session_state["cfg_end_tracking"] is True
