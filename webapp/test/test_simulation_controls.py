"""Tests for :mod:`ui_web.sidebar_config`'s D2 simulation controls.

Covers the pure, unit-testable callbacks (:func:`_load_and_play_clicked`,
:func:`_stop_sim_clicked`, :func:`_reset_buffers_logbook_and_counter`)
with fakes, plus :class:`streamlit.testing.v1.AppTest`-based tests that
drive the real scenario selectbox / speed slider / Load & Play / Stop
Sim buttons in a real Streamlit script context, per the story's
acceptance criteria:

* Scenario selectbox lists Temperature/Vibration/Pressure; speed slider
  ranges 1-10 with an ``f"{v}x"`` label.
* "Load & Play" is disabled unless ``sim_play_enabled``; clicking resets
  session buffers/logbook, loads the selected scenario, and transitions
  ``app_state`` to ``SIMULATION_RUNNING``.
* "Stop Sim" is disabled unless ``sim_stop_enabled``; clicking calls
  ``sim_player.stop()`` and transitions ``app_state`` to ``IDLE``.
* Start/Stop (live streaming) buttons are disabled while
  ``app_state == SIMULATION_RUNNING``.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import pytest

from core.anomaly_detector import AnomalyDetector
from core.simulation_loader import SimulationScenario
from state.app_state import LOGBOOK_COLUMNS, AppState, controls_enabled
from ui_web.sidebar_config import (
    _load_and_play_clicked,
    _reset_buffers_logbook_and_counter,
    _stop_sim_clicked,
)
from utils.ring_buffer import RingBuffer


class _FakeSessionState(dict):
    """Minimal dict-like stand-in for ``st.session_state``.

    Mirrors the ``_FakeSessionState`` pattern used elsewhere in this
    test suite (``test_run_controls.py``, ``test_live_tick.py``).
    """


class _FakeSimPlayer:
    """Stub :class:`~core.simulation_loader.SimulationPlayer` recording calls.

    :ivar load_calls: Scenarios passed to :meth:`load`, in order.
    :vartype load_calls: list[SimulationScenario]
    :ivar speed_calls: Values passed to :meth:`set_speed`, in order.
    :vartype speed_calls: list[float]
    :ivar start_called: True once :meth:`start` has been called.
    :vartype start_called: bool
    :ivar stop_called: True once :meth:`stop` has been called.
    :vartype stop_called: bool
    """

    def __init__(self) -> None:
        self.load_calls: List[SimulationScenario] = []
        self.speed_calls: List[float] = []
        self.start_called = False
        self.stop_called = False

    def load(self, scenario: SimulationScenario) -> None:
        """Record the scenario passed."""
        self.load_calls.append(scenario)

    def set_speed(self, multiplier: float) -> None:
        """Record the speed passed."""
        self.speed_calls.append(multiplier)

    def start(self, now: Optional[float] = None) -> None:
        """Record that start() was called."""
        self.start_called = True

    def stop(self) -> None:
        """Record that stop() was called."""
        self.stop_called = True


def _make_state(app_state: AppState, sim_player: Optional[object] = None) -> _FakeSessionState:
    """Build a fake session-state populated like a real, running session.

    :param app_state: The ``app_state`` to seed.
    :type app_state: AppState
    :param sim_player: The sim_player stub to seed; defaults to a fresh
        :class:`_FakeSimPlayer`.
    :type sim_player: object | None
    :returns: A populated fake session-state mapping.
    :rtype: _FakeSessionState
    """
    state = _FakeSessionState(
        app_state=app_state,
        sim_player=sim_player if sim_player is not None else _FakeSimPlayer(),
        cfg_scenario="Temperature",
        cfg_speed=3,
    )
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
    return state


# ----------------------------------------------------------------------
# _reset_buffers_logbook_and_counter
# ----------------------------------------------------------------------


def test_reset_buffers_logbook_and_counter_clears_all_three() -> None:
    """Ring buffers, logbook_df, and total_samples must all be cleared/zeroed."""
    state = _make_state(AppState.IDLE)

    _reset_buffers_logbook_and_counter(state)

    for key in ("ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected"):
        assert state[key].size == 0
    assert len(state["logbook_df"]) == 0
    assert list(state["logbook_df"].columns) == LOGBOOK_COLUMNS
    assert state["total_samples"] == 0


def test_reset_buffers_logbook_and_counter_does_not_touch_detector() -> None:
    """Unlike Reset Session, this shared helper must not reset the detector."""
    state = _make_state(AppState.IDLE)
    detector = state["detector"]
    detector.feed(0.0, 1.0, False)
    assert detector.sample_count == 1

    _reset_buffers_logbook_and_counter(state)

    assert state["detector"] is detector
    assert detector.sample_count == 1  # untouched


# ----------------------------------------------------------------------
# _load_and_play_clicked
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scenario_name", ["Temperature", "Vibration", "Pressure"])
def test_load_and_play_loads_the_selected_scenario(scenario_name: str) -> None:
    """Load & Play must load the scenario named by cfg_scenario, not always Temperature."""
    sim_player = _FakeSimPlayer()
    state = _make_state(AppState.IDLE, sim_player=sim_player)
    state["cfg_scenario"] = scenario_name

    _load_and_play_clicked(state)

    assert len(sim_player.load_calls) == 1
    assert sim_player.load_calls[0].name == scenario_name


def test_load_and_play_sets_speed_from_cfg_speed() -> None:
    """Load & Play must call set_speed() with the current cfg_speed value."""
    sim_player = _FakeSimPlayer()
    state = _make_state(AppState.IDLE, sim_player=sim_player)
    state["cfg_speed"] = 7

    _load_and_play_clicked(state)

    assert sim_player.speed_calls == [7]


def test_load_and_play_calls_start() -> None:
    """Load & Play must call sim_player.start()."""
    sim_player = _FakeSimPlayer()
    state = _make_state(AppState.IDLE, sim_player=sim_player)

    _load_and_play_clicked(state)

    assert sim_player.start_called is True


def test_load_and_play_resets_ring_buffers_logbook_and_total_samples() -> None:
    """Load & Play must reset session buffers/logbook, per the acceptance criteria."""
    state = _make_state(AppState.IDLE)
    assert len(state["logbook_df"]) == 1
    assert state["total_samples"] == 42

    _load_and_play_clicked(state)

    for key in ("ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected"):
        assert state[key].size == 0
    assert len(state["logbook_df"]) == 0
    assert state["total_samples"] == 0


def test_load_and_play_transitions_app_state_to_simulation_running() -> None:
    """Load & Play must set app_state to SIMULATION_RUNNING."""
    state = _make_state(AppState.IDLE)

    _load_and_play_clicked(state)

    assert state["app_state"] is AppState.SIMULATION_RUNNING


def test_load_and_play_missing_sim_player_is_a_defensive_noop() -> None:
    """A missing sim_player must not raise, and must leave state untouched."""
    state = _make_state(AppState.IDLE)
    del state["sim_player"]
    total_samples_before = state["total_samples"]

    _load_and_play_clicked(state)  # must not raise

    assert state["app_state"] is AppState.IDLE  # unchanged
    assert state["total_samples"] == total_samples_before  # not reset


# ----------------------------------------------------------------------
# _stop_sim_clicked
# ----------------------------------------------------------------------


def test_stop_sim_calls_sim_player_stop() -> None:
    """Stop Sim must call sim_player.stop()."""
    sim_player = _FakeSimPlayer()
    state = _make_state(AppState.SIMULATION_RUNNING, sim_player=sim_player)

    _stop_sim_clicked(state)

    assert sim_player.stop_called is True


def test_stop_sim_transitions_app_state_to_idle() -> None:
    """Stop Sim must set app_state to IDLE."""
    state = _make_state(AppState.SIMULATION_RUNNING)

    _stop_sim_clicked(state)

    assert state["app_state"] is AppState.IDLE


def test_stop_sim_missing_sim_player_still_forces_idle() -> None:
    """A missing sim_player must not raise, and app_state still moves to IDLE defensively."""
    state = _make_state(AppState.SIMULATION_RUNNING)
    del state["sim_player"]

    _stop_sim_clicked(state)  # must not raise

    assert state["app_state"] is AppState.IDLE


# ----------------------------------------------------------------------
# AppTest: exercise the real scenario selectbox / speed slider / buttons.
# ----------------------------------------------------------------------

_APP_SCRIPT_TEMPLATE = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.sidebar_config import (
    render_run_controls,
    render_simulation_controls,
    render_template_and_stream_controls,
)

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
    render_simulation_controls()
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


def test_scenario_selectbox_lists_exactly_the_three_scenarios() -> None:
    """The Scenario selectbox must list exactly Temperature/Vibration/Pressure."""
    at = _run_app("IDLE")

    assert at.exception == []
    selectbox = at.sidebar.selectbox(key="cfg_scenario")
    assert list(selectbox.options) == ["Temperature", "Vibration", "Pressure"]
    assert selectbox.value == "Temperature"


def test_speed_slider_bounds_and_default() -> None:
    """The speed slider must range 1-10, defaulting to 1."""
    at = _run_app("IDLE")

    assert at.exception == []
    slider = at.sidebar.slider(key="cfg_speed")
    assert (slider.min, slider.max) == (1, 10)
    assert slider.value == 1


def test_speed_slider_displays_multiplier_format() -> None:
    """The slider's live value must render as f"{v}x" (handle format + caption)."""
    at = _run_app("IDLE")

    assert at.exception == []
    slider = at.sidebar.slider(key="cfg_speed")
    assert slider.proto.format == "%d×"

    captions = [c.value for c in at.sidebar.caption]
    assert "1×" in captions


def test_moving_speed_slider_updates_caption() -> None:
    """Moving the speed slider must update the f"{v}x" caption underneath it."""
    at = _run_app("IDLE")

    at.sidebar.slider(key="cfg_speed").set_value(7).run()

    assert at.exception == []
    assert at.session_state["cfg_speed"] == 7
    captions = [c.value for c in at.sidebar.caption]
    assert "7×" in captions


@pytest.mark.parametrize("state", list(AppState))
def test_load_and_play_button_disabled_matches_matrix(state: AppState) -> None:
    """Load & Play button's disabled= must exactly match controls_enabled()["sim_play_enabled"]."""
    at = _run_app(state.name)

    assert at.exception == []
    button = at.sidebar.button(key="btn_sim_play")
    assert button.disabled == (not controls_enabled(state)["sim_play_enabled"])


@pytest.mark.parametrize("state", list(AppState))
def test_stop_sim_button_disabled_matches_matrix(state: AppState) -> None:
    """Stop Sim button's disabled= must exactly match controls_enabled()["sim_stop_enabled"]."""
    at = _run_app(state.name)

    assert at.exception == []
    button = at.sidebar.button(key="btn_sim_stop")
    assert button.disabled == (not controls_enabled(state)["sim_stop_enabled"])


def test_clicking_load_and_play_from_idle_transitions_to_simulation_running() -> None:
    """Clicking Load & Play from IDLE must load the scenario and enter SIMULATION_RUNNING."""
    at = _run_app("IDLE")

    at.sidebar.button(key="btn_sim_play").click().run()

    assert at.exception == []
    assert at.session_state["app_state"] is AppState.SIMULATION_RUNNING
    sim_player = at.session_state["sim_player"]
    assert sim_player.finished is False


def test_clicking_load_and_play_resets_session_buffers_and_logbook() -> None:
    """Clicking Load & Play must clear pre-existing ring buffers/logbook/total_samples."""
    at = _run_app("IDLE")

    at.session_state["total_samples"] = 99
    at.session_state["ring_ts"].append(1.0)
    at.session_state["ring_vals"].append(2.0)
    at.session_state["ring_confs"].append(3.0)
    at.session_state["ring_is_anomaly"].append(False)
    at.session_state["ring_is_injected"].append(False)

    at.sidebar.button(key="btn_sim_play").click().run()

    assert at.exception == []
    assert at.session_state["total_samples"] == 0
    assert at.session_state["ring_ts"].size == 0
    assert at.session_state["ring_vals"].size == 0
    assert len(at.session_state["logbook_df"]) == 0


def test_clicking_stop_sim_from_simulation_running_transitions_to_idle() -> None:
    """Clicking Stop Sim from SIMULATION_RUNNING must stop playback and return to IDLE."""
    at = _run_app("IDLE")
    at.sidebar.button(key="btn_sim_play").click().run()
    assert at.session_state["app_state"] is AppState.SIMULATION_RUNNING

    at.sidebar.button(key="btn_sim_stop").click().run()

    assert at.exception == []
    assert at.session_state["app_state"] is AppState.IDLE
    assert at.session_state["sim_player"].finished is True


# ----------------------------------------------------------------------
# Start/Stop (live streaming) buttons must disable while SIMULATION_RUNNING.
# ----------------------------------------------------------------------


def test_start_and_stop_live_buttons_disabled_while_simulation_running() -> None:
    """Once app_state is SIMULATION_RUNNING (e.g. after Load & Play), the live
    Start/Stop buttons must both show disabled=True -- falls out of C3's
    existing controls_enabled()-driven rendering once the transition is
    correct, but proven here directly rather than assumed.
    """
    at = _run_app("IDLE")
    at.sidebar.button(key="btn_sim_play").click().run()
    assert at.session_state["app_state"] is AppState.SIMULATION_RUNNING

    assert at.exception == []
    start_button = at.sidebar.button(key="btn_start")
    stop_button = at.sidebar.button(key="btn_stop")
    assert start_button.disabled is True
    assert stop_button.disabled is True
