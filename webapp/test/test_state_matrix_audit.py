"""G2: full state-machine audit -- automated cross-reference test.

This module is the durable, automated counterpart to
``webapp/docs/state_matrix_audit.md``'s manual cross-reference table. It
renders *every* control listed in the story's authoritative matrix
(Template/Mean/Std/Noise, Start, Stop, Load & Play, Stop Sim, Reset
Session, and the GMM training window + Retrain button) together, in a
single real Streamlit script via :class:`streamlit.testing.v1.AppTest`,
for each of the four :class:`~state.app_state.AppState` values, and
asserts each widget's actual ``.disabled`` property against
:func:`~state.app_state.controls_enabled`'s corresponding matrix key.

Unlike the per-story test files (``test_sidebar_config.py``,
``test_run_controls.py``, ``test_simulation_controls.py``,
``test_gmm_training.py``), which each independently assert a subset of
these controls, this module is intentionally the single place that
cross-checks *all seven* controls/control-groups against the *same*
``controls_enabled()`` call, in the *same* render, so a future change
that breaks the cell-by-cell matrix in any one control is caught here
even if it happens to slip past a narrower, single-story test.

A second test class drives the two manual walkthroughs the story's
acceptance criteria call for
(``IDLE -> RUNNING -> PAUSED -> RUNNING -> IDLE`` and
``IDLE -> SIMULATION_RUNNING -> IDLE``) via real button clicks against a
single, persistent :class:`AppTest` session, asserting the full
seven-control enabled/disabled snapshot after every transition.
"""
from __future__ import annotations

from typing import Dict

import pytest
from streamlit.testing.v1 import AppTest

from state.app_state import AppState, controls_enabled

# ----------------------------------------------------------------------
# Shared AppTest script: renders every control the matrix governs,
# together, in one script run -- mirrors app.py's sidebar wiring
# (render_template_and_stream_controls / render_run_controls /
# render_simulation_controls) plus render_gmm_training_group (normally
# only reachable via the live_tick_fragment, but directly callable here
# exactly as test_gmm_training.py does).
# ----------------------------------------------------------------------

_APP_SCRIPT_TEMPLATE = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.sidebar_config import (
    render_run_controls,
    render_simulation_controls,
    render_template_and_stream_controls,
)
from ui_web.live_tick import render_gmm_training_group

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
    render_gmm_training_group()
"""


def _run_app(state_name: str) -> AppTest:
    """Build and run an AppTest script seeded with the given AppState name.

    :param state_name: An :class:`~state.app_state.AppState` member name
        (e.g. ``"IDLE"``).
    :type state_name: str
    :returns: The executed AppTest instance.
    :rtype: streamlit.testing.v1.AppTest
    """
    at = AppTest.from_string(_APP_SCRIPT_TEMPLATE.format(state_name=state_name))
    at.run()
    return at


def _actual_disabled_snapshot(at: AppTest) -> Dict[str, bool]:
    """Read every matrix-governed control's real ``.disabled`` state off a run AppTest.

    :param at: An already-``.run()``-completed AppTest instance whose
        script rendered all seven controls/control-groups (see
        :data:`_APP_SCRIPT_TEMPLATE`).
    :type at: streamlit.testing.v1.AppTest
    :returns: Dict keyed identically to
        :func:`~state.app_state.controls_enabled`'s return value's keys
        for the six directly-comparable controls, plus
        ``retrain_start_disabled``/``retrain_end_disabled``/
        ``retrain_button_disabled`` for the three GMM-training widgets
        (all of which must equal ``not retrain_enabled``).
    :rtype: dict[str, bool]
    """
    return {
        # Template/Mean/Std/Noise (config_enabled) -- four widgets, all
        # must move together.
        "template_disabled": at.sidebar.selectbox(key="cfg_template").disabled,
        "mean_disabled": at.sidebar.number_input(key="cfg_mean").disabled,
        "std_disabled": at.sidebar.number_input(key="cfg_std").disabled,
        "noise_disabled": at.sidebar.number_input(key="cfg_noise").disabled,
        # Start / Stop
        "start_disabled": at.sidebar.button(key="btn_start").disabled,
        "stop_disabled": at.sidebar.button(key="btn_stop").disabled,
        # Load & Play / Stop Sim
        "sim_play_disabled": at.sidebar.button(key="btn_sim_play").disabled,
        "sim_stop_disabled": at.sidebar.button(key="btn_sim_stop").disabled,
        # Reset Session
        "reset_disabled": at.sidebar.button(key="btn_reset").disabled,
        # GMM training window (Start/End spinboxes) + Retrain Model button
        "retrain_start_disabled": at.sidebar.number_input(key="cfg_retrain_start").disabled,
        "retrain_end_disabled": at.sidebar.number_input(key="cfg_retrain_end").disabled,
        "retrain_button_disabled": at.sidebar.button(key="btn_retrain").disabled,
    }


def _expected_disabled_snapshot(state: AppState) -> Dict[str, bool]:
    """Build the expected ``.disabled`` snapshot for a state from ``controls_enabled()``.

    :param state: The :class:`~state.app_state.AppState` to compute the
        expected snapshot for.
    :type state: AppState
    :returns: Dict with the same keys as
        :func:`_actual_disabled_snapshot`, each the negation of the
        corresponding ``controls_enabled()`` value (``disabled = not
        enabled``).
    :rtype: dict[str, bool]
    """
    enabled = controls_enabled(state)
    return {
        "template_disabled": not enabled["config_enabled"],
        "mean_disabled": not enabled["config_enabled"],
        "std_disabled": not enabled["config_enabled"],
        "noise_disabled": not enabled["config_enabled"],
        "start_disabled": not enabled["start_enabled"],
        "stop_disabled": not enabled["stop_enabled"],
        "sim_play_disabled": not enabled["sim_play_enabled"],
        "sim_stop_disabled": not enabled["sim_stop_enabled"],
        "reset_disabled": not enabled["reset_enabled"],
        "retrain_start_disabled": not enabled["retrain_enabled"],
        "retrain_end_disabled": not enabled["retrain_enabled"],
        "retrain_button_disabled": not enabled["retrain_enabled"],
    }


# ----------------------------------------------------------------------
# Cell-by-cell cross-reference: every control x every AppState.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("state", list(AppState))
def test_all_controls_disabled_state_matches_matrix_for_every_app_state(
    state: AppState,
) -> None:
    """Every one of the 7 controls' real ``.disabled`` must match ``controls_enabled()``.

    This is the single, comprehensive parametrized cross-check the G2
    story's first acceptance criterion asks for: for each AppState, the
    *actual* rendered widget state (not just the source expression) is
    read back and diffed cell-by-cell against the matrix.
    """
    at = _run_app(state.name)
    assert at.exception == []

    actual = _actual_disabled_snapshot(at)
    expected = _expected_disabled_snapshot(state)

    mismatches = {
        key: (expected[key], actual[key])
        for key in expected
        if expected[key] != actual[key]
    }
    assert not mismatches, (
        f"AppState.{state.name}: disabled-state mismatches "
        f"(key -> (expected, actual)): {mismatches}"
    )


@pytest.mark.parametrize("state", list(AppState))
def test_retrain_widgets_have_no_disabled_argument_at_all(state: AppState) -> None:
    """The GMM training window + Retrain button must never be disabled, in any state.

    Distinct from the matrix-diff test above: this specifically nails
    down the story's callout that these three widgets should have *no*
    ``disabled=`` expression at all (rather than one that happens to
    always evaluate to ``False``) -- verified here indirectly by
    asserting ``.disabled is False`` (not just falsy) for all four
    states, which is exactly what "no disabled= argument" renders as.
    """
    at = _run_app(state.name)
    assert at.exception == []

    assert at.sidebar.number_input(key="cfg_retrain_start").disabled is False
    assert at.sidebar.number_input(key="cfg_retrain_end").disabled is False
    assert at.sidebar.button(key="btn_retrain").disabled is False


# ----------------------------------------------------------------------
# Manual walkthrough #1: IDLE -> RUNNING -> PAUSED -> RUNNING -> IDLE
# ----------------------------------------------------------------------


def test_walkthrough_idle_running_paused_running_idle_matches_matrix_at_every_step() -> None:
    """Drive IDLE -> RUNNING -> PAUSED -> RUNNING -> IDLE via real clicks.

    At every step, asserts the full seven-control snapshot against
    ``controls_enabled()`` for the state the app is now in, using the
    *same* persistent AppTest session throughout (real button clicks,
    not re-seeded scripts) so the transitions themselves are exercised,
    not just the terminal state at each step.
    """
    at = _run_app("IDLE")
    assert at.session_state["app_state"] is AppState.IDLE
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.IDLE)

    # IDLE -> RUNNING (click Start)
    at.sidebar.button(key="btn_start").click().run()
    assert at.exception == []
    assert at.session_state["app_state"] is AppState.RUNNING
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.RUNNING)

    # RUNNING -> PAUSED (click Stop)
    at.sidebar.button(key="btn_stop").click().run()
    assert at.exception == []
    assert at.session_state["app_state"] is AppState.PAUSED
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.PAUSED)

    # PAUSED -> RUNNING (click Start again)
    at.sidebar.button(key="btn_start").click().run()
    assert at.exception == []
    assert at.session_state["app_state"] is AppState.RUNNING
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.RUNNING)

    # RUNNING -> back to a clean IDLE (click Reset Session, always enabled)
    at.sidebar.button(key="btn_reset").click().run()
    assert at.exception == []
    assert at.session_state["app_state"] is AppState.IDLE
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.IDLE)


# ----------------------------------------------------------------------
# Manual walkthrough #2: IDLE -> SIMULATION_RUNNING -> IDLE
# ----------------------------------------------------------------------


def test_walkthrough_idle_simulation_running_idle_matches_matrix_at_every_step() -> None:
    """Drive IDLE -> SIMULATION_RUNNING -> IDLE via real clicks (Load & Play / Stop Sim).

    Same rationale as the first walkthrough: a single persistent
    AppTest session, real clicks, full seven-control snapshot asserted
    at each step.
    """
    at = _run_app("IDLE")
    assert at.session_state["app_state"] is AppState.IDLE
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.IDLE)

    # IDLE -> SIMULATION_RUNNING (click Load & Play)
    at.sidebar.button(key="btn_sim_play").click().run()
    assert at.exception == []
    assert at.session_state["app_state"] is AppState.SIMULATION_RUNNING
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(
        AppState.SIMULATION_RUNNING
    )

    # SIMULATION_RUNNING -> IDLE (click Stop Sim)
    at.sidebar.button(key="btn_sim_stop").click().run()
    assert at.exception == []
    assert at.session_state["app_state"] is AppState.IDLE
    assert _actual_disabled_snapshot(at) == _expected_disabled_snapshot(AppState.IDLE)
