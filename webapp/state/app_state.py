"""Application state machine and Streamlit ``session_state`` bootstrap.

This module is the state substrate every other web-app story builds on
top of. It provides two things:

1. :class:`AppState` -- the four high-level states the app can be in
   (mirroring the desktop app's state machine in
   ``ui/main_window.py``, minus the transient ``TRAINING`` state,
   which has no web equivalent since training here is synchronous
   inside :meth:`core.anomaly_detector.AnomalyDetector.feed`) -- and
   :func:`controls_enabled`, a pure function mapping a state to which
   sidebar/toolbar controls should be enabled.
2. :func:`init_session_state`, which lazily creates the long-lived
   objects (`DataGenerator`, `AnomalyDetector`, `SimulationPlayer`,
   ring buffers, the logbook DataFrame, and widget-default keys) in
   ``st.session_state`` exactly once per browser session, so later
   reruns of the Streamlit script reuse the same objects instead of
   resetting them.
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Dict, MutableMapping, Optional

import pandas as pd
import streamlit as st

from core.anomaly_detector import AnomalyDetector
from core.data_generator import TEMPLATES, DataGenerator
from core.simulation_loader import SimulationPlayer
from utils.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

#: Capacity of each rolling chart ring buffer; matches the desktop
#: app's 300-sample chart window (see ``utils/ring_buffer.py`` memory).
_RING_BUFFER_CAPACITY = 300

#: Maximum number of rows ``state["logbook_df"]`` may ever hold at once
#: (story P3). Once the logbook reaches this many rows, appending a new
#: anomaly evicts the single oldest row (FIFO by ``row_id``) rather than
#: growing the DataFrame further, so memory and per-append cost stay
#: bounded no matter how long an unattended session runs. The desktop
#: app's ``LogbookPanel`` has no equivalent cap (see
#: ``ui_web/live_tick.py``'s ``_append_anomaly_to_logbook`` for the
#: eviction logic, and the README's "Known fidelity gaps" section for
#: this deliberate difference). ``row_id``/``next_log_id`` numbering
#: keeps incrementing monotonically forever regardless of eviction --
#: only the DataFrame's row *count* is capped, never the id sequence.
MAX_LOGBOOK_ROWS = 500

#: Column schema for the anomaly logbook, shared with Epic E stories
#: that append rows to it.
LOGBOOK_COLUMNS = [
    "row_id",
    "timestamp",
    "time_str",
    "value",
    "conf_pct",
    "classification",
    "comment",
    "is_injected",
]


class AppState(Enum):
    """High-level application states driving control enable/disable.

    Mirrors the desktop app's ``AppState`` enum in
    ``ui/main_window.py``, with one deliberate difference: there is no
    ``TRAINING`` member. In the desktop app, ``TRAINING`` was a
    momentary state entered while the GMM fit ran on a background-ish
    call and immediately exited back to ``RUNNING``; on the web, GMM
    training happens synchronously inside a single
    :meth:`~core.anomaly_detector.AnomalyDetector.feed` call within one
    Streamlit script run, so there is no observable intermediate state
    to represent.
    """

    #: No stream running, no simulation playing.
    IDLE = auto()
    #: Live synthetic data stream is running.
    RUNNING = auto()
    #: Live stream was started, then paused (can be resumed).
    PAUSED = auto()
    #: A recorded simulation scenario is currently being replayed.
    SIMULATION_RUNNING = auto()


def controls_enabled(state: AppState) -> Dict[str, bool]:
    """Return which controls should be enabled for a given app state.

    Reverse-engineered from the desktop app's per-state
    ``setEnabled`` calls in ``ui/main_window.py``. See the story's
    acceptance-criteria matrix for the authoritative mapping; this
    function must reproduce it exactly for all four :class:`AppState`
    members.

    :param state: The current application state.
    :type state: AppState
    :raises TypeError: If ``state`` is not an :class:`AppState` member.
    :returns: Dict with keys ``config_enabled``, ``start_enabled``,
        ``stop_enabled``, ``sim_play_enabled``, ``sim_stop_enabled``,
        ``reset_enabled``, ``retrain_enabled``, each mapping to a
        ``bool``.
    :rtype: dict[str, bool]
    """
    # Defensive: fail loudly on a bad input rather than silently
    # falling through to some default control matrix.
    if not isinstance(state, AppState):
        raise TypeError(f"state must be an AppState member, got {state!r}")

    # Reset and Retrain are always enabled regardless of state, per
    # the matrix -- pulled out once rather than repeated per branch.
    always_on = {"reset_enabled": True, "retrain_enabled": True}

    if state is AppState.IDLE:
        matrix = {
            "config_enabled": True,
            "start_enabled": True,
            "stop_enabled": False,
            "sim_play_enabled": True,
            "sim_stop_enabled": False,
        }
    elif state is AppState.RUNNING:
        matrix = {
            "config_enabled": False,
            "start_enabled": False,
            "stop_enabled": True,
            "sim_play_enabled": True,
            "sim_stop_enabled": False,
        }
    elif state is AppState.PAUSED:
        matrix = {
            "config_enabled": True,
            "start_enabled": True,
            "stop_enabled": False,
            "sim_play_enabled": True,
            "sim_stop_enabled": False,
        }
    else:  # AppState.SIMULATION_RUNNING
        matrix = {
            "config_enabled": True,
            "start_enabled": False,
            "stop_enabled": False,
            "sim_play_enabled": False,
            "sim_stop_enabled": True,
        }

    matrix.update(always_on)
    return matrix


def _empty_logbook_df() -> pd.DataFrame:
    """Build an empty logbook DataFrame with the Epic E schema.

    :returns: Empty DataFrame with columns :data:`LOGBOOK_COLUMNS`.
    :rtype: pandas.DataFrame
    """
    return pd.DataFrame(columns=LOGBOOK_COLUMNS)


def init_session_state(state: Optional[MutableMapping] = None) -> MutableMapping:
    """Idempotently populate ``st.session_state`` with the app's shared objects.

    Every key is guarded with ``if key not in state`` so that calling
    this function on every Streamlit script rerun (its intended usage
    -- Streamlit re-executes the whole script top-to-bottom on each
    interaction) never resets state that already has data, such as
    ``total_samples``, ``logbook_df``, or samples already collected in
    the ring buffers.

    :param state: The mutable mapping to populate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this function testable outside of a running Streamlit script
        (where the real ``st.session_state`` is unavailable/inert).
    :type state: MutableMapping | None
    :returns: The populated mapping (same object passed in, or
        ``st.session_state``), for convenience/chaining.
    :rtype: MutableMapping
    """
    if state is None:
        state = st.session_state

    # --- Long-lived core objects -----------------------------------
    if "generator" not in state:
        state["generator"] = DataGenerator()
        logger.debug("Initialized session_state['generator']")
    if "detector" not in state:
        state["detector"] = AnomalyDetector()
        logger.debug("Initialized session_state['detector']")
    if "sim_player" not in state:
        state["sim_player"] = SimulationPlayer()
        logger.debug("Initialized session_state['sim_player']")

    # --- Ring buffers for the rolling chart window ------------------
    # Kept as five independent buffers (rather than one buffer of
    # tuples) so each series can be sliced/plotted directly as a numpy
    # array without per-frame unpacking.
    for key in ("ring_ts", "ring_vals", "ring_confs", "ring_is_anomaly", "ring_is_injected"):
        if key not in state:
            state[key] = RingBuffer(_RING_BUFFER_CAPACITY)
            logger.debug("Initialized session_state['%s']", key)

    # --- Session-scoped counters / dataframes ------------------------
    if "total_samples" not in state:
        state["total_samples"] = 0
    if "logbook_df" not in state:
        state["logbook_df"] = _empty_logbook_df()
    if "next_log_id" not in state:
        # Monotonic counter handing out each new logbook row's unique
        # ``row_id`` (Epic E1). A later story (E4, "Clear Log") resets
        # this back to 0 alongside emptying ``logbook_df``; it is seeded
        # here so that reset has a well-defined starting point to write
        # into rather than needing to re-derive it.
        state["next_log_id"] = 0
    if "app_state" not in state:
        state["app_state"] = AppState.IDLE

    # --- Widget-default keys -----------------------------------------
    # Seeded from the Temperature template so the sidebar shows sane
    # defaults on first load, matching the desktop app's default
    # stream (see core.data_generator.TEMPLATES["Temperature"]).
    temperature_template = TEMPLATES["Temperature"]
    widget_defaults = {
        "cfg_template": "Temperature",
        "cfg_mean": temperature_template["mean"],
        "cfg_std": temperature_template["std"],
        "cfg_noise": temperature_template["noise"],
        "cfg_anomaly_pct": 0,
        "cfg_retrain_start": 0,
        "cfg_retrain_end": 0,
        "cfg_end_tracking": True,
        "cfg_scenario": "Temperature",
        "cfg_speed": 1,
    }
    for key, default in widget_defaults.items():
        if key not in state:
            state[key] = default

    logger.info("Session state initialized (idempotent).")
    return state
