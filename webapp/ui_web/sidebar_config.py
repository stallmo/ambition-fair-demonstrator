"""Sidebar controls: sensor template picker, stream parameter inputs,
and the anomaly-injection probability slider.

Ports the desktop app's "Sensor Template", "Stream Parameters", and
"Anomaly Injection" group boxes (``ui/config_panel.py``'s
``_build_template_group`` / ``_build_stream_group`` /
``_build_anomaly_group`` / ``_on_template_changed`` / ``_emit_config``)
to Streamlit widgets bound to ``st.session_state``.

The template selectbox's ``on_change`` callback pushes the selected
template's defaults into the ``cfg_mean``/``cfg_std``/``cfg_noise``
session-state keys *before* Streamlit re-renders the ``number_input``
widgets that are bound to those same keys. This is the only safe way to
programmatically change a widget-bound session-state value in
Streamlit: setting ``st.session_state["cfg_mean"] = ...`` *after* the
``st.number_input(key="cfg_mean", ...)`` call has already executed in
the same script run raises a ``StreamlitAPIException`` ("cannot be
modified after the widget ... is instantiated"). Since ``on_change``
callbacks run *before* the rest of the script body (and before the
widgets below the selectbox are (re-)instantiated), mutating the keys
there is safe and is exactly the pattern Streamlit's own docs
recommend for "linked widgets" / cascading defaults.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import MutableMapping, Optional, Tuple

import pandas as pd
import streamlit as st

from core.anomaly_detector import AnomalyDetector
from core.data_generator import TEMPLATES, StreamConfig
from core.simulation_loader import SimulationScenario, generate_scenarios
from state.app_state import LOGBOOK_COLUMNS, AppState, controls_enabled

logger = logging.getLogger(__name__)

#: Template names in display order; mirrors ``TEMPLATES.keys()`` /
#: the desktop app's ``QComboBox.addItems(list(TEMPLATES.keys()))``.
_TEMPLATE_NAMES = list(TEMPLATES.keys())

#: Scenario names in display order for the D2 simulation selectbox;
#: matches the three CSVs :func:`core.simulation_loader.generate_scenarios`
#: writes (``scenario_temperature.csv`` etc -- lower-cased in the
#: filename, capitalized here for display).
_SCENARIO_NAMES = ["Temperature", "Current", "Product Dimension"]

#: Mirrors ``core.simulation_loader._ASSETS_DIR`` exactly (same
#: ``webapp/assets/simulations`` location, computed independently here
#: since this module lives one directory level away
#: (``webapp/ui_web/`` vs ``webapp/core/``) -- both must resolve to the
#: identical path so ``SimulationScenario.from_csv`` reads the same
#: files :func:`generate_scenarios` writes.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "simulations"

#: Session-state keys of the five ring buffers that must always stay
#: length-synchronized. Kept as a local copy (rather than importing a
#: private constant from ``ui_web.live_tick``) since it is a small,
#: self-contained list of key names, not shared behavior.
_RING_BUFFER_KEYS: Tuple[str, str, str, str, str] = (
    "ring_ts",
    "ring_vals",
    "ring_confs",
    "ring_is_anomaly",
    "ring_is_injected",
)


def _on_template_changed(state: Optional[MutableMapping] = None) -> None:
    """``on_change`` callback for the Sensor Template selectbox.

    Reads the *new* value Streamlit has already written to
    ``state["cfg_template"]`` (widget ``on_change`` callbacks run after
    the bound session-state key has been updated but before the script
    body re-renders downstream widgets) and pushes the matching
    :data:`~core.data_generator.TEMPLATES` entry's ``mean``/``std``/
    ``noise`` into the ``cfg_mean``/``cfg_std``/``cfg_noise`` keys the
    ``number_input`` widgets below are bound to.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    template_name = state.get("cfg_template")
    template = TEMPLATES.get(template_name)
    if template is None:
        # Defensive: an unknown/missing template name should never
        # silently corrupt mean/std/noise -- log and bail instead.
        logger.warning(
            "_on_template_changed: unknown template %r; leaving cfg_mean/"
            "cfg_std/cfg_noise untouched",
            template_name,
        )
        return

    state["cfg_mean"] = template["mean"]
    state["cfg_std"] = template["std"]
    state["cfg_noise"] = template["noise"]
    logger.debug(
        "Template changed to %r; mean=%.4f std=%.4f noise=%.4f",
        template_name,
        template["mean"],
        template["std"],
        template["noise"],
    )


def _sync_generator_config(state: Optional[MutableMapping] = None) -> StreamConfig:
    """Push the current sidebar values into ``state["generator"]``.

    Called at the end of every render so that *any* change to the
    template, mean, std, or noise widgets -- since Streamlit reruns the
    whole script top-to-bottom on every widget interaction -- results
    in the live :class:`~core.data_generator.DataGenerator` picking up
    the new :class:`~core.data_generator.StreamConfig` immediately, the
    same way the desktop app's ``_emit_config`` did on every
    ``valueChanged``/``currentTextChanged`` signal.

    ``anomaly_magnitude`` and ``unit`` are *not* independently editable
    (there is no widget for them in this story) and always follow the
    currently-selected template, matching ``_emit_config`` in
    ``ui/config_panel.py``. ``anomaly_probability`` is read from
    ``cfg_anomaly_pct`` (seeded by A3, rendered as a slider by
    :func:`render_anomaly_control`) and converted from a 0-100
    percentage to a 0-1 fraction, matching the desktop app's
    ``self._anomaly_slider.value() / 100.0``.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``.
    :type state: MutableMapping | None
    :returns: The :class:`StreamConfig` that was pushed to the generator.
    :rtype: StreamConfig
    """
    if state is None:
        state = st.session_state

    template_name = state.get("cfg_template", _TEMPLATE_NAMES[0])
    template = TEMPLATES.get(template_name, TEMPLATES[_TEMPLATE_NAMES[0]])

    # Defensive default: cfg_anomaly_pct may not be customized yet (C2
    # lands the slider itself), so fall back to init_session_state()'s
    # seeded default of 2 rather than raising a KeyError.
    anomaly_pct = state.get("cfg_anomaly_pct", 0)

    config = StreamConfig(
        mean=state.get("cfg_mean", template["mean"]),
        std=state.get("cfg_std", template["std"]),
        noise_amplitude=state.get("cfg_noise", template["noise"]),
        anomaly_probability=anomaly_pct / 100.0,
        anomaly_magnitude=template["anomaly_magnitude"],
        unit=template["unit"],
    )

    generator = state.get("generator")
    if generator is not None:
        generator.configure(config)
        logger.debug("Synced generator config: %r", config)
    else:
        # Defensive: session state not yet initialized (init_session_state()
        # hasn't run). Log and skip rather than raising AttributeError.
        logger.warning(
            "_sync_generator_config: 'generator' missing from session "
            "state; skipping configure() (was init_session_state() called?)"
        )

    return config


def render_anomaly_control(state: Optional[MutableMapping] = None) -> int:
    """Render the "Anomaly Injection" sidebar slider.

    Ports ``ui/config_panel.py``'s ``_build_anomaly_group`` (a
    ``QSlider(0, 100)`` paired with a live ``"{val}%"`` label) to a
    single Streamlit ``st.slider``. Streamlit's slider already renders
    the live value as a floating tooltip/handle label, so the printf
    ``format="%d%%"`` string is passed to reproduce the desktop app's
    ``f"{val}%"`` display for the handle itself; a redundant
    ``st.caption`` below the slider mirrors the desktop app's separate
    ``QLabel`` and guarantees an exact, easily-testable ``"{v}%"``
    rendering that does not depend on the browser-side sprintf.js
    formatting of the slider handle (which this test suite cannot
    exercise headlessly).

    The slider is bound via ``key="cfg_anomaly_pct"`` only -- *not*
    also ``value=`` -- because ``cfg_anomaly_pct`` is already seeded in
    ``st.session_state`` by :func:`state.app_state.init_session_state`
    (default ``2``); passing both ``key`` and ``value`` for a key that
    already exists in session state raises a Streamlit warning about
    the widget's default being overridden by session state.

    :param state: Session-state mapping to read the current slider
        value from for the caption. Defaults to ``st.session_state``;
        accepting an injectable mapping keeps the post-render caption
        logic consistent with the rest of this module's testable
        helpers. The widget itself always binds to the real
        ``st.session_state`` (Streamlit widgets cannot be bound to an
        arbitrary mapping).
    :type state: MutableMapping | None
    :returns: The current anomaly-probability percentage (0-100) shown
        by the slider this run.
    :rtype: int
    """
    if state is None:
        state = st.session_state

    st.subheader("Anomaly Injection")
    st.slider(
        "Anomaly Probability",
        min_value=0,
        max_value=100,
        key="cfg_anomaly_pct",
        format="%d%%",
        help="Per-tick probability of injecting a synthetic spike anomaly.",
        label_visibility="collapsed",
    )

    # Defensive default matches _sync_generator_config()'s fallback so
    # the caption never raises even if called before init_session_state().
    anomaly_pct = state.get("cfg_anomaly_pct", 0)
    st.caption(f"{anomaly_pct}%")
    return anomaly_pct


def render_template_and_stream_controls(
    state: Optional[MutableMapping] = None,
) -> StreamConfig:
    """Render the "Sensor Template", "Stream Parameters", and "Anomaly
    Injection" sidebar controls.

    Ports ``ui/config_panel.py``'s ``_build_template_group``,
    ``_build_stream_group``, and (via :func:`render_anomaly_control`)
    ``_build_anomaly_group`` to Streamlit. Intended to be called from
    within a ``with st.sidebar:`` block.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps the
        post-render sync step unit-testable. The widgets themselves
        always bind to the real ``st.session_state`` (Streamlit widgets
        cannot be bound to an arbitrary mapping), so this parameter
        only affects the non-widget bookkeeping around them.
    :type state: MutableMapping | None
    :returns: The :class:`StreamConfig` pushed to the generator this run.
    :rtype: StreamConfig
    """
    # C3: the Template/Mean/Std/Noise widgets must be disabled exactly
    # when the run lifecycle's `config_enabled` flag is False (i.e. only
    # while RUNNING -- see controls_enabled()'s matrix). Read the *real*
    # st.session_state here (not the injectable `state` param) since this
    # directly gates widget rendering, not post-render bookkeeping -- the
    # injectable `state` param only exists for the bookkeeping helpers
    # below (see module/function docstrings).
    current_app_state = st.session_state.get("app_state", AppState.IDLE)
    config_disabled = not controls_enabled(current_app_state)["config_enabled"]

    st.subheader("Sensor Template")
    st.selectbox(
        "Sensor Template",
        options=_TEMPLATE_NAMES,
        key="cfg_template",
        on_change=_on_template_changed,
        label_visibility="collapsed",
        disabled=config_disabled,
    )

    st.subheader("Stream Parameters")
    st.number_input(
        "Mean",
        key="cfg_mean",
        min_value=0.0,
        max_value=9999.0,
        step=0.1,
        disabled=config_disabled,
    )
    st.number_input(
        "Std Dev",
        key="cfg_std",
        min_value=0.01,
        max_value=100.0,
        step=0.1,
        disabled=config_disabled,
    )
    st.number_input(
        "Noise",
        key="cfg_noise",
        min_value=0.0,
        max_value=50.0,
        step=0.1,
        disabled=config_disabled,
    )

    # C2: Anomaly Probability slider, rendered here so it lives in the
    # same sidebar block and its change (like the widgets above) is
    # picked up by the _sync_generator_config() call below on rerun.
    render_anomaly_control(state)

    # Any change above (template switch or manual edit) triggers a
    # full script rerun; syncing here means the live generator always
    # reflects the values currently shown in the sidebar.
    return _sync_generator_config(state)


# ----------------------------------------------------------------------
# C3: Start / Stop / Reset Session run-lifecycle controls
# ----------------------------------------------------------------------


def _on_start_clicked(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the "Start" button.

    Transitions ``app_state`` from ``IDLE`` or ``PAUSED`` to
    ``RUNNING``, matching the desktop app's ``MainWindow._on_start``.
    A no-op for any other current state (defensive: the button is
    already ``disabled`` in those states per :func:`controls_enabled`,
    but the guard here means a stray/late click can never leave the
    state machine in an invalid transition).

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    current = state.get("app_state", AppState.IDLE)
    if current in (AppState.IDLE, AppState.PAUSED):
        state["app_state"] = AppState.RUNNING
        logger.info("Start clicked: app_state %s -> RUNNING", current)
    else:
        logger.debug("Start clicked while app_state=%s; no-op", current)


def _on_stop_clicked(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the "Stop" button.

    Transitions ``app_state`` from ``RUNNING`` to ``PAUSED``, matching
    the desktop app's ``MainWindow._on_stop``. A no-op for any other
    current state (same defensive rationale as :func:`_on_start_clicked`).

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    current = state.get("app_state", AppState.IDLE)
    if current is AppState.RUNNING:
        state["app_state"] = AppState.PAUSED
        logger.info("Stop clicked: app_state RUNNING -> PAUSED")
    else:
        logger.debug("Stop clicked while app_state=%s; no-op", current)


def _reset_ring_buffers(state: MutableMapping) -> None:
    """Clear all five ring buffers in place, without reallocating them.

    Uses each :class:`~utils.ring_buffer.RingBuffer`'s ``clear()``
    method (O(1), keeps the same backing numpy array) rather than
    replacing the objects, so any other code holding a reference to a
    buffer (e.g. the chart) keeps seeing the same, now-empty, instance.

    :param state: Session-state mapping holding the ring buffer objects.
    :type state: MutableMapping
    :returns: None
    :rtype: None
    """
    for key in _RING_BUFFER_KEYS:
        buf = state.get(key)
        if buf is not None and hasattr(buf, "clear"):
            buf.clear()
        else:
            # Defensive: session state not (fully) initialized -- log
            # and skip rather than raising AttributeError/KeyError.
            logger.warning(
                "_reset_ring_buffers: session_state['%s'] missing or has "
                "no clear() method; skipping (was init_session_state() "
                "called?)",
                key,
            )


def _reset_buffers_logbook_and_counter(state: MutableMapping) -> None:
    """Clear the five ring buffers, the logbook, and ``total_samples``.

    Factored out of :func:`_on_reset_clicked` (Reset Session) so
    :func:`_load_and_play_clicked` (D2's "Load & Play") can reuse the
    identical "clear the session's collected samples" logic without
    duplicating it -- Load & Play deliberately does *not* also reset
    the detector or the GMM training-window widget keys the way Reset
    Session does (a scripted demo is meant to keep reusing whatever
    model state/training window the operator already has), so this
    helper only covers the subset both callers share.

    :param state: Session-state mapping holding the ring buffers,
        ``logbook_df``, and ``total_samples``.
    :type state: MutableMapping
    :returns: None
    :rtype: None
    """
    _reset_ring_buffers(state)
    state["logbook_df"] = pd.DataFrame(columns=LOGBOOK_COLUMNS)
    state["total_samples"] = 0


def _on_reset_clicked(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the "Reset Session" button.

    Performs a full run-lifecycle reset, matching the desktop app's
    ``MainWindow._on_reset``:

    * Clears all five chart ring buffers (see :func:`_reset_ring_buffers`).
    * Resets the anomaly detector (``detector.reset()``), discarding
      the trained GMM and its sample buffer.
    * Clears the anomaly logbook back to an empty DataFrame with the
      Epic E column schema (:data:`~state.app_state.LOGBOOK_COLUMNS`).
    * Resets ``total_samples`` to 0.
    * Resets the GMM training-window widget state (``cfg_retrain_start``/
      ``cfg_retrain_end`` to 0, ``cfg_end_tracking`` back to True so the
      End spinbox resumes auto-tracking ``total_samples``) -- this is
      what makes the sidebar "GMM Training" group (story B4) show its
      progress bar back at 0% / status "Not trained" on the next render,
      since that group reads these same keys plus ``detector.is_trained``/
      ``training_progress``, both of which :meth:`AnomalyDetector.reset`
      also resets.
    * Sets ``app_state`` back to ``IDLE``.

    Always enabled (``reset_enabled`` is unconditionally True in
    :func:`~state.app_state.controls_enabled`'s matrix), so this
    callback is safe to invoke from any :class:`~state.app_state.AppState`.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    _reset_buffers_logbook_and_counter(state)

    detector = state.get("detector")
    if detector is not None and hasattr(detector, "reset"):
        detector.reset()
    else:
        # Defensive fallback: no detector (or a stub without reset())
        # in session state -- construct a fresh one so the invariant
        # "state['detector'] is always a usable AnomalyDetector" holds
        # after a reset, matching how init_session_state() first builds it.
        logger.warning(
            "_on_reset_clicked: 'detector' missing or lacks reset(); "
            "constructing a fresh AnomalyDetector instead"
        )
        state["detector"] = AnomalyDetector()

    state["cfg_retrain_start"] = 0
    state["cfg_retrain_end"] = 0
    state["cfg_end_tracking"] = True
    state["app_state"] = AppState.IDLE

    logger.info("Reset Session clicked: buffers/detector/logbook cleared, app_state -> IDLE")


def render_run_controls(state: Optional[MutableMapping] = None) -> None:
    """Render the "Session Controls" Start / Stop / Reset Session buttons.

    Ports the desktop app's toolbar Start/Stop/Reset actions
    (``ui/main_window.py``'s ``_on_start``/``_on_stop``/``_on_reset``,
    gated by ``AppState``-driven ``setEnabled`` calls) to Streamlit
    buttons whose ``disabled=`` argument is driven directly by
    :func:`~state.app_state.controls_enabled`, so this group can never
    drift out of sync with the matrix that governs every other control.

    Intended to be called from within a ``with st.sidebar:`` block,
    typically right after :func:`render_template_and_stream_controls`.

    :param state: Session-state mapping to read the current
        ``app_state`` from. Defaults to ``st.session_state``; accepting
        an injectable mapping keeps the ``disabled=`` computation
        unit-testable outside a running Streamlit script. The buttons
        themselves always bind their ``on_click`` callbacks to the real
        ``st.session_state`` (Streamlit widgets cannot be bound to an
        arbitrary mapping) via the callbacks' own default-argument
        fallback.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    app_state = state.get("app_state", AppState.IDLE)
    enabled = controls_enabled(app_state)

    st.subheader("Session Controls")
    col_start, col_stop = st.columns(2)
    with col_start:
        st.button(
            "▶ Start",
            key="btn_start",
            disabled=not enabled["start_enabled"],
            on_click=_on_start_clicked,
            use_container_width=True,
        )
    with col_stop:
        st.button(
            "■ Stop",
            key="btn_stop",
            disabled=not enabled["stop_enabled"],
            on_click=_on_stop_clicked,
            use_container_width=True,
        )
    st.button(
        "↺ Reset Session",
        key="btn_reset",
        disabled=not enabled["reset_enabled"],
        on_click=_on_reset_clicked,
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# D2: Simulation scenario controls (scenario picker, speed, Load & Play,
# Stop Sim), feeding the same live-tick fragment as live streaming via
# ``AppState.SIMULATION_RUNNING`` (see ui_web/live_tick.py's
# _process_simulation_tick).
# ----------------------------------------------------------------------


def _load_and_play_clicked(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the "Load & Play" button.

    Loads the scenario currently selected in ``state["cfg_scenario"]``
    at the speed currently selected in ``state["cfg_speed"]``, resets
    the session's collected samples (ring buffers/logbook/
    ``total_samples`` -- see :func:`_reset_buffers_logbook_and_counter`),
    and transitions ``app_state`` to ``SIMULATION_RUNNING`` so the next
    ``live_tick_fragment`` poll starts delivering rows via
    :func:`ui_web.live_tick._process_simulation_tick`.

    Order of operations is deliberately defensive: ``sim_player`` is
    checked and the scenario CSV is loaded *before* any session state is
    reset, so a missing ``sim_player`` or an unreadable/malformed CSV
    leaves the session's existing buffers/logbook untouched rather than
    clearing them and then failing to actually start playback.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    sim_player = state.get("sim_player")
    if sim_player is None:
        # Defensive: session state not yet initialized.
        logger.warning(
            "_load_and_play_clicked: 'sim_player' missing from session "
            "state; skipping (was init_session_state() called?)"
        )
        return

    scenario_name = state.get("cfg_scenario", _SCENARIO_NAMES[0])
    speed = state.get("cfg_speed", 1)

    # Cheap/idempotent -- guarantees the CSV exists even if app.py's own
    # startup call to generate_scenarios() has not run yet (e.g. a test
    # harness that only imports this module).
    generate_scenarios()
    csv_path = _ASSETS_DIR / f"scenario_{str(scenario_name).lower()}.csv"
    scenario = SimulationScenario.from_csv(csv_path)

    _reset_buffers_logbook_and_counter(state)

    sim_player.load(scenario)
    sim_player.set_speed(speed)
    sim_player.start()
    state["app_state"] = AppState.SIMULATION_RUNNING

    logger.info(
        "Load & Play clicked: scenario=%r speed=%sx -> SIMULATION_RUNNING",
        scenario_name,
        speed,
    )


def _stop_sim_clicked(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the "Stop Sim" button.

    Calls ``sim_player.stop()`` (making further ``due_rows()`` polls
    return nothing) and transitions ``app_state`` back to ``IDLE``,
    matching the story's acceptance criterion for manually stopping a
    scripted demo before it finishes naturally.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    sim_player = state.get("sim_player")
    if sim_player is not None and hasattr(sim_player, "stop"):
        sim_player.stop()
    else:
        # Defensive: still force app_state back to IDLE even if
        # sim_player is missing/stub, so the button never leaves the
        # app stuck in SIMULATION_RUNNING.
        logger.warning(
            "_stop_sim_clicked: 'sim_player' missing or lacks stop(); "
            "app_state will still be forced back to IDLE"
        )

    state["app_state"] = AppState.IDLE
    logger.info("Stop Sim clicked: sim_player.stop() called, app_state -> IDLE")


def render_simulation_controls(state: Optional[MutableMapping] = None) -> None:
    """Render the "Simulation" sidebar group (scenario, speed, Load & Play, Stop Sim).

    Ports the desktop app's simulation controls (scenario combo box,
    speed slider, Load/Play and Stop actions in ``ui/config_panel.py``
    and ``ui/main_window.py``) to Streamlit widgets. The scenario
    selectbox is bound via ``key="cfg_scenario"`` and the speed slider
    via ``key="cfg_speed"`` -- both already seeded by
    :func:`state.app_state.init_session_state` (``"Temperature"`` / ``1``
    respectively), so neither widget also passes ``value=`` (same
    rationale as :func:`render_anomaly_control`'s ``cfg_anomaly_pct``
    slider).

    "Load & Play" is disabled unless
    ``controls_enabled(app_state)["sim_play_enabled"]``; "Stop Sim" is
    disabled unless ``controls_enabled(app_state)["sim_stop_enabled"]``
    -- both driven directly by :func:`~state.app_state.controls_enabled`
    so this group can never drift out of sync with the matrix that
    governs every other control (mirrors :func:`render_run_controls`).

    Intended to be called from within a ``with st.sidebar:`` block,
    typically right after :func:`render_run_controls`.

    :param state: Session-state mapping to read the current
        ``app_state``/``cfg_speed`` from. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps the
        ``disabled=``/caption computation unit-testable outside a
        running Streamlit script. The widgets themselves always bind to
        the real ``st.session_state`` (Streamlit widgets cannot be
        bound to an arbitrary mapping).
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    app_state = state.get("app_state", AppState.IDLE)
    enabled = controls_enabled(app_state)

    st.subheader("Simulation")
    st.selectbox(
        "Scenario",
        options=_SCENARIO_NAMES,
        key="cfg_scenario",
        label_visibility="collapsed",
    )
    st.slider(
        "Speed",
        min_value=1,
        max_value=10,
        key="cfg_speed",
        format="%d×",
        help="Playback speed multiplier (1x = real-time).",
        label_visibility="collapsed",
    )
    # Redundant, directly-testable f"{v}×" caption underneath the
    # slider -- same rationale as render_anomaly_control()'s "{v}%"
    # caption: guarantees an exactly-assertable rendering that does not
    # depend on the browser-side sprintf.js formatting of the slider
    # handle (which this test suite cannot exercise headlessly).
    speed = state.get("cfg_speed", 1)
    st.caption(f"{speed}×")

    col_play, col_stop = st.columns(2)
    with col_play:
        st.button(
            "▶ Load & Play",
            key="btn_sim_play",
            disabled=not enabled["sim_play_enabled"],
            on_click=_load_and_play_clicked,
            use_container_width=True,
        )
    with col_stop:
        st.button(
            "■ Stop Sim",
            key="btn_sim_stop",
            disabled=not enabled["sim_stop_enabled"],
            on_click=_stop_sim_clicked,
            use_container_width=True,
        )
