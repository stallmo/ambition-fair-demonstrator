"""Live data-tick fragment: generator -> detector -> ring buffers -> chart.

Conceptually mirrors the desktop app's ``QTimer(100ms)`` ->
``DataGenerator._generate_tick()`` -> ``AnomalyDetector.feed()`` ->
panels wiring in ``ui/main_window.py``, but driven by a
``st.fragment(run_every=LIVE_TICK_INTERVAL)`` poll instead of a Qt timer
signal (there is no real background thread in Streamlit; the fragment
simply re-executes itself on a fixed cadence within the browser
session).

The 300 ms cadence here (see :data:`LIVE_TICK_INTERVAL`, story P2 --
raised from an original 150 ms to roughly halve per-session redraw/
bandwidth cost) versus the original's 100 ms is an intentional,
documented fidelity gap (Streamlit fragment reruns have more overhead
than a Qt timer tick), not a bug. Raising this poll interval changes
only the wall-clock redraw cadence -- it has no effect on
:class:`~core.data_generator.DataGenerator`'s simulated time step
(still a fixed 0.1 s advance per ``tick()`` call) or on the number of
samples the GMM trains on per fragment cycle.

The tick-processing logic is deliberately split out of the
``@st.fragment``-decorated function into :func:`process_one_tick` so it
can be unit-tested directly (``st.fragment`` cannot be meaningfully
exercised outside a live Streamlit script run / ``AppTest`` context).
Likewise, redrawing the persistent chart from the current ring-buffer
windows is split into :func:`update_chart_from_state`, kept separate
from the ``st.plotly_chart`` render call itself (story B3; the render
call was ``st.pyplot`` before story Q1 migrated the chart to Plotly)
for the same testability reason.

Story B4 adds the sidebar "GMM Training" group (progress bar, status
text, training-window spinboxes, "Retrain Model" button) as
:func:`render_gmm_training_group`, called from inside
:func:`live_tick_fragment` (into ``st.sidebar``, which Streamlit
fragments are able to render into) so the training progress/status
stays live-updating on the same 300 ms cadence as the chart, and the
GMM overlay bands/mean/boundary lines it wires into the chart via
:meth:`ui_web.chart.ChartState.update_gmm_overlay`.

Story E1 adds logbook capture: every :class:`~core.anomaly_detector.
DetectionResult` with ``is_anomaly=True`` produced inside
:func:`process_one_tick` is appended as one row to ``state["logbook_df"]``
via :func:`_append_anomaly_to_logbook`, so flagged anomalies persist for
review by later logbook-panel stories (Epic E). Non-anomalous samples
leave the logbook untouched.

Story P3 caps ``state["logbook_df"]`` at
:data:`~state.app_state.MAX_LOGBOOK_ROWS` rows so memory and per-append
cost stay bounded on a long-running unattended session:
:func:`_append_anomaly_to_logbook` grows the DataFrame via ``pd.concat``
only while under that cap (a bounded, one-time cost); once the cap is
reached, every further anomaly overwrites the single oldest row in place
(a cheap ``.iloc[]`` row write keyed by ``row_id % MAX_LOGBOOK_ROWS``,
not a fresh ``pd.concat`` copy of the whole frame) so the FIFO-oldest row
is evicted without ever re-allocating/copying the full DataFrame.
``row_id``/``next_log_id`` keep incrementing monotonically forever
regardless of eviction -- only the DataFrame's row count is capped.

Story D2 adds the ``AppState.SIMULATION_RUNNING`` branch to
:func:`process_one_tick`: instead of pulling exactly one sample per
call from ``generator.tick()`` (the RUNNING branch), it polls
``state["sim_player"].due_rows(now)`` -- which can return zero, one, or
several rows in a single call, since a Streamlit fragment poll interval
does not line up 1:1 with the scenario's own scheduled row cadence --
and feeds/buffers/logs every returned row through the exact same
per-sample pipeline as the RUNNING branch (factored into
:func:`_feed_and_buffer_one_sample` so neither branch duplicates the
detector-feed / ring-buffer-append / logbook-append sequence). Once
``sim_player.finished`` becomes True (either because every row was
delivered, or because "Stop Sim" was clicked), ``app_state`` is
auto-transitioned back to ``IDLE`` on the very next tick that observes
it, so a scripted demo cleanly stops itself without operator action.

Story P1 adds IDLE/PAUSED gating around the chart redraw itself: while
``app_state`` is ``IDLE`` or ``PAUSED``, :func:`live_tick_fragment`
skips *both* :func:`update_chart_from_state` and the chart-render call
entirely for that cycle, so an idle browser tab stops spending server
CPU/bandwidth on a redraw + re-render of an unchanged chart every
300 ms. This intentionally supersedes the "IDLE/PAUSED still render a
static chart" framing from story B3's docstrings above -- but note that
skipping the render this cycle only avoids blanking the chart *because*
of a two-part mechanism, not because of any inherent "fragments leave
earlier elements in place" guarantee (an earlier version of this
docstring incorrectly claimed the latter):

1. ``app.py``'s ``main()`` (re)creates an ``st.empty()`` chart
   placeholder and renders the persistent chart's *current* Figure into
   it on **every** full (non-fragment) script run -- see
   :func:`ui_web.chart.render_chart_into_placeholder`. This guarantees
   the last frame reappears immediately after *any* full rerun (e.g.
   clicking "Pause"/"Stop", itself a full rerun, not a fragment-only
   one), regardless of this cycle's ``app_state``. Empirically (via
   ``streamlit.testing.v1.AppTest``, while implementing this fix), a
   *cached/reused* placeholder -- rather than one recreated every full
   run -- loses its previously-rendered content the moment any
   ``@st.fragment``-decorated function is called later in that same
   script run, on the *next* full rerun; hence it is deliberately not
   cached.
2. This fragment, on cycles where ``app_state`` is ``RUNNING`` or
   ``SIMULATION_RUNNING``, looks up that same placeholder (via
   :func:`ui_web.chart.get_chart_placeholder`, never creating its own)
   and writes the freshly redrawn frame into it
   (``placeholder.plotly_chart(...)`` -- story Q1; this was
   ``placeholder.pyplot(...)`` before the chart migrated to Plotly).
   Because true 300 ms fragment-only auto-reruns do not re-execute
   ``app.py``'s outer script at all,
   elements outside the fragment's own scope -- including this
   placeholder and whatever was last written into it -- are left
   completely untouched by Streamlit on a cycle that skips the write,
   so the last RUNNING/SIMULATION_RUNNING frame simply stays visible.

``render_gmm_training_group`` (story B4) is deliberately *not* gated by
this -- its progress bar/status text keep updating every cycle
regardless of ``app_state``, since training-state visibility while
paused is expected, not part of this optimization.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitAPIException

from core.anomaly_detector import DetectionResult
from state.app_state import MAX_LOGBOOK_ROWS, AppState
from ui_web.chart import ChartState, get_chart_placeholder, get_or_create_chart_state

logger = logging.getLogger(__name__)

#: Poll cadence for the ``@st.fragment``-decorated :func:`live_tick_fragment`
#: (single source of truth passed to ``run_every`` below). Story P2 raised
#: this from an original ``"150ms"`` to ``"300ms"`` to roughly halve
#: per-session redraw/bandwidth cost while the chart still reads as "live"
#: for a demo viewer; it affects only wall-clock redraw frequency, not
#: :class:`~core.data_generator.DataGenerator`'s fixed 0.1 s simulated-time
#: step per ``tick()`` call.
LIVE_TICK_INTERVAL: str = "300ms"

#: Stable ``st.plotly_chart`` element keys (story Q1) for this module's
#: two chart-render call sites inside :func:`live_tick_fragment`. Unlike
#: matplotlib's ``st.pyplot``, Plotly's ``st.plotly_chart`` auto-derives
#: an internal element ID from (element type, parameters) and raises
#: ``StreamlitDuplicateElementId`` if two calls land in the same script
#: run with the same derived ID -- which can happen here because this
#: ``@st.fragment``-decorated function runs inline as part of the same
#: outer script run that calls ``ui_web.chart.render_chart_into_placeholder``
#: (see :data:`ui_web.chart._CHART_KEY_OUTER`). ``_CHART_KEY_LIVE`` is
#: used for the normal placeholder-write path (RUNNING/SIMULATION_RUNNING
#: with a placeholder already set up by ``app.py``'s ``main()``);
#: ``_CHART_KEY_FALLBACK`` for the defensive bare ``st.plotly_chart(...)``
#: call used only when no placeholder exists yet (test-only scenario).
_CHART_KEY_LIVE = "chart_plotly_live"
_CHART_KEY_FALLBACK = "chart_plotly_fallback"

#: Session-state keys of the five ring buffers that must always stay
#: length-synchronized (one append per processed sample, across all five).
_RING_BUFFER_KEYS: Tuple[str, str, str, str, str] = (
    "ring_ts",
    "ring_vals",
    "ring_confs",
    "ring_is_anomaly",
    "ring_is_injected",
)

#: Initial classification label every freshly-logged anomaly row starts
#: with, matching the desktop app's ``LogbookPanel`` default (Epic E).
#: A later story (E2/E3) lets the operator change this via a dropdown.
_UNCLASSIFIED_LABEL = "⚪ Unclassified"


def _append_sample_to_ring_buffers(
    state: MutableMapping, values: Sequence[object]
) -> None:
    """Append one value to each of the five ring buffers, as a single unit.

    Appending all five together (rather than interleaved with other
    logic) keeps the buffers from ever drifting out of length sync --
    every call site either appends to all five or none of them.

    :param state: Session-state mapping holding the ring buffer objects.
    :type state: MutableMapping
    :param values: Values to append, in the same order as
        :data:`_RING_BUFFER_KEYS` (``ts``, ``value``, ``confidence``,
        ``is_anomaly``, ``is_injected``).
    :type values: Sequence[object]
    :raises KeyError: If a required ring buffer key is missing from ``state``.
    :returns: None
    :rtype: None
    """
    # Defensive: resolve all five buffer objects *before* appending
    # anything, so a missing/misconfigured key aborts before any
    # buffer is mutated (avoids a partial, desynchronized append).
    buffers = [state[key] for key in _RING_BUFFER_KEYS]
    for buf, value in zip(buffers, values):
        buf.append(value)


def _append_anomaly_to_logbook(state: MutableMapping, result: DetectionResult) -> bool:
    """Append one row to ``state["logbook_df"]`` for an anomalous detection result.

    Builds a row with a unique, monotonically incrementing ``row_id``
    taken from ``state["next_log_id"]`` (bumped by exactly one on
    success), a human-readable ``time_str`` derived from
    ``result.timestamp`` (which is a wall-clock ``time.time()`` epoch
    float -- see :class:`core.data_generator.DataGenerator.tick` and
    :class:`core.anomaly_detector.DetectionResult` -- so
    ``datetime.fromtimestamp`` correctly converts it to local time, no
    ``datetime.now()`` substitution needed), and the fixed initial
    classification/comment every freshly-logged anomaly starts with.

    Story P3 caps ``logbook_df`` at
    :data:`~state.app_state.MAX_LOGBOOK_ROWS` rows, oldest evicted first
    (FIFO): while the DataFrame holds fewer than the cap, the new row is
    added via ``pd.concat`` (not the removed ``DataFrame.append``,
    matching the modern pandas append idiom) -- a bounded, one-time cost
    while the logbook is still growing towards the cap. Once the cap is
    reached, every further call instead overwrites the single oldest row
    *in place* via a cheap ``.iloc[]`` row write (no ``pd.concat``, no
    full-frame copy): because every row ever appended has a unique
    ``row_id`` handed out by the same monotonic ``next_log_id`` counter
    that grew ``logbook_df`` from empty up to exactly
    ``MAX_LOGBOOK_ROWS`` rows one-for-one, ``row_id % MAX_LOGBOOK_ROWS``
    is a stable ring-buffer slot: it lands on position 0 for the first
    post-cap row (evicting ``row_id=0``, the oldest), position 1 for the
    next (evicting the new oldest, ``row_id=1``), and so on -- always
    the FIFO-oldest surviving row, never the newest, with no reuse of an
    evicted row's id (``next_log_id`` itself never resets or wraps).

    Caller's responsibility: only call this for a result where
    ``result.is_anomaly`` is True -- this function itself does not
    check ``is_anomaly`` again, to keep it a single-purpose "append a
    row" helper mirroring :func:`_append_sample_to_ring_buffers`.

    :param state: Session-state mapping holding ``logbook_df`` and
        ``next_log_id``.
    :type state: MutableMapping
    :param result: The anomalous detection result to log.
    :type result: DetectionResult
    :returns: True if a row was appended/written; False if
        ``logbook_df`` or ``next_log_id`` was missing from ``state``
        (defensive no-op, e.g. session not yet initialized).
    :rtype: bool
    """
    # Defensive: resolve both required pieces of state up front so a
    # missing key is a clean, logged no-op rather than a partial write
    # (e.g. a bumped counter with no corresponding row, or vice versa).
    logbook_df = state.get("logbook_df")
    if logbook_df is None or "next_log_id" not in state:
        logger.warning(
            "_append_anomaly_to_logbook: 'logbook_df'/'next_log_id' missing "
            "from session state; skipping logbook append (was "
            "init_session_state() called?)"
        )
        return False

    row_id = state["next_log_id"]
    new_row = {
        "row_id": row_id,
        "timestamp": result.timestamp,
        "time_str": datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S"),
        "value": result.value,
        "conf_pct": result.confidence * 100,
        "classification": _UNCLASSIFIED_LABEL,
        "comment": "",
        "is_injected": result.is_injected,
    }

    if len(logbook_df) < MAX_LOGBOOK_ROWS:
        # Still growing towards the cap: a plain pd.concat is cheap here
        # since the frame never exceeds MAX_LOGBOOK_ROWS rows even in the
        # worst case, and this branch runs at most MAX_LOGBOOK_ROWS times
        # per session (or per "Clear Log" reset) before the cap kicks in.
        state["logbook_df"] = pd.concat([logbook_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # At cap: evict the FIFO-oldest row via a single in-place row
        # write (no pd.concat, no full-frame copy) -- see docstring for
        # why row_id % MAX_LOGBOOK_ROWS always targets the oldest slot.
        position = row_id % MAX_LOGBOOK_ROWS
        logbook_df.iloc[position] = [new_row[col] for col in logbook_df.columns]
        state["logbook_df"] = logbook_df
        logger.debug(
            "_append_anomaly_to_logbook: logbook_df at cap (%d rows); evicted oldest "
            "row via in-place overwrite at position %d",
            MAX_LOGBOOK_ROWS,
            position,
        )

    state["next_log_id"] = row_id + 1
    logger.debug("_append_anomaly_to_logbook: logged row_id=%d (value=%.4f)", row_id, result.value)
    return True


def _feed_and_buffer_one_sample(
    state: MutableMapping,
    detector: object,
    timestamp: float,
    value: float,
    is_injected: bool,
) -> Optional[DetectionResult]:
    """Feed one sample through the detector and buffer/log the result.

    Shared by both the ``RUNNING`` and ``SIMULATION_RUNNING`` branches
    of :func:`process_one_tick` so the detector-feed -> ring-buffer-append
    -> logbook-append -> ``total_samples`` bump sequence exists in
    exactly one place, regardless of whether the sample came from
    ``generator.tick()`` (live) or a ``sim_player.due_rows()`` row
    (scripted playback).

    :param state: Session-state mapping to mutate (ring buffers,
        ``total_samples``, ``logbook_df``/``next_log_id``).
    :type state: MutableMapping
    :param detector: The :class:`~core.anomaly_detector.AnomalyDetector`
        (or a test double exposing the same ``feed()`` signature) to
        score the sample with.
    :type detector: object
    :param timestamp: The sample's timestamp.
    :type timestamp: float
    :param value: The sample's value.
    :type value: float
    :param is_injected: Whether this sample is a known/ground-truth
        injected anomaly (the live generator's ``is_injected`` flag, or
        a scripted scenario row's ``is_anomaly`` CSV column playing the
        same role) -- passed through to ``detector.feed()``, which
        independently computes ``DetectionResult.is_anomaly`` via the
        trained GMM regardless of this flag's value.
    :type is_injected: bool
    :returns: The :class:`~core.anomaly_detector.DetectionResult` if the
        sample was successfully buffered; None if a ring buffer key was
        missing from ``state`` (defensive no-op -- nothing was appended
        or counted, see :func:`_append_sample_to_ring_buffers`).
    :rtype: DetectionResult | None
    """
    result = detector.feed(timestamp, value, is_injected)

    # --- Append to all five ring buffers, atomically as a group ------
    try:
        _append_sample_to_ring_buffers(
            state,
            (
                result.timestamp,
                result.value,
                result.confidence,
                result.is_anomaly,
                result.is_injected,
            ),
        )
    except KeyError:
        # Defensive: ring buffers missing entirely -- nothing was
        # appended (see _append_sample_to_ring_buffers docstring), so
        # buffers remain in sync; just skip the counter bump/logbook too.
        logger.warning(
            "_feed_and_buffer_one_sample: one or more ring buffer keys "
            "missing from session state; skipping sample (was "
            "init_session_state() called?)"
        )
        return None

    state["total_samples"] = state.get("total_samples", 0) + 1
    logger.debug(
        "_feed_and_buffer_one_sample: processed sample #%d (value=%.4f, is_anomaly=%s)",
        state["total_samples"],
        result.value,
        result.is_anomaly,
    )

    # --- Log anomalous detections to the logbook (Epic E1) -----------
    # Non-anomalous samples must leave logbook_df/next_log_id untouched;
    # only is_anomaly=True results are ever logged.
    if result.is_anomaly:
        _append_anomaly_to_logbook(state, result)

    return result


def _process_running_tick(state: MutableMapping) -> bool:
    """Advance the live stream by exactly one sample (the ``RUNNING`` branch).

    :param state: Session-state mapping to read/mutate.
    :type state: MutableMapping
    :returns: True if a sample was generated, scored, and buffered;
        False if the call was a defensive no-op (missing
        generator/detector/ring buffers).
    :rtype: bool
    """
    generator = state.get("generator")
    detector = state.get("detector")
    if generator is None or detector is None:
        # Defensive: session state not yet initialized (init_session_state()
        # hasn't run). Log and skip rather than raising AttributeError.
        logger.warning(
            "_process_running_tick: 'generator'/'detector' missing from "
            "session state; skipping tick (was init_session_state() called?)"
        )
        return False

    timestamp, value, is_injected = generator.tick()
    result = _feed_and_buffer_one_sample(state, detector, timestamp, value, is_injected)
    return result is not None


def _process_simulation_tick(state: MutableMapping) -> bool:
    """Advance scripted scenario playback by every currently-due row (D2).

    Polls ``state["sim_player"].due_rows(now)`` -- using
    ``time.monotonic()`` to match the time domain
    :meth:`~core.simulation_loader.SimulationPlayer.start`/``due_rows``
    expect -- and feeds/buffers/logs *every* row returned (there can be
    zero, one, or several per poll; see ``due_rows``'s docstring for why
    it bursts rather than delivering at most one row per call). Once
    ``sim_player.finished`` is True after processing this poll's rows
    (either because the scenario naturally ran out of rows, or because
    "Stop Sim" already called :meth:`~core.simulation_loader.SimulationPlayer.stop`),
    ``state["app_state"]`` is transitioned back to ``IDLE`` so a
    scripted demo cleanly stops itself without the operator clicking
    anything.

    :param state: Session-state mapping to read/mutate.
    :type state: MutableMapping
    :returns: True if at least one row was processed this call; False
        if no rows were due yet, or the call was a defensive no-op
        (missing detector/sim_player).
    :rtype: bool
    """
    detector = state.get("detector")
    sim_player = state.get("sim_player")
    if detector is None or sim_player is None:
        # Defensive: session state not yet initialized (init_session_state()
        # hasn't run). Log and skip rather than raising AttributeError.
        logger.warning(
            "_process_simulation_tick: 'detector'/'sim_player' missing "
            "from session state; skipping tick (was init_session_state() "
            "called?)"
        )
        return False

    due_rows: List[Tuple[float, float, bool]] = sim_player.due_rows(time.monotonic())
    processed_any = False
    for timestamp, value, is_row_anomaly in due_rows:
        result = _feed_and_buffer_one_sample(state, detector, timestamp, value, is_row_anomaly)
        if result is not None:
            processed_any = True

    if sim_player.finished:
        state["app_state"] = AppState.IDLE
        logger.info("Simulation finished (or stopped); app_state -> IDLE")

    return processed_any


def process_one_tick(state: Optional[MutableMapping] = None) -> bool:
    """Advance either the live stream or scripted playback by one poll's worth of samples.

    Dispatches on ``state["app_state"]``:

    * ``RUNNING`` -- calls ``generator.tick()`` exactly once and feeds
      the single resulting sample through the detector/ring-buffers/
      logbook pipeline (see :func:`_process_running_tick`). Unchanged
      behavior/contract versus prior stories: exactly one sample per
      call, ``True`` returned iff it was successfully buffered.
    * ``SIMULATION_RUNNING`` -- polls ``sim_player.due_rows()`` and
      feeds *every* due row (zero, one, or many) through the same
      per-sample pipeline (see :func:`_process_simulation_tick`), then
      auto-transitions ``app_state`` back to ``IDLE`` once the scenario
      is exhausted or was stopped.
    * ``IDLE`` / ``PAUSED`` -- a deliberate no-op (generates nothing,
      feeds nothing, appends nothing, leaves ``total_samples`` unchanged).

    Because the ``SIMULATION_RUNNING`` branch can process anywhere from
    zero to many samples in a single call, this function's boolean
    return value means "at least one sample was processed on this
    call" (not "exactly one sample was processed", which is now only
    true for the ``RUNNING`` branch) -- callers that need the exact
    count should read ``state["total_samples"]`` before/after instead.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps this
        function unit-testable outside a running Streamlit script (see
        module docstring).
    :type state: MutableMapping | None
    :returns: True if at least one sample was generated/pulled, scored,
        and buffered on this call; False if the call was a no-op.
    :rtype: bool
    """
    if state is None:
        state = st.session_state

    # Defensive default: treat a missing app_state as IDLE rather than
    # raising, so a stray call before init_session_state() has run is a
    # silent no-op instead of a crash.
    current_state = state.get("app_state", AppState.IDLE)
    if current_state is AppState.RUNNING:
        return _process_running_tick(state)
    if current_state is AppState.SIMULATION_RUNNING:
        return _process_simulation_tick(state)

    logger.debug("process_one_tick: no-op (app_state=%s)", current_state)
    return False


def _build_index_x_axis(total_samples: int, window_length: int) -> npt.NDArray:
    """Build an absolute, monotonically increasing sample-index x-axis.

    Mirrors the desktop app's ``ui/visualization_panel.py`` convention
    of plotting by *absolute* sample index (``np.arange(total_count -
    n, total_count)``) rather than by wall-clock timestamp or by a
    window-relative ``0..n-1`` index. Using the absolute count (not a
    window-relative index) matters once the ring buffer starts
    dropping old samples: the x-axis keeps climbing sample-by-sample
    instead of resetting to ``0..capacity-1`` forever once the buffer
    first fills up, which is what "monotonically increasing" means in
    the acceptance criteria.

    :param total_samples: Cumulative number of samples processed so
        far this session (``state["total_samples"]``).
    :type total_samples: int
    :param window_length: Number of samples currently held in the
        ring buffers (``<= total_samples``, capped at buffer capacity).
    :type window_length: int
    :returns: Array of ``window_length`` consecutive absolute sample
        indices, oldest first, ending at ``total_samples - 1``. Empty
        if ``window_length`` is 0.
    :rtype: numpy.ndarray
    """
    # Defensive: a non-positive window length has no meaningful index
    # range -- return an empty array rather than letting np.arange
    # produce a negative-length/backwards range.
    if window_length <= 0:
        return np.empty(0, dtype=float)
    return np.arange(total_samples - window_length, total_samples, dtype=float)


def update_chart_from_state(
    state: Optional[MutableMapping] = None, chart: Optional[ChartState] = None
) -> ChartState:
    """Redraw the persistent chart's artists from the current ring-buffer windows.

    Pulls the full current window (oldest-to-newest) out of the five
    ring buffers, builds an absolute sample-index x-axis (see
    :func:`_build_index_x_axis`), and mutates the chart's artists via
    :meth:`~ui_web.chart.ChartState.update`. Deliberately does **not**
    render the figure itself (neither ``st.plotly_chart`` nor the P1
    chart placeholder's ``.plotly_chart(...)`` call) -- that belongs in
    the ``@st.fragment`` wrapper (:func:`live_tick_fragment`) so this
    function stays unit-testable with an injected fake ``state``/
    ``chart`` outside of a live Streamlit script context.

    Called by :func:`live_tick_fragment` only while ``app_state`` is
    ``RUNNING`` or ``SIMULATION_RUNNING`` (story P1 skips this call
    entirely while IDLE/PAUSED, since nothing in the ring buffers has
    changed since the last redraw -- see that function's docstring for
    why skipping it does not blank the chart).

    :param state: Session-state mapping to read the ring buffers and
        ``total_samples`` from. Defaults to ``st.session_state``.
    :type state: MutableMapping | None
    :param chart: Chart instance to update. Defaults to the session's
        singleton via :func:`~ui_web.chart.get_or_create_chart_state`;
        accepting an injectable chart keeps this function testable
        without a real Streamlit session/figure.
    :type chart: ChartState | None
    :returns: The chart instance that was updated (whichever was
        passed in, or the session singleton).
    :rtype: ChartState
    """
    if state is None:
        state = st.session_state
    if chart is None:
        chart = get_or_create_chart_state()

    # Defensive: resolve all five ring buffers up front (same pattern as
    # _append_sample_to_ring_buffers) so a partially-initialized session
    # state produces a logged no-op rather than a half-applied redraw.
    try:
        buffers = [state[key] for key in _RING_BUFFER_KEYS]
    except KeyError:
        logger.warning(
            "update_chart_from_state: one or more ring buffer keys missing "
            "from session state; skipping chart redraw (was "
            "init_session_state() called?)"
        )
        return chart

    _ts_buf, vals_buf, confs_buf, is_anomaly_buf, is_injected_buf = buffers
    vals_window = vals_buf.get_window()
    confs_window = confs_buf.get_window()
    is_anomaly_window = is_anomaly_buf.get_window()
    is_injected_window = is_injected_buf.get_window()

    total_samples = state.get("total_samples", 0)
    xs = _build_index_x_axis(total_samples, len(vals_window))

    chart.update(
        timestamps=xs,
        values=vals_window,
        confidences=confs_window,
        is_anomaly=is_anomaly_window,
        is_injected=is_injected_window,
    )
    return chart


def _safe_widget_state_set(state: MutableMapping, key: str, value: object) -> None:
    """Set ``state[key]``, tolerating Streamlit's "already instantiated" guard.

    Streamlit raises :class:`~streamlit.errors.StreamlitAPIException` if a
    session-state key bound to a widget is written *after* that widget
    has already been instantiated during the current script run. In
    normal operation this never triggers here (each fragment tick is a
    fresh script run, and this helper is only ever called before the
    corresponding widget is created within that run) -- it exists as a
    defensive fallback for callers that invoke
    :func:`render_gmm_training_group` more than once within a single
    script execution (e.g. certain test harnesses that call the
    ``@st.fragment``-decorated function repeatedly in a loop without an
    intervening real rerun), where re-instantiating the same key a
    second time would otherwise crash instead of just keeping the
    already-rendered value.

    :param state: Session-state mapping to write into. Only real
        ``st.session_state`` (or an object raising the same exception
        type) exercises the guarded path; a plain dict/fake mapping
        used in unit tests simply gets the value set unconditionally.
    :type state: MutableMapping
    :param key: The session-state key to set.
    :type key: str
    :param value: The value to assign.
    :type value: object
    :returns: None
    :rtype: None
    """
    try:
        state[key] = value
    except StreamlitAPIException:
        logger.debug(
            "_safe_widget_state_set: skipped %s=%r (widget already "
            "instantiated this script run)",
            key,
            value,
        )


def _on_end_spinbox_manual_edit(state: Optional[MutableMapping] = None) -> None:
    """``on_change`` callback for the training-window "End" number_input.

    Flips ``cfg_end_tracking`` to False so the auto-tracking sync in
    :func:`render_gmm_training_group` stops overwriting the user's
    manually-entered value on the next fragment tick. Mirrors the
    desktop app's ``ConfigPanel._on_end_spin_changed``, which set
    ``self._end_tracking = False`` on the spinbox's ``valueChanged``.

    Streamlit calls ``on_change`` callbacks with no positional
    arguments by default (same pattern as
    ``ui_web.sidebar_config._on_template_changed``), so ``state``
    defaults to ``st.session_state`` when used as a real widget
    callback; the parameter exists purely so this function stays
    unit-testable with an injected fake mapping.

    :param state: Session-state mapping to mutate. Defaults to
        ``st.session_state``.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state
    state["cfg_end_tracking"] = False
    logger.debug("End spinbox manually edited; cfg_end_tracking set to False")


def _on_retrain_clicked(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the "Retrain Model" button.

    Calls ``detector.retrain(cfg_retrain_start, cfg_retrain_end)`` and
    immediately recomputes the chart's GMM overlay from a fresh
    ``detector.get_model_params()`` call, mirroring the desktop app's
    ``ConfigPanel._on_retrain_clicked`` -> ``retrain_requested`` signal
    -> ``MainWindow``'s handler that called ``detector.retrain(...)``
    then pushed the new params into ``VisualizationPanel.set_gmm_overlay``.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this callback unit-testable outside a running Streamlit script.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    detector = state.get("detector")
    if detector is None:
        # Defensive: session state not yet initialized.
        logger.warning(
            "_on_retrain_clicked: 'detector' missing from session state; "
            "skipping (was init_session_state() called?)"
        )
        return

    start = int(state.get("cfg_retrain_start", 0))
    end = int(state.get("cfg_retrain_end", 0))
    detector.retrain(start, end)
    logger.info("Retrain Model clicked: detector.retrain(start=%d, end=%d)", start, end)

    chart = get_or_create_chart_state()
    chart.update_gmm_overlay(detector.get_model_params())


def render_gmm_training_group(state: Optional[MutableMapping] = None) -> None:
    """Render the sidebar "GMM Training" group and sync the chart's GMM overlay.

    Ports ``ui/config_panel.py``'s ``_build_training_group`` (progress
    bar, status text, training-window Start/End spinboxes with
    end-auto-tracking, "Retrain Model" button) to Streamlit widgets, and
    -- since there is no separate Qt-style ``set_gmm_overlay`` call site
    on the web -- also keeps the persistent chart's overlay
    (:meth:`ui_web.chart.ChartState.update_gmm_overlay`) synced with the
    detector's current trained state on every call. Intended to be
    called from within a ``with st.sidebar:`` block inside
    :func:`live_tick_fragment`, so this whole group re-renders on the
    same 300 ms cadence as the live chart.

    End auto-tracking: while ``state["cfg_end_tracking"]`` is True, the
    End spinbox's value is overwritten from ``state["total_samples"]``
    *before* the ``number_input`` widget bound to ``cfg_retrain_end`` is
    instantiated this run (the only safe way to programmatically change
    a widget-bound session-state value in Streamlit -- see
    ``ui_web.sidebar_config`` module docstring for the same pattern).
    Editing the End field directly fires :func:`_on_end_spinbox_manual_edit`,
    which sets ``cfg_end_tracking = False`` so this sync stops fighting
    the user's edit on subsequent ticks.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps the
        non-widget bookkeeping unit-testable. The widgets themselves
        always bind to the real ``st.session_state`` (Streamlit widgets
        cannot be bound to an arbitrary mapping).
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    detector = state.get("detector")
    st.subheader("GMM Training")

    if detector is None:
        # Defensive: session state not yet initialized. Still show the
        # group header (for layout stability) but skip the rest.
        logger.warning(
            "render_gmm_training_group: 'detector' missing from session "
            "state; skipping (was init_session_state() called?)"
        )
        return

    # --- Progress bar + status text ----------------------------------
    is_trained = detector.is_trained
    pct = min(100, max(0, int(round(detector.training_progress * 100))))
    if is_trained:
        st.progress(100, text="Trained ✓")
        st.markdown(
            '<span style="color:#50fa7b">Model ready</span>', unsafe_allow_html=True
        )
    else:
        st.progress(pct, text=f"Collecting… {pct}%")
        st.markdown(
            '<span style="color:#ff5555">Not trained</span>', unsafe_allow_html=True
        )

    # --- Training-window Start/End spinboxes -------------------------
    st.caption("Training Window")

    total_samples = int(state.get("total_samples", 0))

    # Auto-track the End value from total_samples *before* the widget
    # below is instantiated this run -- the only safe point to mutate a
    # widget-bound session-state key (see docstring / sidebar_config.py).
    if state.get("cfg_end_tracking", True):
        _safe_widget_state_set(state, "cfg_retrain_end", total_samples)

    # Defensive: clamp both bounds into their valid [0, max] range
    # before instantiating the number_input widgets below -- Streamlit
    # raises if a widget's session-state value falls outside
    # min_value/max_value (e.g. after a session reset shrinks
    # total_samples back to 0 while a stale larger value lingers).
    start_max = max(total_samples - 1, 0)
    end_max = total_samples
    _safe_widget_state_set(
        state, "cfg_retrain_start", min(max(int(state.get("cfg_retrain_start", 0)), 0), start_max)
    )
    _safe_widget_state_set(
        state, "cfg_retrain_end", min(max(int(state.get("cfg_retrain_end", 0)), 0), end_max)
    )

    col_start, col_end = st.columns(2)
    with col_start:
        st.number_input("Start", key="cfg_retrain_start", min_value=0, max_value=start_max, step=1)
    with col_end:
        st.number_input(
            "End",
            key="cfg_retrain_end",
            min_value=0,
            max_value=end_max,
            step=1,
            on_change=_on_end_spinbox_manual_edit,
        )

    # G1: key="btn_retrain" gives this button a stable ``.st-key-btn_retrain``
    # CSS class (Streamlit prefixes any widget ``key=`` with ``st-key-`` on
    # its wrapping DOM element) so ui_web.styling.inject_theme_css() can
    # scope purple styling to it; the "⟳" icon lets the button's purpose
    # ("Retrain") remain recognizable even if that injected CSS ever fails
    # to apply, matching the icon convention used by the other
    # semantically-colored buttons in sidebar_config.py (e.g. "▶ Start").
    st.button(
        "⟳ Retrain Model",
        key="btn_retrain",
        on_click=_on_retrain_clicked,
        use_container_width=True,
    )

    # --- Keep the chart's GMM overlay synced with the trained state --
    # Called every tick, but update_gmm_overlay() is a cheap no-op
    # unless the underlying model_params actually changed (signature
    # check inside ChartState), so this does not force a redraw churn
    # on every single 300ms cycle.
    chart = get_or_create_chart_state()
    chart.update_gmm_overlay(detector.get_model_params())


@st.fragment(run_every=LIVE_TICK_INTERVAL)
def live_tick_fragment() -> None:
    """Streamlit fragment that ticks the live data stream every 300 ms.

    This is the actual live loop entry point wired into the app script.
    It re-runs on its own :data:`LIVE_TICK_INTERVAL` cadence (300 ms,
    independent of full-script reruns triggered by widget interaction)
    and, on every run:

    1. Delegates to :func:`process_one_tick` (generate + score + buffer
       exactly one new sample, if RUNNING; a no-op otherwise).
    2. Delegates to :func:`render_gmm_training_group` (story B4),
       rendered into ``st.sidebar`` -- progress bar, status text,
       training-window spinboxes, "Retrain Model" button -- and syncs
       the chart's GMM overlay with the detector's current trained
       state.
    3. While ``app_state`` is ``RUNNING`` or ``SIMULATION_RUNNING``,
       delegates to :func:`update_chart_from_state` to redraw the
       persistent chart's traces from whatever is currently in the
       ring buffers, then renders the chart's persistent Plotly
       ``Figure`` (story Q1) by writing it into the session's chart
       placeholder (looked up via :func:`ui_web.chart.get_chart_placeholder`
       -- created by ``app.py``'s ``main()``, never by this fragment
       itself) via ``placeholder.plotly_chart(chart.fig,
       use_container_width=True, key=...)`` (mutating the same
       persistent ``Figure`` object every cycle -- asserting the story
       B1 invariant that it is never rebuilt from scratch, only mutated
       in place across the whole session -- unlike matplotlib's
       ``st.pyplot``, ``st.plotly_chart`` requires a stable, distinct
       ``key=`` per call site sharing a script run to avoid a
       ``StreamlitDuplicateElementId`` error, see
       :data:`_CHART_KEY_LIVE`/:data:`_CHART_KEY_FALLBACK`). If no
       placeholder exists yet (e.g. this fragment was invoked directly
       without ``app.py``'s ``main()`` having run first -- a test-only
       scenario), falls back to a bare ``st.plotly_chart(...)`` call so
       the chart still renders, just without story P1's persistence
       guarantee for that session.
    4. While ``app_state`` is ``IDLE`` or ``PAUSED`` (story P1), *skips*
       both of those calls entirely -- neither the chart traces nor
       the placeholder's ``plotly_chart`` call are touched this cycle,
       since nothing in the ring buffers changed since the last redraw. This
       does not blank the chart: during a true 300 ms fragment-only
       auto-rerun, Streamlit never re-executes ``app.py``'s outer
       script at all, so the placeholder -- created outside this
       fragment's scope -- and whatever was last written into it are
       left completely untouched; and on the (much rarer) full-script
       rerun that actually changes ``app_state`` to IDLE/PAUSED (e.g. a
       "Pause"/"Stop" click), ``app.py``'s ``main()`` itself
       unconditionally re-renders the chart's current Figure into a
       freshly (re)created placeholder before this fragment even runs
       -- see :func:`ui_web.chart.render_chart_into_placeholder`'s
       docstring for the full two-part mechanism and why the
       placeholder must be recreated every full run rather than
       cached.

    :returns: None
    :rtype: None
    """
    process_one_tick()
    with st.sidebar:
        render_gmm_training_group()

    # P1: only redraw/re-render the chart while actively streaming.
    # IDLE/PAUSED cycles have nothing new in the ring buffers to show,
    # so skipping both calls saves the redraw + re-render cost every
    # 300ms an idle tab sits open, without hiding the last frame that
    # *was* rendered (see docstring above and ui_web.chart's docstrings
    # for the full mechanism).
    current_state = st.session_state.get("app_state", AppState.IDLE)
    if current_state in (AppState.RUNNING, AppState.SIMULATION_RUNNING):
        chart = update_chart_from_state()
        # Write into the outer-script-created placeholder (see
        # ui_web.chart.get_chart_placeholder) rather than calling a bare
        # top-level st.plotly_chart(...) here -- see this function's and
        # ui_web.chart.render_chart_into_placeholder's docstrings for
        # why that is required for the last frame to survive a fragment
        # cycle that skips this call.
        placeholder = get_chart_placeholder()
        if placeholder is None:
            # Defensive fallback: no placeholder was set up (e.g. a
            # test/caller invoked this fragment without app.py's main()
            # having run first this session). Render directly so the
            # chart still shows up -- degraded (no P1 persistence
            # guarantee), but not a crash. A key distinct from both
            # _CHART_KEY_OUTER and _CHART_KEY_LIVE, since this branch
            # can in principle run in the same script execution as
            # neither, but must never collide with either if a test
            # exercises multiple call sites in one run.
            logger.warning(
                "live_tick_fragment: no chart placeholder in session state "
                "(was app.py's main() -> render_chart_into_placeholder() "
                "called first?); falling back to a bare st.plotly_chart() call"
            )
            st.plotly_chart(chart.fig, use_container_width=True, key=_CHART_KEY_FALLBACK)
        else:
            placeholder.plotly_chart(chart.fig, use_container_width=True, key=_CHART_KEY_LIVE)
