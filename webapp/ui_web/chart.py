"""Persistent, in-place-mutated two-panel Plotly chart component.

Ports the visual design of the desktop app's
``ui/visualization_panel.py`` (a two-panel Qt-embedded matplotlib
figure: a main value line + scatter panel on top, a confidence-score
panel below) to a Streamlit-friendly shape.

Story Q1 replaced the original server-rendered matplotlib figure with a
Plotly :class:`plotly.graph_objects.Figure`, rendered client-side via
``st.plotly_chart``/``placeholder.plotly_chart``. The reason for the
swap: a matplotlib figure has to be *rasterized on the server* (this
process, shared across every concurrently connected browser session)
every single redraw, then shipped down as a PNG -- CPU-bound work that
holds the GIL and does not scale past a handful of simultaneous
viewers. A Plotly figure, by contrast, is shipped down to the browser
as a small JSON spec (trace data + layout) and rendered/redrawn
entirely client-side by plotly.js; the server's job on each ``update()``
call shrinks to "mutate a few numpy arrays and re-serialize JSON",
which is far cheaper and does not compete for the same shared
CPU/GIL resource across sessions.

The key architectural invariant carried over unchanged from the
matplotlib version: the :class:`plotly.graph_objects.Figure` is built
**exactly once** per browser session and stored in
``st.session_state.chart`` (see :func:`get_or_create_chart_state`).
Streamlit re-executes the whole script top-to-bottom on every rerun
(every widget interaction, every ``st.fragment`` tick); if a new
``Figure`` were built on every rerun, this would waste CPU rebuilding
the same static layout/trace scaffolding over and over. Building it
once and mutating the existing traces' ``x``/``y``/``marker.color``
arrays in place (see :meth:`ChartState.update`) keeps each redraw's
cost proportional to "how many samples changed", not "how large the
figure is".

Story B4's GMM overlay (trained-model bands, dashed mean lines, dotted
decision-boundary lines) is ported to Plotly layout *shapes*
(``fig.add_shape``/``fig.add_hline``) rather than traces, mirroring the
matplotlib version's ``axhspan``/``axhline`` -- see
:meth:`ChartState.update_gmm_overlay` for how conditional presence
(absent before training, present after) and the "skip redundant
remove+recreate work when unchanged" signature-check optimization are
preserved.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit.delta_generator import DeltaGenerator

logger = logging.getLogger(__name__)

#: Array-like accepted for update() series arguments -- either a plain
#: Python sequence (e.g. list) or a numpy array (e.g. RingBuffer.get_window()
#: output), so callers do not have to coerce ring-buffer windows themselves.
_ArrayLike = Union[Sequence[float], npt.NDArray]

# ---------------------------------------------------------------------------
# Dracula-derived palette constants
# ---------------------------------------------------------------------------
# Mirrors assets/style.qss / the rcParams set in the desktop app's main.py,
# pulled out as named constants (rather than inline hex literals scattered
# through the class) so later stories reusing the same palette (B4 GMM
# overlay, dashboard pie chart) have a single source of truth to import.
# Story Q1: unchanged names/values -- ui_web/dashboard.py imports several of
# these directly and must keep working without modification.
COLOR_FIG_BG = "#1e1e2e"
COLOR_AXES_BG = "#2a2a3e"
COLOR_GRID = "#44475a"
COLOR_TEXT = "#f8f8f2"
COLOR_LINE = "#6272a4"
COLOR_NORMAL = "#50fa7b"
COLOR_ANOMALY = "#ff5555"

#: GMM overlay colors (story B4), matching
#: ``ui/visualization_panel.py``'s ``set_gmm_overlay``: bands/mean lines
#: cycle through these colors per component (purple, then orange, then
#: wrapping around again if a future model ever had >2 components --
#: today ``AnomalyDetector`` always uses exactly 1 component), and
#: decision-boundary crossing lines are always this pink.
GMM_BAND_COLORS: Tuple[str, ...] = ("#bd93f9", "#ffb86c")
GMM_BOUNDARY_COLOR = "#ff79c6"

#: Default rolling-window capacity the chart is sized for. Matches the
#: rolling ring buffer capacity used elsewhere in the app
#: (state.app_state._RING_BUFFER_CAPACITY). Unlike the matplotlib
#: version, Plotly's Bar trace does not need a fixed-size pre-allocated
#: patch pool -- update() simply truncates to the most recent
#: `capacity` samples if the caller passes more (see update()'s
#: docstring) -- but the constant is kept for that truncation contract
#: and as the default ChartState(capacity=...) value.
DEFAULT_CAPACITY = 300

#: Streamlit element key for the ``st.plotly_chart`` call made by
#: :func:`render_chart_into_placeholder` (the *outer*, non-fragment
#: script's write into a freshly (re)created placeholder -- see that
#: function's docstring). Plotly's ``st.plotly_chart`` (unlike
#: matplotlib's ``st.pyplot``) auto-derives an internal element ID from
#: (element type, parameters) and raises
#: ``StreamlitDuplicateElementId`` if two ``plotly_chart`` calls land in
#: the *same* script run with the same derived ID -- which happens here
#: because a ``@st.fragment``-decorated function runs inline as part of
#: the very outer script run that (re)creates the placeholder (see
#: ``ui_web.live_tick.live_tick_fragment``, which writes into the same
#: placeholder via :data:`_CHART_KEY_LIVE` when RUNNING/
#: SIMULATION_RUNNING). Giving each call site its own fixed, distinct
#: key sidesteps that without needing a fresh key per call (which would
#: defeat reusing/looking up the same placeholder across reruns).
_CHART_KEY_OUTER = "chart_plotly_outer"


class ChartState:
    """Two-panel Plotly chart, built once and mutated in place.

    Holds a persistent :class:`plotly.graph_objects.Figure` with two
    stacked subplots (row 1 for the value line + anomaly scatter, row 2
    for the per-sample confidence bars) plus references to every
    plotted trace. :meth:`update` mutates those traces' ``x``/``y``/
    ``marker.color`` attributes in place rather than clearing the
    figure or re-plotting, so the same trace objects can be reused for
    an entire Streamlit browser session -- Plotly traces support
    wholesale array reassignment (``trace.x = new_array``) without ever
    needing to be removed/recreated, so (unlike the matplotlib version)
    no fixed-capacity pre-allocated artist pool is needed for the
    confidence bars either.

    Instances should not be constructed directly by UI code -- use
    :func:`get_or_create_chart_state` so exactly one instance is ever
    created per session.

    :param capacity: Maximum number of samples :meth:`update` will plot
        at once; a longer window passed to :meth:`update` is truncated
        to the most recent ``capacity`` samples (with a logged
        warning). Defaults to :data:`DEFAULT_CAPACITY`.
    :type capacity: int
    :raises ValueError: If ``capacity`` is not a positive integer.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        # Defensive: a non-positive capacity would make the truncation
        # logic in update() meaningless (or misbehave with a
        # negative-length slice).
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity

        # Two stacked subplots, 75%/25% height split, sharing the x-axis
        # so panning/zooming the value panel keeps the confidence panel
        # aligned -- mirrors ui/visualization_panel.py's _setup_figure()
        # and the matplotlib version's GridSpec(2, 1, height_ratios=[3, 1]).
        self.fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.06,
        )

        # --- Dracula palette, applied at the figure/axes level ----------
        self.fig.update_layout(
            paper_bgcolor=COLOR_FIG_BG,
            plot_bgcolor=COLOR_AXES_BG,
            font=dict(color=COLOR_TEXT, size=11),
            legend=dict(
                bgcolor=COLOR_AXES_BG,
                bordercolor=COLOR_GRID,
                borderwidth=1,
                font=dict(color=COLOR_TEXT, size=10),
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
            ),
            margin=dict(l=55, r=20, t=20, b=45),
            height=500,
            showlegend=True,
            uirevision="chart-state",
        )
        self.fig.update_xaxes(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            linecolor=COLOR_GRID,
            color=COLOR_TEXT,
            showgrid=True,
        )
        self.fig.update_yaxes(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_GRID,
            linecolor=COLOR_GRID,
            color=COLOR_TEXT,
            showgrid=True,
        )
        self.fig.update_yaxes(title_text="Value", row=1, col=1)
        self.fig.update_yaxes(title_text="Confidence", range=[0, 1], row=2, col=1)
        self.fig.update_xaxes(title_text="Sample Index", row=2, col=1)

        # --- Persistent traces, created once, mutated by update() -------
        self.fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=COLOR_LINE, width=1.4),
                name="Value",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        self.line = self.fig.data[-1]

        self.fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color=COLOR_NORMAL, size=6),
                name="Normal",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        self.scatter_normal = self.fig.data[-1]

        self.fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(color=COLOR_ANOMALY, size=11, symbol="triangle-up"),
                name="Anomaly",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        self.scatter_anomaly = self.fig.data[-1]

        # A single Bar trace, whole x/y/marker.color arrays replaced on
        # every update() call -- see class docstring for why this needs
        # no fixed-capacity pre-allocated pool the way the matplotlib
        # version's ax_score.bar() patches did.
        self.fig.add_trace(
            go.Bar(
                x=[],
                y=[],
                marker=dict(color=[]),
                width=1.0,
                name="Confidence",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        self.bar_trace = self.fig.data[-1]

        #: Number of samples represented by the most recent update() call.
        self._n_points = 0

        # --- GMM overlay bookkeeping (story B4), populated lazily --------
        # Empty until the first successful update_gmm_overlay(params) call
        # -- unlike the always-present line/scatter/bar traces above,
        # these are conditionally present (see module + method docstrings).
        # Each list holds references into self.fig.layout.shapes (Plotly
        # layout.Shape objects), analogous to the matplotlib version's
        # lists of Artist references.
        self.gmm_band_artists: List[go.layout.Shape] = []
        self.gmm_mean_lines: List[go.layout.Shape] = []
        self.gmm_boundary_lines: List[go.layout.Shape] = []

        #: Number of layout shapes present before any GMM overlay was
        #: ever drawn (always 0 today, since nothing else in this app
        #: adds figure-level shapes) -- used by _clear_gmm_overlay() to
        #: know where the "not ours" prefix of fig.layout.shapes ends,
        #: so clearing never touches a shape this class did not add.
        self._non_gmm_shape_count: int = len(self.fig.layout.shapes)

        #: Whether a GMM overlay is currently drawn on the main panel.
        self.gmm_overlay_active: bool = False

        #: Signature of the model_params the overlay was last drawn from
        #: (means/stds/boundary_values, as tuples), used to skip
        #: redundant remove+recreate work on repeated calls with
        #: unchanged params (see update_gmm_overlay docstring).
        self._gmm_overlay_signature: Optional[tuple] = None

        logger.debug("ChartState initialized (capacity=%d)", capacity)

    def update(
        self,
        timestamps: _ArrayLike,
        values: _ArrayLike,
        confidences: _ArrayLike,
        is_anomaly: _ArrayLike,
        is_injected: Optional[_ArrayLike] = None,
    ) -> None:
        """Mutate the persistent traces in place with a new data window.

        All five series must be the same length, ordered
        oldest-to-newest (the same convention as
        :meth:`utils.ring_buffer.RingBuffer.get_window`). No trace is
        recreated here -- only ``x``/``y``/``marker.color`` attribute
        reassignments on the trace objects created in :meth:`__init__`
        (which Plotly applies as an in-place patch to the existing
        trace, not a new trace), plus a re-assertion of the confidence
        panel's fixed y-range (defensive against anything upstream
        nudging it).

        :param timestamps: X-axis values (e.g. absolute sample index or
            time) for each sample, oldest first.
        :type timestamps: Sequence[float] | numpy.ndarray
        :param values: Sensor value for each sample.
        :type values: Sequence[float] | numpy.ndarray
        :param confidences: Anomaly-detector confidence score in
            ``[0, 1]`` for each sample.
        :type confidences: Sequence[float] | numpy.ndarray
        :param is_anomaly: Boolean flag per sample; True draws the
            sample as an anomaly marker/bar color instead of normal.
        :type is_anomaly: Sequence[bool] | numpy.ndarray
        :param is_injected: Optional boolean flag per sample marking
            synthetically injected anomalies. Accepted for forward
            compatibility with later stories (e.g. a refined visual
            distinction between injected and detected anomalies); this
            story does not yet render it differently.
        :type is_injected: Sequence[bool] | numpy.ndarray | None
        :raises ValueError: If the input series have mismatched lengths.
        :returns: None
        :rtype: None
        """
        xs = np.asarray(timestamps, dtype=float)
        vals = np.asarray(values, dtype=float)
        confs = np.asarray(confidences, dtype=float)
        anom = np.asarray(is_anomaly, dtype=bool)

        # Defensive: mismatched lengths would silently corrupt the plot
        # (e.g. numpy broadcasting/truncating in confusing ways) if left
        # unchecked -- fail loudly instead.
        lengths = {len(xs), len(vals), len(confs), len(anom)}
        if is_injected is not None:
            injected = np.asarray(is_injected, dtype=bool)
            lengths.add(len(injected))
        if len(lengths) > 1:
            raise ValueError(
                f"update() series must all have equal length, got lengths {lengths}"
            )

        n = len(xs)
        self._n_points = n

        # Truncate to the most recent `capacity` samples if the caller
        # passed more than the chart is sized for -- defensive rather
        # than raising, since capacity mismatches are a caller
        # configuration issue, not a reason to crash a live demo.
        if n > self.capacity:
            logger.warning(
                "ChartState.update: got %d samples, exceeds capacity=%d; "
                "truncating to the most recent %d",
                n,
                self.capacity,
                self.capacity,
            )
            xs, vals, confs, anom = (
                xs[-self.capacity :],
                vals[-self.capacity :],
                confs[-self.capacity :],
                anom[-self.capacity :],
            )
            n = self.capacity

        # --- Main line -----------------------------------------------------
        self.line.x = xs
        self.line.y = vals

        # --- Normal / anomaly scatter overlay -------------------------------
        normal_mask = ~anom
        if np.any(normal_mask):
            self.scatter_normal.x = xs[normal_mask]
            self.scatter_normal.y = vals[normal_mask]
        else:
            self.scatter_normal.x = []
            self.scatter_normal.y = []

        if np.any(anom):
            self.scatter_anomaly.x = xs[anom]
            self.scatter_anomaly.y = vals[anom]
        else:
            self.scatter_anomaly.x = []
            self.scatter_anomaly.y = []

        # --- Confidence bars -------------------------------------------------
        # Wholesale array replacement on the same persistent Bar trace
        # object (see class docstring) -- no per-bar patch pool needed.
        bar_colors = [COLOR_ANOMALY if flag else COLOR_NORMAL for flag in anom]
        self.bar_trace.x = xs
        self.bar_trace.y = confs
        self.bar_trace.marker.color = bar_colors

        # Re-assert the fixed confidence-panel y-range defensively, in
        # case anything upstream (e.g. a future story) nudges it via
        # shared-axis interactions.
        self.fig.update_yaxes(range=[0, 1], row=2, col=1)

        logger.debug("ChartState.update: redrew %d samples in place", n)

    def update_gmm_overlay(self, model_params: Optional[dict]) -> None:
        """Draw (or clear) the GMM component/decision-boundary overlay on the main panel.

        Ports ``ui/visualization_panel.py``'s ``set_gmm_overlay`` to the
        web chart: for each GMM component, a translucent rectangle
        (``fig.add_shape(type="rect", ...)``) band spanning ``mean +/-
        2*std`` plus a dashed mean line (``fig.add_hline``), both
        colored by cycling through :data:`GMM_BAND_COLORS`; and for each
        entry in ``boundary_values``, a dotted
        :data:`GMM_BOUNDARY_COLOR` line (also ``fig.add_hline``) marking
        where the score curve crosses the anomaly threshold. Both band
        rectangles and lines are implemented as Plotly layout *shapes*
        (``fig.layout.shapes``), the closest Plotly analog to
        matplotlib's ``axhspan``/``axhline`` artists.

        Unlike :meth:`update` (which unconditionally mutates a fixed
        set of persistent traces every call), this method's shapes are
        conditionally *present* -- absent before the model is ever
        trained, present afterwards. Since Plotly shapes have no
        per-shape ``.remove()`` the way matplotlib artists do, and the
        shape *count* can change between calls (e.g. a retrain with a
        different number of threshold crossings), old overlay shapes
        are cleared by truncating ``fig.layout.shapes`` back to its
        pre-overlay length (see :meth:`_clear_gmm_overlay`) and new
        ones added, rather than mutated in place. To avoid doing that
        clear+recreate churn on *every* call from a caller that invokes
        this once per fragment tick, a cheap signature check
        short-circuits when ``model_params`` is identical to what was
        last drawn.

        :param model_params: The dict returned by
            :meth:`core.anomaly_detector.AnomalyDetector.get_model_params`
            (keys ``means``, ``stds``, ``weights``, ``boundary_values``),
            or ``None`` if the model is not (yet) trained -- passing
            ``None`` clears any existing overlay.
        :type model_params: dict | None
        :returns: None
        :rtype: None
        """
        if model_params is None:
            if self.gmm_overlay_active:
                logger.debug("update_gmm_overlay: model_params is None; clearing overlay")
                self._clear_gmm_overlay()
            return

        # Defensive: tolerate a partial/malformed params dict (e.g. a
        # future caller bug) rather than raising mid-render -- treat
        # missing keys as empty lists.
        means = list(model_params.get("means", []))
        stds = list(model_params.get("stds", []))
        boundary_values = list(model_params.get("boundary_values", []))

        signature = (tuple(means), tuple(stds), tuple(boundary_values))
        if self.gmm_overlay_active and signature == self._gmm_overlay_signature:
            logger.debug("update_gmm_overlay: params unchanged; skipping redraw")
            return

        self._clear_gmm_overlay()

        for i, (mean, std) in enumerate(zip(means, stds)):
            color = GMM_BAND_COLORS[i % len(GMM_BAND_COLORS)]

            # Translucent band spanning the full plot width (x domain
            # coordinates, 0..1) at mean +/- 2*std -- Plotly's analog of
            # matplotlib's ax.axhspan(...).
            self.fig.add_shape(
                type="rect",
                xref="x domain",
                x0=0,
                x1=1,
                yref="y",
                y0=mean - 2 * std,
                y1=mean + 2 * std,
                fillcolor=color,
                opacity=0.12,
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )
            self.gmm_band_artists.append(self.fig.layout.shapes[-1])

            # Dashed mean line -- fig.add_hline conveniently spans the
            # full subplot width automatically, matching axhline.
            self.fig.add_hline(
                y=mean,
                line=dict(color=color, dash="dash", width=1.0),
                opacity=0.8,
                row=1,
                col=1,
            )
            self.gmm_mean_lines.append(self.fig.layout.shapes[-1])

        for boundary in boundary_values:
            self.fig.add_hline(
                y=boundary,
                line=dict(color=GMM_BOUNDARY_COLOR, dash="dot", width=1.5),
                opacity=0.9,
                row=1,
                col=1,
            )
            self.gmm_boundary_lines.append(self.fig.layout.shapes[-1])

        self.gmm_overlay_active = True
        self._gmm_overlay_signature = signature
        logger.info(
            "update_gmm_overlay: drew %d component(s), %d boundary line(s)",
            len(means),
            len(boundary_values),
        )

    def _clear_gmm_overlay(self) -> None:
        """Remove all currently-drawn GMM overlay shapes from the figure.

        Safe to call when no overlay is active (no-op beyond resetting
        already-empty lists). Truncates ``fig.layout.shapes`` back to
        the length recorded in ``self._non_gmm_shape_count`` (captured
        in :meth:`__init__`, before any overlay shape was ever added),
        so only shapes this class itself added are ever removed --
        anything else that might someday add its own layout shapes
        would be preserved.

        :returns: None
        :rtype: None
        """
        if self.gmm_band_artists or self.gmm_mean_lines or self.gmm_boundary_lines:
            self.fig.layout.shapes = self.fig.layout.shapes[: self._non_gmm_shape_count]

        self.gmm_band_artists = []
        self.gmm_mean_lines = []
        self.gmm_boundary_lines = []
        self.gmm_overlay_active = False
        self._gmm_overlay_signature = None


def get_or_create_chart_state(capacity: int = DEFAULT_CAPACITY) -> ChartState:
    """Return the session's singleton :class:`ChartState`, creating it if absent.

    Guards construction with an ``in`` check against
    ``st.session_state`` so :class:`ChartState.__init__` -- and
    therefore the underlying :class:`plotly.graph_objects.Figure` --
    runs exactly once per browser session, regardless of how many times
    the Streamlit script reruns.

    :param capacity: Forwarded to :class:`ChartState` if a new instance
        needs to be created. Ignored if a :class:`ChartState` already
        exists in session state.
    :type capacity: int
    :returns: The session's persistent chart state.
    :rtype: ChartState
    """
    if "chart" not in st.session_state:
        st.session_state.chart = ChartState(capacity=capacity)
        logger.info("Created new ChartState in session_state['chart'] (capacity=%d)", capacity)
    return st.session_state.chart


def render_chart_into_placeholder(chart: Optional[ChartState] = None) -> DeltaGenerator:
    """Create a fresh ``st.empty()`` placeholder and render the chart's current Figure into it.

    Story P1 (chart-redraw skip while IDLE/PAUSED) needs the live
    chart's rendered frame to survive a ``@st.fragment(run_every=...)``
    cycle that *skips* re-rendering, without the previously-rendered
    frame disappearing from the page. A bare, unconditional
    ``st.plotly_chart(...)`` call made directly inside the fragment's
    own body cannot do this: Streamlit clears and redraws elements a
    fragment creates directly in its own body on every fragment rerun,
    so the very next cycle that skips the call would blank the chart
    rather than freeze on the last frame.

    The fix is an ``st.empty()`` placeholder created in the *outer*,
    non-fragment script (``app.py``'s ``main()``, which is expected to
    call this function on **every** full script execution -- not
    guarded/idempotent, unlike :func:`get_or_create_chart_state`).
    :func:`ui_web.live_tick.live_tick_fragment` then writes each new
    frame into that placeholder (via ``st.session_state["chart_placeholder"]``,
    see :func:`get_chart_placeholder`) only while ``RUNNING``/
    ``SIMULATION_RUNNING``, and simply skips the write while
    ``IDLE``/``PAUSED``.

    This function deliberately does **not** reuse a cached placeholder
    across outer script runs (empirically, a placeholder cached via the
    same guarded ``if key not in st.session_state`` singleton pattern
    used elsewhere in this module loses its previously-rendered content
    the moment *any* ``@st.fragment``-decorated function is called
    later in that same script run, on the *next* full rerun -- observed
    via ``streamlit.testing.v1.AppTest`` while implementing story P1's
    bugfix, and still true after story Q1's switch to Plotly -- this
    mechanism is chart-library-agnostic). Recreating the placeholder
    fresh every outer run, and immediately re-rendering the chart's
    *current* (already up to date, no redraw needed) ``Figure`` into it
    right here, instead guarantees the last frame is visible again
    immediately after *any* full outer rerun -- e.g. clicking
    "Pause"/"Stop", which is itself a full script rerun, not a
    fragment-only one -- regardless of whether that particular rerun's
    ``app_state`` causes the fragment to redraw.

    A fixed, distinct ``key=`` (:data:`_CHART_KEY_OUTER`) is passed to
    ``placeholder.plotly_chart(...)`` here because Plotly's
    ``st.plotly_chart`` (unlike matplotlib's ``st.pyplot``) auto-derives
    an internal element ID from the element type and its parameters,
    and raises ``StreamlitDuplicateElementId`` if two ``plotly_chart``
    calls land in the *same* script run with the same derived ID.
    Because a ``@st.fragment``-decorated function runs inline as part of
    the very outer script run that calls this function (see
    ``ui_web.live_tick.live_tick_fragment``, which may also write into
    this same placeholder within that same run), this call site needs
    its own key distinct from the fragment's write -- see that
    function's docstring for its own key.

    :param chart: Chart instance whose current ``Figure`` should be
        rendered. Defaults to the session's singleton via
        :func:`get_or_create_chart_state`.
    :type chart: ChartState | None
    :returns: The freshly created placeholder, also stored in
        ``st.session_state["chart_placeholder"]`` for
        :func:`get_chart_placeholder` (and the fragment) to write into.
    :rtype: streamlit.delta_generator.DeltaGenerator
    """
    if chart is None:
        chart = get_or_create_chart_state()

    placeholder = st.empty()
    st.session_state["chart_placeholder"] = placeholder
    placeholder.plotly_chart(chart.fig, use_container_width=True, key=_CHART_KEY_OUTER)
    logger.debug("render_chart_into_placeholder: (re)created placeholder and rendered current Figure")
    return placeholder


def get_chart_placeholder() -> Optional[DeltaGenerator]:
    """Return the current chart placeholder created by :func:`render_chart_into_placeholder`.

    Intended for use from inside :func:`ui_web.live_tick.live_tick_fragment`,
    which must *not* create the placeholder itself (see
    :func:`render_chart_into_placeholder`'s docstring for why) -- it
    only looks up whatever the outer script's ``main()`` most recently
    created this session.

    :returns: The placeholder, or ``None`` if
        :func:`render_chart_into_placeholder` has not been called yet
        this session (e.g. a test that calls the fragment directly
        without going through ``app.py``'s ``main()`` first -- callers
        should fall back to a safe degraded behavior in that case
        rather than raising).
    :rtype: streamlit.delta_generator.DeltaGenerator | None
    """
    return st.session_state.get("chart_placeholder")
