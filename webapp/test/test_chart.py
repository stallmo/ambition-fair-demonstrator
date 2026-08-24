"""Tests for :mod:`ui_web.chart` (persistent two-panel Plotly chart, story Q1).

Story B1 originally built this chart with matplotlib; story Q1 rewrote
:class:`ui_web.chart.ChartState` to build a :class:`plotly.graph_objects.Figure`
instead, rendered client-side via ``st.plotly_chart``/
``placeholder.plotly_chart`` rather than server-rasterized ``st.pyplot``.
No assertion in this module depends on a matplotlib-specific API
(``Line2D``, ``axhspan``, ``Rectangle``, etc.) any more.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import pytest

from ui_web.chart import (
    COLOR_ANOMALY,
    COLOR_AXES_BG,
    COLOR_FIG_BG,
    COLOR_GRID,
    COLOR_LINE,
    COLOR_NORMAL,
    COLOR_TEXT,
    GMM_BAND_COLORS,
    GMM_BOUNDARY_COLOR,
    ChartState,
    get_chart_placeholder,
    get_or_create_chart_state,
    render_chart_into_placeholder,
)

# webapp/test/test_chart.py -> webapp/
_WEBAPP_ROOT = Path(__file__).resolve().parents[1]


class _FakeSessionState(dict):
    """Dict-like stand-in for ``st.session_state``, same pattern as test_app_state.py.

    Supports both ``state["chart"]`` and ``state.chart`` attribute-style
    access so it can double as a drop-in for real ``st.session_state``
    if a future test wants attribute syntax; :func:`get_or_create_chart_state`
    itself only uses attribute syntax against the real session state, so
    this class adds ``__getattr__``/``__setattr__`` delegating to the dict.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# ---------------------------------------------------------------------------
# Construction / static shape
# ---------------------------------------------------------------------------


def test_chart_state_builds_figure_with_two_subplots() -> None:
    """The figure must have exactly two y-axes (a 2x1 subplot grid)."""
    chart = ChartState()
    assert isinstance(chart.fig, go.Figure)
    assert chart.fig.layout.yaxis is not None
    assert chart.fig.layout.yaxis2 is not None
    # Confirm the x-axes are actually linked (shared_xaxes=True).
    assert chart.fig.layout.xaxis.matches == "x2" or chart.fig.layout.xaxis2.matches == "x"


def test_chart_state_row_heights_approximately_3_to_1() -> None:
    """The main panel's y-domain must be roughly 3x taller than the score panel's."""
    chart = ChartState()
    main_domain = chart.fig.layout.yaxis.domain
    score_domain = chart.fig.layout.yaxis2.domain
    main_height = main_domain[1] - main_domain[0]
    score_height = score_domain[1] - score_domain[0]
    ratio = main_height / score_height
    assert ratio == pytest.approx(3.0, rel=0.15)
    # Main panel must be positioned above the score panel.
    assert main_domain[0] > score_domain[1]


def test_chart_state_dracula_colors() -> None:
    """Figure/plot background, gridlines, and text colors must match the Dracula palette."""
    chart = ChartState()

    assert chart.fig.layout.paper_bgcolor == COLOR_FIG_BG
    assert chart.fig.layout.plot_bgcolor == COLOR_AXES_BG
    assert chart.fig.layout.font.color == COLOR_TEXT
    assert chart.fig.layout.xaxis.gridcolor == COLOR_GRID
    assert chart.fig.layout.yaxis.gridcolor == COLOR_GRID
    assert chart.fig.layout.xaxis.color == COLOR_TEXT
    assert chart.fig.layout.yaxis.color == COLOR_TEXT


def test_chart_state_trace_colors_match_palette() -> None:
    """The persistent line/scatter traces must use the documented palette constants."""
    chart = ChartState()
    assert chart.line.line.color == COLOR_LINE
    assert chart.scatter_normal.marker.color == COLOR_NORMAL
    assert chart.scatter_anomaly.marker.color == COLOR_ANOMALY
    assert chart.scatter_anomaly.marker.symbol == "triangle-up"


def test_chart_state_score_panel_yaxis_range_fixed_0_1() -> None:
    """The confidence-score panel's y-axis must have a fixed range of (0, 1)."""
    chart = ChartState()
    assert tuple(chart.fig.layout.yaxis2.range) == (0.0, 1.0)


def test_chart_state_rejects_non_positive_capacity() -> None:
    """Defensive: a non-positive capacity must raise ValueError, not silently misbehave."""
    with pytest.raises(ValueError):
        ChartState(capacity=0)
    with pytest.raises(ValueError):
        ChartState(capacity=-5)


def test_chart_state_has_four_persistent_traces() -> None:
    """The figure must hold exactly 4 traces: line, normal, anomaly, confidence bars."""
    chart = ChartState()
    assert len(chart.fig.data) == 4
    assert isinstance(chart.line, go.Scatter)
    assert isinstance(chart.scatter_normal, go.Scatter)
    assert isinstance(chart.scatter_anomaly, go.Scatter)
    assert isinstance(chart.bar_trace, go.Bar)


# ---------------------------------------------------------------------------
# update() mutates traces in place (does not recreate them)
# ---------------------------------------------------------------------------


def test_update_mutates_traces_not_recreates_them() -> None:
    """Calling update() twice must keep the exact same trace object identities."""
    chart = ChartState(capacity=10)

    line_id_before = id(chart.line)
    scat_normal_id_before = id(chart.scatter_normal)
    scat_anomaly_id_before = id(chart.scatter_anomaly)
    bar_trace_id_before = id(chart.bar_trace)
    fig_data_ids_before = [id(t) for t in chart.fig.data]

    chart.update(
        timestamps=[0, 1, 2],
        values=[70.0, 71.0, 69.5],
        confidences=[0.9, 0.95, 0.2],
        is_anomaly=[False, False, True],
    )
    chart.update(
        timestamps=[1, 2, 3],
        values=[71.0, 69.5, 68.0],
        confidences=[0.95, 0.2, 0.6],
        is_anomaly=[False, True, False],
    )

    # Same Python objects, verified by id() equality -- this is the crux
    # of the "mutate in place, don't leak new traces per update" contract.
    assert id(chart.line) == line_id_before
    assert id(chart.scatter_normal) == scat_normal_id_before
    assert id(chart.scatter_anomaly) == scat_anomaly_id_before
    assert id(chart.bar_trace) == bar_trace_id_before
    assert [id(t) for t in chart.fig.data] == fig_data_ids_before


def test_update_line_data_reflects_latest_call() -> None:
    """After update(), the line trace's x/y must reflect the values just passed in."""
    chart = ChartState(capacity=10)
    chart.update(
        timestamps=[0, 1, 2],
        values=[10.0, 20.0, 30.0],
        confidences=[0.9, 0.8, 0.7],
        is_anomaly=[False, False, False],
    )
    np.testing.assert_array_equal(np.asarray(chart.line.x), [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(chart.line.y), [10.0, 20.0, 30.0])


def test_update_scatter_splits_normal_and_anomaly_points() -> None:
    """Anomaly-flagged samples must land in scatter_anomaly, others in scatter_normal."""
    chart = ChartState(capacity=10)
    chart.update(
        timestamps=[0, 1, 2, 3],
        values=[10.0, 20.0, 30.0, 40.0],
        confidences=[0.9, 0.1, 0.8, 0.05],
        is_anomaly=[False, True, False, True],
    )
    assert len(chart.scatter_normal.x) == 2
    assert len(chart.scatter_anomaly.x) == 2
    np.testing.assert_array_equal(np.asarray(chart.scatter_normal.x), [0, 2])
    np.testing.assert_array_equal(np.asarray(chart.scatter_normal.y), [10.0, 30.0])
    np.testing.assert_array_equal(np.asarray(chart.scatter_anomaly.x), [1, 3])
    np.testing.assert_array_equal(np.asarray(chart.scatter_anomaly.y), [20.0, 40.0])


def test_update_bar_heights_and_colors_reflect_confidence_and_anomaly() -> None:
    """Bar trace y-values must equal confidences; colors must reflect is_anomaly."""
    chart = ChartState(capacity=5)
    chart.update(
        timestamps=[0, 1, 2],
        values=[1.0, 2.0, 3.0],
        confidences=[0.9, 0.3, 0.6],
        is_anomaly=[False, True, False],
    )
    np.testing.assert_array_equal(np.asarray(chart.bar_trace.y), [0.9, 0.3, 0.6])
    assert tuple(chart.bar_trace.marker.color) == (COLOR_NORMAL, COLOR_ANOMALY, COLOR_NORMAL)


def test_update_score_panel_yaxis_range_stays_fixed_after_update() -> None:
    """The score panel's y-range must remain (0, 1) after update() (not autoscaled away)."""
    chart = ChartState(capacity=5)
    chart.update(
        timestamps=[0, 1],
        values=[1.0, 2.0],
        confidences=[0.9, 0.1],
        is_anomaly=[False, True],
    )
    assert tuple(chart.fig.layout.yaxis2.range) == (0.0, 1.0)


def test_update_rejects_mismatched_lengths() -> None:
    """Defensive: series of unequal length must raise ValueError, not silently misplot."""
    chart = ChartState(capacity=10)
    with pytest.raises(ValueError):
        chart.update(
            timestamps=[0, 1, 2],
            values=[1.0, 2.0],
            confidences=[0.9, 0.8, 0.7],
            is_anomaly=[False, False, False],
        )


def test_update_truncates_when_exceeding_capacity() -> None:
    """A window longer than `capacity` must be truncated to the most recent `capacity` samples, not crash."""
    chart = ChartState(capacity=3)
    chart.update(
        timestamps=[0, 1, 2, 3, 4],
        values=[10.0, 20.0, 30.0, 40.0, 50.0],
        confidences=[0.1, 0.2, 0.3, 0.4, 0.5],
        is_anomaly=[False, False, False, False, True],
    )
    np.testing.assert_array_equal(np.asarray(chart.line.x), [2, 3, 4])
    np.testing.assert_array_equal(np.asarray(chart.line.y), [30.0, 40.0, 50.0])
    # The Bar trace's arrays must also be truncated to exactly 3 entries
    # (no fixed-capacity padding pool the way the matplotlib version had).
    assert len(chart.bar_trace.y) == 3


def test_update_accepts_optional_is_injected() -> None:
    """is_injected is accepted (forward-compat) and must not raise when provided."""
    chart = ChartState(capacity=5)
    chart.update(
        timestamps=[0, 1],
        values=[1.0, 2.0],
        confidences=[0.9, 0.1],
        is_anomaly=[False, True],
        is_injected=[False, False],
    )


def test_update_rejects_mismatched_is_injected_length() -> None:
    """Defensive: an is_injected array of the wrong length must also raise."""
    chart = ChartState(capacity=5)
    with pytest.raises(ValueError):
        chart.update(
            timestamps=[0, 1],
            values=[1.0, 2.0],
            confidences=[0.9, 0.1],
            is_anomaly=[False, True],
            is_injected=[False],
        )


# ---------------------------------------------------------------------------
# Figure remains usable after repeated updates
# ---------------------------------------------------------------------------


def test_figure_still_usable_after_two_updates() -> None:
    """The figure must remain valid/serializable after repeated update() calls."""
    chart = ChartState(capacity=5)
    chart.update([0], [1.0], [0.9], [False])
    chart.update([0, 1], [1.0, 2.0], [0.9, 0.1], [False, True])
    # A malformed/corrupted Figure would fail to serialize -- a simple
    # canary that the figure is still well-formed.
    assert chart.fig.to_dict()["data"]
    assert len(chart.fig.data) == 4


# ---------------------------------------------------------------------------
# update_gmm_overlay() -- story B4 GMM component/decision-boundary overlay,
# ported to Plotly layout shapes by story Q1.
# ---------------------------------------------------------------------------

_SAMPLE_MODEL_PARAMS = {
    "means": [70.0],
    "stds": [2.0],
    "weights": [1.0],
    "boundary_values": [65.0, 75.0],
}


def test_overlay_absent_before_any_update_gmm_overlay_call() -> None:
    """A freshly-built chart must have no GMM overlay shapes at all."""
    chart = ChartState()
    assert chart.gmm_overlay_active is False
    assert chart.gmm_band_artists == []
    assert chart.gmm_mean_lines == []
    assert chart.gmm_boundary_lines == []
    assert len(chart.fig.layout.shapes) == 0


def test_update_gmm_overlay_none_is_a_noop_when_never_trained() -> None:
    """Passing None before any training must not raise or create shapes."""
    chart = ChartState()
    chart.update_gmm_overlay(None)
    assert chart.gmm_overlay_active is False
    assert chart.gmm_band_artists == []
    assert len(chart.fig.layout.shapes) == 0


def test_update_gmm_overlay_draws_band_mean_and_boundary_shapes() -> None:
    """A valid model_params dict must draw exactly one band/mean-line pair per
    component, plus one boundary line per boundary value.
    """
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)

    assert chart.gmm_overlay_active is True
    assert len(chart.gmm_band_artists) == 1
    assert len(chart.gmm_mean_lines) == 1
    assert len(chart.gmm_boundary_lines) == 2
    # 1 band (rect) + 1 mean line + 2 boundary lines = 4 layout shapes.
    assert len(chart.fig.layout.shapes) == 4


def test_update_gmm_overlay_bands_are_rect_shapes_on_the_figure() -> None:
    """Bands must be rect-type layout shapes, present on chart.fig.layout.shapes."""
    chart = ChartState()
    n_shapes_before = len(chart.fig.layout.shapes)
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)
    assert len(chart.fig.layout.shapes) == n_shapes_before + 4
    assert chart.gmm_band_artists[0].type == "rect"
    assert chart.gmm_band_artists[0] in chart.fig.layout.shapes


def test_update_gmm_overlay_mean_and_boundary_lines_use_expected_colors_and_styles() -> None:
    """Mean lines cycle GMM_BAND_COLORS/dash; boundary lines are GMM_BOUNDARY_COLOR/dot."""
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)

    mean_line = chart.gmm_mean_lines[0]
    assert mean_line.type == "line"
    assert mean_line.line.color == GMM_BAND_COLORS[0]
    assert mean_line.line.dash == "dash"

    for boundary_line in chart.gmm_boundary_lines:
        assert boundary_line.type == "line"
        assert boundary_line.line.color == GMM_BOUNDARY_COLOR
        assert boundary_line.line.dash == "dot"


def test_update_gmm_overlay_band_fillcolor_and_range() -> None:
    """Band rects must use the component's cycled fill color and span mean +/- 2*std."""
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)

    band = chart.gmm_band_artists[0]
    assert band.fillcolor == GMM_BAND_COLORS[0]
    assert band.y0 == pytest.approx(70.0 - 2 * 2.0)
    assert band.y1 == pytest.approx(70.0 + 2 * 2.0)


def test_update_gmm_overlay_bands_cycle_colors_per_component() -> None:
    """A second component must use the second GMM_BAND_COLORS entry."""
    chart = ChartState()
    two_component_params = {
        "means": [70.0, 80.0],
        "stds": [2.0, 1.5],
        "weights": [0.6, 0.4],
        "boundary_values": [],
    }
    chart.update_gmm_overlay(two_component_params)

    assert len(chart.gmm_mean_lines) == 2
    colors = [line.line.color for line in chart.gmm_mean_lines]
    assert colors == list(GMM_BAND_COLORS)


def test_update_gmm_overlay_none_after_trained_clears_overlay() -> None:
    """Passing None after a prior training must remove all overlay shapes."""
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)
    assert chart.gmm_overlay_active is True

    chart.update_gmm_overlay(None)

    assert chart.gmm_overlay_active is False
    assert chart.gmm_band_artists == []
    assert chart.gmm_mean_lines == []
    assert chart.gmm_boundary_lines == []
    # The shapes must actually have been removed from the figure, not
    # merely dropped from our own bookkeeping lists.
    assert len(chart.fig.layout.shapes) == 0


def test_update_gmm_overlay_repeated_identical_params_reuses_same_shapes() -> None:
    """Calling update_gmm_overlay() twice with unchanged params must not recreate shapes.

    This is the "signature check" optimization: a caller invoking this
    once per fragment tick should not churn Plotly layout shapes when
    the trained model hasn't actually changed.
    """
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)
    band_id_before = id(chart.gmm_band_artists[0])
    mean_line_id_before = id(chart.gmm_mean_lines[0])
    boundary_ids_before = [id(a) for a in chart.gmm_boundary_lines]
    shape_count_before = len(chart.fig.layout.shapes)

    # A fresh dict with identical values (not the same object) -- must
    # still be recognized as "unchanged" via the value-based signature.
    same_params = {
        "means": [70.0],
        "stds": [2.0],
        "weights": [1.0],
        "boundary_values": [65.0, 75.0],
    }
    chart.update_gmm_overlay(same_params)

    assert id(chart.gmm_band_artists[0]) == band_id_before
    assert id(chart.gmm_mean_lines[0]) == mean_line_id_before
    assert [id(a) for a in chart.gmm_boundary_lines] == boundary_ids_before
    assert len(chart.fig.layout.shapes) == shape_count_before


def test_update_gmm_overlay_changed_params_recreates_shapes() -> None:
    """Calling update_gmm_overlay() with different params must redraw (new shape objects)."""
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)
    band_id_before = id(chart.gmm_band_artists[0])

    changed_params = dict(_SAMPLE_MODEL_PARAMS)
    changed_params["means"] = [72.0]
    chart.update_gmm_overlay(changed_params)

    assert len(chart.gmm_band_artists) == 1
    assert id(chart.gmm_band_artists[0]) != band_id_before
    assert chart.gmm_band_artists[0].y0 == pytest.approx(72.0 - 2 * 2.0)


def test_update_gmm_overlay_retrain_with_fewer_boundaries_removes_stale_lines() -> None:
    """A retrain producing fewer boundary crossings must not leave stale lines behind."""
    chart = ChartState()
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)  # 2 boundary values
    assert len(chart.gmm_boundary_lines) == 2

    fewer_boundaries = dict(_SAMPLE_MODEL_PARAMS)
    fewer_boundaries["boundary_values"] = [70.0]
    chart.update_gmm_overlay(fewer_boundaries)

    assert len(chart.gmm_boundary_lines) == 1
    # 1 band + 1 mean line + 1 boundary line = 3 total shapes remaining.
    assert len(chart.fig.layout.shapes) == 3


def test_update_gmm_overlay_retrain_preserves_other_chart_traces() -> None:
    """Clearing/redrawing the overlay must never touch the persistent data traces."""
    chart = ChartState()
    chart.update(
        timestamps=[0, 1, 2],
        values=[70.0, 71.0, 69.5],
        confidences=[0.9, 0.95, 0.2],
        is_anomaly=[False, False, True],
    )
    line_id_before = id(chart.line)
    n_traces_before = len(chart.fig.data)

    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)
    chart.update_gmm_overlay(None)
    chart.update_gmm_overlay(_SAMPLE_MODEL_PARAMS)

    assert id(chart.line) == line_id_before
    assert len(chart.fig.data) == n_traces_before
    np.testing.assert_array_equal(np.asarray(chart.line.x), [0, 1, 2])


# ---------------------------------------------------------------------------
# get_or_create_chart_state() -- singleton-per-session behavior
# ---------------------------------------------------------------------------


def test_get_or_create_chart_state_creates_once() -> None:
    """First call creates a ChartState; second call returns the same instance."""
    state: Dict[str, Any] = _FakeSessionState()

    import ui_web.chart as chart_module

    original_session_state = chart_module.st.session_state
    chart_module.st.session_state = state
    try:
        first = get_or_create_chart_state()
        second = get_or_create_chart_state()
        assert isinstance(first, ChartState)
        assert first is second
        assert state["chart"] is first
    finally:
        chart_module.st.session_state = original_session_state


def test_get_or_create_chart_state_persists_across_streamlit_reruns() -> None:
    """Integration test: st.session_state must preserve the ChartState/Figure
    object identity across multiple Streamlit script reruns.

    This empirically verifies the pattern the story asks for (rather
    than only unit-testing ChartState in isolation): a real
    ``streamlit.testing.v1.AppTest`` session is run twice, and the
    ``id()`` of the Figure/line-trace objects recorded on each run must
    be identical -- proving Streamlit's real session_state (not just a
    fake dict) actually keeps the same Python objects alive across
    reruns within one session.
    """
    from streamlit.testing.v1 import AppTest

    script = f'''
import sys
sys.path.insert(0, r"{_WEBAPP_ROOT}")
import streamlit as st
from ui_web.chart import get_or_create_chart_state

chart = get_or_create_chart_state()
st.session_state.setdefault("fig_ids", [])
st.session_state["fig_ids"].append(id(chart.fig))
st.session_state.setdefault("line_ids", [])
st.session_state["line_ids"].append(id(chart.line))
'''
    at = AppTest.from_string(script)
    at.run()
    assert at.exception == []
    at.run()
    assert at.exception == []

    fig_ids = at.session_state["fig_ids"]
    line_ids = at.session_state["line_ids"]
    assert len(fig_ids) == 2
    assert fig_ids[0] == fig_ids[1]
    assert line_ids[0] == line_ids[1]


# ---------------------------------------------------------------------------
# render_chart_into_placeholder() / get_chart_placeholder() (story P1;
# updated to Plotly by story Q1)
# ---------------------------------------------------------------------------


def test_render_chart_into_placeholder_stores_and_returns_a_placeholder() -> None:
    """The call must both return the placeholder and store it in session state."""
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from ui_web.chart import get_chart_placeholder, render_chart_into_placeholder

returned = render_chart_into_placeholder()
st.write(str(returned is get_chart_placeholder()))
st.write(str("chart_placeholder" in st.session_state))
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    markdown_values = [m.value for m in at.markdown]
    assert "True" in markdown_values
    assert markdown_values.count("True") == 2


def test_render_chart_into_placeholder_renders_exactly_one_plotly_chart() -> None:
    """A single call must render exactly one Plotly chart element into the placeholder."""
    from streamlit.testing.v1 import AppTest

    script = """
from ui_web.chart import render_chart_into_placeholder

render_chart_into_placeholder()
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    assert len(at.get("plotly_chart")) == 1


def test_render_chart_into_placeholder_is_not_idempotent_a_fresh_placeholder_every_call() -> None:
    """Unlike get_or_create_chart_state(), this must NOT reuse a cached placeholder.

    A cached/reused placeholder was empirically found (while implementing
    story P1's fix) to silently lose its previously-rendered content the
    next time any ``@st.fragment``-decorated function is called later in
    the same script run, on a *subsequent* full rerun -- so this
    function must always call ``st.empty()`` fresh, never guard with an
    ``if key not in st.session_state`` check the way
    :func:`ui_web.chart.get_or_create_chart_state` does. Story Q1's
    switch to Plotly does not change this -- it is chart-library-agnostic.
    """
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from ui_web.chart import render_chart_into_placeholder

st.session_state.setdefault("placeholder_ids", [])
placeholder = render_chart_into_placeholder()
st.session_state["placeholder_ids"].append(id(placeholder))
"""
    at = AppTest.from_string(script)
    at.run()
    assert at.exception == []
    at.run()
    assert at.exception == []

    placeholder_ids = at.session_state["placeholder_ids"]
    assert len(placeholder_ids) == 2
    # A fresh st.empty() instance every call -- ids must differ.
    assert placeholder_ids[0] != placeholder_ids[1]


def test_get_chart_placeholder_returns_none_before_first_render_call() -> None:
    """Before render_chart_into_placeholder() has ever run, the accessor must return None."""
    state: Dict[str, Any] = _FakeSessionState()

    import ui_web.chart as chart_module

    original_session_state = chart_module.st.session_state
    chart_module.st.session_state = state
    try:
        assert get_chart_placeholder() is None
    finally:
        chart_module.st.session_state = original_session_state
