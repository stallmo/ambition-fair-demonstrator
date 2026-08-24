"""Tests for :mod:`ui_web.dashboard` (story F1 -- Dashboard popover with stats + pie chart).

Covers, in order:

1. :func:`~ui_web.dashboard.compute_logbook_stats` -- the pure stats
   helper: counts/avg-confidence math against constructed DataFrames,
   including the empty-logbook zero case and defensive handling of
   malformed input.
2. :func:`~ui_web.dashboard._build_pie_figure` -- pure Figure-building
   logic (story Q2: now a :class:`plotly.graph_objects.Figure`): pie
   slice colors/labels/count for a populated logbook, and the
   "No anomalies yet" placeholder annotation for an all-zero one.
3. :func:`~ui_web.dashboard.render_dashboard_popover` -- exercised via
   ``streamlit.testing.v1.AppTest`` (the same pattern
   ``test_logbook.py``/``test_styling.py`` use for Streamlit-rendering
   functions): the popover trigger renders, the 6 stat rows carry the
   exact colors/labels the story's acceptance criteria require, the
   "Refresh" button exists with a stable key, and re-running after new
   anomalies are logged shows updated (non-stale) values.

Story Q2 replaced the matplotlib ``_build_pie_figure`` with a Plotly
one and removed matplotlib from this app entirely; the old section 4
("disposable figure closed via ``plt.close``") no longer applies -- a
Plotly ``Figure`` is a plain object with no process-global registry to
leak into, so there is nothing left to test there.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import pytest

from state.app_state import LOGBOOK_COLUMNS
from ui_web.dashboard import (
    COLOR_UNCLASSIFIED,
    _COLOR_FP_ROW,
    _PIE_CHART_KEY,
    _REFRESH_BUTTON_KEY,
    _build_pie_figure,
    compute_logbook_stats,
    render_dashboard_popover,
)
from ui_web.chart import COLOR_ANOMALY, COLOR_NORMAL, COLOR_TEXT


def _row(row_id: int, classification: str, conf_pct: float) -> Dict[str, Any]:
    """Build one well-formed logbook row dict for test fixtures.

    :param row_id: The row's ``row_id``.
    :type row_id: int
    :param classification: Emoji-coded classification value (see
        :data:`ui_web.logbook.CLASSIFICATION_OPTIONS`).
    :type classification: str
    :param conf_pct: Confidence, stored 0-100 (matching
        ``ui_web.live_tick._append_anomaly_to_logbook``'s convention).
    :type conf_pct: float
    :returns: A dict with every :data:`state.app_state.LOGBOOK_COLUMNS` key.
    :rtype: dict[str, Any]
    """
    return {
        "row_id": row_id,
        "timestamp": float(row_id) + 1000.0,
        "time_str": f"10:00:{row_id:02d}",
        "value": float(row_id) * 1.5,
        "conf_pct": conf_pct,
        "classification": classification,
        "comment": "",
        "is_injected": False,
    }


def _make_logbook_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a logbook DataFrame from a list of :func:`_row` dicts.

    :param rows: Row dicts, in the row order the returned DataFrame
        should have.
    :type rows: list[dict[str, Any]]
    :returns: DataFrame with columns :data:`state.app_state.LOGBOOK_COLUMNS`.
    :rtype: pandas.DataFrame
    """
    return pd.DataFrame(rows, columns=LOGBOOK_COLUMNS)


# ----------------------------------------------------------------------
# 1. compute_logbook_stats
# ----------------------------------------------------------------------


def test_compute_logbook_stats_with_none_logbook_returns_all_zero() -> None:
    """A ``None`` logbook_df (session not initialized) must yield an all-zero result."""
    stats = compute_logbook_stats(None, total_samples=7)
    assert stats == {
        "total_samples": 7,
        "total_anomalies": 0,
        "tp": 0,
        "fp": 0,
        "pending": 0,
        "avg_confidence": 0.0,
    }


def test_compute_logbook_stats_with_empty_logbook_returns_all_zero() -> None:
    """An empty (schema-only) logbook_df must yield zero anomaly counts."""
    empty_df = _make_logbook_df([])
    stats = compute_logbook_stats(empty_df, total_samples=0)
    assert stats["total_samples"] == 0
    assert stats["total_anomalies"] == 0
    assert stats["tp"] == 0
    assert stats["fp"] == 0
    assert stats["pending"] == 0
    assert stats["avg_confidence"] == 0.0


def test_compute_logbook_stats_counts_tp_fp_pending_correctly() -> None:
    """TP/FP/Unclassified counts and total_anomalies must match the constructed DataFrame."""
    df = _make_logbook_df(
        [
            _row(0, "🟢 TP", 90.0),
            _row(1, "🟢 TP", 70.0),
            _row(2, "🟠 FP", 40.0),
            _row(3, "⚪ Unclassified", 50.0),
            _row(4, "⚪ Unclassified", 60.0),
        ]
    )
    stats = compute_logbook_stats(df, total_samples=100)

    assert stats["total_samples"] == 100
    assert stats["total_anomalies"] == 5
    assert stats["tp"] == 2
    assert stats["fp"] == 1
    assert stats["pending"] == 2


def test_compute_logbook_stats_avg_confidence_math() -> None:
    """avg_confidence must be the mean of conf_pct (0-100), divided down to a [0, 1] fraction."""
    df = _make_logbook_df(
        [
            _row(0, "🟢 TP", 80.0),
            _row(1, "🟠 FP", 40.0),
            _row(2, "⚪ Unclassified", 60.0),
        ]
    )
    stats = compute_logbook_stats(df, total_samples=0)

    # mean(80, 40, 60) = 60.0 -> / 100 = 0.6
    assert stats["avg_confidence"] == pytest.approx(0.6)
    # The story's exact display formatting expression: {avg * 100:.1f}%
    assert f"{stats['avg_confidence'] * 100:.1f}%" == "60.0%"


def test_compute_logbook_stats_missing_classification_column_is_defensive() -> None:
    """A malformed logbook_df missing 'classification' must not raise -- returns all-zero."""
    malformed_df = pd.DataFrame({"row_id": [0, 1]})
    stats = compute_logbook_stats(malformed_df, total_samples=3)
    assert stats["total_anomalies"] == 0
    assert stats["tp"] == 0
    assert stats["total_samples"] == 3


def test_compute_logbook_stats_non_numeric_total_samples_defaults_to_zero() -> None:
    """A non-coercible total_samples (e.g. None) must default to 0, not raise."""
    stats = compute_logbook_stats(None, total_samples=None)
    assert stats["total_samples"] == 0

    stats2 = compute_logbook_stats(None, total_samples="not-a-number")
    assert stats2["total_samples"] == 0


def test_compute_logbook_stats_missing_conf_pct_column_defaults_avg_to_zero() -> None:
    """A logbook_df with classification but no conf_pct must not raise -- avg defaults to 0."""
    df = pd.DataFrame({"classification": ["🟢 TP", "🟠 FP"]})
    stats = compute_logbook_stats(df, total_samples=0)
    assert stats["avg_confidence"] == 0.0
    assert stats["tp"] == 1
    assert stats["fp"] == 1


# ----------------------------------------------------------------------
# 2. _build_pie_figure
# ----------------------------------------------------------------------


def test_build_pie_figure_all_zero_shows_placeholder_text_no_pie_trace() -> None:
    """When TP/FP/Unclassified are all zero, no pie trace is drawn -- a placeholder annotation is shown instead."""
    stats = {"tp": 0, "fp": 0, "pending": 0}
    fig = _build_pie_figure(stats)
    assert isinstance(fig, go.Figure)
    # No Pie trace drawn.
    assert len(fig.data) == 0
    annotation_texts = [a.text for a in fig.layout.annotations]
    assert "No anomalies yet" in annotation_texts
    # The placeholder annotation must use the muted "Unclassified" color.
    placeholder = next(a for a in fig.layout.annotations if a.text == "No anomalies yet")
    assert placeholder.font.color == COLOR_UNCLASSIFIED


def test_build_pie_figure_nonzero_counts_draws_pie_with_expected_labels_and_colors() -> None:
    """A populated stats dict must draw exactly one Pie trace, colored TP=green/FP=red/Unclassified=muted."""
    stats = {"tp": 3, "fp": 2, "pending": 1}
    fig = _build_pie_figure(stats)
    assert len(fig.data) == 1

    pie = fig.data[0]
    assert isinstance(pie, go.Pie)
    assert list(pie.labels) == ["TP", "FP", "Unclassified"]
    assert list(pie.values) == [3, 2, 1]
    assert list(pie.marker.colors) == [COLOR_NORMAL, COLOR_ANOMALY, COLOR_UNCLASSIFIED]


def test_build_pie_figure_returns_a_fresh_figure_each_call() -> None:
    """Each call must return a distinct Figure object (no shared/mutated-in-place state)."""
    stats = {"tp": 1, "fp": 0, "pending": 0}
    fig1 = _build_pie_figure(stats)
    fig2 = _build_pie_figure(stats)
    assert fig1 is not fig2


# ----------------------------------------------------------------------
# 3. render_dashboard_popover (AppTest)
# ----------------------------------------------------------------------

_DASHBOARD_APP_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.dashboard import render_dashboard_popover

init_session_state()
with st.sidebar:
    render_dashboard_popover()
"""


def test_render_dashboard_popover_renders_without_raising() -> None:
    """render_dashboard_popover() must run cleanly inside a real Streamlit script."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_DASHBOARD_APP_SCRIPT)
    at.run()

    assert at.exception == []


def test_render_dashboard_popover_shows_refresh_button_with_stable_key() -> None:
    """A manual 'Refresh' button must exist with the module's stable key."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_DASHBOARD_APP_SCRIPT)
    at.run()

    assert at.exception == []
    button = at.sidebar.button(key=_REFRESH_BUTTON_KEY)
    assert "Refresh" in button.label


def test_render_dashboard_popover_stat_rows_carry_exact_labels_and_colors() -> None:
    """Each of the 6 stat rows must render with the exact label and color the story requires."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_DASHBOARD_APP_SCRIPT)
    at.run()

    assert at.exception == []
    markdown_html = " ".join(m.value for m in at.sidebar.markdown)

    # Fresh session: 0 anomalies -> Total Anomalies row is NOT red.
    assert "Total Samples" in markdown_html
    assert "Total Anomalies" in markdown_html
    assert "True Positives (TP)" in markdown_html
    assert f"color:{COLOR_NORMAL}" in markdown_html
    assert "False Positives (FP)" in markdown_html
    assert f"color:{_COLOR_FP_ROW}" in markdown_html
    assert "Unclassified" in markdown_html
    assert f"color:{COLOR_UNCLASSIFIED}" in markdown_html
    assert "Avg Confidence" in markdown_html
    assert "0.0%" in markdown_html


def test_render_dashboard_popover_total_anomalies_turns_red_when_nonzero() -> None:
    """Total Anomalies must render in COLOR_ANOMALY (#ff5555) once at least one anomaly is logged."""
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.dashboard import render_dashboard_popover

init_session_state()
if "seeded" not in st.session_state:
    import pandas as pd
    st.session_state.logbook_df = pd.DataFrame(
        [
            {
                "row_id": 0,
                "timestamp": 1.0,
                "time_str": "10:00:00",
                "value": 1.0,
                "conf_pct": 80.0,
                "classification": "🟢 TP",
                "comment": "",
                "is_injected": False,
            }
        ],
        columns=[
            "row_id", "timestamp", "time_str", "value", "conf_pct",
            "classification", "comment", "is_injected",
        ],
    )
    st.session_state.total_samples = 42
    st.session_state.seeded = True
with st.sidebar:
    render_dashboard_popover()
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    markdown_html = " ".join(m.value for m in at.sidebar.markdown)
    assert "Total Anomalies" in markdown_html
    assert f"color:{COLOR_ANOMALY};font-weight:bold;'>1<" in markdown_html
    assert "color:#f8f8f2;font-weight:bold;'>42<" in markdown_html  # Total Samples


def test_render_dashboard_popover_shows_updated_values_across_reruns_not_stale() -> None:
    """Re-running after new anomalies are logged must show updated, non-stale values (no caching)."""
    from streamlit.testing.v1 import AppTest

    script_template = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.dashboard import render_dashboard_popover
import pandas as pd

init_session_state()
st.session_state.logbook_df = pd.DataFrame(
    [
        {{
            "row_id": i,
            "timestamp": float(i),
            "time_str": "10:00:00",
            "value": 1.0,
            "conf_pct": 50.0,
            "classification": "🟢 TP",
            "comment": "",
            "is_injected": False,
        }}
        for i in range({n_rows})
    ],
    columns=[
        "row_id", "timestamp", "time_str", "value", "conf_pct",
        "classification", "comment", "is_injected",
    ],
)
with st.sidebar:
    render_dashboard_popover()
"""
    at_before = AppTest.from_string(script_template.format(n_rows=1))
    at_before.run()
    assert at_before.exception == []
    markdown_before = " ".join(m.value for m in at_before.sidebar.markdown)
    # Total Anomalies = 1 (>0, so rendered in COLOR_ANOMALY per the story's criterion).
    assert f"color:{COLOR_ANOMALY};font-weight:bold;'>1<" in markdown_before

    at_after = AppTest.from_string(script_template.format(n_rows=4))
    at_after.run()
    assert at_after.exception == []
    markdown_after = " ".join(m.value for m in at_after.sidebar.markdown)
    assert f"color:{COLOR_ANOMALY};font-weight:bold;'>4<" in markdown_after  # Total Anomalies = 4


def test_render_dashboard_popover_pie_chart_placeholder_when_no_anomalies() -> None:
    """A fresh session (no anomalies) must render the 'No anomalies yet' pie placeholder, not raise."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_DASHBOARD_APP_SCRIPT)
    at.run()

    assert at.exception == []
    # st.plotly_chart renders successfully (no exception) even for the
    # placeholder branch -- covered indirectly by the exception check
    # above; _build_pie_figure's placeholder branch itself is unit
    # tested directly in
    # test_build_pie_figure_all_zero_shows_placeholder_text_no_pie_trace.


def test_render_dashboard_popover_renders_exactly_one_plotly_chart() -> None:
    """render_dashboard_popover() must render exactly one Plotly chart element (story Q2)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_DASHBOARD_APP_SCRIPT)
    at.run()

    assert at.exception == []
    assert len(at.get("plotly_chart")) == 1


def test_render_dashboard_popover_pie_chart_module_key_constant_is_stable() -> None:
    """The module's fixed pie-chart key constant (story Q2) must be a non-empty, stable string.

    A full round-trip of the internal Streamlit element ID is not
    exposed by ``streamlit.testing.v1.AppTest`` in a stable, documented
    way; :func:`test_render_dashboard_popover_renders_exactly_one_plotly_chart`
    already exercises the actual ``st.plotly_chart(..., key=_PIE_CHART_KEY)``
    call end-to-end without raising (a ``StreamlitDuplicateElementId``
    would surface as an exception there). This test just pins the
    constant's value/type so an accidental future removal/blank-out is
    caught.
    """
    assert isinstance(_PIE_CHART_KEY, str)
    assert _PIE_CHART_KEY == "dashboard_pie_chart"
