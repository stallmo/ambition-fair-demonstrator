"""Session dashboard popover: stats + TP/FP/Unclassified pie chart (story F1).

Ports the desktop app's non-modal ``ui/dashboard_dialog.py`` (a
``QDialog`` with a 5s ``QTimer`` auto-refresh) to a Streamlit
``st.popover``. :func:`compute_logbook_stats` is the single source of
truth for every number the popover displays -- total samples, total
anomalies, TP/FP/Unclassified counts, and average confidence -- built
from ``state["logbook_df"]`` (the same DataFrame
``ui_web.logbook``/story E1-E4 read and mutate) and
``state["total_samples"]`` (maintained by ``ui_web.live_tick``, story
B2).

**Auto-refresh vs. manual refresh (the story's explicit fallback
clause):** the story asks for the popover to auto-refresh every 5s via
``st.popover`` + ``@st.fragment(run_every="5s")``, *unless* that
combination "proves to close the popover on the pinned Streamlit
version" -- in which case a manual "Refresh" button should ship
instead. This module ships the **manual Refresh button**, not the
auto-refresh fragment, for two concrete reasons:

1. This sandboxed environment has no browser/Playwright available (only
   ``streamlit.testing.v1.AppTest``, which executes the script and
   inspects the resulting element tree -- it cannot observe frontend-only
   behavior like a popover's open/closed DOM state), so the
   auto-refresh-vs-closes-the-popover question cannot be verified for
   real here. Streamlit's own documentation for ``st.fragment`` warns
   that interactive containers (including ``st.popover``) nested inside
   a fragment can be reset/closed by a fragment rerun that was **not**
   caused by the user interacting with an element inside that same
   container -- which is exactly what a timer-driven ``run_every="5s"``
   tick is. Shipping an unverified auto-refresh that risks visibly
   closing the popover mid-demo would be a worse operator experience
   than one extra click.
2. A plain ``st.button`` *inside* the popover does not have this
   problem: clicking it triggers a normal full-script rerun whose
   triggering widget lives inside the popover's own container, which
   Streamlit keeps open across that rerun (the same pattern already
   relied on elsewhere in this app, e.g. forms/buttons rendered inside
   an ``st.popover``/``st.expander``). Because :func:`render_dashboard_popover`
   recomputes :func:`compute_logbook_stats` fresh on *every* script rerun
   (not just the first time the popover opens), simply re-opening the
   popover -- or clicking "Refresh" inside it -- always shows current,
   non-stale values; no caching/staleness path exists to guard against.

If a future story revisits this with real browser testing and confirms
``st.fragment(run_every=...)`` nested in ``st.popover`` is stable on the
pinned Streamlit version, swapping :func:`render_dashboard_popover`'s
body to a ``@st.fragment(run_every="5s")``-decorated inner function is a
small, isolated change -- :func:`compute_logbook_stats` and
:func:`_build_pie_figure` would not need to change at all.

**Disposable figure lifecycle (story Q2):** unlike ``ui_web.chart.ChartState``'s
persistent, mutated-in-place ``Figure`` (built once per session and
never closed), the pie chart built here by :func:`_build_pie_figure` is
a brand-new :class:`plotly.graph_objects.Figure` on every popover render
(there is no cheap way to "mutate in place" a pie chart's wedge count,
which changes as TP/FP/Unclassified counts change). Story Q1 already
migrated the main chart (``ui_web/chart.py``) from server-rendered
matplotlib to client-rendered Plotly; story Q2 ports this pie chart the
same way so no matplotlib import -- and no server-side rasterization
path -- remains anywhere in the app. Unlike matplotlib's global
``Figure`` registry, a Plotly :class:`~plotly.graph_objects.Figure` is a
plain Python object with no process-global registry to leak into, so
building a fresh one every call and simply letting it be garbage
collected after :func:`render_dashboard_popover` hands it to
``st.plotly_chart`` needs no explicit disposal step (the ``plt.close()``
call this module used to require is gone).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, MutableMapping, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_web.chart import COLOR_ANOMALY, COLOR_AXES_BG, COLOR_FIG_BG, COLOR_LINE, COLOR_NORMAL, COLOR_TEXT
from ui_web.logbook import CLASSIFICATION_OPTIONS

logger = logging.getLogger(__name__)

#: Muted color for the "Unclassified" stat row / pie slice, per the
#: story's acceptance criteria. Deliberately the *same* hex as
#: ``ui_web.chart.COLOR_LINE`` (re-imported under this more
#: semantically-accurate local alias for this module's usage) -- both
#: mean "the Dracula palette's muted purple-grey", just applied to a
#: different visual element.
COLOR_UNCLASSIFIED = COLOR_LINE

#: Orange used for the "False Positives (FP)" *stat row* text color.
#: Deliberately different from the pie chart's FP slice color
#: (:data:`ui_web.chart.COLOR_ANOMALY`, red) -- the story's acceptance
#: criteria specify orange for the stat row and red for the pie slice,
#: matching the original desktop dashboard's
#: ``ui/dashboard_dialog.py`` (``add_row(3, "False Positives (FP):",
#: fp, "#ffb86c")`` vs. ``colors = ["#50fa7b", "#ff5555", "#6272a4"]``).
_COLOR_FP_ROW = "#ffb86c"

#: Fixed widget key for the "Refresh" button, following this project's
#: "stable key" convention (see e.g. ``ui_web.logbook._EDITOR_KEY``) so
#: Streamlit reliably identifies the same widget across reruns.
_REFRESH_BUTTON_KEY = "dashboard_refresh_button"

#: Label for the ``st.popover`` trigger button, rendered in the sidebar.
_POPOVER_LABEL = "📊 Dashboard"

#: Fixed ``st.plotly_chart`` element key for the pie chart (story Q2),
#: following the same "stable key" convention ``ui_web.chart`` uses for
#: its own ``plotly_chart`` calls (see e.g. ``chart._CHART_KEY_OUTER``).
#: :func:`render_dashboard_popover` is called at most once per full
#: script rerun (it is not itself wrapped in an ``@st.fragment``), so a
#: single fixed key never collides within one run; giving it one anyway
#: keeps this module consistent with ``ui_web.chart``'s convention and
#: guards against a future caller invoking it more than once per rerun.
_PIE_CHART_KEY = "dashboard_pie_chart"


def compute_logbook_stats(
    logbook_df: Optional[pd.DataFrame], total_samples: Any = 0
) -> Dict[str, Any]:
    """Compute every number the F1 dashboard popover displays.

    The single source of truth for the dashboard's stats: total
    samples collected so far (passed through from
    ``state["total_samples"]``, not derivable from ``logbook_df``
    itself), total anomalies logged (every row of ``logbook_df`` is one
    flagged anomaly -- see ``ui_web.logbook``'s module docstring),
    TP/FP/Unclassified counts (by :data:`~ui_web.logbook.CLASSIFICATION_OPTIONS`),
    and average confidence across all logged anomalies.

    :param logbook_df: The full anomaly logbook (typically
        ``state["logbook_df"]``). ``None`` or an empty/malformed
        DataFrame (e.g. missing ``classification``) is handled
        defensively and produces an all-zero result rather than
        raising -- a fresh session with no anomalies yet is a normal,
        expected state, not an error.
    :type logbook_df: pandas.DataFrame | None
    :param total_samples: The session's total sample count (typically
        ``state["total_samples"]``). Coerced to ``int``; any
        non-coercible value (e.g. ``None``) defensively falls back to
        ``0`` rather than raising.
    :type total_samples: Any
    :returns: Dict with keys ``"total_samples"``, ``"total_anomalies"``,
        ``"tp"``, ``"fp"``, ``"pending"`` (all ``int``), and
        ``"avg_confidence"`` (``float`` in ``[0, 1]`` -- the caller is
        responsible for the ``{avg * 100:.1f}%`` display formatting per
        the story's acceptance criteria).
    :rtype: dict[str, Any]
    """
    try:
        total_samples_int = int(total_samples)
    except (TypeError, ValueError):
        logger.warning(
            "compute_logbook_stats: non-numeric total_samples=%r; defaulting to 0",
            total_samples,
        )
        total_samples_int = 0

    if logbook_df is None or logbook_df.empty or "classification" not in logbook_df.columns:
        return {
            "total_samples": total_samples_int,
            "total_anomalies": 0,
            "tp": 0,
            "fp": 0,
            "pending": 0,
            "avg_confidence": 0.0,
        }

    classification = logbook_df["classification"]
    tp = int((classification == CLASSIFICATION_OPTIONS[1]).sum())
    fp = int((classification == CLASSIFICATION_OPTIONS[2]).sum())
    pending = int((classification == CLASSIFICATION_OPTIONS[0]).sum())

    if "conf_pct" in logbook_df.columns:
        # conf_pct is stored as a 0-100 float (see
        # ui_web.live_tick._append_anomaly_to_logbook: "conf_pct":
        # result.confidence * 100) -- divide back down to a [0, 1]
        # fraction so avg_confidence matches confidence's natural
        # units, and let the caller apply the *100 display formatting.
        # errors="coerce" + fillna(0.0) defends against any stray
        # non-numeric cell (e.g. a manually corrupted session) rather
        # than letting .mean() raise or silently propagate NaN.
        numeric_conf = pd.to_numeric(logbook_df["conf_pct"], errors="coerce").fillna(0.0)
        avg_confidence = float(numeric_conf.mean()) / 100.0
    else:
        avg_confidence = 0.0

    return {
        "total_samples": total_samples_int,
        "total_anomalies": len(logbook_df),
        "tp": tp,
        "fp": fp,
        "pending": pending,
        "avg_confidence": avg_confidence,
    }


def _render_stats_rows(stats: Dict[str, Any]) -> None:
    """Render the 6 label/value stat rows inside the popover.

    Each row is a small flex-box ``st.markdown`` block (label in the
    muted :data:`COLOR_UNCLASSIFIED`, value in its own semantic color)
    -- ``st.metric``/plain ``st.write`` cannot express the story's
    per-row color requirements, so raw HTML (via ``unsafe_allow_html``,
    the same technique ``ui_web.styling.inject_theme_css`` already uses
    elsewhere in this app) is used instead.

    :param stats: The dict returned by :func:`compute_logbook_stats`.
    :type stats: dict[str, Any]
    :returns: None
    :rtype: None
    """
    total_anomalies = stats.get("total_anomalies", 0)
    # Red only when there is at least one anomaly to draw attention to;
    # plain text color otherwise -- matches the story's acceptance
    # criterion exactly ("red #ff5555 if >0").
    anomalies_color = COLOR_ANOMALY if total_anomalies > 0 else COLOR_TEXT

    rows = (
        ("Total Samples", str(stats.get("total_samples", 0)), COLOR_TEXT),
        ("Total Anomalies", str(total_anomalies), anomalies_color),
        ("True Positives (TP)", str(stats.get("tp", 0)), COLOR_NORMAL),
        ("False Positives (FP)", str(stats.get("fp", 0)), _COLOR_FP_ROW),
        ("Unclassified", str(stats.get("pending", 0)), COLOR_UNCLASSIFIED),
        ("Avg Confidence", f"{stats.get('avg_confidence', 0.0) * 100:.1f}%", COLOR_TEXT),
    )
    for label, value, color in rows:
        st.markdown(
            "<div style='display:flex;justify-content:space-between;padding:2px 0;'>"
            f"<span style='color:{COLOR_UNCLASSIFIED};'>{label}</span>"
            f"<span style='color:{color};font-weight:bold;'>{value}</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def _build_pie_figure(stats: Dict[str, Any]) -> go.Figure:
    """Build the disposable TP/FP/Unclassified pie chart :class:`~plotly.graph_objects.Figure`.

    A brand-new ``Figure`` every call (no persistent trace pool, unlike
    ``ui_web.chart.ChartState`` -- see this module's docstring for why).
    The caller (:func:`render_dashboard_popover`) is responsible for
    ``st.plotly_chart``-rendering the returned figure; unlike the
    matplotlib version this used to be, no explicit disposal step is
    needed (a Plotly ``Figure`` is a plain object, not registered in any
    process-global registry).

    :param stats: The dict returned by :func:`compute_logbook_stats`;
        only the ``"tp"``/``"fp"``/``"pending"`` keys are used.
    :type stats: dict[str, Any]
    :returns: A Dracula-palette-styled pie chart figure, or a
        "No anomalies yet" placeholder panel (no pie trace) when
        TP/FP/Unclassified are all zero.
    :rtype: plotly.graph_objects.Figure
    """
    tp = stats.get("tp", 0)
    fp = stats.get("fp", 0)
    pending = stats.get("pending", 0)
    sizes = [tp, fp, pending]

    fig = go.Figure()

    if sum(sizes) > 0:
        labels = ["TP", "FP", "Unclassified"]
        # Exact colors per the story's acceptance criteria -- note the
        # FP slice is red here, distinct from the orange used for the
        # FP *stat row* text (see :data:`_COLOR_FP_ROW`'s docstring).
        colors = [COLOR_NORMAL, COLOR_ANOMALY, COLOR_UNCLASSIFIED]
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=sizes,
                marker=dict(colors=colors, line=dict(color=COLOR_FIG_BG, width=1)),
                textinfo="percent",
                texttemplate="%{percent:.0%}",
                textfont=dict(color=COLOR_FIG_BG, size=10),
                insidetextfont=dict(color=COLOR_FIG_BG, size=10),
                outsidetextfont=dict(color=COLOR_TEXT, size=10),
                rotation=90,
                sort=False,
                hole=0,
            )
        )
    else:
        # No pie trace at all -- matches the matplotlib version's
        # ax.axis("off") placeholder panel (no visible axes/pie), just a
        # centered text annotation.
        fig.add_annotation(
            text="No anomalies yet",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLOR_UNCLASSIFIED, size=12),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    fig.update_layout(
        title=dict(text="Anomaly Classification", font=dict(color=COLOR_TEXT, size=12)),
        paper_bgcolor=COLOR_FIG_BG,
        plot_bgcolor=COLOR_AXES_BG,
        font=dict(color=COLOR_TEXT),
        legend=dict(font=dict(color=COLOR_TEXT, size=9)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        width=400,
        showlegend=sum(sizes) > 0,
    )
    return fig


def render_dashboard_popover(state: Optional[MutableMapping] = None) -> None:
    """Render the "Dashboard" ``st.popover`` trigger button and its contents.

    Intended to be called once per script rerun from the sidebar (see
    ``app.py``). Every call recomputes :func:`compute_logbook_stats`
    fresh from ``state["logbook_df"]``/``state["total_samples"]`` --
    there is no caching, so opening the popover, or clicking the
    "Refresh" button rendered inside it, always shows current values
    (never stale relative to anomalies logged/classified since the
    popover was last opened). See this module's docstring for why a
    manual "Refresh" button ships instead of a ``run_every="5s"``
    auto-refreshing fragment.

    :param state: Session-state mapping to read. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps
        this signature consistent with the rest of ``ui_web``.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    with st.popover(_POPOVER_LABEL):
        st.subheader("Session Dashboard")

        logbook_df = state.get("logbook_df")
        total_samples = state.get("total_samples", 0)
        stats = compute_logbook_stats(logbook_df, total_samples)

        _render_stats_rows(stats)

        fig = _build_pie_figure(stats)
        st.plotly_chart(fig, use_container_width=True, key=_PIE_CHART_KEY)

        st.button("🔄 Refresh", key=_REFRESH_BUTTON_KEY)
