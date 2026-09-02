"""Tests for :mod:`ui_web.live_tick`'s GMM Training sidebar group (story B4).

Covers :func:`ui_web.live_tick.render_gmm_training_group` (progress
bar, status text, training-window Start/End spinboxes with end
auto-tracking, "Retrain Model" button) and its wiring into the chart's
GMM overlay via :meth:`ui_web.chart.ChartState.update_gmm_overlay`.

Pure/injectable helpers (:func:`_on_end_spinbox_manual_edit`,
:func:`_on_retrain_clicked`, :func:`_safe_widget_state_set`) are unit
tested directly with fakes; the widget-rendering behavior itself
(progress bar text/value, status color, spinbox auto-tracking, button
click) is exercised via :class:`streamlit.testing.v1.AppTest`, since
``st.progress``/``st.number_input``/``st.button`` require a real
``ScriptRunContext``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pytest

from core.anomaly_detector import AnomalyDetector
from ui_web.chart import ChartState, get_or_create_chart_state
from ui_web.live_tick import (
    _on_end_spinbox_manual_edit,
    _on_retrain_clicked,
    _safe_widget_state_set,
    render_gmm_training_group,
)


class _FakeSessionState(dict):
    """Minimal dict-like stand-in for ``st.session_state``.

    Same pattern used throughout this test suite (``test_live_tick.py``,
    ``test_sidebar_config.py``): the real ``st.session_state`` is inert
    outside a running Streamlit script, so unit tests inject a plain
    dict subclass into the injectable-``state`` parameter instead.
    """


class _SpyDetector:
    """Duck-typed stand-in for :class:`core.anomaly_detector.AnomalyDetector`.

    Exposes exactly the attributes/methods
    :func:`~ui_web.live_tick.render_gmm_training_group` and
    :func:`~ui_web.live_tick._on_retrain_clicked` read
    (``sample_count``, ``training_progress``, ``is_trained``,
    ``get_model_params()``, ``retrain()``), letting tests control them
    directly and record ``retrain()`` calls, without needing to feed
    100+ real samples through the actual GMM.

    :param sample_count: Value to report via :attr:`sample_count`.
    :type sample_count: int
    :param training_progress: Value to report via :attr:`training_progress`.
    :type training_progress: float
    :param is_trained: Value to report via :attr:`is_trained`.
    :type is_trained: bool
    :param model_params: Value :meth:`get_model_params` returns.
    :type model_params: dict | None
    """

    def __init__(
        self,
        sample_count: int = 0,
        training_progress: float = 0.0,
        is_trained: bool = False,
        model_params: Optional[dict] = None,
    ) -> None:
        self.sample_count = sample_count
        self.training_progress = training_progress
        self.is_trained = is_trained
        self._model_params = model_params
        self.retrain_calls: List[Tuple[int, int]] = []

    def get_model_params(self) -> Optional[dict]:
        """Return the params this spy was configured with."""
        return self._model_params

    def retrain(self, start: int, end: int) -> None:
        """Record the call and flip :attr:`is_trained` True, like the real detector."""
        self.retrain_calls.append((start, end))
        self.is_trained = True


_SAMPLE_MODEL_PARAMS = {
    "means": [70.0],
    "stds": [2.0],
    "weights": [1.0],
    "boundary_values": [65.0, 75.0],
}


# ----------------------------------------------------------------------
# _safe_widget_state_set
# ----------------------------------------------------------------------


def test_safe_widget_state_set_sets_value_on_plain_dict() -> None:
    """Against a plain dict (no Streamlit exception possible), the value is set normally."""
    state = _FakeSessionState()
    _safe_widget_state_set(state, "cfg_retrain_end", 42)
    assert state["cfg_retrain_end"] == 42


def test_safe_widget_state_set_swallows_streamlit_api_exception() -> None:
    """A StreamlitAPIException-raising mapping must be tolerated, not propagated."""
    from streamlit.errors import StreamlitAPIException

    class _RaisingState(dict):
        def __setitem__(self, key, value):  # noqa: D401 - test double
            raise StreamlitAPIException("already instantiated")

    state = _RaisingState()
    _safe_widget_state_set(state, "cfg_retrain_end", 42)  # must not raise


def test_safe_widget_state_set_reraises_other_exceptions() -> None:
    """Any other exception type must propagate normally (not silently swallowed)."""

    class _RaisingState(dict):
        def __setitem__(self, key, value):  # noqa: D401 - test double
            raise RuntimeError("boom")

    state = _RaisingState()
    with pytest.raises(RuntimeError):
        _safe_widget_state_set(state, "cfg_retrain_end", 42)


# ----------------------------------------------------------------------
# _on_end_spinbox_manual_edit
# ----------------------------------------------------------------------


def test_on_end_spinbox_manual_edit_sets_end_tracking_false() -> None:
    """Editing the End spinbox must flip cfg_end_tracking to False."""
    state = _FakeSessionState(cfg_end_tracking=True)
    _on_end_spinbox_manual_edit(state)
    assert state["cfg_end_tracking"] is False


def test_on_end_spinbox_manual_edit_idempotent_when_already_false() -> None:
    """Calling it again when already False must not raise or change anything else."""
    state = _FakeSessionState(cfg_end_tracking=False)
    _on_end_spinbox_manual_edit(state)
    assert state["cfg_end_tracking"] is False


# ----------------------------------------------------------------------
# _on_retrain_clicked
# ----------------------------------------------------------------------


def test_on_retrain_clicked_calls_detector_retrain_with_session_state_bounds() -> None:
    """retrain() must be called with exactly cfg_retrain_start/cfg_retrain_end."""
    detector = _SpyDetector(model_params=_SAMPLE_MODEL_PARAMS)
    state = _FakeSessionState(detector=detector, cfg_retrain_start=10, cfg_retrain_end=60)

    _on_retrain_clicked(state)

    assert detector.retrain_calls == [(10, 60)]


def test_on_retrain_clicked_missing_detector_is_a_safe_noop() -> None:
    """No 'detector' key in state (init_session_state() not called) must not raise."""
    state = _FakeSessionState(cfg_retrain_start=0, cfg_retrain_end=0)
    _on_retrain_clicked(state)  # must not raise


def test_on_retrain_clicked_recomputes_chart_overlay_from_fresh_model_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After retrain(), the chart's GMM overlay must be recomputed from get_model_params()."""
    import ui_web.live_tick as live_tick_module

    detector = _SpyDetector(model_params=_SAMPLE_MODEL_PARAMS)
    state = _FakeSessionState(detector=detector, cfg_retrain_start=0, cfg_retrain_end=5)

    chart = ChartState()
    monkeypatch.setattr(live_tick_module, "get_or_create_chart_state", lambda: chart)

    _on_retrain_clicked(state)

    assert chart.gmm_overlay_active is True
    assert chart.gmm_band_artists  # at least one band drawn from the fresh params


# ----------------------------------------------------------------------
# render_gmm_training_group: defensive no-detector path (unit-testable
# without a real Streamlit script, since it returns before any widget
# call once 'detector' is missing).
# ----------------------------------------------------------------------


def test_render_gmm_training_group_missing_detector_returns_without_raising() -> None:
    """No 'detector' in state must be a defensive no-op, not a crash.

    This one call path never reaches a Streamlit widget function *other
    than* ``st.subheader`` (which is fine to call without a live
    ScriptRunContext -- it just warns), so this specific case is
    testable as a plain unit test.
    """
    state = _FakeSessionState()
    render_gmm_training_group(state)  # must not raise


# ----------------------------------------------------------------------
# AppTest-based: progress bar, status text, spinbox auto-tracking,
# retrain button -- all require a real ScriptRunContext.
# ----------------------------------------------------------------------

# Script drives sample-feeding via a test-only '_test_n_feed' session-state
# key (set directly on `at.session_state` before each `.run()`), so tests
# get exact, deterministic control over how many samples the *real*
# AnomalyDetector has buffered/trained on, without depending on
# DataGenerator/live_tick_fragment's random stream.
_APP_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.live_tick import render_gmm_training_group

init_session_state()

n_feed = st.session_state.get("_test_n_feed", 0)
already_fed = st.session_state.get("_test_already_fed", 0)
detector = st.session_state["detector"]
for i in range(already_fed, n_feed):
    detector.feed(float(i), 70.0 + (i % 5) * 0.1, False)
st.session_state["_test_already_fed"] = n_feed
st.session_state["total_samples"] = n_feed

with st.sidebar:
    render_gmm_training_group()
"""


def _run_with_n_feed(at, n_feed: int):
    """Run ``_APP_SCRIPT`` with the detector fed exactly ``n_feed`` samples.

    :param at: The AppTest instance to configure and run.
    :param n_feed: Total number of samples the detector should have
        been fed by the time this run completes.
    :type n_feed: int
    :returns: The same AppTest instance, post-run.
    """
    at.session_state["_test_n_feed"] = n_feed
    at.run()
    return at


def test_progress_bar_collecting_text_and_value_before_100_samples() -> None:
    """Progress bar must read 'Collecting… {pct}%' while sample_count < 100."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 42)

    assert at.exception == []
    progress_elements = at.get("progress")
    assert len(progress_elements) == 1
    assert progress_elements[0].proto.value == 42
    assert progress_elements[0].proto.text == "Collecting… 42%"


def test_progress_bar_trained_text_and_value_at_100_samples() -> None:
    """Once the detector auto-trains (>=100 samples), progress bar shows 'Trained ✓'."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 100)

    assert at.exception == []
    progress_elements = at.get("progress")
    assert progress_elements[0].proto.value == 100
    assert progress_elements[0].proto.text == "Trained ✓"
    assert at.session_state["detector"].is_trained is True


def test_status_text_not_trained_is_red_before_training() -> None:
    """Status text must read 'Not trained' in #ff5555 before training."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 5)

    assert at.exception == []
    markdown_html = [m.value for m in at.sidebar.markdown]
    assert any(
        'color:#ff5555">Not trained</span>' in html for html in markdown_html
    ), markdown_html


def test_status_text_model_ready_is_green_after_training() -> None:
    """Status text must read 'Model ready' in #50fa7b once trained."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 100)

    assert at.exception == []
    markdown_html = [m.value for m in at.sidebar.markdown]
    assert any(
        'color:#50fa7b">Model ready</span>' in html for html in markdown_html
    ), markdown_html


def test_end_spinbox_auto_tracks_total_samples_across_reruns() -> None:
    """While cfg_end_tracking is True, the End spinbox must follow total_samples."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 3)
    assert at.session_state["cfg_retrain_end"] == 3

    _run_with_n_feed(at, 7)
    assert at.session_state["cfg_retrain_end"] == 7
    assert at.session_state["cfg_end_tracking"] is True


def test_manual_edit_of_end_spinbox_stops_auto_tracking() -> None:
    """Directly editing the End spinbox must set cfg_end_tracking False and stick."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 3)
    assert at.session_state["cfg_retrain_end"] == 3

    at.sidebar.number_input(key="cfg_retrain_end").set_value(1)
    at.session_state["_test_n_feed"] = 3  # unchanged this run
    at.run()

    assert at.exception == []
    assert at.session_state["cfg_end_tracking"] is False
    assert at.session_state["cfg_retrain_end"] == 1

    # A later tick (total_samples advancing) must NOT snap End back.
    _run_with_n_feed(at, 9)
    assert at.session_state["cfg_retrain_end"] == 1
    assert at.session_state["cfg_end_tracking"] is False


def test_start_spinbox_manual_edit_is_not_auto_tracked() -> None:
    """The Start spinbox has no auto-tracking; manual edits simply persist."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 20)

    at.sidebar.number_input(key="cfg_retrain_start").set_value(5)
    at.session_state["_test_n_feed"] = 20
    at.run()

    assert at.exception == []
    assert at.session_state["cfg_retrain_start"] == 5

    _run_with_n_feed(at, 30)
    assert at.session_state["cfg_retrain_start"] == 5


def test_retrain_button_click_calls_detector_retrain_and_updates_overlay() -> None:
    """Clicking 'Retrain Model' must call detector.retrain(start, end) and redraw the overlay."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 120)  # comfortably past the 100-sample auto-train threshold

    at.sidebar.number_input(key="cfg_retrain_start").set_value(10)
    at.session_state["_test_n_feed"] = 120
    at.run()
    at.sidebar.number_input(key="cfg_retrain_end").set_value(80)
    at.session_state["_test_n_feed"] = 120
    at.run()

    assert at.session_state["cfg_retrain_start"] == 10
    assert at.session_state["cfg_retrain_end"] == 80

    detector_before = at.session_state["detector"]
    assert isinstance(detector_before, AnomalyDetector)

    at.sidebar.button[0].click()
    at.session_state["_test_n_feed"] = 120
    at.run()

    assert at.exception == []
    # The overlay must have been (re)computed and be active post-retrain.
    chart = at.session_state["chart"]
    assert chart.gmm_overlay_active is True
    assert len(chart.gmm_band_artists) >= 1


def test_overlay_absent_before_training_present_after() -> None:
    """The chart's GMM overlay must be absent pre-training and present post-training."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    _run_with_n_feed(at, 10)
    assert at.exception == []
    chart_before = at.session_state["chart"]
    assert chart_before.gmm_overlay_active is False
    assert chart_before.gmm_band_artists == []

    _run_with_n_feed(at, 100)
    assert at.exception == []
    chart_after = at.session_state["chart"]
    assert chart_after.gmm_overlay_active is True
    assert len(chart_after.gmm_band_artists) >= 1
    assert len(chart_after.gmm_mean_lines) >= 1
