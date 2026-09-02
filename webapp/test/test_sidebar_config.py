"""Tests for :mod:`ui_web.sidebar_config` (Sensor Template / Stream Parameters).

Covers the pure, unit-testable helpers (:func:`_on_template_changed`,
:func:`_sync_generator_config`) with fakes, plus
:class:`streamlit.testing.v1.AppTest`-based tests that drive the actual
selectbox/number_input widgets in a real Streamlit script context, per
the story's acceptance criteria:

* the selectbox lists exactly Temperature/Vibration/Pressure;
* selecting a template pushes its exact ``TEMPLATES`` values into
  ``cfg_mean``/``cfg_std``/``cfg_noise`` via the ``on_change`` callback
  (not a post-hoc assignment, which would raise
  ``StreamlitAPIException``);
* number inputs are bounded per the spec;
* manual overrides of mean/std/noise survive an unrelated rerun;
* any change updates the live ``StreamConfig`` passed to
  ``generator.configure()``.
"""
from __future__ import annotations

from typing import List

import pytest

from core.data_generator import TEMPLATES, DataGenerator, StreamConfig
from ui_web.sidebar_config import (
    _on_template_changed,
    _sync_generator_config,
    render_anomaly_control,
)


class _FakeSessionState(dict):
    """Minimal dict-like stand-in for ``st.session_state``.

    Mirrors the ``_FakeSessionState`` pattern used elsewhere in this
    test suite (``test_app_state.py``, ``test_live_tick.py``): the real
    ``st.session_state`` warns/behaves oddly outside of a running
    Streamlit script, so unit tests inject a plain dict subclass into
    the injectable-``state`` parameter instead.
    """


class _FakeGenerator:
    """Stub generator recording every ``configure()`` call's argument.

    :ivar configs: List of :class:`StreamConfig` instances passed to
        :meth:`configure`, in call order.
    :vartype configs: list[StreamConfig]
    """

    def __init__(self) -> None:
        self.configs: List[StreamConfig] = []

    def configure(self, config: StreamConfig) -> None:
        """Record the call.

        :param config: Config passed by the caller.
        :type config: StreamConfig
        """
        self.configs.append(config)


# ----------------------------------------------------------------------
# _on_template_changed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("template_name", list(TEMPLATES.keys()))
def test_on_template_changed_pushes_exact_template_values(template_name: str) -> None:
    """Each template's mean/std/noise are pushed byte-for-byte into cfg_* keys."""
    state = _FakeSessionState(cfg_template=template_name)

    _on_template_changed(state)

    template = TEMPLATES[template_name]
    assert state["cfg_mean"] == template["mean"]
    assert state["cfg_std"] == template["std"]
    assert state["cfg_noise"] == template["noise"]


def test_on_template_changed_unknown_template_is_a_defensive_noop() -> None:
    """An unknown template name must not raise or corrupt existing values."""
    state = _FakeSessionState(
        cfg_template="NotARealTemplate", cfg_mean=1.0, cfg_std=2.0, cfg_noise=3.0
    )

    _on_template_changed(state)  # must not raise

    assert state["cfg_mean"] == 1.0
    assert state["cfg_std"] == 2.0
    assert state["cfg_noise"] == 3.0


# ----------------------------------------------------------------------
# _sync_generator_config
# ----------------------------------------------------------------------


def test_sync_generator_config_pushes_current_values_to_generator() -> None:
    """configure() must be called with the current sidebar values, converted correctly."""
    generator = _FakeGenerator()
    state = _FakeSessionState(
        generator=generator,
        cfg_template="Vibration",
        cfg_mean=0.6,
        cfg_std=0.2,
        cfg_noise=0.07,
        cfg_anomaly_pct=10,
    )

    config = _sync_generator_config(state)

    assert len(generator.configs) == 1
    pushed = generator.configs[0]
    assert pushed is config
    assert pushed.mean == 0.6
    assert pushed.std == 0.2
    assert pushed.noise_amplitude == 0.07
    # anomaly_pct (0-100) must be converted to a 0-1 fraction, matching
    # the desktop app's `self._anomaly_slider.value() / 100.0`.
    assert pushed.anomaly_probability == pytest.approx(0.10)
    # anomaly_magnitude/unit are not independently editable; they always
    # follow the currently-selected template.
    assert pushed.anomaly_magnitude == TEMPLATES["Vibration"]["anomaly_magnitude"]
    assert pushed.unit == TEMPLATES["Vibration"]["unit"]


def test_sync_generator_config_defaults_anomaly_pct_when_absent() -> None:
    """Missing cfg_anomaly_pct (pre-C2) must not crash; falls back to A3's seeded default."""
    generator = _FakeGenerator()
    state = _FakeSessionState(
        generator=generator, cfg_template="Temperature", cfg_mean=70.0, cfg_std=2.0, cfg_noise=0.5
    )

    config = _sync_generator_config(state)

    assert config.anomaly_probability == pytest.approx(0.02)


def test_sync_generator_config_missing_generator_is_a_defensive_noop() -> None:
    """No generator in state (init_session_state() not called) must not raise."""
    state = _FakeSessionState(cfg_template="Temperature", cfg_mean=70.0, cfg_std=2.0, cfg_noise=0.5)

    config = _sync_generator_config(state)  # must not raise

    assert isinstance(config, StreamConfig)


def test_sync_generator_config_works_against_real_generator() -> None:
    """End-to-end sanity check against the real DataGenerator, not just a fake."""
    generator = DataGenerator()
    state = _FakeSessionState(
        generator=generator,
        cfg_template="Pressure",
        cfg_mean=105.0,
        cfg_std=4.0,
        cfg_noise=1.5,
        cfg_anomaly_pct=5,
    )

    _sync_generator_config(state)

    # Internal config isn't publicly exposed; verify indirectly via a tick's
    # magnitude of drift, which is only possible/meaningful with mean/std
    # set -- so instead assert against the private attribute directly,
    # since DataGenerator intentionally has no public getter (mirrors the
    # desktop app, which never reads config back out of the generator).
    assert generator._config.mean == 105.0
    assert generator._config.std == 4.0
    assert generator._config.noise_amplitude == 1.5
    assert generator._config.anomaly_probability == pytest.approx(0.05)
    assert generator._config.anomaly_magnitude == TEMPLATES["Pressure"]["anomaly_magnitude"]
    assert generator._config.unit == TEMPLATES["Pressure"]["unit"]


# ----------------------------------------------------------------------
# AppTest: exercise the real selectbox/number_input widgets end-to-end.
# ----------------------------------------------------------------------

_APP_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.sidebar_config import render_template_and_stream_controls

init_session_state()
with st.sidebar:
    render_template_and_stream_controls()
"""


def test_selectbox_lists_exactly_the_three_templates() -> None:
    """The Sensor Template selectbox must list exactly Temperature/Vibration/Pressure."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    assert at.exception == []
    selectbox = at.sidebar.selectbox(key="cfg_template")
    assert list(selectbox.options) == ["Temperature", "Vibration", "Pressure"]


@pytest.mark.parametrize("template_name", list(TEMPLATES.keys()))
def test_selecting_template_via_widget_sets_mean_std_noise(template_name: str) -> None:
    """Selecting a template through the real widget updates cfg_mean/std/noise exactly."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    at.sidebar.selectbox(key="cfg_template").select(template_name).run()

    assert at.exception == []
    template = TEMPLATES[template_name]
    assert at.session_state["cfg_mean"] == template["mean"]
    assert at.session_state["cfg_std"] == template["std"]
    assert at.session_state["cfg_noise"] == template["noise"]


def test_manual_override_survives_unrelated_rerun() -> None:
    """Editing mean/std/noise after a template selection must not snap back on rerun."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    # Select a template, then manually override all three fields.
    at.sidebar.selectbox(key="cfg_template").select("Vibration").run()
    at.sidebar.number_input(key="cfg_mean").set_value(1.23).run()
    at.sidebar.number_input(key="cfg_std").set_value(0.45).run()
    at.sidebar.number_input(key="cfg_noise").set_value(0.06).run()

    assert at.exception == []
    assert at.session_state["cfg_mean"] == pytest.approx(1.23)
    assert at.session_state["cfg_std"] == pytest.approx(0.45)
    assert at.session_state["cfg_noise"] == pytest.approx(0.06)

    # An unrelated rerun (re-running the same script state, simulating
    # e.g. another widget interaction elsewhere on the page) must not
    # reset the manually-overridden values back to Vibration's defaults.
    at.run()

    assert at.session_state["cfg_mean"] == pytest.approx(1.23)
    assert at.session_state["cfg_std"] == pytest.approx(0.45)
    assert at.session_state["cfg_noise"] == pytest.approx(0.06)


def test_number_input_bounds() -> None:
    """Mean/Std Dev/Noise number_inputs must be bounded per the spec."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    mean_input = at.sidebar.number_input(key="cfg_mean")
    std_input = at.sidebar.number_input(key="cfg_std")
    noise_input = at.sidebar.number_input(key="cfg_noise")

    assert (mean_input.min, mean_input.max) == (0.0, 9999.0)
    assert (std_input.min, std_input.max) == (0.01, 100.0)
    assert (noise_input.min, noise_input.max) == (0.0, 50.0)


def test_changing_widgets_updates_live_generator_config() -> None:
    """Any widget change must update session_state['generator']'s live StreamConfig."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    at.sidebar.number_input(key="cfg_mean").set_value(88.8).run()

    assert at.exception == []
    generator = at.session_state["generator"]
    assert generator._config.mean == pytest.approx(88.8)
    # anomaly_probability should still reflect A3's seeded 2% default
    # converted to a fraction, since this test never touches the C2
    # anomaly slider.
    assert generator._config.anomaly_probability == pytest.approx(0.02)


# ----------------------------------------------------------------------
# C2: Anomaly Probability slider
# ----------------------------------------------------------------------


def test_render_anomaly_control_returns_current_pct_with_fake_state() -> None:
    """render_anomaly_control() reads/returns cfg_anomaly_pct from the injected state.

    This does not exercise the real ``st.slider`` widget (that is
    covered by the AppTest-based tests below); it only verifies the
    non-widget bookkeeping -- the caption text / return value -- against
    a fake session-state mapping, mirroring the pattern used for
    :func:`_on_template_changed` / :func:`_sync_generator_config` above.
    """
    state = _FakeSessionState(cfg_anomaly_pct=42)

    # st.slider/st.subheader/st.caption calls outside a real Streamlit
    # script context just log a "missing ScriptRunContext" warning and
    # no-op rather than raising, so calling this directly is safe here.
    result = render_anomaly_control(state)

    assert result == 42


def test_render_anomaly_control_defaults_when_pct_absent() -> None:
    """Missing cfg_anomaly_pct must not crash; falls back to the seeded default of 2."""
    state = _FakeSessionState()

    result = render_anomaly_control(state)

    assert result == 2


def test_anomaly_slider_bounds_and_default() -> None:
    """The Anomaly Probability slider must be a 0-100 range defaulting to 2."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    assert at.exception == []
    slider = at.sidebar.slider(key="cfg_anomaly_pct")
    assert (slider.min, slider.max) == (0, 100)
    assert slider.value == 2


def test_anomaly_slider_displays_percentage_format() -> None:
    """The slider's live value must render as f"{v}%" (handle format + caption)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    slider = at.sidebar.slider(key="cfg_anomaly_pct")
    # The printf-style format string applied to the slider's own handle
    # label; reproduces the desktop app's f"{val}%" QLabel text.
    assert slider.proto.format == "%d%%"

    # Redundant, directly-testable f"{v}%" caption underneath the
    # slider (see render_anomaly_control()'s docstring for why this is
    # kept alongside the slider's own format string).
    captions = [c.value for c in at.sidebar.caption]
    assert "2%" in captions


def test_moving_anomaly_slider_updates_generator_config() -> None:
    """Moving the slider must update generator._config.anomaly_probability = v/100."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    at.sidebar.slider(key="cfg_anomaly_pct").set_value(37).run()

    assert at.exception == []
    assert at.session_state["cfg_anomaly_pct"] == 37
    generator = at.session_state["generator"]
    assert generator._config.anomaly_probability == pytest.approx(0.37)

    # The caption must reflect the new live value too.
    captions = [c.value for c in at.sidebar.caption]
    assert "37%" in captions


@pytest.mark.parametrize("pct", [0, 1, 50, 99, 100])
def test_moving_anomaly_slider_across_range_updates_config(pct: int) -> None:
    """Any value across the full 0-100 range must convert to v/100 in StreamConfig."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_APP_SCRIPT)
    at.run()

    at.sidebar.slider(key="cfg_anomaly_pct").set_value(pct).run()

    assert at.exception == []
    generator = at.session_state["generator"]
    assert generator._config.anomaly_probability == pytest.approx(pct / 100.0)
