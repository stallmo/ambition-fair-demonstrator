"""Tests for story G1: Dark theme + semantic button styling.

Covers:

* ``.streamlit/config.toml`` exists, parses as valid TOML, and sets
  exactly the five ``[theme]`` keys/values the story requires.
* :func:`ui_web.styling.build_theme_css` (the pure, unit-testable CSS
  string builder) emits the expected ``.st-key-{key} button`` selectors
  scoped to the correct color for each of the five semantically-colored
  buttons.
* :func:`ui_web.styling.inject_theme_css` renders without raising inside
  a real Streamlit script (:class:`streamlit.testing.v1.AppTest`) and
  the ``st.markdown`` element it produces contains the expected CSS.
* The "Retrain Model" button (``ui_web/live_tick.py``) now has
  ``key="btn_retrain"`` and an icon-prefixed label.
* All five target buttons keep an icon character in their label,
  independent of whether the injected CSS applies (the story's "still
  usable/understandable with the CSS temporarily disabled" criterion).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from ui_web.styling import (
    GREEN_BUTTON_KEYS,
    PURPLE_BUTTON_KEYS,
    RED_BUTTON_KEYS,
    build_theme_css,
)

#: Path to the theme config this story adds, resolved relative to this
#: test file so pytest can be invoked from any working directory.
_CONFIG_TOML_PATH = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

#: Expected theme keys/values, exactly as specified by the story's
#: acceptance criteria.
_EXPECTED_THEME = {
    "base": "dark",
    "primaryColor": "#bd93f9",
    "backgroundColor": "#1e1e2e",
    "secondaryBackgroundColor": "#2a2a3e",
    "textColor": "#f8f8f2",
}


# ----------------------------------------------------------------------
# .streamlit/config.toml
# ----------------------------------------------------------------------


def test_config_toml_exists() -> None:
    """The theme config file must exist at webapp/.streamlit/config.toml."""
    assert _CONFIG_TOML_PATH.is_file(), f"missing {_CONFIG_TOML_PATH}"


def test_config_toml_parses_with_exactly_the_expected_theme_keys() -> None:
    """The [theme] table must parse and contain exactly the five required keys/values."""
    with _CONFIG_TOML_PATH.open("rb") as fh:
        parsed = tomllib.load(fh)

    assert "theme" in parsed, parsed
    assert parsed["theme"] == _EXPECTED_THEME


# ----------------------------------------------------------------------
# ui_web.styling.build_theme_css (pure CSS-string builder)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("key", GREEN_BUTTON_KEYS)
def test_green_buttons_scoped_to_green_color(key: str) -> None:
    """Start / Load & Play buttons' st-key selector must be colored green (#50fa7b)."""
    css = build_theme_css()
    assert f".st-key-{key} button" in css
    # The selector and the green color must appear in the same rule block
    # (not merely somewhere else in the stylesheet) -- assert the
    # selector line is immediately followed by the green declaration.
    selector_index = css.index(f".st-key-{key} button")
    rule_block = css[selector_index : selector_index + 200]
    assert "#50fa7b" in rule_block


@pytest.mark.parametrize("key", RED_BUTTON_KEYS)
def test_red_buttons_scoped_to_red_color(key: str) -> None:
    """Stop / Stop Sim buttons' st-key selector must be colored red (#ff5555)."""
    css = build_theme_css()
    assert f".st-key-{key} button" in css
    selector_index = css.index(f".st-key-{key} button")
    rule_block = css[selector_index : selector_index + 200]
    assert "#ff5555" in rule_block


@pytest.mark.parametrize("key", PURPLE_BUTTON_KEYS)
def test_purple_buttons_scoped_to_purple_color(key: str) -> None:
    """Retrain Model button's st-key selector must be colored purple (#bd93f9)."""
    css = build_theme_css()
    assert f".st-key-{key} button" in css
    selector_index = css.index(f".st-key-{key} button")
    rule_block = css[selector_index : selector_index + 200]
    assert "#bd93f9" in rule_block


def test_build_theme_css_wraps_in_a_style_tag() -> None:
    """The generated CSS must be wrapped in a <style>...</style> block."""
    css = build_theme_css()
    assert css.strip().startswith("<style>")
    assert css.strip().endswith("</style>")


def test_all_five_target_button_keys_are_covered_and_disjoint() -> None:
    """The five target buttons must be split across exactly the three color groups, no overlaps."""
    all_keys = GREEN_BUTTON_KEYS + RED_BUTTON_KEYS + PURPLE_BUTTON_KEYS
    assert sorted(all_keys) == sorted(
        ["btn_start", "btn_sim_play", "btn_stop", "btn_sim_stop", "btn_retrain"]
    )
    assert len(set(all_keys)) == len(all_keys)  # no duplicates/overlaps between groups


# ----------------------------------------------------------------------
# ui_web.styling.inject_theme_css via AppTest (real Streamlit script run)
# ----------------------------------------------------------------------

_INJECT_APP_SCRIPT = """
from ui_web.styling import inject_theme_css

inject_theme_css()
"""


def test_inject_theme_css_renders_without_raising() -> None:
    """inject_theme_css() must run cleanly inside a real Streamlit script."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_INJECT_APP_SCRIPT)
    at.run()

    assert at.exception == []


def test_inject_theme_css_markdown_output_contains_all_expected_colors() -> None:
    """The st.markdown element inject_theme_css() renders must carry all three colors."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_INJECT_APP_SCRIPT)
    at.run()

    assert at.exception == []
    markdown_html = [m.value for m in at.markdown]
    assert len(markdown_html) == 1
    css = markdown_html[0]
    assert "#50fa7b" in css
    assert "#ff5555" in css
    assert "#bd93f9" in css
    for key in ("btn_start", "btn_sim_play", "btn_stop", "btn_sim_stop", "btn_retrain"):
        assert f".st-key-{key} button" in css


# ----------------------------------------------------------------------
# Retrain Model button: key + icon-bearing label (ui_web/live_tick.py)
# ----------------------------------------------------------------------

_RETRAIN_APP_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.live_tick import render_gmm_training_group

init_session_state()
with st.sidebar:
    render_gmm_training_group()
"""


def test_retrain_button_has_stable_key_and_icon_label() -> None:
    """The Retrain Model button must have key='btn_retrain' and an icon-prefixed label."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_RETRAIN_APP_SCRIPT)
    at.run()

    assert at.exception == []
    button = at.sidebar.button(key="btn_retrain")
    assert button.label == "⟳ Retrain Model"
    # Icon survives even if the CSS block is disabled -- directly
    # checkable as a string property, independent of any CSS.
    assert "⟳" in button.label
    assert "Retrain" in button.label


# ----------------------------------------------------------------------
# All five buttons: icon-bearing labels, independent of CSS
# ----------------------------------------------------------------------

_RUN_CONTROLS_APP_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.sidebar_config import render_run_controls, render_simulation_controls

init_session_state()
with st.sidebar:
    render_run_controls()
    render_simulation_controls()
"""

#: (button key, expected icon character, expected label substring) for
#: the four sidebar_config.py buttons this story colors. Retrain Model
#: is covered separately above since it lives in a different module.
_EXPECTED_ICON_BUTTONS = [
    ("btn_start", "▶", "Start"),
    ("btn_stop", "■", "Stop"),
    ("btn_sim_play", "▶", "Load & Play"),
    ("btn_sim_stop", "■", "Stop Sim"),
]


@pytest.mark.parametrize("key,icon,text", _EXPECTED_ICON_BUTTONS)
def test_sidebar_config_button_labels_carry_a_distinguishing_icon(
    key: str, icon: str, text: str
) -> None:
    """Start/Stop/Load & Play/Stop Sim labels must keep an icon + distinguishing text."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_RUN_CONTROLS_APP_SCRIPT)
    at.run()

    assert at.exception == []
    button = at.sidebar.button(key=key)
    assert icon in button.label
    assert text in button.label


# ----------------------------------------------------------------------
# app.py wiring: inject_theme_css() must actually be called by main()
# ----------------------------------------------------------------------
#
# G1 was previously marked "done" while build_theme_css()/inject_theme_css()
# existed but were never called from app.py -- meaning the CSS never
# actually reached the rendered page. These tests guard against that
# regression recurring: one exercises the *real* app.py script end-to-end
# (mirrors the pattern in test_logbook.py's
# test_app_main_renders_the_anomaly_logbook_section), the other patches
# ui_web.styling.inject_theme_css directly and asserts app.main() calls it.


def test_app_main_injects_theme_css_into_the_page() -> None:
    """Running the real app.py script (via AppTest) must inject the G1 button CSS."""
    from streamlit.testing.v1 import AppTest

    script = """
import app
app.main()
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    markdown_html = [m.value for m in at.markdown]
    # The <style> block inject_theme_css() renders must be present among
    # app.py's st.markdown() elements, carrying all three semantic colors
    # and all five target button selectors.
    css_blocks = [html for html in markdown_html if "<style>" in html]
    assert len(css_blocks) == 1, markdown_html
    css = css_blocks[0]
    assert "#50fa7b" in css
    assert "#ff5555" in css
    assert "#bd93f9" in css
    for key in ("btn_start", "btn_sim_play", "btn_stop", "btn_sim_stop", "btn_retrain"):
        assert f".st-key-{key} button" in css


def test_app_main_calls_inject_theme_css() -> None:
    """app.main() must call ui_web.styling.inject_theme_css (not just define it)."""
    from unittest.mock import patch

    import app

    with patch("app.inject_theme_css") as mock_inject:
        app.main()

    mock_inject.assert_called_once()
