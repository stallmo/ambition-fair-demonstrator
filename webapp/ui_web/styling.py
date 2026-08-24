"""Semantic button-coloring CSS injection (story G1).

The ``.streamlit/config.toml`` file (also part of story G1) sets the
overall Dracula dark theme (background/text/primary colors), which
Streamlit applies natively to every widget. It does *not*, however,
give a way to color individual buttons differently from one another --
Streamlit's ``theme.primaryColor`` is applied uniformly to every
``type="primary"`` button, and there is no per-widget theming knob in
``config.toml``. To make the Start/Stop/Retrain/Load & Play/Stop Sim
buttons carry the same green="go"/red="stop"/purple="retrain" meaning
the desktop app conveyed via distinct ``QPushButton`` stylesheets
(``ui/config_panel.py`` / ``assets/style.qss``), this module injects a
small scoped ``<style>`` block via ``st.markdown(..., unsafe_allow_html=True)``.

CSS selector mechanism
-----------------------
Every Streamlit widget that is given an explicit ``key=`` argument gets
a CSS class named ``st-key-{key}`` on its wrapping DOM element (this is
documented directly in the installed Streamlit 1.61 source, e.g.
``elements/widgets/button.py``'s ``key`` parameter docstring: "if
``key`` is provided, it will be used as a CSS class name prefixed with
``st-key-``"). This means a plain, already-keyed ``st.button(key="btn_start", ...)``
call (see ``ui_web/sidebar_config.py`` / ``ui_web/live_tick.py``) is
directly targetable as ``.st-key-btn_start button`` -- no extra
``st.container(key=...)`` wrapping is needed for a single widget like a
button (that wrapping pattern matters when scoping a *group* of
widgets that don't each have their own key, which is not the case
here). This keeps the selector chain short and avoids introducing
extra DOM nesting purely for CSS-hook purposes.

Every one of the five colored buttons keeps its icon-prefixed label
(``"▶ Start"``, ``"■ Stop"``, ``"⟳ Retrain Model"``, ``"▶ Load & Play"``,
``"■ Stop Sim"``) regardless of whether this CSS successfully applies,
so the button's purpose is never solely conveyed by color (accessibility
/ "if the CSS hook breaks" resilience called out in the story).
"""
from __future__ import annotations

import logging
from typing import Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dracula-derived palette constants.
# ---------------------------------------------------------------------------
# Intentionally re-declared here (rather than imported from ui_web.chart)
# to keep this styling module decoupled from chart internals -- it deals
# in CSS/DOM, not matplotlib Axes/Figure objects, and the two modules have
# no other reason to depend on each other. Values match ui_web.chart's
# COLOR_NORMAL / COLOR_ANOMALY / GMM_BAND_COLORS[0] and
# .streamlit/config.toml's theme.primaryColor exactly.
_COLOR_GREEN = "#50fa7b"
_COLOR_RED = "#ff5555"
_COLOR_PURPLE = "#bd93f9"
_COLOR_BTN_TEXT = "#1e1e2e"  # dark text for readability on the bright fills above

#: Streamlit widget ``key=`` values of the five buttons this story colors,
#: keyed by the semantic role their color conveys. Mirrors the ``key=``
#: arguments on the ``st.button(...)`` calls in ``ui_web/sidebar_config.py``
#: (``btn_start``, ``btn_stop``, ``btn_sim_play``, ``btn_sim_stop``) and
#: ``ui_web/live_tick.py`` (``btn_retrain``).
GREEN_BUTTON_KEYS: Tuple[str, ...] = ("btn_start", "btn_sim_play")
RED_BUTTON_KEYS: Tuple[str, ...] = ("btn_stop", "btn_sim_stop")
PURPLE_BUTTON_KEYS: Tuple[str, ...] = ("btn_retrain",)


def _selector_rule(keys: Tuple[str, ...], color: str, text_color: str) -> str:
    """Build one CSS rule coloring the buttons for a set of widget keys.

    :param keys: Streamlit widget ``key=`` values whose ``.st-key-{key}``
        wrapper class should be targeted.
    :type keys: tuple[str, ...]
    :param color: CSS background color (hex string) to apply.
    :type color: str
    :param text_color: CSS text/border color (hex string) to apply.
    :type text_color: str
    :returns: A single CSS rule (selector list + declaration block) as
        a string, ready to be embedded inside a ``<style>`` block.
    :rtype: str
    """
    # Defensive: an empty keys tuple would otherwise produce a selector-less
    # (invalid) CSS rule "{ ... }" -- skip it rather than emit malformed CSS.
    if not keys:
        logger.warning("_selector_rule: called with no keys; skipping rule for color=%s", color)
        return ""

    selectors = ", ".join(f".st-key-{key} button" for key in keys)
    return (
        f"{selectors} {{\n"
        f"    background-color: {color} !important;\n"
        f"    color: {text_color} !important;\n"
        f"    border-color: {color} !important;\n"
        f"}}\n"
        f"{selectors}:hover {{\n"
        f"    filter: brightness(1.1);\n"
        f"}}\n"
        f"{selectors}:disabled {{\n"
        f"    filter: grayscale(0.6) opacity(0.6);\n"
        f"}}\n"
    )


def build_theme_css() -> str:
    """Build the full ``<style>...</style>`` block for semantic button colors.

    Pure string-building helper split out from :func:`inject_theme_css` so
    the generated CSS (selectors, color values) can be unit-tested by
    plain substring assertions, without needing a running Streamlit
    script / ``ScriptRunContext``.

    :returns: A complete HTML ``<style>...</style>`` string.
    :rtype: str
    """
    rules = [
        _selector_rule(GREEN_BUTTON_KEYS, _COLOR_GREEN, _COLOR_BTN_TEXT),
        _selector_rule(RED_BUTTON_KEYS, _COLOR_RED, _COLOR_BTN_TEXT),
        _selector_rule(PURPLE_BUTTON_KEYS, _COLOR_PURPLE, _COLOR_BTN_TEXT),
    ]
    css_body = "\n".join(rule for rule in rules if rule)
    return f"<style>\n{css_body}</style>"


def inject_theme_css() -> None:
    """Inject the G1 semantic button-coloring ``<style>`` block into the page.

    Colors the five run-lifecycle/demo buttons that carry a semantic
    action (matching the desktop app's distinct ``QPushButton``
    stylesheets):

    * Green (``#50fa7b``): "▶ Start" (``key="btn_start"``,
      ``ui_web/sidebar_config.py``) and "▶ Load & Play"
      (``key="btn_sim_play"``, ``ui_web/sidebar_config.py``).
    * Red (``#ff5555``): "■ Stop" (``key="btn_stop"``,
      ``ui_web/sidebar_config.py``) and "■ Stop Sim"
      (``key="btn_sim_stop"``, ``ui_web/sidebar_config.py``).
    * Purple (``#bd93f9``): "⟳ Retrain Model" (``key="btn_retrain"``,
      ``ui_web/live_tick.py``).

    "↺ Reset Session" (``key="btn_reset"``) is intentionally left
    unstyled/default -- it is not one of the five buttons named in this
    story's acceptance criteria.

    Uses ``st.markdown(..., unsafe_allow_html=True)`` because Streamlit's
    ``config.toml`` has no per-widget theming knob (see module docstring);
    the CSS is scoped purely to each button's own ``st-key-{key}`` class,
    so it cannot leak onto unrelated widgets that happen to share a
    parent container.

    **Expected call site**: this function is meant to be imported and
    called once, near the top of ``app.py``'s ``main()`` -- e.g.::

        from ui_web.styling import inject_theme_css
        inject_theme_css()

    before the rest of the page is rendered, so the CSS is present in
    the DOM by the time the sidebar/buttons render. ``app.py``'s
    ``main()`` calls this function immediately after ``st.set_page_config``
    and before ``init_session_state``/any widget rendering.

    :returns: None
    :rtype: None
    """
    css = build_theme_css()
    st.markdown(css, unsafe_allow_html=True)
    logger.debug("inject_theme_css: injected %d bytes of scoped button CSS", len(css))
