"""Entry point for the Predictive Maintenance Demonstrator web app.

This is an incrementally-growing skeleton (story A1 onward). It boots
the Streamlit page, initializes shared session state (A3), and wires in
sidebar controls as later stories land (B1-B4, C1-C3, D1-D2, E1-E4, F1,
G1-G2, H1-H3) build out the full sensor-data / anomaly-detection web
experience on top of this scaffold.

:raises None: This module does not raise on import; Streamlit handles
    script execution and re-runs internally.
"""

import logging

import streamlit as st

from core.simulation_loader import generate_scenarios
from state.app_state import init_session_state
from ui_web.chart import render_chart_into_placeholder
from ui_web.dashboard import render_dashboard_popover
from ui_web.live_tick import live_tick_fragment
from ui_web.logbook import render_logbook_table
from ui_web.sidebar_config import (
    render_run_controls,
    render_simulation_controls,
    render_template_and_stream_controls,
)
from ui_web.styling import inject_theme_css

# Configure module-level logging so later stories can extend/reuse it
# instead of each re-inventing logging setup.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Render the Streamlit page.

    :returns: None
    :rtype: None
    """
    st.set_page_config(
        page_title="DigiMach Machine Health Demonstrator: Anomaly Detection",
        page_icon=":gear:",
        layout="wide",
    )

    # G1: inject the semantic button-coloring <style> block before any
    # widgets render, so the CSS is already present in the DOM by the
    # time the Start/Stop/Retrain/Load & Play/Stop Sim buttons are drawn
    # below (see ui_web/styling.py's inject_theme_css docstring for the
    # rationale/selector mechanism).
    inject_theme_css()

    # Idempotent: safe to call on every rerun (see state.app_state docs).
    init_session_state()

    # Idempotent/cheap (only writes CSVs that don't already exist) --
    # guarantees the D2 simulation scenario CSVs exist before "Load &
    # Play" can ever be clicked, without needing a separate first-boot
    # migration step.
    generate_scenarios()

    st.title("DigiMach Machine Health Demonstrator: Live Anomaly Detection")
    st.write(
        "Learning to detect anomalies in machine health data using sensor data from machines. \n"
        "1. **Collect sensor data** from machines during normal operation (training data). \n"
        "2. **Learn a model** of the sensor data's normal behavior (machine learning). \n"
        "3. **Flag** incoming data as **anomalous** using the model of normal behavior (inference). \n"
        "4. **Judge** the flagged anomalies to identify problems with the model and take corrective actions (logbook). \n"
        "5. **Improve** (retrain) the model as new data becomes available, e.g., to account for sensor drift or new operating conditions. \n \n"
        "**Note**: This is a demonstrator. In a real prototype system, the data would be collected from real machines, the model may be multi-dimensional and more complex, the UI tailored to companies' specific needs, and it could be integrated into existing monitoring and alerting systems. "
    )

    with st.sidebar:
        render_template_and_stream_controls()
        render_run_controls()
        #render_simulation_controls()
        # F1: "Dashboard" popover with session-summary stats + pie chart.
        # Rendered last in the sidebar, below the run/simulation controls,
        # since it is a read-only summary rather than a stream control.
        render_dashboard_popover()

    # P1: (re)create the st.empty() chart placeholder here, in the
    # *outer* (non-fragment) script execution, on *every* full script
    # run -- not idempotently guarded -- and immediately render the
    # persistent chart's current Figure into it. This guarantees the
    # last frame is visible right after any full rerun (e.g. clicking
    # "Pause"/"Stop", itself a full rerun, not a fragment-only one),
    # regardless of app_state. live_tick_fragment() then only *writes
    # into* this same placeholder (via
    # ui_web.chart.get_chart_placeholder()) while RUNNING/
    # SIMULATION_RUNNING, skipping the write while IDLE/PAUSED -- see
    # ui_web.chart.render_chart_into_placeholder's docstring for why
    # the placeholder must be recreated here every run rather than
    # cached/reused, and ui_web.live_tick.live_tick_fragment's
    # docstring for the fragment-side half of this mechanism.
    render_chart_into_placeholder()

    # Top-level (not inside `with st.sidebar:`): the fragment itself
    # opens its own `with st.sidebar:` block internally to render the
    # "GMM Training" group (story B4) on the same 300ms cadence as the
    # chart it also renders here in the main content area.
    live_tick_fragment()

    # E3: editable anomaly logbook (filter/search/footer), rendered below
    # the live chart in the main content area (not the sidebar) -- its
    # own @st.fragment(run_every="1s") keeps it independently refreshed.
    render_logbook_table()

    logger.info("Rendered page.")


# `streamlit run app.py` executes this script as `__main__`. Guarding with
# this check keeps `main()` import-safe (e.g. for unit tests that import
# `app` without triggering a page render).
if __name__ == "__main__":
    main()
