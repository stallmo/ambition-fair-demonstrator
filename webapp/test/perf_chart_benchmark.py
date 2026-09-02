"""Story Q3: standalone before/after benchmark for the Plotly chart migration (Q1).

This is a **manual** benchmark script, not a pytest test module (it is
deliberately named without a ``test_`` prefix/``_test`` suffix so pytest's
default discovery does not collect it and it never slows down or adds
flakiness to ``pytest test/ -q``). Run it directly::

    .venv/bin/python test/perf_chart_benchmark.py

Why this exists
----------------
Story Q1 replaced ``webapp/ui_web/chart.py``'s server-rendered matplotlib
``Figure`` (rasterized to a PNG and shipped to the browser via
``st.pyplot`` on every redraw) with a client-rendered Plotly
``go.Figure`` (shipped down once as a small JSON spec, then mutated/
re-serialized -- much cheaper -- and redrawn entirely in the browser by
plotly.js via ``st.plotly_chart``). Story Q3 asks for *measured*, not
assumed, evidence that this actually reduced server-side CPU/memory cost
per session.

``webapp/`` has no prior git history (it was added as a single, already
fully-formed commit), so there is no old commit of ``chart.py`` to check
out and benchmark directly -- see the story's own notes. This script
therefore takes the pragmatic approach the story explicitly sanctions:

Part A -- micro-benchmark (the primary, most reproducible evidence)
    Builds a small, **self-contained, benchmark-only** matplotlib chart
    (mirroring the pre-Q1 design closely: a two-panel
    ``GridSpec(2, 1, height_ratios=[3, 1])`` figure with a persistent
    line + two persistent scatter series in the top panel, and a
    confidence-bar panel that is cleared and rebuilt via ``ax.cla()`` +
    ``ax.bar()`` every redraw -- this exactly mirrors the real desktop
    app's ``ui/visualization_panel.py::VisualizationPanel._redraw()``,
    which the webapp's pre-Q1 ``ChartState`` was itself a port of) next
    to the real, current, post-Q1 Plotly ``ChartState`` imported from
    ``ui_web.chart``. Both are built **once** (mirroring the "persistent
    figure per session" invariant both versions share) and then driven
    through the same number of update+redraw+serialize cycles, over the
    same synthetic data volumes (ramping 1..300 points, then holding at
    the ``DEFAULT_CAPACITY=300`` steady state) -- the matplotlib side
    rasterizes to PNG bytes via ``FigureCanvasAgg.draw()`` +
    ``fig.savefig(..., format="png")`` (what ``st.pyplot`` sends over the
    websocket); the Plotly side serializes via ``plotly.io.to_json(fig)``
    (what ``st.plotly_chart`` sends over the websocket instead of a
    raster image). This isolates the actual per-redraw server-side cost
    difference the Q1 migration targeted, without needing a full
    multi-session Streamlit server harness.

Part B -- AppTest multi-session simulation (secondary, current-implementation-only)
    Uses ``streamlit.testing.v1.AppTest`` to spin up N independent
    sessions (N=10, N=50) each calling ``init_session_state()`` then the
    real, current (post-Q1) ``ui_web.live_tick.live_tick_fragment()``
    repeatedly with ``app_state=RUNNING``, sequentially in this one
    process -- mirroring the fact that a real Streamlit deployment is
    itself one single, GIL-bound process serving every concurrent
    session (see ``web-opt-changes.md``'s "Context" section, root cause
    #2). This *cannot* be run against the pre-Q1 matplotlib
    implementation (that code no longer exists anywhere in this
    checkout -- ``ui_web/chart.py`` and ``ui_web/live_tick.py`` are the
    real, current, already-migrated modules), so Part B only reports
    absolute post-Q1 numbers at N=10/N=50; the README's "before" N=10/N=50
    figures are extrapolated from Part A's measured per-cycle cost ratio
    applied to Part B's measured post-Q1 per-session cost -- this
    extrapolation is called out explicitly, both here and in the README,
    as an estimate rather than a direct measurement.

Caveats (read before trusting these numbers too far)
-----------------------------------------------------
* This is a single-process, single-machine micro/meso-benchmark, **not**
  a real multi-user load test with actual concurrent websocket
  connections, network latency, or browser-side rendering cost -- it
  measures server-side CPU/memory only, which is exactly the dimension
  story Q3 asks about, but it is still an approximation.
* ``resource.getrusage(...).ru_maxrss`` is a **high-water mark** for the
  whole process, not a clean per-phase delta -- Part A works around this
  by running the matplotlib and Plotly halves in separate, freshly
  spawned subprocesses (see :func:`_run_in_subprocess`), so each
  reported ``peak_rss_kb`` reflects only that subprocess's own peak
  (interpreter baseline + workload), not cross-contaminated by the other
  implementation's imports. Part B's RSS figure is a single process's
  peak across all N sessions (no subprocess isolation per session -- a
  real Streamlit server is a single process serving all sessions too, so
  this is arguably a *more* representative number for Part B's purpose
  than an isolated-per-session figure would be).
* "CPU%" here means ``100 * (user+sys CPU seconds consumed during the
  benchmarked window) / (wall-clock seconds of that window)`` for
  *this* process -- i.e. average CPU utilization during that window
  (100% == fully saturating one core the whole time), computed from
  :func:`resource.getrusage` deltas. No ``psutil`` dependency is used
  (not installed in this project's ``.venv``; stdlib ``resource`` is
  sufficient for a single-process, single-window measurement like this).
* Numbers vary by machine/load; treat the reported figures as
  order-of-magnitude, reproducible-on-this-machine evidence, not
  guaranteed absolute figures for any particular deployment target.
"""
from __future__ import annotations

import io
import logging
import multiprocessing
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# This script is run standalone (``python test/perf_chart_benchmark.py``),
# not via pytest -- conftest.py's sys.path setup (which makes ``core``/
# ``ui_web``/``state`` importable) only applies under pytest, so it must be
# replicated here explicitly.
_WEBAPP_ROOT = Path(__file__).resolve().parent.parent
if str(_WEBAPP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WEBAPP_ROOT))

# Keep benchmark output readable -- this module's own code and the
# app modules it imports (ui_web.chart, ui_web.live_tick, ...) log fairly
# verbosely at INFO/DEBUG on every update() call, which would otherwise
# drown out the benchmark's printed results.
logging.disable(logging.CRITICAL)

#: Rolling-window capacity the real chart is sized for (mirrors
#: ui_web.chart.DEFAULT_CAPACITY).
DEFAULT_CAPACITY = 300

#: Number of update+redraw+serialize cycles per Part A benchmark run --
#: chosen to mirror a realistic single RUNNING session: at the story P2
#: 300ms fragment cadence, 300 cycles is a ~90 second demo run (long
#: enough to ramp the ring buffer from empty to full capacity and then
#: hold steady-state for a while).
N_CYCLES = 300


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_window(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one deterministic ``(xs, vals, confs, anom)`` window of length ``n``.

    :param n: Window length (number of samples).
    :type n: int
    :param seed: Base RNG seed; combined with ``n`` so each distinct
        window length in a ramp is still reproducible run-to-run.
    :type seed: int
    :returns: Four same-length arrays: sample-index x-axis, values,
        confidence scores in ``[0, 1]``, and an anomaly boolean mask.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    rng = np.random.default_rng(seed + n)
    xs = np.arange(n, dtype=float)
    vals = rng.normal(20.0, 2.0, size=n)
    confs = rng.uniform(0.0, 1.0, size=n)
    anom = confs < 0.05
    return xs, vals, confs, anom


def _ramp_lengths(n_cycles: int, capacity: int) -> List[int]:
    """Window-length schedule mirroring a real session's ring buffer filling then holding.

    The first ``capacity`` cycles ramp the window length ``1..capacity``
    (buffer filling up from empty, as happens right after clicking
    "Start"); any remaining cycles hold at ``capacity`` (buffer full,
    steady-state redraw cost) -- this matches the actual workload shape
    ``update_chart_from_state`` produces over a real RUNNING session.

    :param n_cycles: Total number of cycles to schedule.
    :type n_cycles: int
    :param capacity: Maximum window length (ring buffer capacity).
    :type capacity: int
    :returns: A list of length ``n_cycles`` of window lengths.
    :rtype: list[int]
    """
    return [min(i + 1, capacity) for i in range(n_cycles)]


def _cpu_pct(rusage_before: Any, rusage_after: Any, wall_seconds: float) -> float:
    """Average CPU utilization (percent of one core) over a measured window.

    :param rusage_before: ``resource.getrusage(...)`` snapshot taken
        immediately before the measured window.
    :param rusage_after: Snapshot taken immediately after.
    :param wall_seconds: Wall-clock duration of the measured window.
    :type wall_seconds: float
    :returns: ``100 * (user+sys CPU seconds consumed) / wall_seconds``;
        ``0.0`` if ``wall_seconds`` is non-positive (defensive, avoids
        a division error on a degenerate/zero-length window).
    :rtype: float
    """
    if wall_seconds <= 0:
        return 0.0
    cpu_before = rusage_before.ru_utime + rusage_before.ru_stime
    cpu_after = rusage_after.ru_utime + rusage_after.ru_stime
    return 100.0 * (cpu_after - cpu_before) / wall_seconds


def _maxrss_kb(ru_maxrss: int) -> float:
    """Normalize ``ru_maxrss`` to kilobytes across platforms.

    :param ru_maxrss: Raw ``ru_maxrss`` value from
        :func:`resource.getrusage` -- reported in bytes on macOS/Darwin
        but kilobytes on Linux (a long-standing platform quirk of the
        underlying ``getrusage(2)`` syscall).
    :type ru_maxrss: int
    :returns: Peak resident set size in kilobytes.
    :rtype: float
    """
    if sys.platform == "darwin":
        return ru_maxrss / 1024.0
    return float(ru_maxrss)


# ---------------------------------------------------------------------------
# Part A -- legacy (pre-Q1) matplotlib chart, benchmark-only reconstruction
# ---------------------------------------------------------------------------


def _build_legacy_matplotlib_chart() -> Dict[str, Any]:
    """Build a standalone matplotlib chart mirroring the pre-Q1 design.

    FOR BENCHMARKING ONLY -- not reintroduced into the live app. Mirrors
    the real desktop app's ``ui/visualization_panel.py::_setup_figure()``
    (which the webapp's original, pre-Q1 ``ChartState`` was itself a
    faithful port of, per that module's current docstring): an (8, 5)in
    ``Figure`` with a ``GridSpec(2, 1, height_ratios=[3, 1])`` main/score
    panel split, a persistent line + two persistent scatter artists in
    the main panel, and the Dracula palette applied via the same hex
    constants.

    :returns: A dict of the figure/axes/artists later mutated by
        :func:`_redraw_and_serialize_legacy`.
    :rtype: dict[str, Any]
    """
    import matplotlib

    matplotlib.use("Agg")  # headless raster backend -- no display needed
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 5))
    fig.patch.set_facecolor("#1e1e2e")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_score = fig.add_subplot(gs[1], sharex=ax_main)

    for ax in (ax_main, ax_score):
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#f8f8f2", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#44475a")

    ax_main.set_ylabel("Value", color="#f8f8f2", fontsize=10)
    ax_score.set_ylabel("Confidence", color="#f8f8f2", fontsize=10)
    ax_score.set_xlabel("Sample Index", color="#f8f8f2", fontsize=10)
    ax_score.set_ylim(0, 1)

    (line,) = ax_main.plot([], [], color="#6272a4", linewidth=1.2, zorder=1)
    scat_normal = ax_main.scatter([], [], c="#50fa7b", s=12, zorder=2, label="Normal")
    scat_anomaly = ax_main.scatter(
        [], [], c="#ff5555", s=60, marker="^", zorder=3, label="Anomaly"
    )
    ax_main.legend(
        facecolor="#2a2a3e", edgecolor="#44475a", labelcolor="#f8f8f2", fontsize=8, loc="upper left"
    )

    canvas = FigureCanvasAgg(fig)
    return {
        "fig": fig,
        "canvas": canvas,
        "ax_main": ax_main,
        "ax_score": ax_score,
        "line": line,
        "scat_normal": scat_normal,
        "scat_anomaly": scat_anomaly,
    }


def _redraw_and_serialize_legacy(
    chart_objs: Dict[str, Any],
    xs: np.ndarray,
    vals: np.ndarray,
    confs: np.ndarray,
    anom: np.ndarray,
) -> int:
    """Mutate the legacy chart's artists and rasterize to PNG bytes.

    Mirrors ``ui/visualization_panel.py::VisualizationPanel._redraw()``:
    the line and two scatter artists are mutated in place, but the
    confidence-score axis is cleared (``ax.cla()``) and rebuilt with a
    fresh ``ax.bar()`` call every single redraw (the real desktop code's
    actual behavior -- it does *not* keep a persistent bar-patch pool),
    then the figure is drawn and rasterized to PNG via
    ``FigureCanvasAgg`` -- the same bytes a pre-Q1 ``st.pyplot(fig)``
    call would have shipped over the websocket.

    :param chart_objs: Dict returned by :func:`_build_legacy_matplotlib_chart`.
    :type chart_objs: dict[str, Any]
    :param xs: Sample-index x-axis values for this window.
    :type xs: numpy.ndarray
    :param vals: Sensor values for this window.
    :type vals: numpy.ndarray
    :param confs: Confidence scores for this window.
    :type confs: numpy.ndarray
    :param anom: Anomaly boolean mask for this window.
    :type anom: numpy.ndarray
    :returns: Size, in bytes, of the rasterized PNG (a proxy for the
        websocket payload size a pre-Q1 redraw would have sent).
    :rtype: int
    """
    fig = chart_objs["fig"]
    ax_main = chart_objs["ax_main"]
    ax_score = chart_objs["ax_score"]
    line = chart_objs["line"]
    scat_normal = chart_objs["scat_normal"]
    scat_anomaly = chart_objs["scat_anomaly"]

    line.set_data(xs, vals)

    norm_idx = np.where(~anom)[0]
    if len(norm_idx):
        scat_normal.set_offsets(np.c_[xs[norm_idx], vals[norm_idx]])
    else:
        scat_normal.set_offsets(np.empty((0, 2)))

    anom_idx = np.where(anom)[0]
    if len(anom_idx):
        scat_anomaly.set_offsets(np.c_[xs[anom_idx], vals[anom_idx]])
    else:
        scat_anomaly.set_offsets(np.empty((0, 2)))

    ax_main.relim()
    ax_main.autoscale_view()

    # Confidence panel: cleared and rebuilt from scratch every redraw,
    # exactly like the real desktop implementation (see docstring).
    ax_score.cla()
    ax_score.set_facecolor("#2a2a3e")
    ax_score.set_ylim(0, 1)
    ax_score.set_ylabel("Confidence", color="#f8f8f2", fontsize=10)
    ax_score.tick_params(colors="#f8f8f2", labelsize=9)
    for spine in ax_score.spines.values():
        spine.set_color("#44475a")
    colors = np.where(anom, "#ff5555", "#50fa7b")
    ax_score.bar(xs, confs, color=colors, width=1.0, align="center")

    chart_objs["canvas"].draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    return len(buf.getvalue())


def run_matplotlib_benchmark(
    n_cycles: int = N_CYCLES, capacity: int = DEFAULT_CAPACITY
) -> Dict[str, float]:
    """Run the legacy matplotlib update+redraw+serialize benchmark.

    :param n_cycles: Number of redraw cycles to run.
    :type n_cycles: int
    :param capacity: Ring buffer capacity the window ramp is sized for.
    :type capacity: int
    :returns: Timing/CPU/memory/payload-size results, see
        :func:`run_plotly_benchmark` for the matching key set.
    :rtype: dict[str, float]
    """
    lengths = _ramp_lengths(n_cycles, capacity)
    # Built once, outside the timed loop -- mirrors the real "persistent
    # Figure per session" invariant both the pre- and post-Q1 chart
    # implementations share (a fresh Figure is not rebuilt every redraw).
    try:
        chart_objs = _build_legacy_matplotlib_chart()
    except ImportError as exc:
        # matplotlib is not a declared webapp/requirements.txt dependency
        # any more once story Q2 lands (it drops matplotlib entirely) --
        # this benchmark's "before" side still needs it importable to
        # reconstruct the pre-Q1 comparison chart. Fail with an actionable
        # message rather than a bare, confusing ModuleNotFoundError deep
        # inside a spawned subprocess.
        raise RuntimeError(
            "matplotlib is required to run the pre-Q1 comparison side of "
            "this benchmark but is not importable in this environment. "
            "It is intentionally not a permanent webapp/requirements.txt "
            "dependency (see README.md's 'Performance notes' section) -- "
            "install it temporarily with `uv pip install matplotlib "
            "--python .venv/bin/python` (or `pip install matplotlib` in "
            "an active venv), re-run this script, then uninstall it again "
            "afterwards if you want to keep the environment matching "
            "requirements.txt exactly."
        ) from exc

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    last_payload_bytes = 0
    for n in lengths:
        xs, vals, confs, anom = _synthetic_window(n)
        last_payload_bytes = _redraw_and_serialize_legacy(chart_objs, xs, vals, confs, anom)
    wall = time.perf_counter() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "impl": "matplotlib (pre-Q1, benchmark reconstruction)",
        "n_cycles": float(n_cycles),
        "total_wall_seconds": wall,
        "avg_ms_per_cycle": (wall / n_cycles) * 1000.0,
        "cpu_pct": _cpu_pct(rusage_before, rusage_after, wall),
        "peak_rss_kb": _maxrss_kb(rusage_after.ru_maxrss),
        "last_payload_bytes": float(last_payload_bytes),
    }


# ---------------------------------------------------------------------------
# Part A -- current (post-Q1) Plotly ChartState, the real production code
# ---------------------------------------------------------------------------


def run_plotly_benchmark(
    n_cycles: int = N_CYCLES, capacity: int = DEFAULT_CAPACITY
) -> Dict[str, float]:
    """Run the real, current post-Q1 ``ChartState`` update+serialize benchmark.

    Uses the actual ``ui_web.chart.ChartState`` class (not a
    reconstruction) so this side of the comparison exercises the real
    production code path.

    :param n_cycles: Number of update+serialize cycles to run.
    :type n_cycles: int
    :param capacity: Forwarded to ``ChartState(capacity=...)``.
    :type capacity: int
    :returns: Timing/CPU/memory/payload-size results (same key set as
        :func:`run_matplotlib_benchmark`, for a side-by-side comparison).
    :rtype: dict[str, float]
    """
    import plotly.io as pio

    from ui_web.chart import ChartState

    lengths = _ramp_lengths(n_cycles, capacity)
    # Built once -- ChartState.__init__ does not touch st.session_state,
    # so it can be instantiated directly outside a running Streamlit
    # script/session (see ui_web.chart module docstring).
    chart = ChartState(capacity=capacity)

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    last_payload_bytes = 0
    for n in lengths:
        xs, vals, confs, anom = _synthetic_window(n)
        chart.update(timestamps=xs, values=vals, confidences=confs, is_anomaly=anom)
        # plotly.io.to_json(fig) is what st.plotly_chart serializes and
        # ships over the websocket on every render call -- the direct
        # Plotly analog of the matplotlib side's fig.savefig(..., "png").
        payload = pio.to_json(chart.fig, validate=False)
        last_payload_bytes = len(payload.encode("utf-8"))
    wall = time.perf_counter() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "impl": "Plotly (post-Q1, real ChartState)",
        "n_cycles": float(n_cycles),
        "total_wall_seconds": wall,
        "avg_ms_per_cycle": (wall / n_cycles) * 1000.0,
        "cpu_pct": _cpu_pct(rusage_before, rusage_after, wall),
        "peak_rss_kb": _maxrss_kb(rusage_after.ru_maxrss),
        "last_payload_bytes": float(last_payload_bytes),
    }


def _subprocess_entry(target_name: str, n_cycles: int, capacity: int, queue: "multiprocessing.Queue") -> None:
    """Run one of the two Part A benchmarks in an isolated subprocess.

    Isolating each implementation in its own freshly spawned process
    means ``peak_rss_kb`` reflects only that implementation's own
    interpreter-baseline + workload footprint, not contaminated by the
    other implementation's imports (e.g. matplotlib's C extension
    footprint leaking into the Plotly side's peak-RSS reading, or vice
    versa, if both ran in-process one after another).

    :param target_name: Either ``"matplotlib"`` or ``"plotly"``.
    :type target_name: str
    :param n_cycles: Forwarded to the chosen benchmark function.
    :type n_cycles: int
    :param capacity: Forwarded to the chosen benchmark function.
    :type capacity: int
    :param queue: Result is put here as a dict (subprocesses cannot
        return values directly to the parent).
    :type queue: multiprocessing.Queue
    :returns: None
    :rtype: None
    """
    logging.disable(logging.CRITICAL)
    if target_name == "matplotlib":
        result = run_matplotlib_benchmark(n_cycles=n_cycles, capacity=capacity)
    elif target_name == "plotly":
        result = run_plotly_benchmark(n_cycles=n_cycles, capacity=capacity)
    else:
        raise ValueError(f"unknown target_name: {target_name!r}")
    queue.put(result)


def run_part_a(n_cycles: int = N_CYCLES, capacity: int = DEFAULT_CAPACITY) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run both Part A benchmarks, each in its own isolated subprocess.

    :param n_cycles: Forwarded to both benchmark functions.
    :type n_cycles: int
    :param capacity: Forwarded to both benchmark functions.
    :type capacity: int
    :returns: ``(matplotlib_result, plotly_result)`` dicts.
    :rtype: tuple[dict[str, float], dict[str, float]]
    """
    ctx = multiprocessing.get_context("spawn")
    results: Dict[str, Dict[str, float]] = {}
    for target_name in ("matplotlib", "plotly"):
        queue: "multiprocessing.Queue" = ctx.Queue()
        proc = ctx.Process(
            target=_subprocess_entry, args=(target_name, n_cycles, capacity, queue)
        )
        proc.start()
        try:
            results[target_name] = queue.get(timeout=120)
        except Exception as exc:
            # The subprocess most likely crashed (e.g. matplotlib not
            # importable -- see _build_legacy_matplotlib_chart's
            # RuntimeError) before ever putting a result on the queue;
            # surface a clear, actionable error instead of a bare
            # multiprocessing.queues.Empty with no context.
            proc.join(timeout=5)
            raise RuntimeError(
                f"Part A '{target_name}' subprocess did not return a result "
                f"(exit code {proc.exitcode}) -- see the traceback printed "
                "above this one for the underlying error."
            ) from exc
        proc.join(timeout=30)
    return results["matplotlib"], results["plotly"]


# ---------------------------------------------------------------------------
# Part B -- AppTest multi-session simulation (post-Q1 implementation only)
# ---------------------------------------------------------------------------


_APPTEST_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state, AppState
from ui_web.live_tick import live_tick_fragment

init_session_state()
st.session_state["app_state"] = AppState.RUNNING
live_tick_fragment()
"""


def run_apptest_multi_session(n_sessions: int, n_ticks_per_session: int = 5) -> Dict[str, float]:
    """Simulate N concurrent RUNNING sessions against the real, current app.

    Each simulated session is a fresh ``streamlit.testing.v1.AppTest``
    instance; per session, ``.run()`` is called ``n_ticks_per_session``
    times in a row -- each call re-executes the whole script (including
    ``live_tick_fragment()``), which is what happens on a real 300ms
    fragment tick during a RUNNING session. All N sessions run
    sequentially, in this one process -- deliberately, since a real
    Streamlit deployment is itself one single, GIL-bound process serving
    every concurrently connected browser session (see
    ``web-opt-changes.md``'s "Context" section), so sequential
    accumulation of per-session cost in one process is a reasonable
    proxy for "how much total server CPU/memory N concurrent sessions
    cost", not an artifact of this benchmark's own implementation.

    Only the current, post-Q1 (Plotly) implementation can be exercised
    this way -- the pre-Q1 matplotlib fragment/chart code no longer
    exists anywhere in this checkout to import (see module docstring).

    :param n_sessions: Number of independent simulated sessions.
    :type n_sessions: int
    :param n_ticks_per_session: Number of fragment ticks to run per
        session (first tick includes one-time session construction
        cost -- generator/detector/ring buffers/chart Figure -- later
        ticks are the repeated steady-state per-redraw cost).
    :type n_ticks_per_session: int
    :returns: Aggregate timing/CPU/memory results across all sessions.
    :rtype: dict[str, float]
    """
    from streamlit.testing.v1 import AppTest

    rusage_before = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    for _ in range(n_sessions):
        at = AppTest.from_string(_APPTEST_SCRIPT)
        for _ in range(n_ticks_per_session):
            at.run()
    wall = time.perf_counter() - t0
    rusage_after = resource.getrusage(resource.RUSAGE_SELF)

    total_ticks = n_sessions * n_ticks_per_session
    return {
        "n_sessions": float(n_sessions),
        "n_ticks_per_session": float(n_ticks_per_session),
        "total_wall_seconds": wall,
        "avg_ms_per_session": (wall / n_sessions) * 1000.0,
        "avg_ms_per_tick": (wall / total_ticks) * 1000.0,
        "cpu_pct": _cpu_pct(rusage_before, rusage_after, wall),
        "peak_rss_kb": _maxrss_kb(rusage_after.ru_maxrss),
    }


# ---------------------------------------------------------------------------
# Report formatting / entry point
# ---------------------------------------------------------------------------


def _print_part_a_report(mpl_result: Dict[str, float], plotly_result: Dict[str, float]) -> None:
    """Print a human-readable Part A comparison table.

    :param mpl_result: Result dict from :func:`run_matplotlib_benchmark`.
    :type mpl_result: dict[str, float]
    :param plotly_result: Result dict from :func:`run_plotly_benchmark`.
    :type plotly_result: dict[str, float]
    :returns: None
    :rtype: None
    """
    print("\n=== Part A: per-redraw-cycle micro-benchmark ===")
    print(f"{'metric':<28} {'matplotlib (pre-Q1)':>22} {'Plotly (post-Q1)':>20}")
    rows = [
        ("cycles", "n_cycles", "{:.0f}"),
        ("total wall time (s)", "total_wall_seconds", "{:.3f}"),
        ("avg time / cycle (ms)", "avg_ms_per_cycle", "{:.3f}"),
        ("CPU % (of 1 core)", "cpu_pct", "{:.1f}"),
        ("peak RSS (KB)", "peak_rss_kb", "{:.0f}"),
        ("last payload size (bytes)", "last_payload_bytes", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        mpl_val = fmt.format(mpl_result[key])
        plotly_val = fmt.format(plotly_result[key])
        print(f"{label:<28} {mpl_val:>22} {plotly_val:>20}")

    speedup = mpl_result["avg_ms_per_cycle"] / plotly_result["avg_ms_per_cycle"]
    payload_ratio = mpl_result["last_payload_bytes"] / max(plotly_result["last_payload_bytes"], 1.0)
    print(f"\n-> Plotly redraw+serialize is ~{speedup:.1f}x faster per cycle than the pre-Q1 matplotlib path.")
    print(f"-> Pre-Q1 PNG payload is ~{payload_ratio:.1f}x the size of the post-Q1 Plotly JSON payload (last cycle).")


def _print_part_b_report(results_by_n: Dict[int, Dict[str, float]]) -> None:
    """Print a human-readable Part B (AppTest multi-session) report.

    :param results_by_n: Mapping of ``n_sessions`` to the result dict
        returned by :func:`run_apptest_multi_session`.
    :type results_by_n: dict[int, dict[str, float]]
    :returns: None
    :rtype: None
    """
    print("\n=== Part B: AppTest multi-session simulation (post-Q1 implementation only) ===")
    print(f"{'N sessions':<12} {'total wall (s)':>16} {'ms/session':>12} {'CPU %':>8} {'peak RSS (KB)':>15}")
    for n, result in sorted(results_by_n.items()):
        print(
            f"{n:<12} {result['total_wall_seconds']:>16.2f} "
            f"{result['avg_ms_per_session']:>12.1f} {result['cpu_pct']:>8.1f} "
            f"{result['peak_rss_kb']:>15.0f}"
        )


def main() -> None:
    """Run Part A and Part B and print a combined report.

    :returns: None
    :rtype: None
    """
    print(f"Running Part A micro-benchmark ({N_CYCLES} cycles each, capacity={DEFAULT_CAPACITY})...")
    mpl_result, plotly_result = run_part_a()
    _print_part_a_report(mpl_result, plotly_result)

    print("\nRunning Part B AppTest multi-session simulation (N=10, N=50)...")
    results_by_n = {}
    for n in (10, 50):
        results_by_n[n] = run_apptest_multi_session(n, n_ticks_per_session=5)
    _print_part_b_report(results_by_n)


if __name__ == "__main__":
    main()
