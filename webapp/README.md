# Predictive Maintenance Demonstrator — Web App

A browser-based port of the PyQt5 **Predictive Maintenance Demonstrator**
built for the AMBITION Industry Fair 2026 / DIGIMACH workshop. It shows a
Gaussian-Mixture-Model (GMM) anomaly detector running live over a simulated
sensor stream — with a live chart, an operator logbook (classify findings as
true/false positives, add notes, export to CSV), and a stats dashboard — all
served as a normal web page via [Streamlit](https://streamlit.io/) so
attendees can try it from any browser, without installing Python, PyQt5, or
any dependency locally.

The original desktop app (`main.py`, `core/`, `ui/`, `utils/` at the
repository root) is untouched and keeps working exactly as before; this
`webapp/` directory is a fully self-contained sibling application that reuses
the same detection algorithms and constants.

## What's inside

- **Sensor template selector** — Temperature / Vibration / Pressure presets
  (mean, std-dev, noise), independently editable once selected.
- **Live sensor stream** — Start/Stop/Reset controls drive a simulated data
  generator with configurable anomaly-injection probability.
- **GMM anomaly detector** — trains automatically once 100 samples have been
  collected, overlays its component bands/decision boundaries on the chart,
  and can be retrained over a chosen sample window on demand.
- **Simulation replay** — pre-generated, seeded CSV scenarios (one per
  sensor template) that replay at 1×–10× speed for repeatable, scripted
  demos.
- **Operator logbook** — every detected anomaly is logged with an editable
  Unclassified / TP / FP classification and a free-text note, with
  filtering, note search, CSV export, and a clear-log action.
- **Dashboard** — a popover summarizing total samples/anomalies, TP/FP/
  unclassified counts, average confidence, and a pie chart.
- **Dracula-derived dark theme** — matches the look of the original desktop
  app's Qt stylesheet.

## Local run

Requirements: Python 3.9+ (the app is developed/tested against Python 3.12
locally and Python 3.11 in the Docker image; any modern CPython 3.9+ should
work since no version-specific language features are used).

From inside this directory (`webapp/`):

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL — open **http://localhost:8501** in your
browser. On first boot the app auto-generates the three simulation scenario
CSVs into `assets/simulations/` (no manual setup, no volume mount, no
additional command needed); this happens the same way in Docker (see below).

If you use [`uv`](https://docs.astral.sh/uv/) instead of a bare `venv`, the
equivalent is:

```bash
uv venv
uv pip install -r requirements.txt
uv run streamlit run app.py
```

To stop the app, press `Ctrl+C` in the terminal running `streamlit run`.

## Docker

A `Dockerfile` and `docker-compose.yml` are provided so the app can be
deployed with a single command and no local Python install at all.

### Option 1 — `docker build` / `docker run`

Run these commands from inside this `webapp/` directory:

```bash
docker build -t pm-demonstrator-web .
docker run -p 8501:8501 pm-demonstrator-web
```

Then open **http://localhost:8501**. The container generates the simulation
CSVs into its own filesystem on first boot — no volume mount is required for
the app to be fully functional (note that any logbook data or generated
CSVs are ephemeral and will be lost if the container is removed).

### Option 2 — `docker compose` (recommended for an unattended booth laptop)

Also run from inside this `webapp/` directory:

```bash
docker compose up -d --build
```

(The standalone, older `docker-compose` v1 binary works identically with
this same `docker-compose.yml` file, if that's what's installed instead of
the newer `docker compose` v2 plugin — i.e. `docker-compose up -d --build`.)

This builds the image (if needed) and starts the container in the
background, mapped to the same `8501` port as Option 1. The compose service
is configured with `restart: unless-stopped`, so if the Streamlit process
inside the container crashes mid-demo, Docker automatically restarts it
without anyone needing to notice or intervene — useful for an unattended
fair/booth laptop running for hours unsupervised.

To stop it:

```bash
docker compose down
```

## Known fidelity gaps

This is a faithful port, but Streamlit's execution model (server-side
"reruns" of the whole script rather than a persistent, hand-rolled event
loop) is fundamentally different from the original PyQt5 desktop app's Qt
event loop and signal/slot wiring. The following differences are
**expected, deliberate trade-offs — not bugs**:

1. **300ms update cadence instead of the original's 100ms/50ms timers.**
   The original desktop app used two separate Qt timers: one every 100ms to
   generate a new sensor sample, and one every 50ms to redraw the chart.
   Streamlit can't reproduce that exact dual-timer precision, so this web
   app instead re-runs its live-update logic on a single ~300ms cycle
   (raised from an initial 150ms, which roughly halves the per-session
   redraw/bandwidth cost). In practice this is imperceptible to someone
   watching a demo — the chart still animates smoothly — but it is
   technically a slightly coarser, single-cadence approximation of the
   original's finer-grained timing.

2. **Simulation replay looks stepped/bursty above ~5× speed.** Scripted
   scenario replay ("Load & Play") is smooth up to about 5× speed. At
   higher speeds (up to the maximum of 10×), multiple rows of the scenario
   can become due within a single ~300ms update cycle and are drawn to the
   chart together in a burst rather than one at a time, which can look
   visibly "steppy" rather than fluid. No data is lost or skipped — every
   row is still processed in the correct order — but the visual smoothness
   degrades at high speed. This is an accepted, deliberate trade-off rather
   than something to fix.

3. **No colored row backgrounds in the logbook table.** In the original
   desktop app, classified logbook rows could be tinted with a background
   color. Streamlit's editable table widget (`st.data_editor`) does not
   support tinting the background of individual editable rows, so instead
   each row's classification is conveyed with a plain-text emoji label in
   the Classification column: "⚪ Unclassified", "🟢 TP" (true positive), or
   "🟠 FP" (false positive). The information is the same; only the visual
   presentation differs.

4. **The dashboard is a popover, not a free-floating window.** In the
   original desktop app, the stats dashboard was a separate, non-modal
   window that could stay open and visible on screen alongside the main
   window at all times. In a browser there is no equivalent concept of a
   second, independently-positioned always-on-top window tied to the same
   page, so the web version instead uses a "Dashboard" button that opens a
   popover anchored to that button. It shows the same statistics and pie
   chart, but it closes if you click elsewhere on the page, and you must
   re-open it (or use the popover's own refresh) to see updated numbers
   rather than having it passively update in the background at all times.

5. **No keyboard shortcuts.** The original desktop app wired up menu items
   and keyboard shortcuts (e.g. for Start/Stop) via Qt's menu system. This
   web port has no keyboard shortcuts at all — every action (Start, Stop,
   Reset, Load & Play, Retrain, etc.) must be triggered by clicking its
   corresponding on-page button.

6. **Each browser tab is an independent session (no shared live state).**
   The original desktop app was a single process with one shared in-memory
   state, so there was only ever "one instance" running. This web app's
   state (the current run, ring buffers, logbook, etc.) lives in a
   Streamlit *session* tied to one browser tab. If you open the same URL in
   a second tab or a second browser window, you get a brand-new, independent
   session — not a live view of the same running demo — starting back at
   the idle state with no data. For a demo, always drive the app from a
   single browser tab.

7. **The anomaly logbook retains only the most recent 500 anomalies.** The
   original desktop app's logbook could grow without bound for the lifetime
   of the process. To keep memory usage and per-append cost bounded during a
   long unattended booth/multi-user session, this web app caps
   `state["logbook_df"]` at 500 rows (`MAX_LOGBOOK_ROWS` in
   `state/app_state.py`); once that many anomalies have been logged, each
   new anomaly evicts the single oldest logged row (FIFO by `row_id`).
   `row_id`/`next_log_id` numbering itself keeps incrementing forever and is
   never reused, so classification/notes edits and CSV exports made before
   an eviction remain internally consistent — only the oldest rows silently
   drop out of the table, filters, export, and dashboard stats once the cap
   is reached.

## Performance notes

Story P2 raised the live-tick fragment's poll cadence from an initial
150ms to the current **300ms** (`LIVE_TICK_INTERVAL` in
`ui_web/live_tick.py`; see "Known fidelity gaps" item 1 above), roughly
halving per-session redraw/bandwidth cost. Story Q1 then replaced the
main chart's server-rendered matplotlib `Figure` (rasterized to a PNG
and shipped to the browser via `st.pyplot` on every redraw) with a
client-rendered Plotly `go.Figure` (shipped down as a small JSON spec
and redrawn entirely in the browser by plotly.js via `st.plotly_chart`),
so the redraw/rasterization cost that used to run on the shared server
process for every connected session now runs once, client-side, per
viewer's own browser. This section records the actual measurements
taken to verify that the Plotly migration reduces server-side cost,
rather than just asserting it — see `test/perf_chart_benchmark.py` for
the full benchmark script and its methodology docstring.

**Methodology and caveats** (read before over-trusting these numbers):
this repository's `webapp/` directory has no prior git history (it was
added as a single already-complete commit), so there is no old commit
of the pre-Q1 matplotlib `chart.py` to check out and benchmark directly.
`test/perf_chart_benchmark.py` instead (a) rebuilds a small,
self-contained, benchmark-only matplotlib chart mirroring the pre-Q1
design (two-panel layout, persistent line/scatter, `ax.cla()` +
`ax.bar()` confidence panel every redraw — closely mirroring the
original desktop app's `ui/visualization_panel.py`, which the webapp's
chart was ported from) and runs it through 300 update+rasterize cycles
against a ramping 1→300-point data window (roughly a 90-second RUNNING
session at the 300ms cadence), timed in an isolated subprocess; (b) runs
the exact same cycle count/data volume against the real, current,
production `ChartState` (Plotly), serializing via `plotly.io.to_json()`
— the same call `st.plotly_chart` makes internally — also in an isolated
subprocess; and (c) separately simulates N=10 and N=50 concurrent
`RUNNING`-state sessions against the real, current app using
`streamlit.testing.v1.AppTest` instances run sequentially in one
process (a reasonable proxy for a real deployment, since Streamlit
itself is a single, GIL-bound process serving every session — see the
"Context" section of the optimization backlog this work came from).
Because the pre-Q1 matplotlib chart/fragment code no longer exists
anywhere in this checkout to import, the AppTest simulation can only
exercise the current (post-Q1) implementation directly; the "before"
N=10/N=50 figures below are an **extrapolation** (not a direct
measurement) that applies the micro-benchmark's measured per-cycle
matplotlib-vs-Plotly delta to every `RUNNING`-state tick, clearly
labeled as such. All numbers are single-machine, single-process
measurements — an approximation of server-side CPU/memory cost, not a
substitute for a real multi-user load test with actual concurrent
websocket connections.

### Measured: per-redraw-cycle cost (Part A)

| metric | matplotlib (pre-Q1 reconstruction) | Plotly (post-Q1, real code) |
| --- | --- | --- |
| avg time / redraw cycle | ~66 ms | ~1.4 ms |
| CPU (of 1 core) during the loop | ~99–100% | ~99–100% |
| serialized payload size (last cycle) | ~76 KB (PNG) | ~29 KB (JSON) |

**~47x faster per redraw cycle, ~2.7x smaller payload shipped to the
browser.** The CPU% figures are both ~100% because each benchmark is a
tight loop saturating one core throughout — the meaningful comparison is
the wall-clock time per cycle (i.e. how many CPU-milliseconds the server
spends per redraw), not the percentage.

### Measured: N concurrent sessions, current (post-Q1) implementation (Part B)

| N sessions | total wall time (5 ticks/session) | avg time / session | CPU (of 1 core) | peak RSS |
| --- | --- | --- | --- | --- |
| 10 | ~1.7 s | ~166 ms | ~99% | ~237 MB |
| 50 | ~3.9 s | ~78 ms | ~99% | ~245 MB |

(Peak RSS is dominated by Streamlit/pandas/scikit-learn/numpy's own
import footprint, not by per-session state — the ~8 MB growth from
N=10 to N=50 sessions confirms each additional session's ring
buffers/chart/logbook are cheap relative to that fixed baseline.)

### Extrapolated: N concurrent sessions, pre-Q1 matplotlib implementation

Applying Part A's measured per-cycle delta (~65 ms extra per redraw,
matplotlib vs Plotly) to every `RUNNING`-state tick in Part B's N=10/
N=50 simulation (**estimated, not measured** — see methodology above):

| N sessions | estimated total wall time | approximate slowdown vs. Plotly |
| --- | --- | --- |
| 10 | ~4.9 s | ~3x |
| 50 | ~20.2 s | ~5x |

### What this means for deployment sizing (approximate)

The measured numbers above indicate the Plotly migration (Q1) cut the
server-side cost of each `RUNNING`-state redraw by roughly an order of
magnitude (~47x per cycle) and roughly halved the bytes shipped per
redraw (~2.7x smaller payload) — this is the change that removes the
"server rendering is the multi-user scaling ceiling" bottleneck
identified in the optimization backlog, not just a tuning adjustment.
Combined with story P2's cadence change (300ms, already reflected in
the numbers above), a single vCPU running this app should comfortably
sustain dozens of concurrently `RUNNING` sessions before chart redraw
becomes the bottleneck again (Part B's own measurements show N=50
sessions completing 5 ticks each in well under 4 seconds of total
CPU-bound wall time on a single core). Actual Azure sizing should still
budget headroom above this floor for other per-request costs (widget
rendering, GMM training/scoring, logbook growth up to `MAX_LOGBOOK_ROWS`)
and for real network/browser-side variance this single-process
benchmark does not capture — treat "dozens of sessions per vCPU" as a
starting point for load-testing a real deployment, not a hard guarantee.

To reproduce these numbers yourself: `.venv/bin/python
test/perf_chart_benchmark.py` (not part of the automated `pytest test/`
suite — see that script's module docstring for why, and its full
methodology). Note it temporarily needs `matplotlib` importable to run
the pre-Q1 comparison side; since `matplotlib` may no longer be a
declared dependency in `requirements.txt` (story Q2 removes it once its
own dashboard-pie-chart migration lands), you may need to
`uv pip install matplotlib` into the local `.venv` first, then uninstall
it again afterwards if you want to keep the environment matching
`requirements.txt` exactly.

## Troubleshooting

- **Port 8501 already in use** — either stop whatever else is using it, or
  pass a different port, e.g. `streamlit run app.py --server.port=8502` (or
  add `--server.port=8502` to the `ENTRYPOINT`/override the port mapping,
  e.g. `docker run -p 8502:8501 pm-demonstrator-web`, when running via
  Docker).
- **Simulation scenarios missing/stale** — they are regenerated
  automatically on every app start if not already present; delete
  `assets/simulations/*.csv` and restart the app to force regeneration.
- **Blank/broken theme** — confirm `.streamlit/config.toml` is present in
  this directory; Streamlit only picks up theme settings from a
  `.streamlit/config.toml` relative to the directory `streamlit run` is
  launched from.
