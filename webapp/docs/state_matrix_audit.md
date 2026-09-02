# G2 -- Full State-Machine Audit

**Story:** As a developer, I need every control's `disabled=` argument
checked against the `controls_enabled()` matrix across all four
`AppState` values, so no control is left enabled/disabled incorrectly at
any stage.

**Source of truth:** `state/app_state.py`, function `controls_enabled(state)`.

**Automated backing test:** `test/test_state_matrix_audit.py` (10 tests:
7 controls x cross-reference across all 4 states, a dedicated
"no-disabled-argument-at-all" check for the GMM training group across
all 4 states, and the two manual walkthroughs below, executed as real
`AppTest` button clicks against a single persistent session).

This document is the human-readable, cell-by-cell counterpart to that
test file. Both were produced by independently reading every widget
call site listed below (not by copying the matrix and assuming it
holds).

## 1. Cell-by-cell cross-reference

For every control, the table below states:

* **Source expression** -- the exact `disabled=` argument (or its
  absence) at the widget's call site, and the file/line it lives in.
* For each `AppState`, **expected** (from the story's matrix) vs.
  **actual** (what the source expression evaluates to for that state,
  confirmed by reading `controls_enabled()`'s branches directly and, for
  the automated cross-check, by asserting the real rendered widget's
  `.disabled` property via `AppTest`).
* **Match** -- yes/no.

### 1.1 Template / Mean / Std / Noise (`config_enabled`)

*Source:* `ui_web/sidebar_config.py`, `render_template_and_stream_controls()`
(lines ~260-297). All four widgets (`st.selectbox(key="cfg_template")`,
`st.number_input(key="cfg_mean")`, `key="cfg_std"`, `key="cfg_noise"`)
share one `disabled=config_disabled` argument, where
`config_disabled = not controls_enabled(current_app_state)["config_enabled"]`
-- i.e. this reads `controls_enabled()` directly, so it cannot drift
from the matrix by construction.

| AppState | Matrix expects | `config_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | enabled | `True` | `False` (not disabled) | yes |
| RUNNING | disabled | `False` | `True` (disabled) | yes |
| PAUSED | enabled | `True` | `False` (not disabled) | yes |
| SIMULATION_RUNNING | enabled | `True` | `False` (not disabled) | yes |

### 1.2 Start button (`start_enabled`)

*Source:* `ui_web/sidebar_config.py`, `render_run_controls()` (~line 516):
`st.button("▶ Start", key="btn_start", disabled=not enabled["start_enabled"], ...)`
where `enabled = controls_enabled(app_state)`.

| AppState | Matrix expects | `start_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | enabled | `True` | `False` | yes |
| RUNNING | disabled | `False` | `True` | yes |
| PAUSED | enabled | `True` | `False` | yes |
| SIMULATION_RUNNING | disabled | `False` | `True` | yes |

### 1.3 Stop button (`stop_enabled`)

*Source:* `ui_web/sidebar_config.py`, `render_run_controls()` (~line 524):
`st.button("■ Stop", key="btn_stop", disabled=not enabled["stop_enabled"], ...)`

| AppState | Matrix expects | `stop_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | disabled | `False` | `True` | yes |
| RUNNING | enabled | `True` | `False` | yes |
| PAUSED | disabled | `False` | `True` | yes |
| SIMULATION_RUNNING | disabled | `False` | `True` | yes |

### 1.4 Load & Play button (`sim_play_enabled`)

*Source:* `ui_web/sidebar_config.py`, `render_simulation_controls()`
(~line 708): `st.button("▶ Load & Play", key="btn_sim_play",
disabled=not enabled["sim_play_enabled"], ...)`

| AppState | Matrix expects | `sim_play_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | enabled | `True` | `False` | yes |
| RUNNING | enabled | `True` | `False` | yes |
| PAUSED | enabled | `True` | `False` | yes |
| SIMULATION_RUNNING | disabled | `False` | `True` | yes |

### 1.5 Stop Sim button (`sim_stop_enabled`)

*Source:* `ui_web/sidebar_config.py`, `render_simulation_controls()`
(~line 716): `st.button("■ Stop Sim", key="btn_sim_stop",
disabled=not enabled["sim_stop_enabled"], ...)`

| AppState | Matrix expects | `sim_stop_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | disabled | `False` | `True` | yes |
| RUNNING | disabled | `False` | `True` | yes |
| PAUSED | disabled | `False` | `True` | yes |
| SIMULATION_RUNNING | enabled | `True` | `False` | yes |

### 1.6 Reset Session button (`reset_enabled`)

*Source:* `ui_web/sidebar_config.py`, `render_run_controls()` (~line 531):
`st.button("↺ Reset Session", key="btn_reset",
disabled=not enabled["reset_enabled"], ...)`. `controls_enabled()`
always sets `reset_enabled: True` (the `always_on` dict, merged into
every branch's matrix before returning), so `not True` = `False` in
every state -- the button is unconditionally enabled, expressed via a
correct (always-`False`-evaluating) `disabled=` argument rather than by
omitting the argument.

| AppState | Matrix expects | `reset_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | enabled | `True` | `False` | yes |
| RUNNING | enabled | `True` | `False` | yes |
| PAUSED | enabled | `True` | `False` | yes |
| SIMULATION_RUNNING | enabled | `True` | `False` | yes |

### 1.7 GMM Training window (Start/End spinboxes) + Retrain Model button (`retrain_enabled`)

*Source:* `ui_web/live_tick.py`, `render_gmm_training_group()`
(~lines 678-701):

```python
st.number_input("Start", key="cfg_retrain_start", min_value=0, max_value=start_max, step=1)
st.number_input("End", key="cfg_retrain_end", min_value=0, max_value=end_max, step=1,
                 on_change=_on_end_spinbox_manual_edit)
...
st.button("⟳ Retrain Model", key="btn_retrain", on_click=_on_retrain_clicked,
          use_container_width=True)
```

None of these three widget calls has a `disabled=` argument at all --
confirmed by reading every argument passed to each call site above.
Since `controls_enabled()`'s `always_on` dict sets `retrain_enabled:
True` in every branch, the matrix requires these widgets to be enabled
in all four states; the absence of a `disabled=` argument means
Streamlit's own default (`disabled=False`) applies unconditionally,
which is exactly what "always enabled" requires -- there is no
possible drift here because there is no expression to drift.
`test_retrain_widgets_have_no_disabled_argument_at_all` in
`test/test_state_matrix_audit.py` asserts `.disabled is False`
(not just falsy) for all three widgets across all four states as the
automated, durable version of this observation.

| AppState | Matrix expects | `retrain_enabled` | Actual `disabled=` | Match |
|---|---|---|---|---|
| IDLE | enabled | `True` | not present -> `False` | yes |
| RUNNING | enabled | `True` | not present -> `False` | yes |
| PAUSED | enabled | `True` | not present -> `False` | yes |
| SIMULATION_RUNNING | enabled | `True` | not present -> `False` | yes |

## 2. Result: zero mismatches found

All 7 controls/control-groups x 4 `AppState` values = 28 cells were
audited above. **Zero mismatches were found.** No source code changes
were required as part of this story. This was independently confirmed
by running `test/test_state_matrix_audit.py`'s
`test_all_controls_disabled_state_matches_matrix_for_every_app_state`
(parametrized over all 4 `AppState` values, diffing all 12 widget keys'
real `.disabled` property against `controls_enabled()` in the same
render) and `test_retrain_widgets_have_no_disabled_argument_at_all` --
all pass with no mismatches reported.

## 3. Manual walkthrough #1: `IDLE -> RUNNING -> PAUSED -> RUNNING -> IDLE`

Executed as `test_walkthrough_idle_running_paused_running_idle_matches_matrix_at_every_step`
in `test/test_state_matrix_audit.py`, driving a single, persistent
`AppTest` session with real button `.click().run()` calls (not
re-seeded scripts per step), and asserting the full seven-control
snapshot against `controls_enabled()` after every transition. Observed
behavior:

1. **Start at IDLE.** Template/Mean/Std/Noise, Start, and Load & Play
   are enabled; Stop and Stop Sim are disabled; Reset Session and the
   GMM training window/Retrain button are enabled. Matches the IDLE
   column exactly.
2. **Click Start -> RUNNING.** `app_state` observed to transition to
   `AppState.RUNNING`. On the next render: Template/Mean/Std/Noise and
   Start become disabled; Stop becomes enabled; Load & Play stays
   enabled (per the matrix, Load & Play is enabled while RUNNING, only
   disabled during `SIMULATION_RUNNING`); Stop Sim stays disabled;
   Reset Session and the GMM training controls remain enabled
   throughout. Matches the RUNNING column exactly.
3. **Click Stop -> PAUSED.** `app_state` observed to transition to
   `AppState.PAUSED`. Template/Mean/Std/Noise and Start become enabled
   again; Stop becomes disabled again; Load & Play stays enabled; Stop
   Sim stays disabled; Reset Session and GMM training controls remain
   enabled. Matches the PAUSED column exactly -- and confirms PAUSED and
   IDLE produce an identical control snapshot (the only difference
   between the two states is *not* control-enablement but which
   transition Start performs internally, which is out of this story's
   `disabled=` scope).
4. **Click Start again -> RUNNING.** Same disabled/enabled snapshot as
   step 2 was re-observed, confirming the RUNNING state's control
   pattern is reproduced identically on a second entry into it (no
   stale state left over from the PAUSED excursion).
5. **Click Reset Session -> IDLE.** `app_state` observed to transition
   back to `AppState.IDLE`; the resulting snapshot exactly reproduced
   step 1's IDLE snapshot. (Reset Session is used here, rather than
   Stop, to additionally exercise the always-enabled Reset control's
   own state-independent availability at each visited state, per
   `controls_enabled()`'s `always_on` dict -- Stop -> PAUSED already
   exercised the plain RUNNING -> PAUSED leg in step 3.)

No control was observed to be incorrectly clickable (i.e. enabled when
the matrix says disabled) or incorrectly greyed out (disabled when the
matrix says enabled) at any of the five observed states in this
walkthrough.

## 4. Manual walkthrough #2: `IDLE -> SIMULATION_RUNNING -> IDLE`

Executed as `test_walkthrough_idle_simulation_running_idle_matches_matrix_at_every_step`
in `test/test_state_matrix_audit.py`, same real-click methodology.
Observed behavior:

1. **Start at IDLE.** Same snapshot as walkthrough #1's step 1.
2. **Click Load & Play -> SIMULATION_RUNNING.** `app_state` observed to
   transition to `AppState.SIMULATION_RUNNING`. On the next render:
   Template/Mean/Std/Noise remain **enabled** (per the matrix, config
   editing is allowed during a running simulation, unlike during a live
   RUNNING stream); Start becomes disabled; Load & Play becomes
   disabled; Stop remains disabled; Stop Sim becomes enabled; Reset
   Session and the GMM training controls remain enabled. Matches the
   SIMULATION_RUNNING column exactly -- this is the one state whose
   `config_enabled` value (`True`) differs from RUNNING's (`False`)
   while `start_enabled`/`stop_enabled` are both `False` in either
   state, so it was checked with particular care since it is the
   easiest cell to accidentally conflate with RUNNING's row.
3. **Click Stop Sim -> IDLE.** `app_state` observed to transition back
   to `AppState.IDLE`; the resulting snapshot exactly reproduced step
   1's IDLE snapshot.

No control was observed to be incorrectly clickable or incorrectly
greyed out at any of the three observed states in this walkthrough.

## 5. Conclusion

The audit is complete for all 7 controls/control-groups across all 4
`AppState` values (28 cells), with **zero mismatches** found between
each control's actual rendered `.disabled` behavior and the
`controls_enabled()` matrix. No source fixes were necessary. Both
required manual walkthroughs were executed for real (via `AppTest`
button clicks against a persistent session, not reasoned about on
paper) and produced exactly the enabled/disabled control states the
matrix specifies at every step, with no control incorrectly clickable
or incorrectly greyed out.
