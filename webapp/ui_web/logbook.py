"""Anomaly logbook table: editable ``st.data_editor`` merged back by ``row_id``.

Story E1 (``ui_web.live_tick._append_anomaly_to_logbook``) appends one row
per flagged anomaly to ``state["logbook_df"]``, each with a unique,
monotonically increasing ``row_id`` (see ``state.app_state.LOGBOOK_COLUMNS``).
This module (story E2) renders that DataFrame as an editable
``st.data_editor`` -- Classification as a ``SelectboxColumn`` (emoji-coded
``⚪ Unclassified`` / ``🟢 TP`` / ``🟠 FP``, since Streamlit's data editor
cannot tint an editable row's background the way the desktop app's
``LogbookPanel`` did) and Note/comment as a free-text ``TextColumn``, with
Time/Value/Conf% shown but not editable -- refreshed every second via
``@st.fragment(run_every="1s")``.

**The row_id merge-back mechanism (the story's core technical requirement):**
:func:`_build_display_df` re-indexes the DataFrame handed to
``st.data_editor`` on ``row_id`` (hidden from the rendered grid via
``hide_index=True``, but preserved as the DataFrame's actual pandas index).
``st.data_editor`` applies any cell edits in place to a *copy* of that same
DataFrame -- same row order, same index -- and returns the edited copy
directly (see ``streamlit.elements.widgets.data_editor._apply_cell_edits``,
which indexes by *row position*, not label, but never reorders rows itself).
That means the *label* at each returned row's index is still reliably its
originating ``row_id``, even though Streamlit's internal ``edited_rows``
diff dict is keyed by position. :func:`merge_editor_edits_into_logbook`
then merges the edited Classification/Note values back into
``state["logbook_df"]`` purely by ``row_id`` label lookup
(``DataFrame.loc[row_id, ...]``), never by positional index or ``.iloc``.

This makes the merge robust to a *future* filtered/reordered view (story
E3): whatever subset/order of rows a filter passes into
:func:`_build_display_df` (or directly into ``st.data_editor``), as long as
that view's index is still ``row_id``, :func:`merge_editor_edits_into_logbook`
merges each edited row back onto the correct underlying row regardless of
its visual position -- the two functions are deliberately kept decoupled
from :func:`render_logbook_table` so a future filter only needs to build a
different display DataFrame, not touch the merge logic at all.

**Story E3 (classification filter + notes search + stats footer):**
:func:`_compute_filter_mask` computes a boolean row mask over the FULL
``logbook_df`` (classification filter AND notes search, combined with
``&``) which :func:`render_logbook_table` applies via
``logbook_df[mask]`` *only* when building the display DataFrame handed to
:func:`_build_display_df` -- ``state["logbook_df"]`` itself is never
filtered/mutated, and ``merge_editor_edits_into_logbook`` is always given
the full, unfiltered ``logbook_df`` as its first argument so edits made
while a filter is active still merge back onto the complete logbook. The
stats footer (:func:`_compute_footer_counts`) is likewise always computed
from the full, post-merge ``logbook_df`` so it never reflects a filtered
subset and never goes stale on the same rerun as an edit.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, MutableMapping, Optional, Sequence

import pandas as pd
import streamlit as st

from state.app_state import LOGBOOK_COLUMNS, _empty_logbook_df

logger = logging.getLogger(__name__)

#: Exact Classification dropdown options, emoji-coded per the story's
#: acceptance criteria (Streamlit's data editor can't tint an editable
#: row's background the way the desktop app's ``LogbookPanel`` colored
#: TP/FP rows, so the emoji carries that signal instead).
CLASSIFICATION_OPTIONS: Sequence[str] = ("⚪ Unclassified", "🟢 Real Anomaly", "🟠 No Anomaly")

#: Columns shown in the editor grid, in display order. ``row_id`` is
#: deliberately excluded here (it becomes the DataFrame's hidden index --
#: see :func:`_build_display_df`) and ``timestamp``/``is_injected`` are
#: internal bookkeeping columns not meant for operator review.
DISPLAY_COLUMNS: Sequence[str] = ("time_str", "value", "conf_pct", "classification", "comment")

#: Columns that are editable by the operator; every other displayed
#: column is rendered read-only (``disabled=True``).
_EDITABLE_COLUMNS: Sequence[str] = ("classification", "comment")

#: Fixed ``st.data_editor`` widget key. A stable key (rather than
#: ``None``) is required for Streamlit to preserve in-progress edits
#: across the fragment's 1s auto-reruns instead of resetting the grid
#: every tick.
_EDITOR_KEY = "logbook_data_editor"

#: Plain-text classification filter options exposed in the E3 filter
#: selectbox. ``"All"`` bypasses classification filtering entirely; the
#: other three are deliberately plain (no emoji) so the filter UI stays
#: readable/greppable -- :data:`_CLASSIFICATION_FILTER_VALUES` maps each
#: one to the exact emoji-coded value actually stored in
#: ``logbook_df["classification"]`` (see :data:`CLASSIFICATION_OPTIONS`).
CLASSIFICATION_FILTER_OPTIONS: Sequence[str] = ("All", "TP", "FP", "Unclassified")

#: Maps each non-``"All"`` :data:`CLASSIFICATION_FILTER_OPTIONS` entry to
#: the exact emoji-coded value it must equal in
#: ``logbook_df["classification"]``. Keeping this mapping separate (rather
#: than duplicating the emoji strings inline in the filter logic) means
#: :data:`CLASSIFICATION_OPTIONS` remains the single source of truth for
#: the emoji-coding.
_CLASSIFICATION_FILTER_VALUES: Dict[str, str] = {
    "Unclassified": CLASSIFICATION_OPTIONS[0],
    "TP": CLASSIFICATION_OPTIONS[1],
    "FP": CLASSIFICATION_OPTIONS[2],
}

#: Fixed widget keys for the E3 filter/search controls, following the
#: same "stable key so the fragment's 1s auto-reruns don't reset operator
#: input" rationale as :data:`_EDITOR_KEY`.
_FILTER_KEY = "logbook_classification_filter"
_SEARCH_KEY = "logbook_notes_search"

#: Exact column set/order required by story E4's "Export CSV" acceptance
#: criterion. Deliberately differs from :data:`LOGBOOK_COLUMNS`: ``row_id``
#: is included (unlike :data:`DISPLAY_COLUMNS`, which hides it as the
#: editor's index), ``time_str`` is dropped entirely (the export keeps the
#: raw ``timestamp`` instead, not the pre-formatted display string), and
#: ``conf_pct`` is renamed to ``confidence``.
EXPORT_COLUMNS: Sequence[str] = (
    "row_id",
    "timestamp",
    "value",
    "confidence",
    "classification",
    "comment",
    "is_injected",
)

#: Maps each emoji-coded :data:`CLASSIFICATION_OPTIONS` value to the plain
#: text required in the CSV export (story E4: "classification values have
#: emoji prefixes stripped to plain Unclassified/TP/FP"). An explicit
#: mapping (rather than positional string-splitting on the first space) is
#: used deliberately -- it is robust regardless of how the emoji is
#: encoded/rendered and is a single source of truth alongside
#: :data:`CLASSIFICATION_OPTIONS`.
_CLASSIFICATION_PLAIN_TEXT: Dict[str, str] = {
    CLASSIFICATION_OPTIONS[0]: "Unclassified",
    CLASSIFICATION_OPTIONS[1]: "TP",
    CLASSIFICATION_OPTIONS[2]: "FP",
}

#: Fixed widget keys for the E4 "Export CSV"/"Clear Log" buttons, following
#: the same "stable key" rationale as :data:`_EDITOR_KEY`/:data:`_FILTER_KEY`.
_EXPORT_BUTTON_KEY = "logbook_export_csv_button"
_CLEAR_LOG_BUTTON_KEY = "logbook_clear_log_button"


def _build_column_config() -> Dict[str, Any]:
    """Build the ``st.data_editor`` ``column_config`` mapping for the logbook grid.

    :returns: Mapping of column name to ``st.column_config`` column type,
        matching the story's acceptance criteria exactly: Classification
        is a ``SelectboxColumn`` with exactly
        :data:`CLASSIFICATION_OPTIONS`; Note/``comment`` is an editable
        ``TextColumn``; Time/Value/Conf% are ``disabled=True``.
    :rtype: dict[str, Any]
    """
    return {
        "time_str": st.column_config.TextColumn("Time", disabled=True),
        "value": st.column_config.NumberColumn("Value", disabled=True, format="%.3f"),
        "conf_pct": st.column_config.NumberColumn("Conf %", disabled=True, format="%.1f"),
        "classification": st.column_config.SelectboxColumn(
            "Classification",
            options=list(CLASSIFICATION_OPTIONS),
            required=True,
        ),
        "comment": st.column_config.TextColumn("Note"),
    }


def _build_display_df(logbook_df: pd.DataFrame) -> pd.DataFrame:
    """Build the DataFrame handed to ``st.data_editor``, indexed by ``row_id``.

    Selects only :data:`DISPLAY_COLUMNS` (hiding the internal
    ``timestamp``/``is_injected`` bookkeeping columns from the grid) and
    re-indexes on ``row_id`` -- ``row_id`` itself is *not* one of the
    displayed columns (it becomes the DataFrame's index instead, hidden
    from the rendered grid via ``hide_index=True`` in
    :func:`render_logbook_table`, but preserved as real data for the
    row_id-based merge-back in :func:`merge_editor_edits_into_logbook`).

    A future filter/search story (E3) can build its own reduced/reordered
    subset the same way (e.g. ``_build_display_df(logbook_df[mask])``)
    without needing any change to the merge-back logic.

    :param logbook_df: Source DataFrame, must contain a ``row_id`` column
        plus every name in :data:`DISPLAY_COLUMNS`.
    :type logbook_df: pandas.DataFrame
    :raises KeyError: If ``row_id`` or any of :data:`DISPLAY_COLUMNS` is
        missing from ``logbook_df`` (defensive -- signals a corrupted/
        mis-initialized ``logbook_df``, caught by the caller).
    :returns: A new DataFrame with :data:`DISPLAY_COLUMNS` as columns and
        ``row_id`` (cast to ``int64`` for stable cross-dtype matching) as
        the index.
    :rtype: pandas.DataFrame
    """
    display_df = logbook_df[list(DISPLAY_COLUMNS)].copy()
    # Cast explicitly to int64: logbook_df's columns are all `object` dtype
    # (see state.app_state._empty_logbook_df -- an empty DataFrame with no
    # explicit dtypes, which pd.concat then preserves as `object` even once
    # rows are appended), so row_id's underlying values are plain Python
    # ints held in an object-dtype Series. Casting here keeps this index's
    # dtype consistent with what st.data_editor's Arrow round-trip returns,
    # so index-label matching in merge_editor_edits_into_logbook is exact.
    display_df.index = logbook_df["row_id"].astype("int64")
    display_df.index.name = "row_id"
    return display_df


def merge_editor_edits_into_logbook(
    logbook_df: pd.DataFrame, edited_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge Classification/Note edits from ``edited_df`` into ``logbook_df``, by ``row_id``.

    Merges strictly by ``edited_df.index`` (which holds ``row_id`` values
    -- see :func:`_build_display_df`) matched against ``logbook_df["row_id"]``,
    **never** by positional/integer row order. This is what makes the merge
    correct even if ``edited_df`` is a reordered or filtered subset of
    ``logbook_df``'s row order (the future E3 filter scenario): a row's
    *position* within ``edited_df`` is irrelevant here, only its ``row_id``
    label is used to locate the matching row in ``logbook_df``.

    Rows present in ``logbook_df`` but absent from ``edited_df`` (e.g. a
    filtered-out row under a future filter) are left completely untouched.
    Any ``row_id`` present in ``edited_df`` but not found in ``logbook_df``
    is defensively skipped (logged, not raised) rather than silently
    inventing a row via ``.loc`` auto-creation.

    :param logbook_df: The full, authoritative logbook DataFrame
        (``state["logbook_df"]``), in its own row order.
    :type logbook_df: pandas.DataFrame
    :param edited_df: The (possibly reordered/subset) DataFrame returned
        by ``st.data_editor``, indexed by ``row_id`` (see
        :func:`_build_display_df`).
    :type edited_df: pandas.DataFrame
    :returns: A new DataFrame with the same columns/row order as
        ``logbook_df``, with ``classification``/``comment`` updated for
        every matched ``row_id``. Returns ``logbook_df`` unchanged (by
        reference) if it is empty or missing a ``row_id`` column, or if
        ``edited_df`` is empty.
    :rtype: pandas.DataFrame
    """
    if "row_id" not in logbook_df.columns:
        logger.warning(
            "merge_editor_edits_into_logbook: 'row_id' column missing from "
            "logbook_df; skipping merge (was init_session_state() called?)"
        )
        return logbook_df

    if logbook_df.empty or edited_df.empty:
        return logbook_df

    # Index by row_id (keeping the row_id *column* too, via drop=False) so
    # the final result can be restored to logbook_df's original column set
    # without a separate re-merge step.
    merged = logbook_df.copy()
    merged["row_id"] = merged["row_id"].astype("int64")
    merged = merged.set_index("row_id", drop=False)

    edited_index = edited_df.index.astype("int64")
    matched_ids = merged.index.intersection(edited_index)
    unmatched_ids = edited_index.difference(merged.index)
    if len(unmatched_ids) > 0:
        # Defensive: an edited row_id with no counterpart in logbook_df
        # (e.g. a stale editor key or a bug in a future filter view) must
        # never silently create a new row via .loc auto-insertion.
        logger.warning(
            "merge_editor_edits_into_logbook: %d edited row_id(s) not found "
            "in logbook_df, skipped: %s",
            len(unmatched_ids),
            list(unmatched_ids),
        )

    if len(matched_ids) > 0:
        for col in _EDITABLE_COLUMNS:
            if col in edited_df.columns:
                # .loc keyed by the row_id label on both sides -- this is
                # the merge-by-row_id step; it is never expressed in terms
                # of .iloc / integer position.
                merged.loc[matched_ids, col] = edited_df.loc[matched_ids, col].to_numpy()

    return merged.reset_index(drop=True)[list(LOGBOOK_COLUMNS)]


def _compute_filter_mask(
    logbook_df: pd.DataFrame, classification_filter: str, search_text: str
) -> pd.Series:
    """Compute the E3 row-visibility mask: classification filter AND notes search.

    Both conditions are computed independently and combined with ``&``
    (AND logic, per the story's acceptance criteria): a row must satisfy
    *both* the classification filter and the notes search to be shown.

    :param logbook_df: Full ``logbook_df`` to compute the mask over. The
        mask is aligned to this DataFrame's index; it is never applied
        in-place, so ``logbook_df`` itself is left untouched (the caller
        is expected to apply it via ``logbook_df[mask]`` only when
        building a *display* subset, never onto ``state["logbook_df"]``).
    :type logbook_df: pandas.DataFrame
    :param classification_filter: One of :data:`CLASSIFICATION_FILTER_OPTIONS`.
        ``"All"`` (or any unrecognized value, defensively) matches every
        row regardless of classification.
    :type classification_filter: str
    :param search_text: Case-insensitive substring to match against the
        ``comment`` column. An empty string matches every row.
    :type search_text: str
    :returns: Boolean Series aligned to ``logbook_df.index``, ``True``
        for rows that pass both filters.
    :rtype: pandas.Series
    """
    if classification_filter in _CLASSIFICATION_FILTER_VALUES:
        target = _CLASSIFICATION_FILTER_VALUES[classification_filter]
        classification_mask = logbook_df["classification"] == target
    else:
        # "All" (or any unexpected value) -- no classification filtering.
        classification_mask = pd.Series(True, index=logbook_df.index)

    search_text = search_text or ""
    if search_text == "":
        search_mask = pd.Series(True, index=logbook_df.index)
    else:
        # case=False -> case-insensitive; na=False -> missing/NaN comments
        # never match a non-empty search (rather than raising or matching
        # everything); regex=False -> a literal substring match, so
        # operator-typed search text with regex metacharacters (e.g. "?")
        # is matched literally, not interpreted as a pattern.
        search_mask = (
            logbook_df["comment"].astype(str).str.contains(search_text, case=False, na=False, regex=False)
        )

    return classification_mask & search_mask


def _compute_footer_counts(logbook_df: pd.DataFrame) -> Dict[str, int]:
    """Compute the E3 stats footer's total/TP/FP/pending counts.

    Always computed from the FULL, unfiltered ``logbook_df`` -- the
    footer must reflect every logged anomaly regardless of the current
    classification filter/search, per the story's acceptance criteria.

    :param logbook_df: Full ``logbook_df`` (typically ``state["logbook_df"]``,
        read *after* any same-rerun editor merge, so the counts never go
        stale relative to an edit that just happened).
    :type logbook_df: pandas.DataFrame
    :returns: Dict with integer keys ``"total"``, ``"tp"``, ``"fp"``,
        ``"pending"``.
    :rtype: dict[str, int]
    """
    if logbook_df.empty or "classification" not in logbook_df.columns:
        return {"total": 0, "tp": 0, "fp": 0, "pending": 0}

    classification = logbook_df["classification"]
    return {
        "total": len(logbook_df),
        "tp": int((classification == CLASSIFICATION_OPTIONS[1]).sum()),
        "fp": int((classification == CLASSIFICATION_OPTIONS[2]).sum()),
        "pending": int((classification == CLASSIFICATION_OPTIONS[0]).sum()),
    }


def _build_export_csv(logbook_df: Optional[pd.DataFrame]) -> str:
    """Build the story E4 "Export CSV" payload from ``logbook_df``.

    Transforms ``logbook_df`` (whose own columns are
    :data:`state.app_state.LOGBOOK_COLUMNS`) into the exact column
    set/order/names required by the story's acceptance criteria
    (:data:`EXPORT_COLUMNS`): ``row_id`` is kept as-is, ``time_str`` is
    dropped (only the raw ``timestamp`` is exported), ``conf_pct`` is
    renamed to ``confidence``, and ``classification`` has its emoji
    prefix stripped to plain text via :data:`_CLASSIFICATION_PLAIN_TEXT`
    (unrecognized values -- which should not occur in practice, since
    the editor's ``SelectboxColumn`` constrains input to
    :data:`CLASSIFICATION_OPTIONS` -- are passed through unchanged rather
    than raising, so a malformed row never blocks the whole export).

    :param logbook_df: Source DataFrame (typically
        ``state["logbook_df"]``). ``None`` or empty is handled
        defensively and produces a header-only CSV rather than raising.
    :type logbook_df: pandas.DataFrame | None
    :returns: CSV text (including a trailing newline, per
        ``DataFrame.to_csv``'s default), with header
        ``row_id,timestamp,value,confidence,classification,comment,is_injected``.
    :rtype: str
    """
    if logbook_df is None or logbook_df.empty:
        # Header-only CSV: still a valid, non-crashing export for a fresh
        # or just-cleared logbook.
        return pd.DataFrame(columns=list(EXPORT_COLUMNS)).to_csv(index=False)

    export_df = pd.DataFrame(
        {
            "row_id": logbook_df["row_id"],
            "timestamp": logbook_df["timestamp"],
            "value": logbook_df["value"],
            "confidence": logbook_df["conf_pct"],
            "classification": logbook_df["classification"].map(
                lambda value: _CLASSIFICATION_PLAIN_TEXT.get(value, value)
            ),
            "comment": logbook_df["comment"],
            "is_injected": logbook_df["is_injected"],
        },
        columns=list(EXPORT_COLUMNS),
    )
    return export_df.to_csv(index=False)


def _clear_logbook(state: Optional[MutableMapping] = None) -> None:
    """``on_click`` callback for the story E4 "Clear Log" button.

    Resets ``state["logbook_df"]`` to a fresh, empty
    :data:`~state.app_state.LOGBOOK_COLUMNS`-schema DataFrame (reusing
    :func:`state.app_state._empty_logbook_df`, the single source of
    truth for that schema, rather than duplicating it here) and resets
    ``state["next_log_id"]`` back to ``0``, matching the story's
    acceptance criterion. Follows the same "callback defaults its
    ``state`` param to ``st.session_state``" convention as
    ``ui_web.sidebar_config``'s button callbacks (e.g.
    ``_on_reset_clicked``), so it can be wired directly as
    ``on_click=_clear_logbook`` without an explicit ``args=`` tuple.

    :param state: Session-state mapping to mutate. Defaults to
        ``st.session_state``.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    state["logbook_df"] = _empty_logbook_df()
    state["next_log_id"] = 0
    logger.info("Logbook cleared: logbook_df emptied, next_log_id reset to 0.")


@st.fragment(run_every="1s")
def render_logbook_table(state: Optional[MutableMapping] = None) -> None:
    """Render the editable anomaly logbook grid and merge edits back by ``row_id``.

    Renders an ``st.data_editor`` over ``state["logbook_df"]`` (via
    :func:`_build_display_df`/:func:`_build_column_config`) and writes the
    merged result (via :func:`merge_editor_edits_into_logbook`) back into
    ``state["logbook_df"]`` on every call. Shows a friendly empty-state
    message instead of an empty grid when no anomalies have been logged
    yet (an empty, all-``object``-dtype DataFrame is a poor fit for
    ``st.data_editor``'s Arrow-based type inference, and there is nothing
    useful for the operator to edit, filter, or search yet -- so, by
    design, the E3 filter/search controls and the ``st.data_editor`` grid
    itself are *not* shown in this state; they only appear once the
    logbook holds at least one row. The E4 stats footer caption, however,
    IS still shown in this empty state (reading all-zero), and so are the
    E4 "Export CSV"/"Clear Log" buttons -- both are meaningful and safe
    to use on an empty logbook (a header-only CSV export, or a no-op
    clear), and showing the all-zero footer immediately confirms to the
    operator that "Clear Log" actually took effect rather than the grid
    merely disappearing).

    **Story E3 (filter/search/footer):** above the editor, renders a
    classification filter (:data:`CLASSIFICATION_FILTER_OPTIONS`) and a
    notes search box, combines them into a boolean mask via
    :func:`_compute_filter_mask`, and applies that mask *only* when
    building the display subset (``logbook_df[mask]``) handed to
    :func:`_build_display_df` -- ``logbook_df``/``state["logbook_df"]``
    itself is never filtered or mutated, and
    :func:`merge_editor_edits_into_logbook` is always called with the
    full, unfiltered ``logbook_df`` as its first argument so an edit made
    while a filter/search is active still merges back onto the complete
    logbook (any row currently hidden by the filter is simply left
    untouched, per :func:`merge_editor_edits_into_logbook`'s own
    contract). Below the editor, a stats footer caption is computed via
    :func:`_compute_footer_counts` from ``state["logbook_df"]`` *after*
    the merge-back assignment above it, so an edit made on this very
    rerun is already reflected in the footer's counts -- never stale.

    **Story E4 (Export CSV / Clear Log):** an "Export CSV" download
    button and a "Clear Log" button are rendered directly below the
    subheader, before either the empty-state message or the populated
    editor/filter/footer, so both are always available regardless of
    whether the logbook currently holds any rows.
    :func:`_build_export_csv` builds the download payload from the
    CURRENT, pre-merge ``logbook_df`` (any edit made earlier THIS same
    rerun is already applied to ``state["logbook_df"]`` by the time the
    next 1s fragment tick re-renders this button, so the exported CSV
    is never more than one tick stale). "Clear Log" is wired via
    ``on_click=_clear_logbook``, which resets ``state["logbook_df"]`` to
    empty and ``state["next_log_id"]`` to ``0``; because ``on_click``
    callbacks run *before* the rerun they trigger, the very same rerun
    that follows a "Clear Log" click already sees the emptied
    ``logbook_df`` -- so the stats footer (rendered unconditionally,
    including in the empty-state branch -- see this function's main
    docstring) reads all-zero immediately, with no extra 1s tick of lag.

    Decorated ``@st.fragment(run_every="1s")`` so the grid (and any
    classification/comment edits already applied) stays live-refreshed
    once per second, independent of full-script reruns triggered
    elsewhere in the app -- mirroring ``ui_web.live_tick.live_tick_fragment``'s
    ``run_every=LIVE_TICK_INTERVAL`` (``"300ms"``) pattern for the chart. Note that a
    fragment-decorated function's body only executes within a real
    ``ScriptRunContext`` (Streamlit no-ops it entirely -- returns ``None``
    without calling through -- outside of one); tests exercising this
    function's actual widget-rendering behavior therefore use
    ``streamlit.testing.v1.AppTest``, while the pure logic it delegates to
    (:func:`_build_display_df`, :func:`merge_editor_edits_into_logbook`,
    :func:`_build_column_config`, :func:`_compute_filter_mask`,
    :func:`_compute_footer_counts`) is unit-tested directly.

    :param state: Session-state mapping to read/mutate. Defaults to
        ``st.session_state``; accepting an injectable mapping keeps this
        signature consistent with the rest of ``ui_web`` even though, in
        practice, only real ``st.session_state`` can back the
        ``st.data_editor`` widget itself.
    :type state: MutableMapping | None
    :returns: None
    :rtype: None
    """
    if state is None:
        state = st.session_state

    st.subheader("Anomaly Logbook")

    logbook_df = state.get("logbook_df")
    if logbook_df is None:
        # Defensive: session not yet initialized (init_session_state()
        # hasn't run). Still show the header for layout stability.
        logger.warning(
            "render_logbook_table: 'logbook_df' missing from session state; "
            "skipping (was init_session_state() called?)"
        )
        st.info("Logbook is not initialized yet.")
        return

    # E4: "Export CSV" / "Clear Log", rendered unconditionally (even for
    # an empty logbook -- see the docstring's rationale) so both are
    # always reachable regardless of whether any anomalies are logged.
    export_col, clear_col = st.columns(2)
    with export_col:
        st.download_button(
            "Export CSV",
            data=_build_export_csv(logbook_df),
            file_name="anomaly_logbook.csv",
            mime="text/csv",
            key=_EXPORT_BUTTON_KEY,
        )
    with clear_col:
        st.button("Clear Log", key=_CLEAR_LOG_BUTTON_KEY, on_click=_clear_logbook)

    if logbook_df.empty:
        # No filter/search/data_editor for an empty logbook -- see the
        # docstring's rationale -- but the E4 stats footer is still shown
        # (reading all-zero) so a just-cleared log visibly confirms it.
        st.info("No anomalies logged yet.")
        counts = _compute_footer_counts(logbook_df)
        st.caption(
            f"{counts['total']} anomalies | TP: {counts['tp']} | FP: {counts['fp']} | "
            f"Pending: {counts['pending']}"
        )
        return

    # E3: classification filter + notes search, rendered above the editor.
    # Fixed keys (_FILTER_KEY/_SEARCH_KEY) so the operator's current
    # filter/search selection survives this fragment's 1s auto-reruns
    # instead of resetting back to defaults every tick (same rationale as
    # _EDITOR_KEY above).
    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        classification_filter = st.selectbox(
            "Classification filter",
            options=list(CLASSIFICATION_FILTER_OPTIONS),
            index=0,
            key=_FILTER_KEY,
        )
    with search_col:
        search_text = st.text_input(
            "Search notes",
            value="",
            key=_SEARCH_KEY,
            placeholder="Filter by note text...",
        )

    mask = _compute_filter_mask(logbook_df, classification_filter, search_text)

    try:
        # Apply the mask only to the DISPLAY subset -- logbook_df (and,
        # below, the merge-back target) always stays the full, unfiltered
        # DataFrame, per the story's "without deleting or mutating
        # logbook_df" acceptance criterion.
        display_df = _build_display_df(logbook_df[mask])
    except KeyError:
        logger.warning(
            "render_logbook_table: logbook_df is missing one or more "
            "required columns (%s); skipping render",
            ["row_id", *DISPLAY_COLUMNS],
        )
        st.error("Logbook data is malformed; cannot render the editor.")
        return

    edited_df = st.data_editor(
        display_df,
        column_config=_build_column_config(),
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        key=_EDITOR_KEY,
    )

    # Merge back onto the FULL, unfiltered logbook_df -- never the
    # filtered display_df -- so rows currently hidden by the filter/search
    # are preserved untouched (see merge_editor_edits_into_logbook's own
    # "filtered subset" contract).
    state["logbook_df"] = merge_editor_edits_into_logbook(logbook_df, edited_df)

    # E3 stats footer: computed from state["logbook_df"] (i.e. AFTER the
    # merge-back assignment immediately above), not the pre-merge local
    # `logbook_df`, so a classification edit applied on this very rerun
    # is already reflected -- no stale counts on the same rerun.
    counts = _compute_footer_counts(state["logbook_df"])
    st.caption(
        f"{counts['total']} anomalies | TP: {counts['tp']} | FP: {counts['fp']} | "
        f"Pending: {counts['pending']}"
    )
