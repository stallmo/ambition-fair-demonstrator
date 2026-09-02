"""Tests for :mod:`ui_web.logbook` (story E2 -- editable logbook data editor).

Covers, in order:

1. The exported constants (:data:`~ui_web.logbook.CLASSIFICATION_OPTIONS`,
   :data:`~ui_web.logbook.DISPLAY_COLUMNS`).
2. :func:`~ui_web.logbook._build_column_config` -- exact ``SelectboxColumn``
   options and ``disabled=True`` on Time/Value/Conf%.
3. :func:`~ui_web.logbook._build_display_df` -- the ``row_id``-indexed
   DataFrame handed to ``st.data_editor``.
4. :func:`~ui_web.logbook.merge_editor_edits_into_logbook` -- the crux of
   the story: merge-by-``row_id`` correctness, including under a
   reordered/subset "future filtered view" of the editor's input, tested
   as pure DataFrame logic (no Streamlit context needed).
5. :func:`~ui_web.logbook.render_logbook_table` -- exercised via
   ``streamlit.testing.v1.AppTest`` (a real ``ScriptRunContext`` is
   required both because it is ``@st.fragment``-decorated -- which no-ops
   entirely outside of one, see point 6 -- and because it calls
   ``st.data_editor`` directly): empty-logbook friendly message with no
   crash, correct column config reaching the real widget proto, and a
   full round-trip edit-merge-by-row_id check (including a reordered
   "future filter" scenario) driven by directly setting the data editor's
   widget-state session key, mirroring how ``AppTest`` widgets' own
   ``set_value()`` helpers work under the hood.
6. The ``@st.fragment(run_every="1s")`` decoration itself.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pandas as pd
import pytest

from state.app_state import LOGBOOK_COLUMNS
from ui_web.logbook import (
    CLASSIFICATION_FILTER_OPTIONS,
    CLASSIFICATION_OPTIONS,
    DISPLAY_COLUMNS,
    EXPORT_COLUMNS,
    _CLEAR_LOG_BUTTON_KEY,
    _EDITOR_KEY,
    _EXPORT_BUTTON_KEY,
    _FILTER_KEY,
    _SEARCH_KEY,
    _build_column_config,
    _build_display_df,
    _build_export_csv,
    _clear_logbook,
    _compute_filter_mask,
    _compute_footer_counts,
    merge_editor_edits_into_logbook,
    render_logbook_table,
)


def _row(row_id: int, classification: str = "⚪ Unclassified", comment: str = "") -> Dict[str, Any]:
    """Build one well-formed logbook row dict for test fixtures.

    :param row_id: The row's ``row_id``.
    :type row_id: int
    :param classification: Value for the ``classification`` column.
    :type classification: str
    :param comment: Value for the ``comment`` column.
    :type comment: str
    :returns: A dict with every :data:`state.app_state.LOGBOOK_COLUMNS` key.
    :rtype: dict[str, Any]
    """
    return {
        "row_id": row_id,
        "timestamp": float(row_id) + 1000.0,
        "time_str": f"10:00:{row_id:02d}",
        "value": float(row_id) * 1.5,
        "conf_pct": float(row_id) * 2.0,
        "classification": classification,
        "comment": comment,
        "is_injected": row_id % 2 == 0,
    }


def _make_logbook_df(row_ids: List[int]) -> pd.DataFrame:
    """Build a logbook DataFrame with one row per ``row_ids`` entry, in that order.

    :param row_ids: ``row_id`` values, in the row order the returned
        DataFrame should have.
    :type row_ids: list[int]
    :returns: DataFrame with columns :data:`state.app_state.LOGBOOK_COLUMNS`.
    :rtype: pandas.DataFrame
    """
    return pd.DataFrame([_row(rid) for rid in row_ids], columns=LOGBOOK_COLUMNS)


# ----------------------------------------------------------------------
# 1. Constants
# ----------------------------------------------------------------------


def test_classification_options_are_exactly_the_three_required_values() -> None:
    """CLASSIFICATION_OPTIONS must be exactly the three emoji-coded labels, in order."""
    assert tuple(CLASSIFICATION_OPTIONS) == ("⚪ Unclassified", "🟢 TP", "🟠 FP")


def test_display_columns_excludes_row_id_timestamp_and_is_injected() -> None:
    """The displayed grid must never surface row_id/timestamp/is_injected raw columns."""
    assert "row_id" not in DISPLAY_COLUMNS
    assert "timestamp" not in DISPLAY_COLUMNS
    assert "is_injected" not in DISPLAY_COLUMNS
    assert set(DISPLAY_COLUMNS) == {"time_str", "value", "conf_pct", "classification", "comment"}


# ----------------------------------------------------------------------
# 2. _build_column_config
# ----------------------------------------------------------------------


def test_column_config_classification_is_selectbox_with_exact_options() -> None:
    """Classification must be a SelectboxColumn with exactly the 3 required options."""
    config = _build_column_config()
    classification_cfg = config["classification"]
    assert classification_cfg["type_config"]["type"] == "selectbox"
    assert classification_cfg["type_config"]["options"] == [
        "⚪ Unclassified",
        "🟢 TP",
        "🟠 FP",
    ]


def test_column_config_classification_is_not_disabled() -> None:
    """Classification must be editable (not disabled=True)."""
    config = _build_column_config()
    assert config["classification"].get("disabled") is not True


def test_column_config_comment_is_editable_text_column() -> None:
    """Note/comment must be a TextColumn and must not be disabled."""
    config = _build_column_config()
    comment_cfg = config["comment"]
    assert comment_cfg["type_config"]["type"] == "text"
    assert comment_cfg.get("disabled") is not True


@pytest.mark.parametrize("column_name", ["time_str", "value", "conf_pct"])
def test_column_config_time_value_conf_are_disabled(column_name: str) -> None:
    """Time/Value/Conf% columns must all be disabled=True (not editable)."""
    config = _build_column_config()
    assert config[column_name]["disabled"] is True


def test_column_config_covers_every_display_column() -> None:
    """Every DISPLAY_COLUMNS entry must have a column_config entry (no silent gaps)."""
    config = _build_column_config()
    assert set(config.keys()) == set(DISPLAY_COLUMNS)


# ----------------------------------------------------------------------
# 3. _build_display_df
# ----------------------------------------------------------------------


def test_build_display_df_indexes_by_row_id() -> None:
    """The returned DataFrame's index must be exactly the row_id values, as int64."""
    logbook_df = _make_logbook_df([10, 20, 30])
    display_df = _build_display_df(logbook_df)
    assert list(display_df.index) == [10, 20, 30]
    assert display_df.index.name == "row_id"
    assert display_df.index.dtype == "int64"


def test_build_display_df_only_contains_display_columns() -> None:
    """row_id/timestamp/is_injected must not appear as columns in the display DataFrame."""
    logbook_df = _make_logbook_df([1])
    display_df = _build_display_df(logbook_df)
    assert list(display_df.columns) == list(DISPLAY_COLUMNS)
    assert "row_id" not in display_df.columns
    assert "timestamp" not in display_df.columns
    assert "is_injected" not in display_df.columns


def test_build_display_df_preserves_row_order() -> None:
    """The display DataFrame's row order must exactly match logbook_df's row order."""
    logbook_df = _make_logbook_df([30, 10, 20])  # deliberately non-ascending
    display_df = _build_display_df(logbook_df)
    assert list(display_df.index) == [30, 10, 20]


def test_build_display_df_raises_keyerror_when_row_id_missing() -> None:
    """A logbook_df without a 'row_id' column must raise KeyError (caller's job to catch)."""
    logbook_df = _make_logbook_df([1]).drop(columns=["row_id"])
    with pytest.raises(KeyError):
        _build_display_df(logbook_df)


# ----------------------------------------------------------------------
# 4. merge_editor_edits_into_logbook -- the crux of story E2.
# ----------------------------------------------------------------------


def test_merge_classification_edit_lands_on_correct_row() -> None:
    """Editing classification for one row_id must only change that row."""
    logbook_df = _make_logbook_df([10, 20, 30])
    edited = _build_display_df(logbook_df)
    edited.loc[20, "classification"] = "🟢 TP"

    merged = merge_editor_edits_into_logbook(logbook_df, edited)

    result = merged.set_index("row_id")["classification"].to_dict()
    assert result == {10: "⚪ Unclassified", 20: "🟢 TP", 30: "⚪ Unclassified"}


def test_merge_comment_edit_lands_on_correct_row() -> None:
    """Editing the comment/note for one row_id must only change that row's comment."""
    logbook_df = _make_logbook_df([10, 20, 30])
    edited = _build_display_df(logbook_df)
    edited.loc[30, "comment"] = "looks like a sensor glitch"

    merged = merge_editor_edits_into_logbook(logbook_df, edited)

    result = merged.set_index("row_id")["comment"].to_dict()
    assert result == {10: "", 20: "", 30: "looks like a sensor glitch"}


def test_merge_multiple_independent_edits_all_land_correctly() -> None:
    """Simultaneous edits to different rows/columns must all merge independently."""
    logbook_df = _make_logbook_df([1, 2, 3, 4])
    edited = _build_display_df(logbook_df)
    edited.loc[1, "classification"] = "🟠 FP"
    edited.loc[3, "classification"] = "🟢 TP"
    edited.loc[4, "comment"] = "checked twice"

    merged = merge_editor_edits_into_logbook(logbook_df, edited).set_index("row_id")

    assert merged.loc[1, "classification"] == "🟠 FP"
    assert merged.loc[2, "classification"] == "⚪ Unclassified"
    assert merged.loc[3, "classification"] == "🟢 TP"
    assert merged.loc[4, "comment"] == "checked twice"
    assert merged.loc[1, "comment"] == ""  # untouched


def test_merge_with_reordered_edited_df_still_matches_correct_row_id() -> None:
    """An edited_df in a DIFFERENT row order than logbook_df must still merge by row_id.

    This is the acceptance criterion's core scenario: a future filter
    (E3) may hand st.data_editor a reordered view, so the *visual
    position* of a row in the editor must never be confused with its
    row_id.
    """
    logbook_df = _make_logbook_df([10, 20, 30])  # logbook_df's own order: 10, 20, 30
    edited = _build_display_df(logbook_df)
    # Reorder to simulate a filtered/sorted view: position 0->30, 1->10, 2->20.
    edited = edited.loc[[30, 10, 20]]
    # Edit whatever sits at *visual position 1* in this reordered view (row_id=10),
    # NOT row_id=20 (which sat at position 1 in logbook_df's own order).
    edited.iloc[1, edited.columns.get_loc("classification")] = "🟢 TP"

    merged = merge_editor_edits_into_logbook(logbook_df, edited).set_index("row_id")

    assert merged.loc[10, "classification"] == "🟢 TP"
    assert merged.loc[20, "classification"] == "⚪ Unclassified"
    assert merged.loc[30, "classification"] == "⚪ Unclassified"
    # And logbook_df's own row order (10, 20, 30) must be unchanged by the merge.
    assert list(merged.reset_index()["row_id"]) == [10, 20, 30]


def test_merge_with_filtered_subset_edited_df_only_updates_matched_rows() -> None:
    """A filtered subset (fewer rows than logbook_df) must leave excluded rows untouched."""
    logbook_df = _make_logbook_df([10, 20, 30, 40])
    edited = _build_display_df(logbook_df)
    # Simulate a filter that only shows row_id 20 and 40.
    edited_subset = edited.loc[[20, 40]].copy()
    edited_subset.loc[20, "classification"] = "🟠 FP"
    edited_subset.loc[40, "comment"] = "confirmed false positive"

    merged = merge_editor_edits_into_logbook(logbook_df, edited_subset).set_index("row_id")

    assert merged.loc[20, "classification"] == "🟠 FP"
    assert merged.loc[40, "comment"] == "confirmed false positive"
    # Rows never shown to the (simulated) filtered editor must be untouched.
    assert merged.loc[10, "classification"] == "⚪ Unclassified"
    assert merged.loc[30, "classification"] == "⚪ Unclassified"
    # All 4 original rows must still be present -- the filter must not have
    # dropped rows from the authoritative logbook_df.
    assert len(merged) == 4


def test_merge_unmatched_row_id_in_edited_df_is_skipped_not_inserted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An edited_df row_id absent from logbook_df must be skipped, not silently inserted."""
    logbook_df = _make_logbook_df([10, 20])
    edited = _build_display_df(logbook_df)
    stray_row = edited.loc[[10]].copy()
    stray_row.index = pd.Index([999], name="row_id")  # row_id not in logbook_df
    edited_with_stray = pd.concat([edited, stray_row])

    with caplog.at_level(logging.WARNING):
        merged = merge_editor_edits_into_logbook(logbook_df, edited_with_stray)

    assert len(merged) == 2  # no row was inserted
    assert list(merged["row_id"]) == [10, 20]
    assert any("not found in logbook_df" in record.message for record in caplog.records)


def test_merge_preserves_non_display_columns_and_original_row_order() -> None:
    """timestamp/is_injected (not shown in the editor) must survive the merge untouched."""
    logbook_df = _make_logbook_df([10, 20, 30])
    edited = _build_display_df(logbook_df)
    edited.loc[20, "classification"] = "🟢 TP"

    merged = merge_editor_edits_into_logbook(logbook_df, edited)

    assert list(merged["row_id"]) == [10, 20, 30]  # row order preserved
    pd.testing.assert_series_equal(
        merged["timestamp"].reset_index(drop=True),
        logbook_df["timestamp"].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        merged["is_injected"].reset_index(drop=True),
        logbook_df["is_injected"].reset_index(drop=True),
        check_dtype=False,
    )
    assert list(merged.columns) == list(LOGBOOK_COLUMNS)


def test_merge_returns_unchanged_on_empty_logbook() -> None:
    """An empty logbook_df must be returned unchanged (no crash, no-op)."""
    logbook_df = _make_logbook_df([])
    edited = pd.DataFrame(columns=DISPLAY_COLUMNS)
    result = merge_editor_edits_into_logbook(logbook_df, edited)
    assert result is logbook_df


def test_merge_returns_unchanged_on_empty_edited_df() -> None:
    """An empty edited_df (nothing to merge) must leave logbook_df unchanged."""
    logbook_df = _make_logbook_df([10, 20])
    empty_edited = pd.DataFrame(columns=DISPLAY_COLUMNS)
    empty_edited.index.name = "row_id"
    result = merge_editor_edits_into_logbook(logbook_df, empty_edited)
    assert result is logbook_df


def test_merge_missing_row_id_column_is_a_safe_noop(caplog: pytest.LogCaptureFixture) -> None:
    """A logbook_df missing 'row_id' entirely must be a safe, logged no-op."""
    logbook_df = _make_logbook_df([10]).drop(columns=["row_id"])
    edited = pd.DataFrame({"classification": ["🟢 TP"]}, index=pd.Index([10], name="row_id"))

    with caplog.at_level(logging.WARNING):
        result = merge_editor_edits_into_logbook(logbook_df, edited)

    assert result is logbook_df
    assert any("row_id" in record.message for record in caplog.records)


# ----------------------------------------------------------------------
# 5. render_logbook_table -- AppTest-based (fragment + st.data_editor).
# ----------------------------------------------------------------------

_EMPTY_LOGBOOK_SCRIPT = """
import streamlit as st
from state.app_state import init_session_state
from ui_web.logbook import render_logbook_table

init_session_state()
render_logbook_table()
"""


def test_render_logbook_table_empty_logbook_shows_info_and_no_data_editor() -> None:
    """An empty (freshly initialized) logbook must show a friendly message, not a crash."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_EMPTY_LOGBOOK_SCRIPT)
    at.run()

    assert at.exception == []
    assert len(at.get("dataframe")) == 0  # no data_editor rendered for an empty logbook
    info_values = [i.value for i in at.info]
    assert any("No anomalies logged yet" in v for v in info_values)


def _populated_script(row_ids: List[int]) -> str:
    """Build an AppTest script that seeds logbook_df with the given row_ids, then renders it.

    :param row_ids: ``row_id`` values, in the row order ``logbook_df``
        should hold them (simulating, for a non-ascending order, what a
        future filtered/reordered view (E3) might hand the editor).
    :type row_ids: list[int]
    :returns: A script string usable with ``AppTest.from_string``.
    :rtype: str
    """
    rows_literal = repr(row_ids)
    return f"""
import pandas as pd
import streamlit as st
from state.app_state import LOGBOOK_COLUMNS, init_session_state
from ui_web.logbook import render_logbook_table

init_session_state()

def _row(row_id):
    return {{
        "row_id": row_id,
        "timestamp": float(row_id) + 1000.0,
        "time_str": f"10:00:{{row_id:02d}}",
        "value": float(row_id) * 1.5,
        "conf_pct": float(row_id) * 2.0,
        "classification": "\\u26aa Unclassified",
        "comment": "",
        "is_injected": row_id % 2 == 0,
    }}

row_ids = {rows_literal}
st.session_state["logbook_df"] = pd.DataFrame([_row(r) for r in row_ids], columns=LOGBOOK_COLUMNS)

render_logbook_table()

result = st.session_state["logbook_df"][["row_id", "classification", "comment"]].to_dict(orient="records")
st.json(result)
"""


def test_render_logbook_table_populated_renders_data_editor_with_correct_column_config() -> None:
    """A populated logbook must render exactly one data_editor with the required column config."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_populated_script([10, 20, 30]))
    at.run()

    assert at.exception == []
    dataframes = at.get("dataframe")
    assert len(dataframes) == 1

    columns_config = json.loads(dataframes[0].proto.columns)
    assert columns_config["classification"]["type_config"]["type"] == "selectbox"
    assert columns_config["classification"]["type_config"]["options"] == [
        "⚪ Unclassified",
        "🟢 TP",
        "🟠 FP",
    ]
    assert columns_config["comment"]["type_config"]["type"] == "text"
    assert columns_config["comment"].get("disabled") is not True
    for disabled_col in ("time_str", "value", "conf_pct"):
        assert columns_config[disabled_col]["disabled"] is True
    # row_id must not be exposed as a visible column, and the index (which
    # IS row_id) must be hidden from the rendered grid.
    assert "row_id" not in columns_config
    assert columns_config["_index"]["hidden"] is True


def test_render_logbook_table_edit_merges_by_row_id_in_ascending_order() -> None:
    """Editing the row at visual position 1 (ascending logbook order) must update row_id=20."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_populated_script([10, 20, 30]))
    at.run()
    assert at.exception == []

    # Simulate the frontend reporting an edit at *position* 1 (0-indexed),
    # which in ascending logbook_df order corresponds to row_id=20.
    at.session_state[_EDITOR_KEY] = {
        "edited_rows": {1: {"classification": "🟢 TP"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert at.exception == []

    result = json.loads(at.get("json")[-1].value)
    by_row_id = {r["row_id"]: r["classification"] for r in result}
    assert by_row_id == {10: "⚪ Unclassified", 20: "🟢 TP", 30: "⚪ Unclassified"}


def test_render_logbook_table_edit_merges_by_row_id_under_reordered_future_filter_view() -> None:
    """Editing position 1 of a NON-ascending logbook_df order must still hit the right row_id.

    ``logbook_df`` here is seeded in the order [30, 10, 20] -- standing in
    for a filtered/reordered view a future story (E3) might produce.
    Position 1 in *this* order is row_id=10, not row_id=20 -- if the merge
    logic ever regressed to using positional/``.iloc``-based matching
    instead of ``row_id``, this test would catch it by asserting the edit
    landed on row_id=10.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_populated_script([30, 10, 20]))
    at.run()
    assert at.exception == []

    at.session_state[_EDITOR_KEY] = {
        "edited_rows": {1: {"comment": "flagged during review"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert at.exception == []

    result = json.loads(at.get("json")[-1].value)
    by_row_id = {r["row_id"]: r["comment"] for r in result}
    assert by_row_id == {30: "", 10: "flagged during review", 20: ""}


def test_render_logbook_table_missing_logbook_df_shows_info_not_crash() -> None:
    """A session with no 'logbook_df' key at all (init not called) must not crash."""
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from ui_web.logbook import render_logbook_table

render_logbook_table()
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    info_values = [i.value for i in at.info]
    assert any("not initialized" in v for v in info_values)


# ----------------------------------------------------------------------
# 6b (story E3). _compute_filter_mask -- classification filter + notes
# search, combined with AND logic, as pure DataFrame logic.
# ----------------------------------------------------------------------


def test_classification_filter_options_are_exactly_the_four_required_values() -> None:
    """CLASSIFICATION_FILTER_OPTIONS must be exactly the four plain-text labels, in order."""
    assert tuple(CLASSIFICATION_FILTER_OPTIONS) == ("All", "TP", "FP", "Unclassified")


def test_filter_mask_all_matches_every_row_regardless_of_classification() -> None:
    """The 'All' filter option must not hide any row."""
    logbook_df = _make_logbook_df([1, 2, 3])
    logbook_df.loc[0, "classification"] = "🟢 TP"
    logbook_df.loc[1, "classification"] = "🟠 FP"
    logbook_df.loc[2, "classification"] = "⚪ Unclassified"

    mask = _compute_filter_mask(logbook_df, "All", "")

    assert list(mask) == [True, True, True]


@pytest.mark.parametrize(
    "filter_option,expected_classification",
    [
        ("TP", "🟢 TP"),
        ("FP", "🟠 FP"),
        ("Unclassified", "⚪ Unclassified"),
    ],
)
def test_filter_mask_each_classification_option_matches_only_that_classification(
    filter_option: str, expected_classification: str
) -> None:
    """Each of TP/FP/Unclassified must match ONLY rows with that exact classification."""
    logbook_df = _make_logbook_df([1, 2, 3])
    logbook_df.loc[0, "classification"] = "🟢 TP"
    logbook_df.loc[1, "classification"] = "🟠 FP"
    logbook_df.loc[2, "classification"] = "⚪ Unclassified"

    mask = _compute_filter_mask(logbook_df, filter_option, "")

    expected = logbook_df["classification"] == expected_classification
    assert list(mask) == list(expected)
    # Sanity: exactly one row should match in this 3-row, 3-distinct-value fixture.
    assert mask.sum() == 1


def test_filter_mask_classification_filter_does_not_mutate_logbook_df() -> None:
    """Computing the mask must never mutate/delete rows from logbook_df itself."""
    logbook_df = _make_logbook_df([1, 2, 3])
    logbook_df.loc[0, "classification"] = "🟢 TP"
    original = logbook_df.copy()

    _compute_filter_mask(logbook_df, "FP", "")

    pd.testing.assert_frame_equal(logbook_df, original)
    assert len(logbook_df) == 3


def test_filter_mask_search_is_case_insensitive_substring_match_on_comment() -> None:
    """Notes search must match 'comment' case-insensitively, as a substring."""
    logbook_df = _make_logbook_df([1, 2, 3])
    logbook_df.loc[0, "comment"] = "Sensor GLITCH detected"
    logbook_df.loc[1, "comment"] = "confirmed spike"
    logbook_df.loc[2, "comment"] = ""

    mask = _compute_filter_mask(logbook_df, "All", "glitch")

    assert list(mask) == [True, False, False]


def test_filter_mask_empty_search_matches_every_row() -> None:
    """An empty search string must not hide any row."""
    logbook_df = _make_logbook_df([1, 2, 3])
    logbook_df.loc[0, "comment"] = "abc"
    logbook_df.loc[1, "comment"] = ""
    logbook_df.loc[2, "comment"] = "xyz"

    mask = _compute_filter_mask(logbook_df, "All", "")

    assert list(mask) == [True, True, True]


def test_filter_mask_combines_classification_and_search_with_and_logic() -> None:
    """A row matching the classification filter but NOT the search must be excluded, and vice versa."""
    logbook_df = _make_logbook_df([1, 2, 3, 4])
    # Matches classification (TP) but NOT the search text.
    logbook_df.loc[0, "classification"] = "🟢 TP"
    logbook_df.loc[0, "comment"] = "unrelated note"
    # Matches the search text but NOT the classification filter (FP, not TP).
    logbook_df.loc[1, "classification"] = "🟠 FP"
    logbook_df.loc[1, "comment"] = "confirmed spike"
    # Matches BOTH -- the only row that should pass.
    logbook_df.loc[2, "classification"] = "🟢 TP"
    logbook_df.loc[2, "comment"] = "confirmed spike"
    # Matches NEITHER.
    logbook_df.loc[3, "classification"] = "⚪ Unclassified"
    logbook_df.loc[3, "comment"] = "nothing interesting"

    mask = _compute_filter_mask(logbook_df, "TP", "spike")

    assert list(mask) == [False, False, True, False]


def test_filter_mask_search_ignores_none_search_text_defensively() -> None:
    """A None search_text (defensive edge case) must behave like an empty search, not raise."""
    logbook_df = _make_logbook_df([1, 2])
    mask = _compute_filter_mask(logbook_df, "All", None)  # type: ignore[arg-type]
    assert list(mask) == [True, True]


# ----------------------------------------------------------------------
# 6c (story E3). _compute_footer_counts -- total/TP/FP/pending counts.
# ----------------------------------------------------------------------


def test_footer_counts_all_zero_for_empty_logbook() -> None:
    """An empty logbook_df must report all-zero counts, not raise."""
    counts = _compute_footer_counts(_make_logbook_df([]))
    assert counts == {"total": 0, "tp": 0, "fp": 0, "pending": 0}


def test_footer_counts_are_computed_from_full_unfiltered_logbook() -> None:
    """Counts must reflect every row, matching each classification exactly."""
    logbook_df = _make_logbook_df([1, 2, 3, 4, 5])
    logbook_df["classification"] = [
        "🟢 TP",
        "🟢 TP",
        "🟠 FP",
        "⚪ Unclassified",
        "⚪ Unclassified",
    ]

    counts = _compute_footer_counts(logbook_df)

    assert counts == {"total": 5, "tp": 2, "fp": 1, "pending": 2}


def test_footer_counts_missing_classification_column_is_a_safe_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed logbook_df missing 'classification' must return safe zeros, not raise."""
    logbook_df = _make_logbook_df([1]).drop(columns=["classification"])
    counts = _compute_footer_counts(logbook_df)
    assert counts == {"total": 0, "tp": 0, "fp": 0, "pending": 0}


# ----------------------------------------------------------------------
# 6d (story E3). render_logbook_table's filter/search UI + footer caption,
# via AppTest (real widget rendering + interaction).
# ----------------------------------------------------------------------

def _populated_script_with_classifications(rows: List[Dict[str, Any]]) -> str:
    """Build an AppTest script seeding logbook_df with explicit classification/comment per row.

    :param rows: List of dicts, each with ``row_id``, ``classification``,
        and ``comment`` keys.
    :type rows: list[dict[str, Any]]
    :returns: A script string usable with ``AppTest.from_string``.
    :rtype: str
    """
    rows_literal = repr(rows)
    return f"""
import pandas as pd
import streamlit as st
from state.app_state import LOGBOOK_COLUMNS, init_session_state
from ui_web.logbook import render_logbook_table

init_session_state()

def _row(row_id, classification, comment):
    return {{
        "row_id": row_id,
        "timestamp": float(row_id) + 1000.0,
        "time_str": f"10:00:{{row_id:02d}}",
        "value": float(row_id) * 1.5,
        "conf_pct": float(row_id) * 2.0,
        "classification": classification,
        "comment": comment,
        "is_injected": row_id % 2 == 0,
    }}

rows = {rows_literal}
st.session_state["logbook_df"] = pd.DataFrame(
    [_row(r["row_id"], r["classification"], r["comment"]) for r in rows],
    columns=LOGBOOK_COLUMNS,
)

render_logbook_table()

result = st.session_state["logbook_df"][["row_id", "classification", "comment"]].to_dict(orient="records")
st.json(result)
"""


def test_render_logbook_table_shows_filter_selectbox_and_search_input_when_populated() -> None:
    """A populated logbook must render the classification filter selectbox and search box."""
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "⚪ Unclassified", "comment": ""},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()

    assert at.exception == []
    selectboxes = at.selectbox
    assert len(selectboxes) == 1
    assert list(selectboxes[0].options) == ["All", "TP", "FP", "Unclassified"]
    assert selectboxes[0].value == "All"
    assert len(at.text_input) == 1


def test_render_logbook_table_no_filter_search_when_logbook_empty() -> None:
    """A genuinely empty logbook must show neither the filter selectbox nor the search box.

    Story E4 changed this scenario slightly from its original E3 shape: the
    stats footer caption IS now shown even for an empty logbook (reading
    all-zero -- see :func:`test_render_logbook_table_footer_caption_is_all_zero_when_empty`),
    so that a just-cleared "Clear Log" click visibly confirms the reset
    rather than the whole footer disappearing. Only the filter/search
    controls and the ``st.data_editor`` grid remain absent when empty.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_EMPTY_LOGBOOK_SCRIPT)
    at.run()

    assert at.exception == []
    assert len(at.selectbox) == 0
    assert len(at.text_input) == 0


def test_render_logbook_table_footer_caption_is_all_zero_when_empty() -> None:
    """Story E4: the stats footer caption must read all-zero for a genuinely empty logbook."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_EMPTY_LOGBOOK_SCRIPT)
    at.run()

    assert at.exception == []
    captions = [c.value for c in at.caption]
    assert captions == ["0 anomalies | TP: 0 | FP: 0 | Pending: 0"]


def test_render_logbook_table_footer_caption_format_is_exact() -> None:
    """The footer caption must exactly match the required format string, character-for-character."""
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "\U0001F7E2 TP", "comment": ""},
        {"row_id": 2, "classification": "\U0001F7E2 TP", "comment": ""},
        {"row_id": 3, "classification": "\U0001F7E0 FP", "comment": ""},
        {"row_id": 4, "classification": "⚪ Unclassified", "comment": ""},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()

    assert at.exception == []
    captions = [c.value for c in at.caption]
    assert "4 anomalies | TP: 2 | FP: 1 | Pending: 1" in captions


def test_render_logbook_table_classification_filter_hides_non_matching_rows_in_editor() -> None:
    """Selecting a classification filter must hide non-matching rows from the data_editor view."""
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "\U0001F7E2 TP", "comment": ""},
        {"row_id": 2, "classification": "\U0001F7E0 FP", "comment": ""},
        {"row_id": 3, "classification": "⚪ Unclassified", "comment": ""},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()
    assert at.exception == []

    at.selectbox(key=_FILTER_KEY).set_value("TP")
    at.run()
    assert at.exception == []

    dataframes = at.get("dataframe")
    assert len(dataframes) == 1
    assert list(dataframes[0].value.index) == [1]

    # The underlying (full, unfiltered) logbook_df must still hold all 3 rows.
    result = json.loads(at.get("json")[-1].value)
    assert sorted(r["row_id"] for r in result) == [1, 2, 3]


def test_render_logbook_table_notes_search_hides_non_matching_rows_in_editor() -> None:
    """Typing a search term must hide rows whose comment does not contain it."""
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "⚪ Unclassified", "comment": "sensor glitch"},
        {"row_id": 2, "classification": "⚪ Unclassified", "comment": "confirmed spike"},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()
    assert at.exception == []

    at.text_input(key=_SEARCH_KEY).set_value("GLITCH")  # case-insensitive
    at.run()
    assert at.exception == []

    dataframes = at.get("dataframe")
    assert len(dataframes) == 1
    assert list(dataframes[0].value.index) == [1]


def test_render_logbook_table_filter_and_search_combine_with_and_logic() -> None:
    """A row matching only the classification filter (not the search) must still be excluded."""
    from streamlit.testing.v1 import AppTest

    rows = [
        # Matches classification filter (TP) but not search text -- must be excluded.
        {"row_id": 1, "classification": "\U0001F7E2 TP", "comment": "unrelated"},
        # Matches search text but not classification filter (FP, filter is TP) -- excluded.
        {"row_id": 2, "classification": "\U0001F7E0 FP", "comment": "confirmed spike"},
        # Matches BOTH -- the only row that should remain visible.
        {"row_id": 3, "classification": "\U0001F7E2 TP", "comment": "confirmed spike"},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()
    assert at.exception == []

    at.selectbox(key=_FILTER_KEY).set_value("TP")
    at.text_input(key=_SEARCH_KEY).set_value("spike")
    at.run()
    assert at.exception == []

    dataframes = at.get("dataframe")
    assert len(dataframes) == 1
    assert list(dataframes[0].value.index) == [3]


def test_render_logbook_table_filter_hiding_all_rows_does_not_crash() -> None:
    """A filter combination matching zero rows must render an empty editor, not crash."""
    from streamlit.testing.v1 import AppTest

    rows = [{"row_id": 1, "classification": "⚪ Unclassified", "comment": "abc"}]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()

    at.selectbox(key=_FILTER_KEY).set_value("TP")  # no TP rows exist
    at.run()

    assert at.exception == []
    dataframes = at.get("dataframe")
    assert len(dataframes) == 1
    assert len(dataframes[0].value) == 0
    # The footer must still reflect the FULL logbook, not the (empty) filtered view.
    captions = [c.value for c in at.caption]
    assert "1 anomalies | TP: 0 | FP: 0 | Pending: 1" in captions


def test_render_logbook_table_filtering_does_not_mutate_underlying_logbook_df() -> None:
    """Applying a filter must never delete rows from state['logbook_df'] itself."""
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "\U0001F7E2 TP", "comment": ""},
        {"row_id": 2, "classification": "\U0001F7E0 FP", "comment": ""},
        {"row_id": 3, "classification": "⚪ Unclassified", "comment": ""},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()

    at.selectbox(key=_FILTER_KEY).set_value("Unclassified")
    at.run()
    assert at.exception == []

    result = json.loads(at.get("json")[-1].value)
    assert len(result) == 3  # all 3 rows still present in the underlying logbook_df
    assert sorted(r["row_id"] for r in result) == [1, 2, 3]


def test_render_logbook_table_footer_recomputes_on_same_rerun_as_classification_edit() -> None:
    """The footer must reflect a classification edit on the SAME rerun it happens, not stale counts.

    This is the acceptance criterion's trickiest scenario: editing
    ``edited_rows`` in the data_editor's own widget-state session key
    (the same mechanism ``AppTest``'s ``set_value()`` uses under the
    hood for ``st.data_editor``, per the existing E2 tests above) and
    confirming the footer caption -- rendered further down the SAME
    script run -- already shows the post-edit counts.
    """
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "⚪ Unclassified", "comment": ""},
        {"row_id": 2, "classification": "⚪ Unclassified", "comment": ""},
        {"row_id": 3, "classification": "⚪ Unclassified", "comment": ""},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()
    assert at.exception == []
    pre_edit_captions = [c.value for c in at.caption]
    assert "3 anomalies | TP: 0 | FP: 0 | Pending: 3" in pre_edit_captions

    # Simulate the frontend reporting an edit at visual position 1
    # (row_id=2, ascending order) reclassifying it as TP.
    at.session_state[_EDITOR_KEY] = {
        "edited_rows": {1: {"classification": "🟢 TP"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert at.exception == []

    post_edit_captions = [c.value for c in at.caption]
    # Must show the NEW counts (1 TP / 2 pending), not the pre-edit ones,
    # on this very rerun -- no stale counts.
    assert "3 anomalies | TP: 1 | FP: 0 | Pending: 2" in post_edit_captions
    assert "3 anomalies | TP: 0 | FP: 0 | Pending: 3" not in post_edit_captions


def test_render_logbook_table_footer_recomputes_on_same_rerun_as_edit_while_filtered() -> None:
    """The footer must reflect an edit even while a classification filter is currently active."""
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "⚪ Unclassified", "comment": ""},
        {"row_id": 2, "classification": "⚪ Unclassified", "comment": ""},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()
    assert at.exception == []

    # Filter down to "Unclassified" only -- both rows currently visible.
    at.selectbox(key=_FILTER_KEY).set_value("Unclassified")
    at.run()
    assert at.exception == []

    # Edit the row at visual position 0 (row_id=1, the only column shown
    # under this filter in ascending order) to TP.
    at.session_state[_EDITOR_KEY] = {
        "edited_rows": {0: {"classification": "🟠 FP"}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert at.exception == []

    captions = [c.value for c in at.caption]
    assert "2 anomalies | TP: 0 | FP: 1 | Pending: 1" in captions


# ----------------------------------------------------------------------
# 6e (story E3). app.py wiring.
# ----------------------------------------------------------------------


def test_app_module_imports_cleanly_with_logbook_wired_in() -> None:
    """`import app` must not raise now that render_logbook_table is wired into main()."""
    import app  # noqa: F401 -- import-safety check only.

    assert hasattr(app, "render_logbook_table")


def test_app_main_renders_the_anomaly_logbook_section() -> None:
    """Running the real app.py script (via AppTest) must render the Anomaly Logbook section."""
    from streamlit.testing.v1 import AppTest

    script = """
import app
app.main()
"""
    at = AppTest.from_string(script)
    at.run()

    assert at.exception == []
    subheaders = [s.value for s in at.subheader]
    assert "Anomaly Logbook" in subheaders
    # A freshly-initialized session has an empty logbook -- friendly
    # empty-state message, not a crash or a bare editor.
    info_values = [i.value for i in at.info]
    assert any("No anomalies logged yet" in v for v in info_values)


# ----------------------------------------------------------------------
# 6e (story E4). "Export CSV" / "Clear Log".
# ----------------------------------------------------------------------


def test_export_columns_are_exactly_the_required_header_in_order() -> None:
    """EXPORT_COLUMNS must be exactly the required CSV header, in order."""
    assert tuple(EXPORT_COLUMNS) == (
        "row_id",
        "timestamp",
        "value",
        "confidence",
        "classification",
        "comment",
        "is_injected",
    )


def test_build_export_csv_header_is_exact_character_for_character() -> None:
    """The CSV's first line must be exactly the required header string."""
    logbook_df = _make_logbook_df([1])
    csv_text = _build_export_csv(logbook_df)
    header_line = csv_text.splitlines()[0]
    assert header_line == "row_id,timestamp,value,confidence,classification,comment,is_injected"


def test_build_export_csv_on_empty_logbook_is_header_only_and_does_not_crash() -> None:
    """An empty logbook_df must export a valid header-only CSV, never raise."""
    empty_df = pd.DataFrame(columns=LOGBOOK_COLUMNS)
    csv_text = _build_export_csv(empty_df)
    lines = csv_text.splitlines()
    assert lines == ["row_id,timestamp,value,confidence,classification,comment,is_injected"]


def test_build_export_csv_on_none_is_header_only_and_does_not_crash() -> None:
    """A None logbook_df (defensive) must also export a valid header-only CSV."""
    csv_text = _build_export_csv(None)
    lines = csv_text.splitlines()
    assert lines == ["row_id,timestamp,value,confidence,classification,comment,is_injected"]


@pytest.mark.parametrize(
    "classification,expected_plain",
    [
        ("⚪ Unclassified", "Unclassified"),
        ("🟢 TP", "TP"),
        ("🟠 FP", "FP"),
    ],
)
def test_build_export_csv_strips_emoji_prefix_from_classification(
    classification: str, expected_plain: str
) -> None:
    """Each of the 3 emoji-coded classification values must map to its plain-text equivalent."""
    logbook_df = _make_logbook_df([1])
    logbook_df.loc[0, "classification"] = classification
    csv_text = _build_export_csv(logbook_df)
    data_row = csv_text.splitlines()[1]
    assert data_row.split(",")[4] == expected_plain


def test_build_export_csv_renames_conf_pct_to_confidence_and_drops_time_str() -> None:
    """The exported row's values must come from conf_pct (renamed) and never include time_str."""
    logbook_df = _make_logbook_df([7])
    csv_text = _build_export_csv(logbook_df)
    header, data_row = csv_text.splitlines()[0], csv_text.splitlines()[1]
    assert "time_str" not in header
    fields = dict(zip(header.split(","), data_row.split(",")))
    row = _row(7)
    assert fields["row_id"] == str(row["row_id"])
    assert fields["timestamp"] == str(row["timestamp"])
    assert fields["value"] == str(row["value"])
    assert fields["confidence"] == str(row["conf_pct"])
    assert fields["classification"] == "Unclassified"
    assert fields["comment"] == row["comment"]
    assert fields["is_injected"] == str(row["is_injected"])


def test_build_export_csv_multiple_rows_preserve_row_order() -> None:
    """Multiple rows must appear in the export in the same order as logbook_df."""
    logbook_df = _make_logbook_df([3, 1, 2])
    csv_text = _build_export_csv(logbook_df)
    data_lines = csv_text.splitlines()[1:]
    assert len(data_lines) == 3
    row_ids_in_order = [line.split(",")[0] for line in data_lines]
    assert row_ids_in_order == ["3", "1", "2"]


def test_clear_logbook_resets_logbook_df_to_empty_correct_schema_and_next_log_id_to_zero() -> None:
    """_clear_logbook must empty logbook_df (correct schema) and reset next_log_id to 0."""
    state: Dict[str, Any] = {
        "logbook_df": _make_logbook_df([1, 2, 3]),
        "next_log_id": 42,
    }
    _clear_logbook(state)

    assert state["next_log_id"] == 0
    cleared_df = state["logbook_df"]
    assert len(cleared_df) == 0
    assert list(cleared_df.columns) == LOGBOOK_COLUMNS


def test_clear_logbook_defaults_state_to_session_state_when_none() -> None:
    """_clear_logbook(None) must not raise outside a real Streamlit context.

    Exercising the `state is None` defensive branch directly; a real
    `st.session_state` is inert/unusable outside a ScriptRunContext, but
    this at least confirms the function does not blow up attempting the
    attribute access itself (mirrors the "no crash outside context" style
    used elsewhere in this test module for fragment/session-state code).
    """
    import streamlit as st

    try:
        _clear_logbook(None)
    except Exception as exc:  # pragma: no cover -- defensive assertion only
        pytest.fail(f"_clear_logbook(None) raised unexpectedly: {exc!r}")

    # Whatever bare st.session_state resolves to outside a real script
    # context, the call above must have attempted (not skipped) the reset.
    assert "next_log_id" in st.session_state
    assert st.session_state["next_log_id"] == 0


# ----------------------------------------------------------------------
# 6f (story E4). Export CSV / Clear Log via AppTest (real widgets + click).
# ----------------------------------------------------------------------


def test_render_logbook_table_shows_export_and_clear_buttons_when_populated() -> None:
    """A populated logbook must render both the Export CSV download button and Clear Log button."""
    from streamlit.testing.v1 import AppTest

    rows = [{"row_id": 1, "classification": "⚪ Unclassified", "comment": ""}]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()

    assert at.exception == []
    download_buttons = at.download_button
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "Export CSV"

    buttons = at.button
    assert len(buttons) == 1
    assert buttons[0].label == "Clear Log"


def test_render_logbook_table_shows_export_and_clear_buttons_when_empty() -> None:
    """Both buttons must also be present (and usable) when the logbook is empty."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_EMPTY_LOGBOOK_SCRIPT)
    at.run()

    assert at.exception == []
    assert len(at.download_button) == 1
    assert len(at.button) == 1


def test_export_csv_download_button_uses_expected_label_and_filename() -> None:
    """The rendered download_button widget must carry the expected label.

    ``AppTest``'s ``DownloadButton`` proto exposes ``label``/``url`` (the
    ``url`` referencing an internal, version-coupled mock media file store
    -- not something to assert byte-for-byte against here), but not the
    raw ``data``/``file_name``/``mime`` kwargs passed to
    ``st.download_button`` directly; those are exercised precisely via
    :func:`_build_export_csv`'s own dedicated unit tests above (exact
    header, emoji-stripping, empty-logbook, column renaming/order), which
    is what actually feeds this button's ``data=`` argument in
    :func:`~ui_web.logbook.render_logbook_table`.
    """
    from streamlit.testing.v1 import AppTest

    rows = [
        {"row_id": 1, "classification": "🟢 TP", "comment": "confirmed"},
        {"row_id": 2, "classification": "🟠 FP", "comment": "false alarm"},
    ]
    at = AppTest.from_string(_populated_script_with_classifications(rows))
    at.run()

    assert at.exception == []
    download_buttons = at.download_button
    assert len(download_buttons) == 1
    assert download_buttons[0].label == "Export CSV"
    assert download_buttons[0].proto.id.endswith(f"-{_EXPORT_BUTTON_KEY}")


_CLEAR_LOG_SCRIPT = """
import pandas as pd
import streamlit as st
from state.app_state import LOGBOOK_COLUMNS, init_session_state
from ui_web.logbook import render_logbook_table

init_session_state()


def _row(row_id):
    return {
        "row_id": row_id,
        "timestamp": float(row_id) + 1000.0,
        "time_str": f"10:00:{row_id:02d}",
        "value": float(row_id) * 1.5,
        "conf_pct": float(row_id) * 2.0,
        "classification": "\\U0001F7E2 TP",
        "comment": "note",
        "is_injected": False,
    }


# Guard the seeding to run exactly once (first script execution only) --
# on_click callbacks (e.g. "Clear Log"'s) run BEFORE the rerun they
# trigger, so unconditionally re-seeding logbook_df on every rerun (as
# earlier E2/E3 test fixtures in this module do, since none of them
# exercise a callback whose effect must survive into the next rerun)
# would silently clobber whatever the callback just did.
if "_seeded" not in st.session_state:
    st.session_state["logbook_df"] = pd.DataFrame(
        [_row(r) for r in [1, 2, 3]], columns=LOGBOOK_COLUMNS
    )
    st.session_state["next_log_id"] = 17
    st.session_state["_seeded"] = True

render_logbook_table()
"""


def test_clicking_clear_log_button_empties_logbook_and_resets_next_log_id() -> None:
    """A real AppTest click on 'Clear Log' must empty logbook_df and reset next_log_id to 0."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_CLEAR_LOG_SCRIPT)
    at.run()
    assert at.exception == []
    assert len(at.session_state["logbook_df"]) == 3
    assert at.session_state["next_log_id"] == 17

    at.button(key=_CLEAR_LOG_BUTTON_KEY).click()
    at.run()
    assert at.exception == []

    cleared_df = at.session_state["logbook_df"]
    assert len(cleared_df) == 0
    assert list(cleared_df.columns) == LOGBOOK_COLUMNS
    assert at.session_state["next_log_id"] == 0


def test_clicking_clear_log_button_makes_footer_caption_read_all_zero_immediately() -> None:
    """Immediately after a real 'Clear Log' click, the footer caption must read all-zero.

    Drives the click through a real AppTest interaction (not just reasoning
    about the code), then asserts the exact footer caption string rendered
    on that very same rerun -- no extra 1s fragment tick needed, since
    `_clear_logbook` runs as the button's `on_click` callback *before* the
    rerun it triggers.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_CLEAR_LOG_SCRIPT)
    at.run()
    assert at.exception == []
    pre_clear_captions = [c.value for c in at.caption]
    assert "3 anomalies | TP: 3 | FP: 0 | Pending: 0" in pre_clear_captions

    at.button(key=_CLEAR_LOG_BUTTON_KEY).click()
    at.run()
    assert at.exception == []

    post_clear_captions = [c.value for c in at.caption]
    assert post_clear_captions == ["0 anomalies | TP: 0 | FP: 0 | Pending: 0"]


def test_clicking_clear_log_then_exporting_produces_header_only_csv() -> None:
    """After Clear Log, _build_export_csv over the (now-empty) logbook_df must be header-only."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_CLEAR_LOG_SCRIPT)
    at.run()
    assert at.exception == []

    at.button(key=_CLEAR_LOG_BUTTON_KEY).click()
    at.run()
    assert at.exception == []

    csv_text = _build_export_csv(at.session_state["logbook_df"])
    assert csv_text.splitlines() == [
        "row_id,timestamp,value,confidence,classification,comment,is_injected"
    ]


# ----------------------------------------------------------------------
# 6. @st.fragment(run_every="1s") decoration.
# ----------------------------------------------------------------------


def test_render_logbook_table_is_fragment_decorated_with_run_every_1s() -> None:
    """render_logbook_table's closure must carry run_every='1s' (the @st.fragment argument).

    ``st.fragment`` is itself a decorator factory returning a wrapping
    closure; the ``run_every`` value it was called with is captured as a
    free variable in that closure, inspectable via
    ``__closure__``/``__code__.co_freevars`` without needing a live
    Streamlit script run. This directly verifies the decoration itself,
    complementing the AppTest-based tests above (which verify the
    decorated function's *behavior* within a real script run, since a
    fragment-decorated function's body does not execute at all outside of
    one -- see the module docstring on render_logbook_table).
    """
    freevars = render_logbook_table.__code__.co_freevars
    closure = render_logbook_table.__closure__
    assert closure is not None
    values = dict(zip(freevars, (cell.cell_contents for cell in closure)))
    assert values.get("run_every") == "1s"


def test_render_logbook_table_noops_outside_a_real_script_context() -> None:
    """Calling the fragment-decorated function with no ScriptRunContext must not raise.

    Streamlit's fragment wrapper simply returns None without executing the
    wrapped body when there is no live ``ScriptRunContext`` (confirmed
    against streamlit==1.61.1's actual behavior) -- this is exactly why
    this module's actual rendering behavior is tested via AppTest above,
    not via bare calls.
    """
    result = render_logbook_table({"logbook_df": pd.DataFrame(columns=LOGBOOK_COLUMNS)})
    assert result is None
