"""Right sidebar: anomaly logbook with classification and filtering."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from PyQt5.QtCore import QModelIndex, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.anomaly_detector import DetectionResult

_COL_TIME = 0
_COL_VALUE = 1
_COL_CONF = 2
_COL_CLASS = 3
_COL_NOTE = 4

_CLASSIFICATIONS = ["Unclassified", "TP", "FP"]

_BG_TP = QColor("#1a3a2a")
_BG_FP = QColor("#3a2a1a")
_BG_UNCLASSIFIED = QColor("#2a2a3e")


@dataclass
class LogbookEntry:
    row_id: int
    timestamp: float
    value: float
    confidence: float
    classification: str = "Unclassified"
    comment: str = ""
    is_injected: bool = False


class _ClassificationDelegate(QStyledItemDelegate):
    """Shows a persistent QComboBox in the Classification column."""

    classification_changed = pyqtSignal(int, str)  # row, value

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(_CLASSIFICATIONS)
        combo.currentTextChanged.connect(
            lambda text, r=index.row(): self._on_changed(r, text)
        )
        return combo

    def setEditorData(self, editor, index):
        val = index.data(Qt.EditRole) or "Unclassified"
        idx = _CLASSIFICATIONS.index(val) if val in _CLASSIFICATIONS else 0
        editor.setCurrentIndex(idx)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)

    def _on_changed(self, row: int, text: str) -> None:
        self.classification_changed.emit(row, text)


class LogbookPanel(QWidget):
    """Right panel: anomaly log with TP/FP classification and filtering.

    Signals
    -------
    stats_updated(dict)
    """

    stats_updated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list[LogbookEntry] = []
        self._next_id = 0
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_anomaly(self, result: DetectionResult) -> None:
        if not result.is_anomaly:
            return
        entry = LogbookEntry(
            row_id=self._next_id,
            timestamp=result.timestamp,
            value=result.value,
            confidence=result.confidence,
            is_injected=result.is_injected,
        )
        self._next_id += 1
        self._entries.append(entry)
        self._add_table_row(entry)
        self._update_footer()

    def clear(self) -> None:
        self._entries.clear()
        self._next_id = 0
        self._table.setRowCount(0)
        self._update_footer()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("Anomaly Logbook")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #bd93f9;")
        layout.addWidget(title)

        # Filter bar
        filter_row = QHBoxLayout()
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "TP", "FP", "Unclassified"])
        self._filter_combo.currentTextChanged.connect(self._apply_filter)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search notes…")
        self._search_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_combo)
        filter_row.addWidget(self._search_edit)
        layout.addLayout(filter_row)

        # Table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Time", "Value", "Conf%", "Classification", "Note"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(_COL_TIME, 72)
        self._table.setColumnWidth(_COL_VALUE, 68)
        self._table.setColumnWidth(_COL_CONF, 52)
        self._table.setColumnWidth(_COL_CLASS, 100)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self._table.verticalHeader().setVisible(False)
        self._table.itemChanged.connect(self._on_item_changed)

        # Classification delegate
        self._delegate = _ClassificationDelegate(self._table)
        self._table.setItemDelegateForColumn(_COL_CLASS, self._delegate)
        self._delegate.classification_changed.connect(self._on_classification_changed)

        layout.addWidget(self._table)

        # Footer
        self._footer_label = QLabel("0 anomalies | TP: 0 | FP: 0 | Pending: 0")
        self._footer_label.setStyleSheet("color: #6272a4; font-size: 11px;")
        layout.addWidget(self._footer_label)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear)
        layout.addWidget(clear_btn)

    def _add_table_row(self, entry: LogbookEntry) -> None:
        self._table.blockSignals(True)
        row = self._table.rowCount()
        self._table.insertRow(row)

        ts_str = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
        items = [
            QTableWidgetItem(ts_str),
            QTableWidgetItem(f"{entry.value:.2f}"),
            QTableWidgetItem(f"{entry.confidence * 100:.0f}%"),
            QTableWidgetItem(entry.classification),
            QTableWidgetItem(entry.comment),
        ]
        for col, item in enumerate(items):
            item.setData(Qt.UserRole, entry.row_id)
            if col not in (_COL_CLASS, _COL_NOTE):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, col, item)

        self._set_row_color(row, entry.classification)
        self._table.blockSignals(False)
        self._table.scrollToBottom()

    def _set_row_color(self, row: int, classification: str) -> None:
        color = {
            "TP": _BG_TP,
            "FP": _BG_FP,
        }.get(classification, _BG_UNCLASSIFIED)
        for col in range(self._table.columnCount()):
            item = self._table.item(row, col)
            if item:
                item.setBackground(color)

    def _on_classification_changed(self, row: int, text: str) -> None:
        if row >= len(self._entries):
            return
        # Find entry by row_id stored in item data
        id_item = self._table.item(row, 0)
        if id_item is None:
            return
        entry_id = id_item.data(Qt.UserRole)
        for entry in self._entries:
            if entry.row_id == entry_id:
                entry.classification = text
                break
        self._table.blockSignals(True)
        class_item = self._table.item(row, _COL_CLASS)
        if class_item:
            class_item.setText(text)
        self._set_row_color(row, text)
        self._table.blockSignals(False)
        self._update_footer()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != _COL_NOTE:
            return
        row = item.row()
        id_item = self._table.item(row, 0)
        if id_item is None:
            return
        entry_id = id_item.data(Qt.UserRole)
        for entry in self._entries:
            if entry.row_id == entry_id:
                entry.comment = item.text()
                break
        self._update_footer()

    def _apply_filter(self) -> None:
        filt = self._filter_combo.currentText()
        search = self._search_edit.text().lower()
        for row in range(self._table.rowCount()):
            class_item = self._table.item(row, _COL_CLASS)
            note_item = self._table.item(row, _COL_NOTE)
            classification = class_item.text() if class_item else "Unclassified"
            note = note_item.text().lower() if note_item else ""
            class_match = filt == "All" or classification == filt
            note_match = not search or search in note
            self._table.setRowHidden(row, not (class_match and note_match))

    def _update_footer(self) -> None:
        total = len(self._entries)
        tp = sum(1 for e in self._entries if e.classification == "TP")
        fp = sum(1 for e in self._entries if e.classification == "FP")
        pending = sum(1 for e in self._entries if e.classification == "Unclassified")
        self._footer_label.setText(
            f"{total} anomalies | TP: {tp} | FP: {fp} | Pending: {pending}"
        )
        stats = {
            "total_anomalies": total,
            "tp": tp,
            "fp": fp,
            "pending": pending,
        }
        self.stats_updated.emit(stats)

    def get_stats(self) -> dict:
        total = len(self._entries)
        tp = sum(1 for e in self._entries if e.classification == "TP")
        fp = sum(1 for e in self._entries if e.classification == "FP")
        pending = sum(1 for e in self._entries if e.classification == "Unclassified")
        return {
            "total_anomalies": total,
            "tp": tp,
            "fp": fp,
            "pending": pending,
            "entries": self._entries,
        }

    def export_csv(self, path: str) -> None:
        import pandas as pd
        rows = [
            {
                "row_id": e.row_id,
                "timestamp": e.timestamp,
                "value": e.value,
                "confidence": e.confidence,
                "classification": e.classification,
                "comment": e.comment,
                "is_injected": e.is_injected,
            }
            for e in self._entries
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
