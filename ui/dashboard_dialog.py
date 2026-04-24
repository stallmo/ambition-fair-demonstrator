"""Non-modal summary dashboard dialog."""
from __future__ import annotations

from typing import Callable

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_REFRESH_MS = 5000


class DashboardDialog(QDialog):
    """Auto-refreshing summary statistics popup."""

    def __init__(self, stats_provider: Callable[[], dict], total_samples_provider: Callable[[], int], parent=None):
        super().__init__(parent)
        self._stats_fn = stats_provider
        self._samples_fn = total_samples_provider
        self.setWindowTitle("Session Dashboard")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.resize(480, 560)
        self._setup_ui()
        self._refresh()
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Session Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #bd93f9;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Stats grid
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(6)
        layout.addWidget(self._grid_widget)

        # Pie chart
        self._fig = Figure(figsize=(4, 3), tight_layout=True)
        self._fig.patch.set_facecolor("#1e1e2e")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#2a2a3e")
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas)

        # Buttons
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _refresh(self) -> None:
        stats = self._stats_fn()
        total_samples = self._samples_fn()

        # Clear and rebuild grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        def add_row(r, label, value, color="#f8f8f2"):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #6272a4;")
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {color}; font-weight: bold;")
            self._grid.addWidget(lbl, r, 0)
            self._grid.addWidget(val, r, 1)

        total_anomalies = stats.get("total_anomalies", 0)
        tp = stats.get("tp", 0)
        fp = stats.get("fp", 0)
        pending = stats.get("pending", 0)

        entries = stats.get("entries", [])
        avg_conf = (
            sum(e.confidence for e in entries) / len(entries) if entries else 0.0
        )

        add_row(0, "Total Samples:", total_samples)
        add_row(1, "Total Anomalies:", total_anomalies, "#ff5555" if total_anomalies else "#f8f8f2")
        add_row(2, "True Positives (TP):", tp, "#50fa7b")
        add_row(3, "False Positives (FP):", fp, "#ffb86c")
        add_row(4, "Unclassified:", pending, "#6272a4")
        add_row(5, "Avg Confidence:", f"{avg_conf * 100:.1f}%")

        # Pie chart
        self._ax.cla()
        self._ax.set_facecolor("#2a2a3e")
        labels = ["TP", "FP", "Unclassified"]
        sizes = [tp, fp, pending]
        colors = ["#50fa7b", "#ff5555", "#6272a4"]

        if sum(sizes) > 0:
            wedges, texts, autotexts = self._ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.0f%%",
                startangle=90,
                textprops={"color": "#f8f8f2", "fontsize": 10},
            )
            for at in autotexts:
                at.set_color("#1e1e2e")
                at.set_fontweight("bold")
        else:
            self._ax.text(
                0.5, 0.5, "No anomalies yet",
                ha="center", va="center",
                color="#6272a4", fontsize=12,
                transform=self._ax.transAxes,
            )
            self._ax.set_xlim(0, 1)
            self._ax.set_ylim(0, 1)

        self._ax.set_title("Anomaly Classification", color="#f8f8f2", fontsize=12, pad=10)
        self._canvas.draw_idle()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)
