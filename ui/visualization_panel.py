"""Center panel: embedded matplotlib chart with dirty-flag redraw."""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.anomaly_detector import DetectionResult
from utils.ring_buffer import RingBuffer

_WINDOW = 300   # samples visible at once


class VisualizationPanel(QWidget):
    """Displays the rolling sensor value + confidence score chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._needs_redraw = False
        self._unit = "°C"
        self._total_count = 0

        # Ring buffers
        self._ts = RingBuffer(_WINDOW, dtype=float)
        self._vals = RingBuffer(_WINDOW, dtype=float)
        self._confs = RingBuffer(_WINDOW, dtype=float)
        self._is_anomaly = RingBuffer(_WINDOW, dtype=bool)
        self._is_injected = RingBuffer(_WINDOW, dtype=bool)

        self._gmm_band_artists: list = []
        self._gmm_mean_lines: list = []
        self._gmm_boundary_lines: list = []
        self._gmm_overlay_active = False

        self._setup_figure()
        self._setup_redraw_timer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_point(self, result: DetectionResult) -> None:
        self._ts.append(result.timestamp)
        self._vals.append(result.value)
        self._confs.append(result.confidence)
        self._is_anomaly.append(result.is_anomaly)
        self._is_injected.append(result.is_injected)
        self._total_count += 1
        self._needs_redraw = True

    def set_unit(self, unit: str) -> None:
        self._unit = unit

    def set_gmm_overlay(self, means: list, stds: list, weights: list,
                        boundary_values: list | None = None) -> None:
        _COLORS = ["#bd93f9", "#ffb86c"]
        for artist in self._gmm_band_artists + self._gmm_mean_lines + self._gmm_boundary_lines:
            try:
                artist.remove()
            except Exception:
                pass
        self._gmm_band_artists = []
        self._gmm_mean_lines = []
        self._gmm_boundary_lines = []

        for i, (m, s, c) in enumerate(zip(means, stds, _COLORS)):
            band = self._ax_main.axhspan(m - 2 * s, m + 2 * s, alpha=0.12, color=c,
                                         zorder=0, label=f"GMM component {i + 1}")
            line = self._ax_main.axhline(m, color=c, linestyle="--", linewidth=1.0,
                                          alpha=0.7, zorder=1)
            self._gmm_band_artists.append(band)
            self._gmm_mean_lines.append(line)

        for i, bv in enumerate(boundary_values or []):
            label = "Decision boundary" if i == 0 else "_nolegend_"
            line = self._ax_main.axhline(bv, color="#ff79c6", linestyle=":",
                                          linewidth=1.5, alpha=0.9, zorder=2, label=label)
            self._gmm_boundary_lines.append(line)

        self._ax_main.legend(
            facecolor="#2a2a3e", edgecolor="#44475a", labelcolor="#f8f8f2",
            fontsize=8, loc="upper left"
        )
        self._gmm_overlay_active = True
        self._needs_redraw = True

    def clear(self) -> None:
        for buf in (self._ts, self._vals, self._confs, self._is_anomaly, self._is_injected):
            buf.clear()
        for artist in self._gmm_band_artists + self._gmm_mean_lines + self._gmm_boundary_lines:
            try:
                artist.remove()
            except Exception:
                pass
        self._gmm_band_artists = []
        self._gmm_mean_lines = []
        self._gmm_boundary_lines = []
        self._gmm_overlay_active = False
        self._total_count = 0
        self._needs_redraw = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_figure(self) -> None:
        self._fig = Figure(figsize=(8, 5), constrained_layout=True)
        self._fig.patch.set_facecolor("#1e1e2e")

        # 2 stacked subplots: 75% / 25% height ratio
        gs = self._fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        self._ax_main = self._fig.add_subplot(gs[0])
        self._ax_score = self._fig.add_subplot(gs[1], sharex=self._ax_main)

        for ax in (self._ax_main, self._ax_score):
            ax.set_facecolor("#2a2a3e")
            ax.tick_params(colors="#f8f8f2", labelsize=9)
            ax.spines[:].set_color("#44475a")
            for spine in ax.spines.values():
                spine.set_color("#44475a")

        self._ax_main.set_ylabel("Value", color="#f8f8f2", fontsize=10)
        self._ax_score.set_ylabel("Confidence", color="#f8f8f2", fontsize=10)
        self._ax_score.set_xlabel("Sample Index", color="#f8f8f2", fontsize=10)
        self._ax_score.set_ylim(0, 1)

        # Persistent plot objects (updated in-place for performance)
        self._line, = self._ax_main.plot([], [], color="#6272a4", linewidth=1.2, zorder=1)
        self._scat_normal = self._ax_main.scatter([], [], c="#50fa7b", s=12, zorder=2, label="Normal")
        self._scat_anomaly = self._ax_main.scatter([], [], c="#ff5555", s=60, marker="^", zorder=3, label="Anomaly")
        self._ax_main.legend(
            facecolor="#2a2a3e", edgecolor="#44475a", labelcolor="#f8f8f2", fontsize=8, loc="upper left"
        )

        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def _setup_redraw_timer(self) -> None:
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(50)  # 20 fps
        self._redraw_timer.timeout.connect(self._redraw)
        self._redraw_timer.start()

    def _redraw(self) -> None:
        if not self._needs_redraw:
            return
        self._needs_redraw = False

        n = self._vals.size
        if n == 0:
            return

        xs = np.arange(self._total_count - n, self._total_count)
        vals = self._vals.get_window(n)
        confs = self._confs.get_window(n)
        is_anom = self._is_anomaly.get_window(n).astype(bool)

        # Main line
        self._line.set_data(xs, vals)

        # Normal scatter
        norm_idx = np.where(~is_anom)[0]
        if len(norm_idx):
            self._scat_normal.set_offsets(np.c_[xs[norm_idx], vals[norm_idx]])
        else:
            self._scat_normal.set_offsets(np.empty((0, 2)))

        # Anomaly scatter
        anom_idx = np.where(is_anom)[0]
        if len(anom_idx):
            self._scat_anomaly.set_offsets(np.c_[xs[anom_idx], vals[anom_idx]])
        else:
            self._scat_anomaly.set_offsets(np.empty((0, 2)))

        # Rescale main axis
        self._ax_main.relim()
        self._ax_main.autoscale_view()

        # Score bars (confidence)
        self._ax_score.cla()
        self._ax_score.set_facecolor("#2a2a3e")
        self._ax_score.set_ylim(0, 1)
        self._ax_score.set_ylabel("Confidence", color="#f8f8f2", fontsize=10)
        self._ax_score.tick_params(colors="#f8f8f2", labelsize=9)
        for spine in self._ax_score.spines.values():
            spine.set_color("#44475a")

        colors = np.where(is_anom, "#ff5555", "#50fa7b")
        self._ax_score.bar(xs, confs, color=colors, width=1.0, align="center")

        if self._gmm_overlay_active:
            self._ax_score.axhline(0.5, color="#ff79c6", linestyle="--",
                                   linewidth=1.0, alpha=0.8, label="Decision boundary")
            self._ax_score.legend(facecolor="#2a2a3e", edgecolor="#44475a",
                                  labelcolor="#f8f8f2", fontsize=8, loc="upper right")

        self._canvas.draw_idle()
