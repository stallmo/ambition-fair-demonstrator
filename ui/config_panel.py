"""Left sidebar: sensor configuration, training controls, simulation mode."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.data_generator import TEMPLATES, StreamConfig


class ConfigPanel(QWidget):
    """Left sidebar panel.

    Signals
    -------
    config_changed(StreamConfig)
    start_requested()
    stop_requested()
    reset_requested()
    train_requested()
    simulation_load_requested(str)   # scenario name
    """

    config_changed = pyqtSignal(object)   # StreamConfig
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    retrain_requested = pyqtSignal(int, int)   # (start_index, end_index)
    simulation_load_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = False
        self._end_tracking = True
        self._setup_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model_status(self, trained: bool) -> None:
        if trained:
            self._model_status.setText("Model ready")
            self._model_status.setStyleSheet("color: #50fa7b;")
        else:
            self._model_status.setText("Not trained")
            self._model_status.setStyleSheet("color: #ff5555;")

    def set_training_progress(self, progress: float) -> None:
        self._progress_bar.setValue(int(progress * 100))
        if progress >= 1.0:
            self._progress_bar.setFormat("Trained ✓")
        else:
            self._progress_bar.setFormat(f"Collecting… {int(progress * 100)}%")

    def set_controls_enabled(self, streaming: bool) -> None:
        """Lock / unlock controls during active streaming."""
        self._template_combo.setEnabled(not streaming)
        self._mean_spin.setEnabled(not streaming)
        self._std_spin.setEnabled(not streaming)
        self._noise_spin.setEnabled(not streaming)
        self._start_btn.setEnabled(not streaming)
        self._stop_btn.setEnabled(streaming)

    def set_simulation_controls_enabled(self, running: bool) -> None:
        self._sim_play_btn.setEnabled(not running)
        self._sim_stop_btn.setEnabled(running)
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(False)

    def reset_ui(self) -> None:
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Collecting… 0%")
        self.set_model_status(False)
        self.set_controls_enabled(False)
        self._end_tracking = True
        self._retrain_start_spin.setValue(0)
        self._retrain_end_spin.setValue(0)

    def update_sample_count(self, n: int) -> None:
        self._retrain_start_spin.setMaximum(max(0, n - 1))
        self._retrain_end_spin.setMaximum(n)
        if self._end_tracking:
            self._retrain_end_spin.setValue(n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(self._build_template_group())
        layout.addWidget(self._build_stream_group())
        layout.addWidget(self._build_anomaly_group())
        layout.addWidget(self._build_training_group())
        layout.addWidget(self._build_simulation_group())
        layout.addWidget(self._build_controls_group())
        layout.addStretch()

    # ── Template picker ───────────────────────────────────────────────

    def _build_template_group(self) -> QGroupBox:
        grp = QGroupBox("Sensor Template")
        lay = QVBoxLayout(grp)
        self._template_combo = QComboBox()
        self._template_combo.addItems(list(TEMPLATES.keys()))
        self._template_combo.currentTextChanged.connect(self._on_template_changed)
        lay.addWidget(self._template_combo)
        return grp

    def _on_template_changed(self, name: str) -> None:
        self._building = True
        tmpl = TEMPLATES[name]
        self._mean_spin.setValue(tmpl["mean"])
        self._std_spin.setValue(tmpl["std"])
        self._noise_spin.setValue(tmpl["noise"])
        self._building = False
        self._emit_config()

    # ── Stream parameters ─────────────────────────────────────────────

    def _build_stream_group(self) -> QGroupBox:
        grp = QGroupBox("Stream Parameters")
        lay = QVBoxLayout(grp)

        def row(label_text, spin):
            r = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(70)
            r.addWidget(lbl)
            r.addWidget(spin)
            return r

        self._mean_spin = self._make_spin(0, 9999, TEMPLATES["Temperature"]["mean"])
        self._std_spin = self._make_spin(0.01, 100, TEMPLATES["Temperature"]["std"])
        self._noise_spin = self._make_spin(0, 50, TEMPLATES["Temperature"]["noise"])

        for spin in (self._mean_spin, self._std_spin, self._noise_spin):
            spin.valueChanged.connect(self._emit_config)

        lay.addLayout(row("Mean:", self._mean_spin))
        lay.addLayout(row("Std Dev:", self._std_spin))
        lay.addLayout(row("Noise:", self._noise_spin))
        return grp

    def _make_spin(self, lo, hi, val, decimals=2) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setValue(val)
        s.setSingleStep(0.1)
        return s

    # ── Anomaly injection ─────────────────────────────────────────────

    def _build_anomaly_group(self) -> QGroupBox:
        grp = QGroupBox("Anomaly Injection")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Probability:"))
        self._anomaly_label = QLabel("2%")
        self._anomaly_label.setFixedWidth(36)
        row.addWidget(self._anomaly_label)
        lay.addLayout(row)

        self._anomaly_slider = QSlider(Qt.Horizontal)
        self._anomaly_slider.setRange(0, 100)
        self._anomaly_slider.setValue(2)
        self._anomaly_slider.setTickInterval(10)
        self._anomaly_slider.valueChanged.connect(self._on_anomaly_slider)
        lay.addWidget(self._anomaly_slider)
        return grp

    def _on_anomaly_slider(self, val: int) -> None:
        self._anomaly_label.setText(f"{val}%")
        self._emit_config()

    # ── GMM Training ──────────────────────────────────────────────────

    def _build_training_group(self) -> QGroupBox:
        grp = QGroupBox("GMM Training")
        lay = QVBoxLayout(grp)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Collecting… 0%")
        lay.addWidget(self._progress_bar)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self._model_status = QLabel("Not trained")
        self._model_status.setStyleSheet("color: #ff5555;")
        status_row.addWidget(self._model_status)
        status_row.addStretch()
        lay.addLayout(status_row)

        lay.addWidget(QLabel("Training Window"))

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Start:"))
        self._retrain_start_spin = QSpinBox()
        self._retrain_start_spin.setRange(0, 0)
        self._retrain_start_spin.setValue(0)
        window_row.addWidget(self._retrain_start_spin)
        window_row.addWidget(QLabel("End:"))
        self._retrain_end_spin = QSpinBox()
        self._retrain_end_spin.setRange(0, 0)
        self._retrain_end_spin.setValue(0)
        self._retrain_end_spin.valueChanged.connect(self._on_end_spin_changed)
        window_row.addWidget(self._retrain_end_spin)
        lay.addLayout(window_row)

        self._retrain_btn = QPushButton("Retrain Model")
        self._retrain_btn.setObjectName("train_btn")
        self._retrain_btn.clicked.connect(self._on_retrain_clicked)
        lay.addWidget(self._retrain_btn)
        return grp

    def _on_end_spin_changed(self, _value: int) -> None:
        self._end_tracking = False

    def _on_retrain_clicked(self) -> None:
        start = self._retrain_start_spin.value()
        end = self._retrain_end_spin.value()
        self.retrain_requested.emit(start, end)

    # ── Simulation mode ───────────────────────────────────────────────

    def _build_simulation_group(self) -> QGroupBox:
        grp = QGroupBox("Simulation Mode")
        lay = QVBoxLayout(grp)

        self._scenario_combo = QComboBox()
        self._scenario_combo.addItems(["Temperature", "Vibration", "Pressure"])
        lay.addWidget(self._scenario_combo)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self._speed_label = QLabel("1×")
        self._speed_label.setFixedWidth(28)
        speed_row.addWidget(self._speed_label)
        lay.addLayout(speed_row)

        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setRange(1, 10)
        self._speed_slider.setValue(1)
        self._speed_slider.valueChanged.connect(lambda v: self._speed_label.setText(f"{v}×"))
        lay.addWidget(self._speed_slider)

        btn_row = QHBoxLayout()
        self._sim_play_btn = QPushButton("Load & Play")
        self._sim_play_btn.setObjectName("start_btn")
        self._sim_play_btn.clicked.connect(self._on_sim_play)
        self._sim_stop_btn = QPushButton("Stop Sim")
        self._sim_stop_btn.setObjectName("stop_btn")
        self._sim_stop_btn.setEnabled(False)
        self._sim_stop_btn.clicked.connect(self.stop_requested)
        btn_row.addWidget(self._sim_play_btn)
        btn_row.addWidget(self._sim_stop_btn)
        lay.addLayout(btn_row)
        return grp

    def _on_sim_play(self) -> None:
        self.simulation_load_requested.emit(self._scenario_combo.currentText())

    def get_speed_multiplier(self) -> float:
        return float(self._speed_slider.value())

    # ── Controls ──────────────────────────────────────────────────────

    def _build_controls_group(self) -> QGroupBox:
        grp = QGroupBox("Controls")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("start_btn")
        self._start_btn.clicked.connect(self.start_requested)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested)
        row1.addWidget(self._start_btn)
        row1.addWidget(self._stop_btn)
        lay.addLayout(row1)

        self._reset_btn = QPushButton("Reset Session")
        self._reset_btn.clicked.connect(self.reset_requested)
        lay.addWidget(self._reset_btn)
        return grp

    # ── Signal emission ───────────────────────────────────────────────

    def _emit_config(self) -> None:
        if self._building:
            return
        tmpl_name = self._template_combo.currentText()
        tmpl = TEMPLATES[tmpl_name]
        cfg = StreamConfig(
            mean=self._mean_spin.value(),
            std=self._std_spin.value(),
            noise_amplitude=self._noise_spin.value(),
            anomaly_probability=self._anomaly_slider.value() / 100.0,
            anomaly_magnitude=tmpl["anomaly_magnitude"],
            unit=tmpl["unit"],
        )
        self.config_changed.emit(cfg)
