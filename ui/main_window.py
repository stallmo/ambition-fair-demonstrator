"""Main application window: owns all objects and wires signals."""
from __future__ import annotations

import time
from enum import Enum, auto
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
)

from core.anomaly_detector import AnomalyDetector, DetectionResult
from core.data_generator import DataGenerator, StreamConfig
from core.simulation_loader import SimulationPlayer, SimulationScenario, generate_scenarios
from ui.config_panel import ConfigPanel
from ui.dashboard_dialog import DashboardDialog
from ui.logbook_panel import LogbookPanel
from ui.visualization_panel import VisualizationPanel


class AppState(Enum):
    IDLE = auto()
    TRAINING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SIMULATION_RUNNING = auto()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._state = AppState.IDLE
        self._total_samples = 0
        self._dashboard: DashboardDialog | None = None

        self._create_objects()
        self._setup_ui()
        self._wire_signals()
        self._build_menus()
        self._build_status_bar()

        generate_scenarios()

    # ------------------------------------------------------------------
    # Object creation
    # ------------------------------------------------------------------

    def _create_objects(self) -> None:
        self._generator = DataGenerator(self)
        self._detector = AnomalyDetector(self)
        self._sim_player = SimulationPlayer(self)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        self.setWindowTitle("Predictive Maintenance Demonstrator")
        self.resize(1400, 820)

        self._config_panel = ConfigPanel()
        self._vis_panel = VisualizationPanel()
        self._log_panel = LogbookPanel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._config_panel)
        splitter.addWidget(self._vis_panel)
        splitter.addWidget(self._log_panel)
        splitter.setSizes([300, 760, 340])
        splitter.setChildrenCollapsible(False)

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _wire_signals(self) -> None:
        # Generator → Detector
        self._generator.new_sample.connect(self._on_new_sample)
        # SimPlayer → Detector (same handler)
        self._sim_player.new_sample.connect(self._on_new_sample)
        self._sim_player.finished.connect(self._on_simulation_finished)

        # Detector → Panels
        self._detector.detection_ready.connect(self._vis_panel.add_point)
        self._detector.detection_ready.connect(self._log_panel.add_anomaly)
        self._detector.model_trained.connect(self._on_model_trained)
        self._detector.training_progress_changed.connect(
            self._config_panel.set_training_progress
        )

        # ConfigPanel → Generator/MainWindow
        self._config_panel.config_changed.connect(self._generator.configure)
        self._config_panel.start_requested.connect(self._on_start)
        self._config_panel.stop_requested.connect(self._on_stop)
        self._config_panel.reset_requested.connect(self._session_reset)
        self._config_panel.retrain_requested.connect(self._on_retrain)
        self._config_panel.simulation_load_requested.connect(self._on_load_simulation)

        # Logbook stats → status bar
        self._log_panel.stats_updated.connect(self._update_status)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        new_act = QAction("New Session", self)
        new_act.setShortcut("Ctrl+N")
        new_act.triggered.connect(self._session_reset)
        file_menu.addAction(new_act)

        load_act = QAction("Load Simulation…", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self._on_load_simulation_dialog)
        file_menu.addAction(load_act)

        export_act = QAction("Export Log…", self)
        export_act.setShortcut("Ctrl+S")
        export_act.triggered.connect(self._on_export_log)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menubar.addMenu("View")
        dashboard_act = QAction("Dashboard", self)
        dashboard_act.setShortcut("Ctrl+D")
        dashboard_act.triggered.connect(self._show_dashboard)
        view_menu.addAction(dashboard_act)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self._status_state = QLabel("State: IDLE")
        self._status_samples = QLabel("Samples: 0")
        self._status_anomalies = QLabel("Anomalies: 0")
        bar.addWidget(self._status_state)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._status_samples)
        bar.addWidget(QLabel(" | "))
        bar.addWidget(self._status_anomalies)
        self.setStatusBar(bar)

    def _set_state(self, state: AppState) -> None:
        self._state = state
        self._status_state.setText(f"State: {state.name}")

    # ------------------------------------------------------------------
    # Sample handler (single entry point for both sources)
    # ------------------------------------------------------------------

    def _on_new_sample(self, timestamp: float, value: float, is_injected: bool) -> None:
        self._total_samples += 1
        self._status_samples.setText(f"Samples: {self._total_samples}")
        self._detector.feed(timestamp, value, is_injected)
        self._config_panel.update_sample_count(self._total_samples)

    # ------------------------------------------------------------------
    # Session controls
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        if self._state not in (AppState.IDLE, AppState.PAUSED):
            return
        self._generator.start()
        self._set_state(AppState.RUNNING)
        self._config_panel.set_controls_enabled(True)

    def _on_stop(self) -> None:
        if self._state == AppState.RUNNING:
            self._generator.stop()
            self._set_state(AppState.PAUSED)
            self._config_panel.set_controls_enabled(False)
        elif self._state == AppState.SIMULATION_RUNNING:
            self._sim_player.stop()
            self._set_state(AppState.IDLE)
            self._config_panel.set_simulation_controls_enabled(False)

    def _session_reset(self) -> None:
        self._generator.reset()
        self._sim_player.stop()
        self._detector.reset()
        self._vis_panel.clear()
        self._log_panel.clear()
        self._total_samples = 0
        self._status_samples.setText("Samples: 0")
        self._status_anomalies.setText("Anomalies: 0")
        self._set_state(AppState.IDLE)
        self._config_panel.reset_ui()
        self._config_panel.update_sample_count(0)
        self._config_panel.set_controls_enabled(False)
        self._config_panel.set_simulation_controls_enabled(False)

    def _on_model_trained(self) -> None:
        self._config_panel.set_model_status(True)
        if self._state == AppState.TRAINING:
            self._set_state(AppState.RUNNING)
        self._refresh_gmm_overlay()

    def _refresh_gmm_overlay(self) -> None:
        params = self._detector.get_model_params()
        if params:
            self._vis_panel.set_gmm_overlay(
                params["means"], params["stds"], params["weights"],
                params.get("boundary_values", [])
            )

    def _on_retrain(self, start: int, end: int) -> None:
        self._detector.retrain(start, end)
        self._refresh_gmm_overlay()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _on_load_simulation(self, scenario_name: str) -> None:
        from pathlib import Path
        assets_dir = Path(__file__).parent.parent / "assets" / "simulations"
        filename = f"scenario_{scenario_name.lower()}.csv"
        path = assets_dir / filename
        if not path.exists():
            generate_scenarios()
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"Could not find {path}")
            return
        self._session_reset()
        try:
            scenario = SimulationScenario.from_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        speed = self._config_panel.get_speed_multiplier()
        self._sim_player.load(scenario)
        self._sim_player.set_speed(speed)
        self._set_state(AppState.SIMULATION_RUNNING)
        self._config_panel.set_simulation_controls_enabled(True)
        self._sim_player.play()

    def _on_load_simulation_dialog(self) -> None:
        assets_dir = str(Path(__file__).parent.parent / "assets" / "simulations")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Simulation CSV", assets_dir, "CSV Files (*.csv)"
        )
        if not path:
            return
        self._session_reset()
        try:
            scenario = SimulationScenario.from_csv(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._sim_player.load(scenario)
        self._sim_player.set_speed(1.0)
        self._set_state(AppState.SIMULATION_RUNNING)
        self._config_panel.set_simulation_controls_enabled(True)
        self._sim_player.play()

    def _on_simulation_finished(self) -> None:
        self._set_state(AppState.IDLE)
        self._config_panel.set_simulation_controls_enabled(False)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _show_dashboard(self) -> None:
        if self._dashboard is None or not self._dashboard.isVisible():
            self._dashboard = DashboardDialog(
                stats_provider=self._log_panel.get_stats,
                total_samples_provider=lambda: self._total_samples,
                parent=self,
            )
        self._dashboard.show()
        self._dashboard.raise_()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Anomaly Log", "anomaly_log.csv", "CSV Files (*.csv)"
        )
        if path:
            self._log_panel.export_csv(path)
            QMessageBox.information(self, "Export", f"Log exported to:\n{path}")

    # ------------------------------------------------------------------
    # Status bar update
    # ------------------------------------------------------------------

    def _update_status(self, stats: dict) -> None:
        total = stats.get("total_anomalies", 0)
        self._status_anomalies.setText(f"Anomalies: {total}")

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            "<b>Predictive Maintenance Demonstrator</b><br>"
            "Interactive fair presentation tool.<br><br>"
            "Uses GMM-based anomaly detection on live synthetic sensor streams.",
        )
