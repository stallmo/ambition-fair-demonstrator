"""Entry point for the Predictive Maintenance Demonstrator."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

# Configure matplotlib for dark theme before any windows open
plt.rcParams.update({
    "axes.facecolor": "#2a2a3e",
    "figure.facecolor": "#1e1e2e",
    "text.color": "#f8f8f2",
    "axes.labelcolor": "#f8f8f2",
    "xtick.color": "#f8f8f2",
    "ytick.color": "#f8f8f2",
    "axes.edgecolor": "#44475a",
    "grid.color": "#44475a",
    "axes.grid": False,
    "figure.dpi": 100,
})

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load QSS stylesheet
    qss_path = Path(__file__).parent / "assets" / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
