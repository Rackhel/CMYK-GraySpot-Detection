"""Main PyQt6 window for the CMYK engineering GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from gui.tabs import DataTab, EmbeddingTab, EvaluationTab, OptunaTab, SettingsTab, TrainingTab


def default_config() -> dict[str, Any]:
    """Return GUI defaults without reaching into backend internals."""

    return {
        "data": {"channels": ["Y", "M", "C", "K"], "num_levels": 6},
        "storage": {"labeled_dir": "data_set/labeled"},
        "phase2": {"epochs": 10},
    }


class MainWindow(QMainWindow):
    """Main window responsible only for tab orchestration."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = default_config()
        self.setWindowTitle("CMYK AI - Engineering GUI")
        self.resize(1280, 820)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tab_widget)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._add_tabs()
        self.status.showMessage("Ready")

    def _add_tabs(self) -> None:
        """Create the six contract-defined tabs."""

        self.data_tab = DataTab(self.cfg)
        self.training_tab = TrainingTab(self.cfg)
        self.evaluation_tab = EvaluationTab(self.cfg)
        self.settings_tab = SettingsTab(self.cfg)
        self.optuna_tab = OptunaTab(self.cfg)
        self.embedding_tab = EmbeddingTab(self.cfg, labels_dir=Path("data_set"))

        self.tab_widget.addTab(self.data_tab, "Data")
        self.tab_widget.addTab(self.training_tab, "Training")
        self.tab_widget.addTab(self.evaluation_tab, "Evaluation")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.optuna_tab, "Optuna HPO")
        self.tab_widget.addTab(self.embedding_tab, "Embedding")

    def _on_tab_changed(self, index: int) -> None:
        """Refresh the activated tab and update status text."""

        widget = self.tab_widget.widget(index)
        if hasattr(widget, "refresh"):
            widget.refresh()
        self.status.showMessage(f"{self.tab_widget.tabText(index)} ready")

    def closeEvent(self, event) -> None:
        """Cancel active worker threads before closing."""

        for tab_index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(tab_index)
            worker = getattr(tab, "worker", None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(2000)
        event.accept()
