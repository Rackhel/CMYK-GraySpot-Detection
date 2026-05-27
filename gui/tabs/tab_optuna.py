"""Optuna HPO tab."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout

from gui.components.metric_card import MetricCard
from gui.components.progress_panel import ProgressPanel
from gui.services.tuning_service import TuningService
from gui.tabs.base_tab import BaseTab
from gui.workers.tuning_worker import TuningWorker


class OptunaTab(BaseTab):
    """Hyperparameter tuning controls and best-trial summary."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = TuningService()
        self.worker: TuningWorker | None = None
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K"])
        self.trials = QSpinBox()
        self.trials.setRange(1, 200)
        self.trials.setValue(10)
        self.best_card = MetricCard("Best Value", "-")
        self.progress = ProgressPanel()
        start_button = QPushButton("Start HPO")
        start_button.clicked.connect(self.start_tuning)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_box)
        controls.addWidget(QLabel("Trials"))
        controls.addWidget(self.trials)
        controls.addWidget(start_button)
        layout.addLayout(controls)
        layout.addWidget(self.best_card)
        layout.addWidget(self.progress)

    def start_tuning(self) -> None:
        """Start tuning through the service boundary."""

        self.worker = self.service.start_tuning(self.cfg, self.channel_box.currentText(), self.trials.value())
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(lambda msg: self.progress.append_log(f"ERROR: {msg}"))
        self.worker.start()

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        self.best_card.set_value(f"{result.get('best_value', 0):.4f}")
        self.progress.append_log(f"Best params: {result.get('best_params', {})}")
