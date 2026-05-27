"""Evaluation tab."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.components.metric_card import MetricCard
from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.services.evaluation_service import EvaluationService
from gui.tabs.base_tab import BaseTab
from gui.workers.evaluation_worker import EvaluationWorker


class EvaluationTab(BaseTab):
    """Run evaluation and display report-oriented metrics."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = EvaluationService()
        self.worker: EvaluationWorker | None = None
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K"])
        self.accuracy_card = MetricCard("Accuracy", "-")
        self.chart = PlotlyWidget()
        self.progress = ProgressPanel()
        run_button = QPushButton("Run Evaluation")
        run_button.clicked.connect(self.start_evaluation)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_box)
        controls.addWidget(run_button)
        layout.addLayout(controls)
        layout.addWidget(self.accuracy_card)
        layout.addWidget(self.chart)
        layout.addWidget(self.progress)

    def start_evaluation(self) -> None:
        """Start evaluation through the service boundary."""

        self.worker = self.service.start_evaluation(self.cfg, self.channel_box.currentText(), "")
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(lambda msg: self.progress.append_log(f"ERROR: {msg}"))
        self.worker.start()

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        self.accuracy_card.set_value(f"{result.get('accuracy', 0):.3f}")
        matrix = result.get("confusion_matrix")
        if matrix:
            self.chart.show_matrix(matrix)
