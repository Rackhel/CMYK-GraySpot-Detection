from typing import Any

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .base_tab import BaseTab
from ..components.progress_panel import ProgressPanel
from ..services.training_service import TrainingService
from ..workers.training_worker import TrainingWorker


class TrainingTab(BaseTab):
    """Training controls and progress display."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = TrainingService()
        self.worker: TrainingWorker | None = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.phase_box = QComboBox()
        self.phase_box.addItems(["2", "0"])
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K", "All"])
        controls.addWidget(QLabel("Phase"))
        controls.addWidget(self.phase_box)
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_box)
        layout.addLayout(controls)

        self.progress = ProgressPanel()
        layout.addWidget(self.progress)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)

    def refresh(self) -> None:
        """Refresh selectable state."""

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        """Display completed training metrics."""

        self.progress.append_log(f"Training complete: {result}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

    def start_training(self) -> None:
        """Start training through the service boundary."""

        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = self.service.start_training(
            self.cfg,
            phase=int(self.phase_box.currentText()),
            channel=self.channel_box.currentText(),
        )
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def stop_training(self) -> None:
        """Stop active training safely."""

        if self.worker is not None:
            self.service.stop_training()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, message: str) -> None:
        self.progress.append_log(f"ERROR: {message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
