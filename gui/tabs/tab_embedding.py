from pathlib import Path
import csv
from typing import Any

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout

from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.services.embedding_service import EmbeddingService
from gui.workers.embedding_worker import EmbeddingWorker
from .base_tab import BaseTab


class EmbeddingTab(BaseTab):
    """Embedding visualization and versioned label correction tab."""

    def __init__(self, cfg: dict[str, Any] | None = None, labels_dir: Path | None = None) -> None:
        super().__init__(cfg)
        self.service = EmbeddingService()
        self.worker: EmbeddingWorker | None = None
        self.labels_dir = Path(labels_dir) if labels_dir is not None else Path("data_set")
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K"])
        self.level_box = QSpinBox()
        self.level_box.setRange(1, 6)
        self.chart = PlotlyWidget()
        self.progress = ProgressPanel()
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channel"))
        controls.addWidget(self.channel_box)
        run_btn = QPushButton("Extract Embeddings")
        run_btn.clicked.connect(self.start_embedding)
        controls.addWidget(run_btn)
        layout.addLayout(controls)
        layout.addWidget(self.chart)
        layout.addWidget(QLabel("Correction Level"))
        layout.addWidget(self.level_box)
        self.save_btn = QPushButton("Save Label Correction")
        self.save_btn.clicked.connect(lambda: self.save_label_correction("sample.png", self.level_box.value()))
        layout.addWidget(self.save_btn)
        layout.addWidget(self.progress)

    def refresh(self) -> None:
        """Refresh label review state."""

    def start_embedding(self) -> None:
        """Start embedding extraction through the service boundary."""

        self.worker = self.service.start_embedding(self.cfg, self.channel_box.currentText(), "")
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(lambda msg: self.progress.append_log(f"ERROR: {msg}"))
        self.worker.start()

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        points = result.get("embeddings_2d", [])
        labels = result.get("labels", [])
        self.chart.show_scatter(points, labels)

    def save_label_correction(self, path: str, new_level: int) -> None:
        """Append or create a new labels_vN.csv file with the correction.

        This implements the version-increment behavior described in the Contract and BDD.
        """
        if self.labels_dir is None:
            raise ValueError("labels_dir not configured")
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Determine next version
        existing = list(self.labels_dir.glob("labels_v*.csv"))
        max_v = -1
        for p in existing:
            name = p.stem
            try:
                v = int(name.split("_v")[-1])
                if v > max_v:
                    max_v = v
            except Exception:
                continue
        next_v = max_v + 1
        out = self.labels_dir / f"labels_v{next_v}.csv"

        # Write a minimal CSV with path and level
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "level"])
            writer.writerow([path, str(new_level)])
        self.progress.append_log(f"Saved {out.name}")
