"""Embedding tab — t-SNE 시각화 및 라벨 교정.
t-SNE embedding visualization and versioned label correction.

Contract: Contract_gui.md §3.6  /  SSOT_GUI.md §6.6
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gui.components.log_panel import LogPanel
from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.services.embedding_service import EmbeddingService
from gui.workers.embedding_worker import EmbeddingWorker
from .base_tab import BaseTab


class EmbeddingTab(BaseTab):
    """Embedding visualization and versioned label correction tab."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        labels_dir: Path | None = None,
        settings_tab=None,          # SettingsTab 참조 (checkpoint 경로 공유)
    ) -> None:
        super().__init__(cfg)
        self.service = EmbeddingService()
        self.worker: EmbeddingWorker | None = None
        self._settings_tab = settings_tab
        self.labels_dir = Path(labels_dir) if labels_dir is not None else Path("data_set")

        # 현재 scatter 클릭으로 선택된 이미지 경로
        self._selected_path: str = ""
        # t-SNE 결과 이미지 경로 목록 (worker finished 시 저장)
        self._embedding_paths: list[str] = []

        # ── 컨트롤 / Controls ─────────────────────────────────────────────
        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K"])

        run_btn  = QPushButton("▶  Extract Embeddings")
        stop_btn = QPushButton("■  Stop")
        run_btn.clicked.connect(self.start_embedding)
        stop_btn.clicked.connect(self.stop_embedding)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel"))
        ctrl_row.addWidget(self.channel_box)
        ctrl_row.addWidget(run_btn)
        ctrl_row.addWidget(stop_btn)
        ctrl_row.addStretch()

        # ── Scatter 차트 / Scatter chart ──────────────────────────────────
        self.chart = PlotlyWidget()
        # PlotlyWidget이 점 클릭 시그널을 지원하면 연결한다
        if hasattr(self.chart, "point_clicked"):
            self.chart.point_clicked.connect(self._on_point_clicked)

        # ── 라벨 교정 / Label correction ──────────────────────────────────
        self.level_box = QSpinBox()
        self.level_box.setRange(0, self.cfg.get("data", {}).get("num_levels", 6) - 1)

        self._selected_label = QLabel("Selected: (none)")

        self.save_btn = QPushButton("💾  Save Label Correction")
        self.save_btn.clicked.connect(self._save_correction)

        correction_row = QHBoxLayout()
        correction_row.addWidget(QLabel("New Level"))
        correction_row.addWidget(self.level_box)
        correction_row.addWidget(self.save_btn)
        correction_row.addStretch()

        # ── 진행 / Progress ───────────────────────────────────────────────
        self.progress = ProgressPanel()
        self.log = LogPanel()

        # ── 최상위 레이아웃 / Top-level layout ───────────────────────────
        layout = QVBoxLayout(self)
        layout.addLayout(ctrl_row)
        layout.addWidget(self.chart, stretch=1)
        layout.addWidget(self._selected_label)
        layout.addLayout(correction_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        """탭 활성화 시 상태 갱신."""

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        """EmbeddingWorker 완료 — scatter 표시 및 경로 목록 저장."""
        points = result.get("embeddings_2d", [])
        labels = result.get("labels", [])
        self._embedding_paths = result.get("paths", [])
        self.chart.show_scatter(points, labels)
        self.log.append(f"✅ {len(points)}개 포인트 시각화 완료")

    # ── Public API ────────────────────────────────────────────────────────────

    def start_embedding(self) -> None:
        """중복 실행 방지 후 EmbeddingWorker 시작."""
        if self.worker is not None and self.worker.isRunning():
            self.log.append("⚠️  Embedding extraction already running — stop it first")
            return

        ckpt = self._get_checkpoint()
        if not ckpt:
            self.log.append("⚠️  Settings 탭에서 체크포인트를 먼저 지정하세요")

        self.worker = self.service.start_embedding(
            self.cfg, self.channel_box.currentText(), ckpt
        )
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(
            lambda msg: self.log.append(f"ERROR: {msg}")
        )
        self.worker.start()

    def stop_embedding(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(2000)
        self.log.append("Stopped.")

    def save_label_correction(self, path: str, new_level: int) -> None:
        """path/level 쌍을 새 버전 CSV(labels_vN.csv)에 기록한다.

        Contract §3.6 — 버전 증가 방식 저장.
        """
        if self.labels_dir is None:
            raise ValueError("labels_dir not configured")
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 다음 버전 결정 / Determine next version
        existing = list(self.labels_dir.glob("labels_v*.csv"))
        max_v = -1
        for p in existing:
            try:
                v = int(p.stem.split("_v")[-1])
                if v > max_v:
                    max_v = v
            except Exception:
                continue
        next_v = max_v + 1
        out = self.labels_dir / f"labels_v{next_v}.csv"

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "level"])
            writer.writerow([path, str(new_level)])

        self.log.append(f"✅ Saved {out.name}  ({Path(path).name} → L{new_level})")

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_point_clicked(self, index: int) -> None:
        """PlotlyWidget에서 scatter 포인트 클릭 시 호출된다."""
        if 0 <= index < len(self._embedding_paths):
            self._selected_path = self._embedding_paths[index]
            self._selected_label.setText(f"Selected: {Path(self._selected_path).name}")
        else:
            self._selected_path = ""
            self._selected_label.setText("Selected: (none)")

    def _save_correction(self) -> None:
        """Save 버튼 핸들러 — 선택된 경로가 없으면 파일 다이얼로그 열기."""
        path = self._selected_path
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image to Correct",
                str(self.labels_dir),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
        if not path:
            self.log.append("⚠️  저장할 이미지를 선택하세요 / No image selected")
            return
        self.save_label_correction(path, self.level_box.value())

    def _get_checkpoint(self) -> str:
        """SettingsTab에서 체크포인트 경로를 읽는다 (없으면 빈 문자열)."""
        if self._settings_tab is not None and hasattr(self._settings_tab, "get_checkpoint_path"):
            return self._settings_tab.get_checkpoint_path()
        return ""
