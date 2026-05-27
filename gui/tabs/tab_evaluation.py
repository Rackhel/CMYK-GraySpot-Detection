"""Evaluation tab — 모델 평가 및 단일 이미지 추론.
Run dataset evaluation and single-image inference with preview.

Contract: Contract_gui.md §3.3  /  SSOT_GUI.md §6.3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.components.image_viewer import ImageViewer
from gui.components.log_panel import LogPanel
from gui.components.metric_card import MetricCard
from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.services.evaluation_service import EvaluationService
from gui.tabs.base_tab import BaseTab
from gui.workers.evaluation_worker import EvaluationWorker
from gui.workers.inference_worker import InferenceWorker


class EvaluationTab(BaseTab):
    """Run dataset evaluation and single-image inference.

    - 데이터셋 평가: EvaluationWorker → accuracy + confusion matrix
    - 이미지 추론: InferenceWorker → predicted level + confidence + top-3
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        settings_tab=None,          # SettingsTab 참조 (checkpoint 경로 공유)
    ) -> None:
        super().__init__(cfg)
        self.service = EvaluationService()
        self.eval_worker: EvaluationWorker | None = None
        self.infer_worker: InferenceWorker | None = None
        self._settings_tab = settings_tab
        self._selected_image: str = ""

        # ── 데이터셋 평가 섹션 / Dataset evaluation section ──────────────────
        eval_group = QGroupBox("Dataset Evaluation")
        eval_v = QVBoxLayout(eval_group)

        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K"])

        run_btn  = QPushButton("▶  Run Evaluation")
        stop_btn = QPushButton("■  Stop")
        run_btn.clicked.connect(self.start_evaluation)
        stop_btn.clicked.connect(self.stop_evaluation)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Channel"))
        ctrl_row.addWidget(self.channel_box)
        ctrl_row.addWidget(run_btn)
        ctrl_row.addWidget(stop_btn)
        ctrl_row.addStretch()

        self.accuracy_card = MetricCard("Accuracy", "-")
        self.chart = PlotlyWidget()
        self.progress = ProgressPanel()

        eval_v.addLayout(ctrl_row)
        eval_v.addWidget(self.accuracy_card)
        eval_v.addWidget(self.chart)
        eval_v.addWidget(self.progress)

        # ── 단일 이미지 추론 섹션 / Single-image inference section ───────────
        infer_group = QGroupBox("Single Image Inference")
        infer_v = QVBoxLayout(infer_group)

        browse_btn = QPushButton("📂  Browse Image…")
        browse_btn.clicked.connect(self._browse_image)
        infer_btn  = QPushButton("▶  Run Inference")
        infer_btn.clicked.connect(self.start_inference)

        img_ctrl_row = QHBoxLayout()
        img_ctrl_row.addWidget(browse_btn)
        img_ctrl_row.addWidget(infer_btn)
        img_ctrl_row.addStretch()

        self.image_viewer = ImageViewer()
        self.image_viewer.setFixedHeight(160)

        # 결과 레이블
        self._pred_label  = QLabel("Predicted Level: —")
        self._conf_label  = QLabel("Confidence: —")
        self._top3_label  = QLabel("Top-3: —")
        self._pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._top3_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for lbl in (self._pred_label, self._conf_label, self._top3_label):
            lbl.setStyleSheet("font-size: 13px; padding: 4px;")
        self._pred_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #2563EB; padding: 4px;"
        )

        self.infer_progress = ProgressPanel()
        self.infer_log = LogPanel()

        infer_v.addLayout(img_ctrl_row)
        infer_v.addWidget(self.image_viewer)
        infer_v.addWidget(self._pred_label)
        infer_v.addWidget(self._conf_label)
        infer_v.addWidget(self._top3_label)
        infer_v.addWidget(self.infer_progress)
        infer_v.addWidget(self.infer_log)

        # ── 최상위 스크롤 레이아웃 ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        main_v = QVBoxLayout(content)
        main_v.addWidget(eval_group)
        main_v.addWidget(infer_group)
        main_v.addStretch()
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        """탭 활성화 시 상태 갱신 (BaseTab 인터페이스)."""

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        """EvaluationWorker 완료 처리."""
        self.accuracy_card.set_value(f"{result.get('accuracy', 0):.3f}")
        matrix = result.get("confusion_matrix")
        if matrix:
            self.chart.show_matrix(matrix)

    # ── Dataset evaluation ────────────────────────────────────────────────────

    def start_evaluation(self) -> None:
        """중복 실행 방지 후 EvaluationWorker 시작."""
        if self.eval_worker is not None and self.eval_worker.isRunning():
            self.progress.append_log("⚠️  Evaluation already running — stop it first")
            return

        ckpt = self._get_checkpoint()
        self.eval_worker = self.service.start_evaluation(
            self.cfg, self.channel_box.currentText(), ckpt
        )
        self.eval_worker.progress_updated.connect(self.progress.set_progress)
        self.eval_worker.log_emitted.connect(self.progress.append_log)
        self.eval_worker.finished.connect(self.on_worker_finished)
        self.eval_worker.error_occurred.connect(
            lambda msg: self.progress.append_log(f"ERROR: {msg}")
        )
        self.eval_worker.start()

    def stop_evaluation(self) -> None:
        self.service.stop_evaluation()
        self.progress.append_log("Evaluation stopped.")

    # ── Single-image inference ────────────────────────────────────────────────

    def start_inference(self) -> None:
        """중복 실행 방지 후 InferenceWorker 시작."""
        if not self._selected_image:
            self.infer_log.append("⚠️  먼저 이미지를 선택하세요 / Select an image first")
            return
        if self.infer_worker is not None and self.infer_worker.isRunning():
            self.infer_log.append("⚠️  Inference already running")
            return

        ckpt = self._get_checkpoint()
        if not ckpt:
            self.infer_log.append("⚠️  Settings 탭에서 체크포인트를 먼저 지정하세요")

        self.infer_worker = InferenceWorker(self.cfg, self._selected_image, ckpt)
        self.infer_worker.progress_updated.connect(self.infer_progress.set_progress)
        self.infer_worker.log_emitted.connect(self.infer_progress.append_log)
        self.infer_worker.finished.connect(self._on_inference_finished)
        self.infer_worker.error_occurred.connect(
            lambda msg: self.infer_log.append(f"ERROR: {msg}")
        )
        self.infer_worker.start()

    def _on_inference_finished(self, result: dict[str, Any]) -> None:
        pred   = result.get("pred_level", "?")
        conf   = result.get("confidence", 0.0)
        top3   = result.get("top3", [])

        self._pred_label.setText(f"Predicted Level: {pred}")
        self._conf_label.setText(f"Confidence: {conf:.1%}")

        if top3:
            top3_str = "  |  ".join(f"L{lvl}: {p:.1%}" for lvl, p in top3)
            self._top3_label.setText(f"Top-3: {top3_str}")

        self.infer_log.append(
            f"✅ Level {pred}  conf={conf:.3f}  "
            f"image={Path(result.get('image_path', '')).name}"
        )

    def _browse_image(self) -> None:
        """파일 다이얼로그로 이미지 선택."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Inference",
            "data_set/labeled",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self._selected_image = path
            self.image_viewer.load_image(path)
            self._pred_label.setText("Predicted Level: —")
            self._conf_label.setText("Confidence: —")
            self._top3_label.setText("Top-3: —")
            self.infer_log.append(f"Selected: {Path(path).name}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_checkpoint(self) -> str:
        """SettingsTab에서 체크포인트 경로를 읽는다 (없으면 빈 문자열)."""
        if self._settings_tab is not None and hasattr(self._settings_tab, "get_checkpoint_path"):
            return self._settings_tab.get_checkpoint_path()
        return ""
