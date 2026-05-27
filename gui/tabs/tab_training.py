"""Training tab — 모델 설정 + Phase 0/2 학습 실행.
Model configuration and Phase 0 / Phase 2 training controls.

Contract: Contract_gui.md §3.2  /  SSOT_GUI.md §6.2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.components.log_panel import LogPanel
from gui.components.metric_card import MetricCard
from gui.components.progress_panel import ProgressPanel
from gui.services.training_service import TrainingService
from gui.tabs.base_tab import BaseTab
from gui.workers.training_worker import TrainingWorker

_ROOT       = Path(__file__).resolve().parents[2]
_SRC_CONFIG = _ROOT / "src" / "config" / "config.json"

_FIELD_W = 180


class TrainingTab(BaseTab):
    """Model configuration and training run controls."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = TrainingService()
        self.worker: TrainingWorker | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── 모델 설정 / Model configuration ──────────────────────────────
        layout.addWidget(self._build_model_group())

        # ── 학습 실행 컨트롤 / Run controls ──────────────────────────────
        run_group = QGroupBox("Run Training")
        run_v = QVBoxLayout(run_group)

        self.phase_box = QComboBox()
        self.phase_box.addItems(["2 — Supervised", "0 — SimCLR"])
        self.phase_box.setMaximumWidth(160)

        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K"])
        self.channel_box.setMaximumWidth(80)

        self.start_btn = QPushButton("▶  Start Training")
        self.stop_btn  = QPushButton("■  Stop")
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Phase"))
        ctrl_row.addWidget(self.phase_box)
        ctrl_row.addWidget(QLabel("Channel"))
        ctrl_row.addWidget(self.channel_box)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch()
        run_v.addLayout(ctrl_row)
        layout.addWidget(run_group)

        # ── 결과 카드 / Result cards ──────────────────────────────────────
        card_row = QHBoxLayout()
        self.val_acc_card  = MetricCard("Val Acc", "-")
        self.test_acc_card = MetricCard("Test Acc", "-")
        card_row.addWidget(self.val_acc_card)
        card_row.addWidget(self.test_acc_card)
        card_row.addStretch()
        layout.addLayout(card_row)

        # ── 진행 / Progress ───────────────────────────────────────────────
        self.progress = ProgressPanel()
        layout.addWidget(self.progress, stretch=1)

        self.log = LogPanel()
        self.log.setMaximumHeight(80)
        layout.addWidget(self.log)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        """cfg에서 모델 설정을 다시 읽는다."""
        m = self.cfg.get("model", {})
        self._backbone.setCurrentText(m.get("backbone", "efficientnet_b0"))
        self._frozen.setChecked(bool(m.get("frozen_backbone", False)))

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        phase = result.get("phase", "?")
        if phase == 0:
            self.val_acc_card.set_value(f"loss {result.get('val_acc', 0):.4f}")
        else:
            self.val_acc_card.set_value(f"{result.get('val_acc', 0):.3f}")
            self.test_acc_card.set_value(f"{result.get('test_acc', 0):.3f}")
        ckpt = result.get("checkpoint", "")
        self.progress.append_log(f"✅ 완료 — checkpoint: {Path(ckpt).name if ckpt else '-'}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.progress.append_log("⚠️ Training already running — stop it first")
            return

        # 폼에서 읽은 모델 설정을 cfg에 반영한 뒤 worker에 전달
        self._apply_model_to_cfg()

        phase   = int(self.phase_box.currentText()[0])
        channel = self.channel_box.currentText()

        self.worker = self.service.start_training(self.cfg, phase=phase, channel=channel)
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self.progress.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(self._on_error)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def stop_training(self) -> None:
        if self.worker is not None:
            self.service.stop_training()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def save_model_config(self) -> None:
        """backbone / frozen_backbone을 src/config/config.json에 저장."""
        try:
            src_cfg: dict = json.loads(_SRC_CONFIG.read_text(encoding="utf-8"))
            src_cfg.setdefault("model", {}).update({
                "backbone":        self._backbone.currentText(),
                "frozen_backbone": self._frozen.isChecked(),
            })
            _SRC_CONFIG.write_text(
                json.dumps(src_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.cfg.update(src_cfg)
            self.log.append("✅ Model config saved → src/config/config.json")
        except Exception as exc:
            self.log.append(f"❌ Save failed: {exc}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_model_group(self) -> QGroupBox:
        g = QGroupBox("Model Configuration")
        f = self._make_form(g)

        m = self.cfg.get("model", {})

        self._backbone = QComboBox()
        self._backbone.addItems(["efficientnet_b0", "resnet50"])
        self._backbone.setCurrentText(m.get("backbone", "efficientnet_b0"))
        self._backbone.setMaximumWidth(_FIELD_W)

        self._frozen = QCheckBox("Frozen backbone")
        self._frozen.setChecked(bool(m.get("frozen_backbone", False)))

        save_btn = QPushButton("Save Model Config")
        save_btn.setMaximumWidth(160)
        save_btn.clicked.connect(self.save_model_config)

        f.addRow("Backbone", self._backbone)
        f.addRow(self._frozen)     # full-span
        f.addRow(save_btn)         # full-span
        return g

    def _apply_model_to_cfg(self) -> None:
        """학습 시작 전 폼 값을 in-memory cfg에 반영 (파일 저장 없음)."""
        self.cfg.setdefault("model", {}).update({
            "backbone":        self._backbone.currentText(),
            "frozen_backbone": self._frozen.isChecked(),
        })

    def _on_error(self, message: str) -> None:
        self.progress.append_log(f"ERROR: {message}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None

    @staticmethod
    def _make_form(parent: QGroupBox) -> QFormLayout:
        f = QFormLayout(parent)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        f.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        f.setContentsMargins(8, 6, 8, 6)
        return f
