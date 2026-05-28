"""Training tab — 모델/head/phase 설정 + 학습 실행 + 학습 곡선 시각화.
Model config, head config, phase-specific hyperparameters, training run, learning curves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.components.log_panel import LogPanel
from gui.components.metric_card import MetricCard
from gui.components.progress_panel import ProgressPanel
from gui.components.training_chart import TrainingChart
from gui.services.training_service import TrainingService
from gui.tabs.base_tab import BaseTab
from gui.workers.training_worker import TrainingWorker

_ROOT = Path(__file__).resolve().parents[2]
_SRC_CONFIG = _ROOT / "src" / "config" / "config.json"
_FIELD_W = 180


class TrainingTab(BaseTab):
    """Model + head + phase configuration and training execution with live charts."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = TrainingService()
        self.worker: TrainingWorker | None = None

        # ── 왼쪽 패널: 설정 / Left panel: configuration ──────────────────
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setSpacing(10)
        left_v.addWidget(self._build_backbone_group())
        left_v.addWidget(self._build_head_group())
        left_v.addWidget(self._build_phase_group())
        left_v.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(300)

        # ── 오른쪽 패널: 실행 + 차트 / Right panel: run + chart ──────────
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setSpacing(8)
        right_v.addWidget(self._build_run_group())

        card_row = QHBoxLayout()
        self.val_acc_card = MetricCard("Best Val Acc", "—")
        self.test_acc_card = MetricCard("Test Acc", "—")
        self.mae_card = MetricCard("MAE", "—")
        for c in (self.val_acc_card, self.test_acc_card, self.mae_card):
            card_row.addWidget(c)
        card_row.addStretch()
        right_v.addLayout(card_row)

        self.chart = TrainingChart()
        self.progress = ProgressPanel()
        self.log = LogPanel()
        self.log.setMaximumHeight(80)

        right_v.addWidget(QLabel("<b>학습 곡선 / Learning Curves</b>"))
        right_v.addWidget(self.chart, stretch=1)
        right_v.addWidget(self.progress)
        right_v.addWidget(self.log)

        # ── Splitter ──────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        m = self.cfg.get("model", {})
        p2 = self.cfg.get("phase2", {})
        p0 = self.cfg.get("phase0", {})

        self._backbone.setCurrentText(m.get("backbone", "efficientnet_b0"))
        self._frozen.setChecked(bool(m.get("frozen_backbone", False)))

        self._num_levels.setValue(self.cfg.get("data", {}).get("num_levels", 6))
        self._dropout.setValue(float(p2.get("dropout", 0.3)))

        self._p2_epochs.setValue(int(p2.get("epochs", 30)))
        self._p2_lr.setValue(float(p2.get("learning_rate", 1e-4)))
        self._p2_bs.setValue(int(p2.get("batch_size", 32)))
        self._p2_wd.setValue(float(p2.get("weight_decay", 1e-4)))

        self._p0_epochs.setValue(int(p0.get("epochs", 10)))
        self._p0_lr.setValue(float(p0.get("learning_rate", 1e-3)))
        self._p0_bs.setValue(int(p0.get("batch_size", 16)))
        self._p0_temp.setValue(float(p0.get("temperature", 0.1)))
        self._p0_proj.setValue(int(p0.get("projection_dim", 128)))

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        phase = result.get("phase", "?")
        if phase == 0:
            self.val_acc_card.set_value(f"loss {result.get('val_acc', 0):.4f}")
        else:
            self.val_acc_card.set_value(
                f"{result.get('best_val_acc', result.get('val_acc', 0)):.3f}"
            )
            self.test_acc_card.set_value(f"{result.get('test_acc', 0):.3f}")
            self.mae_card.set_value(f"{result.get('mae', 0):.3f}")
        self.chart.load_history_from_result(result)
        self.chart.render()
        ckpt = result.get("checkpoint_path", result.get("checkpoint", ""))
        self.progress.append_log(f"✅ 완료 — {Path(ckpt).name if ckpt else '—'}")
        self._set_running(False)

    # ── Public API ────────────────────────────────────────────────────────────

    def start_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.progress.append_log("⚠️  이미 실행 중입니다.")
            return
        self._apply_cfg()
        phase = int(self._phase_box.currentData())
        channel = self._channel_box.currentText()
        self.chart.reset()
        self.worker = self.service.start_training(
            self.cfg, phase=phase, channel=channel
        )
        self.worker.progress_updated.connect(self.progress.set_progress)
        self.worker.log_emitted.connect(self._on_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(self._on_error)
        self._set_running(True)
        self.worker.start()

    def stop_training(self) -> None:
        if self.worker is not None:
            self.service.stop_training()
            self.worker = None
        self._set_running(False)

    def save_config(self) -> None:
        try:
            self._apply_cfg()
            src_cfg: dict = json.loads(_SRC_CONFIG.read_text(encoding="utf-8"))
            for key in ("model", "phase2", "phase0", "data"):
                src_cfg[key] = self.cfg.get(key, src_cfg.get(key, {}))
            _SRC_CONFIG.write_text(
                json.dumps(src_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.log.append("✅ 설정 저장 → src/config/config.json")
        except Exception as exc:
            self.log.append(f"❌ {exc}")

    # ── Private — build UI ────────────────────────────────────────────────────

    def _build_backbone_group(self) -> QGroupBox:
        g = QGroupBox("Backbone 설정")
        f = self._form(g)

        self._backbone = QComboBox()
        self._backbone.addItems(["efficientnet_b0", "resnet50"])
        self._backbone.setCurrentText(
            self.cfg.get("model", {}).get("backbone", "efficientnet_b0")
        )
        self._backbone.setMaximumWidth(_FIELD_W)

        self._frozen = QCheckBox("Frozen (백본 가중치 고정)")
        self._frozen.setChecked(
            bool(self.cfg.get("model", {}).get("frozen_backbone", False))
        )

        save_btn = QPushButton("💾  설정 저장")
        save_btn.setMaximumWidth(140)
        save_btn.clicked.connect(self.save_config)

        f.addRow("Backbone", self._backbone)
        f.addRow(self._frozen)
        f.addRow(save_btn)
        return g

    def _build_head_group(self) -> QGroupBox:
        g = QGroupBox("Head 설정 / Classification Head")
        f = self._form(g)

        d = self.cfg.get("data", {})
        p2 = self.cfg.get("phase2", {})

        self._num_levels = QSpinBox()
        self._num_levels.setRange(2, 20)
        self._num_levels.setValue(d.get("num_levels", 6))
        self._num_levels.setMaximumWidth(100)

        self._dropout = QDoubleSpinBox()
        self._dropout.setRange(0.0, 0.9)
        self._dropout.setSingleStep(0.05)
        self._dropout.setDecimals(2)
        self._dropout.setValue(float(p2.get("dropout", 0.3)))
        self._dropout.setMaximumWidth(100)

        f.addRow("Num Levels (클래스 수)", self._num_levels)
        f.addRow("Dropout Rate", self._dropout)
        return g

    def _build_phase_group(self) -> QGroupBox:
        g = QGroupBox("Phase 하이퍼파라미터")
        outer = QVBoxLayout(g)

        # Phase 2 — Supervised
        p2_box = QGroupBox("Phase 2 — Supervised Classification")
        f2 = self._form(p2_box)
        p2 = self.cfg.get("phase2", {})

        self._p2_epochs = QSpinBox()
        self._p2_epochs.setRange(1, 500)
        self._p2_epochs.setValue(int(p2.get("epochs", 30)))
        self._p2_epochs.setMaximumWidth(100)

        self._p2_lr = QDoubleSpinBox()
        self._p2_lr.setRange(1e-6, 1.0)
        self._p2_lr.setDecimals(6)
        self._p2_lr.setSingleStep(1e-4)
        self._p2_lr.setValue(float(p2.get("learning_rate", 1e-4)))
        self._p2_lr.setMaximumWidth(130)

        self._p2_bs = QSpinBox()
        self._p2_bs.setRange(1, 512)
        self._p2_bs.setValue(int(p2.get("batch_size", 32)))
        self._p2_bs.setMaximumWidth(100)

        self._p2_wd = QDoubleSpinBox()
        self._p2_wd.setRange(0, 1)
        self._p2_wd.setDecimals(6)
        self._p2_wd.setSingleStep(1e-5)
        self._p2_wd.setValue(float(p2.get("weight_decay", 1e-4)))
        self._p2_wd.setMaximumWidth(130)

        f2.addRow("Epochs", self._p2_epochs)
        f2.addRow("Learning Rate", self._p2_lr)
        f2.addRow("Batch Size", self._p2_bs)
        f2.addRow("Weight Decay", self._p2_wd)

        # Phase 0 — SimCLR
        p0_box = QGroupBox("Phase 0 — SimCLR Contrastive")
        f0 = self._form(p0_box)
        p0 = self.cfg.get("phase0", {})

        self._p0_epochs = QSpinBox()
        self._p0_epochs.setRange(1, 500)
        self._p0_epochs.setValue(int(p0.get("epochs", 10)))
        self._p0_epochs.setMaximumWidth(100)

        self._p0_lr = QDoubleSpinBox()
        self._p0_lr.setRange(1e-6, 1.0)
        self._p0_lr.setDecimals(6)
        self._p0_lr.setSingleStep(1e-4)
        self._p0_lr.setValue(float(p0.get("learning_rate", 1e-3)))
        self._p0_lr.setMaximumWidth(130)

        self._p0_bs = QSpinBox()
        self._p0_bs.setRange(1, 512)
        self._p0_bs.setValue(int(p0.get("batch_size", 16)))
        self._p0_bs.setMaximumWidth(100)

        self._p0_temp = QDoubleSpinBox()
        self._p0_temp.setRange(0.01, 1.0)
        self._p0_temp.setDecimals(3)
        self._p0_temp.setSingleStep(0.01)
        self._p0_temp.setValue(float(p0.get("temperature", 0.1)))
        self._p0_temp.setMaximumWidth(100)

        self._p0_proj = QSpinBox()
        self._p0_proj.setRange(32, 1024)
        self._p0_proj.setValue(int(p0.get("projection_dim", 128)))
        self._p0_proj.setMaximumWidth(100)

        f0.addRow("Epochs", self._p0_epochs)
        f0.addRow("Learning Rate", self._p0_lr)
        f0.addRow("Batch Size", self._p0_bs)
        f0.addRow("Temperature", self._p0_temp)
        f0.addRow("Projection Dim", self._p0_proj)

        outer.addWidget(p2_box)
        outer.addWidget(p0_box)
        return g

    def _build_run_group(self) -> QGroupBox:
        g = QGroupBox("학습 실행 / Run Training")
        v = QVBoxLayout(g)

        self._phase_box = QComboBox()
        self._phase_box.addItem("Phase 2 — Supervised", 2)
        self._phase_box.addItem("Phase 0 — SimCLR", 0)
        self._phase_box.setMaximumWidth(200)

        self._channel_box = QComboBox()
        self._channel_box.addItems(["Y", "M", "C", "K"])
        self._channel_box.setMaximumWidth(80)

        self._start_btn = QPushButton("▶  학습 시작 / Start Training")
        self._stop_btn = QPushButton("■  중지 / Stop")
        self._start_btn.clicked.connect(self.start_training)
        self._stop_btn.clicked.connect(self.stop_training)
        self._stop_btn.setEnabled(False)

        row = QHBoxLayout()
        row.addWidget(QLabel("Phase"))
        row.addWidget(self._phase_box)
        row.addWidget(QLabel("Channel"))
        row.addWidget(self._channel_box)
        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        row.addStretch()
        v.addLayout(row)
        return g

    # ── Private — helpers ─────────────────────────────────────────────────────

    def _apply_cfg(self) -> None:
        self.cfg.setdefault("model", {}).update(
            {
                "backbone": self._backbone.currentText(),
                "frozen_backbone": self._frozen.isChecked(),
            }
        )
        self.cfg.setdefault("data", {})["num_levels"] = self._num_levels.value()
        self.cfg.setdefault("phase2", {}).update(
            {
                "epochs": self._p2_epochs.value(),
                "learning_rate": self._p2_lr.value(),
                "batch_size": self._p2_bs.value(),
                "weight_decay": self._p2_wd.value(),
                "dropout": self._dropout.value(),
            }
        )
        self.cfg.setdefault("phase0", {}).update(
            {
                "epochs": self._p0_epochs.value(),
                "learning_rate": self._p0_lr.value(),
                "batch_size": self._p0_bs.value(),
                "temperature": self._p0_temp.value(),
                "projection_dim": self._p0_proj.value(),
            }
        )

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)

    def _on_log(self, line: str) -> None:
        self.progress.append_log(line)
        if self.chart.parse_log_line(line):
            self.chart.render()

    def _on_error(self, message: str) -> None:
        self.progress.append_log(f"ERROR: {message}")
        self._set_running(False)
        self.worker = None

    @staticmethod
    def _form(parent: QGroupBox) -> QFormLayout:
        f = QFormLayout(parent)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        f.setContentsMargins(8, 6, 8, 6)
        return f
