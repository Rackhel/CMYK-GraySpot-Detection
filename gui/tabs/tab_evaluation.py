"""Evaluation tab — F1/MAE/Confusion Matrix, 채널별/전체 평가.
Dataset evaluation: Accuracy, F1, MAE, Confusion Matrix per channel + overall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.components.log_panel import LogPanel
from gui.components.metric_card import MetricCard
from gui.components.plotly_widget import PlotlyWidget
from gui.components.progress_panel import ProgressPanel
from gui.services.evaluation_service import EvaluationService
from gui.tabs.base_tab import BaseTab
from gui.workers.evaluation_worker import EvaluationWorker
from gui.workers._ckpt_utils import auto_find_checkpoint

_CHANNELS = ["Y", "M", "C", "K"]


class EvaluationTab(BaseTab):
    """Dataset evaluation with F1, MAE, Confusion Matrix per channel."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.service = EvaluationService()
        self.eval_worker: EvaluationWorker | None = None
        self._results: dict[str, dict] = {}   # channel → metrics dict

        # ── 상단 컨트롤 / Top controls ────────────────────────────────────
        ctrl_group = QGroupBox("평가 실행 / Run Evaluation")
        ctrl_v = QVBoxLayout(ctrl_group)

        self.channel_box = QComboBox()
        self.channel_box.addItems(["Y", "M", "C", "K", "전체 (All)"])
        self.channel_box.setMaximumWidth(180)

        self._run_btn  = QPushButton("▶  평가 실행 / Run Evaluation")
        self._stop_btn = QPushButton("■  중지 / Stop")
        self._run_btn.clicked.connect(self.start_evaluation)
        self._stop_btn.clicked.connect(self.stop_evaluation)
        self._stop_btn.setEnabled(False)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("채널 / Channel:"))
        ctrl_row.addWidget(self.channel_box)
        ctrl_row.addWidget(self._run_btn)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch()
        ctrl_v.addLayout(ctrl_row)

        # ── 메트릭 카드 / Metric cards ────────────────────────────────────
        card_row = QHBoxLayout()
        self.acc_card  = MetricCard("Accuracy", "—")
        self.f1_card   = MetricCard("Macro F1",  "—")
        self.mae_card  = MetricCard("MAE",       "—")
        self.n_card    = MetricCard("Samples",   "—")
        for c in (self.acc_card, self.f1_card, self.mae_card, self.n_card):
            card_row.addWidget(c)
        card_row.addStretch()

        # ── 채널별 비교 테이블 / Per-channel comparison table ─────────────
        self._ch_table = QTableWidget(len(_CHANNELS), 4)
        self._ch_table.setHorizontalHeaderLabels(["Channel", "Accuracy", "Macro F1", "MAE"])
        self._ch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._ch_table.setMaximumHeight(160)
        for r, ch in enumerate(_CHANNELS):
            self._ch_table.setItem(r, 0, QTableWidgetItem(ch))
            for c in range(1, 4):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._ch_table.setItem(r, c, item)
        self._ch_table.resizeColumnsToContents()

        # ── Confusion Matrix 차트 / Confusion matrix chart ────────────────
        self.chart = PlotlyWidget()
        self.chart.setMinimumHeight(320)

        # ── 진행 / Progress ───────────────────────────────────────────────
        self.progress = ProgressPanel()
        self.log      = LogPanel()
        self.log.setMaximumHeight(70)

        # ── 스크롤 레이아웃 / Scroll layout ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)
        v.setSpacing(10)
        v.addWidget(ctrl_group)
        v.addLayout(card_row)
        v.addWidget(QLabel("<b>채널별 비교 / Per-Channel Summary</b>"))
        v.addWidget(self._ch_table)
        v.addWidget(QLabel("<b>Confusion Matrix</b>"))
        v.addWidget(self.chart)
        v.addWidget(self.progress)
        v.addWidget(self.log)
        v.addStretch()
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(scroll)

    # ── BaseTab interface ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        pass

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        ch = result.get("channel", "?")
        self._results[ch] = result
        self._update_cards(result)
        self._update_ch_table(ch, result)
        matrix = result.get("confusion_matrix")
        if matrix:
            self.chart.show_matrix(matrix, title=f"Confusion Matrix — {ch}")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.eval_worker = None

    # ── Evaluation control ────────────────────────────────────────────────────

    def start_evaluation(self) -> None:
        if self.eval_worker is not None and self.eval_worker.isRunning():
            self.progress.append_log("⚠️  평가가 이미 실행 중입니다.")
            return

        sel = self.channel_box.currentText()
        channels = _CHANNELS if sel.startswith("전체") else [sel]

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._pending = list(channels)
        self._run_next_channel()

    def _run_next_channel(self) -> None:
        if not self._pending:
            self._run_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            if len(self._results) >= 2:
                self._show_overall_avg()
            return

        ch   = self._pending.pop(0)
        ckpt = auto_find_checkpoint(self.cfg, ch)
        self.eval_worker = self.service.start_evaluation(self.cfg, ch, ckpt)
        self.eval_worker.progress_updated.connect(self.progress.set_progress)
        self.eval_worker.log_emitted.connect(self.progress.append_log)
        self.eval_worker.finished.connect(self._on_ch_done)
        self.eval_worker.error_occurred.connect(
            lambda msg, c=ch: self.log.append(f"❌ [{c}] {msg.splitlines()[0]}")
        )
        self.eval_worker.start()

    def _on_ch_done(self, result: dict[str, Any]) -> None:
        self.on_worker_finished(result)
        self._run_next_channel()

    def stop_evaluation(self) -> None:
        self._pending = []
        self.service.stop_evaluation()
        self.progress.append_log("평가 중지됨.")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_cards(self, r: dict) -> None:
        self.acc_card.set_value(f"{r.get('accuracy', 0):.3f}")
        self.f1_card.set_value(f"{r.get('macro_f1', 0):.3f}")
        self.mae_card.set_value(f"{r.get('mae', 0):.3f}")
        self.n_card.set_value(str(r.get("n_samples", "—")))

    def _update_ch_table(self, ch: str, r: dict) -> None:
        row = _CHANNELS.index(ch) if ch in _CHANNELS else -1
        if row < 0:
            return
        for col, key in enumerate(("accuracy", "macro_f1", "mae"), start=1):
            val = r.get(key, None)
            text = f"{val:.3f}" if val is not None else "—"
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._ch_table.setItem(row, col, item)

    def _show_overall_avg(self) -> None:
        accs  = [v.get("accuracy", 0) for v in self._results.values()]
        f1s   = [v.get("macro_f1",  0) for v in self._results.values()]
        maes  = [v.get("mae",       0) for v in self._results.values()]
        n     = len(accs)
        avg   = {"accuracy": sum(accs)/n, "macro_f1": sum(f1s)/n, "mae": sum(maes)/n}
        self.log.append(
            f"📊 전체 평균 — Acc: {avg['accuracy']:.3f}  F1: {avg['macro_f1']:.3f}  MAE: {avg['mae']:.3f}"
        )
        self._update_cards({**avg, "n_samples": sum(v.get("n_samples", 0) for v in self._results.values())})
