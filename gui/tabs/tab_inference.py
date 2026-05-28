"""InferenceTab — 학습된 모델로 이미지 추론 (단일 / 배치).
Test images with a trained model: single-image and batch-folder modes.

Contract: Contract_gui.md §3.6  /  SSOT_GUI.md §6.7
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.components.log_panel import LogPanel
from gui.components.level_accuracy_table import LevelAccuracyTable
from gui.i18n import t
from gui.tabs.base_tab import BaseTab
from gui.workers.batch_inference_worker import BatchInferenceWorker
from gui.workers.inference_worker import InferenceWorker
from gui.workers.gradcam_worker import GradCAMWorker
from gui.workers._ckpt_utils import auto_find_checkpoint, auto_find_all_checkpoints

_IMG_FILTER  = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
_CKPT_FILTER = "Checkpoint (*.pt *.pth)"
_CHANNELS    = ["Y", "M", "C", "K"]
_DATA_SOURCES = {
    "labeled": "data_set/labeled",
    "raw":     "data_set/raw",
    "roi":     "data_set/roi",
}


class InferenceTab(BaseTab):
    """단일 이미지 + 배치 폴더 추론 탭.

    상단: 채널 선택 + 체크포인트 행 (수동 or 자동 탐지)
    좌측: 단일 이미지 — 선택 → 미리보기 → 추론 → 레벨 배지 + Top-3
    우측: 배치 폴더 — 선택 → 일괄 추론 → 실시간 테이블 → CSV 내보내기
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self._selected_image: str  = ""
        self._selected_folder: str = ""
        self._checkpoint_path: str = self._load_saved_checkpoint()
        self._batch_results: list[dict] = []

        self.infer_worker: InferenceWorker | None  = None
        self.batch_worker: BatchInferenceWorker | None = None
        self.gradcam_worker: GradCAMWorker | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        root_layout.addWidget(self._build_checkpoint_group())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_single_panel())
        splitter.addWidget(self._build_batch_panel())
        splitter.setSizes([480, 560])

        root_layout.addWidget(splitter, stretch=1)

    # ══════════════════════════════════════════════════════════════════════════
    # Build — 공유 체크포인트 + 채널 행
    # ══════════════════════════════════════════════════════════════════════════

    def _build_checkpoint_group(self) -> QGroupBox:
        self._grp_ckpt = QGroupBox(t("grp_ckpt"))
        row = QHBoxLayout(self._grp_ckpt)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        # 채널 선택 / Channel selector
        ch_lbl = QLabel("Channel:")
        self._channel_combo = QComboBox()
        self._channel_combo.addItem("Y",             userData="Y")
        self._channel_combo.addItem("M",             userData="M")
        self._channel_combo.addItem("C",             userData="C")
        self._channel_combo.addItem("K",             userData="K")
        self._channel_combo.addItem("전체 앙상블 (All Channels)", userData="all")
        self._channel_combo.setFixedWidth(180)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)

        # 체크포인트 경로 필드
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText(t("lbl_no_ckpt"))
        self._ckpt_edit.setReadOnly(True)
        if self._checkpoint_path:
            self._ckpt_edit.setText(self._checkpoint_path)

        # 버튼
        auto_btn   = QPushButton(t("btn_auto_detect"))
        auto_btn.setFixedWidth(110)
        auto_btn.clicked.connect(self._auto_detect_checkpoint)

        browse_btn = QPushButton(t("btn_browse_ckpt"))
        browse_btn.setFixedWidth(160)
        browse_btn.clicked.connect(self._browse_checkpoint)

        # 데이터 소스 선택 / Data source selector
        src_lbl = QLabel("데이터:")
        self._src_combo = QComboBox()
        self._src_combo.addItem("Labeled",  userData="labeled")
        self._src_combo.addItem("Raw",      userData="raw")
        self._src_combo.addItem("ROI",      userData="roi")
        self._src_combo.setFixedWidth(100)
        self._src_combo.currentIndexChanged.connect(self._on_source_changed)

        row.addWidget(ch_lbl)
        row.addWidget(self._channel_combo)
        row.addWidget(src_lbl)
        row.addWidget(self._src_combo)
        row.addWidget(self._ckpt_edit, stretch=1)
        row.addWidget(auto_btn)
        row.addWidget(browse_btn)
        return self._grp_ckpt

    # ── 단일 이미지 패널 ──────────────────────────────────────────────────────

    def _build_single_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        self._grp_single = QGroupBox(t("grp_single_infer"))
        ctrl_layout = QVBoxLayout(self._grp_single)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(6)

        btn_row = QHBoxLayout()
        self._browse_img_btn = QPushButton(t("btn_browse_img"))
        self._run_infer_btn  = QPushButton(t("btn_run_infer"))
        self._stop_infer_btn = QPushButton(t("btn_stop"))
        self._stop_infer_btn.setEnabled(False)
        btn_row.addWidget(self._browse_img_btn)
        btn_row.addWidget(self._run_infer_btn)
        btn_row.addWidget(self._stop_infer_btn)
        btn_row.addStretch()

        self._img_name_lbl = QLabel(t("lbl_selected"))
        self._img_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._img_preview = QLabel()
        self._img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_preview.setMinimumHeight(220)
        self._img_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._img_preview.setStyleSheet(
            "border: 1px dashed #4a6080; border-radius: 4px; background: transparent;"
        )
        self._img_preview.setText("이미지를 선택하세요\nSelect an image")

        self._infer_progress = QProgressBar()
        self._infer_progress.setRange(0, 100)
        self._infer_progress.setValue(0)
        self._infer_progress.setMaximumHeight(16)

        ctrl_layout.addLayout(btn_row)
        ctrl_layout.addWidget(self._img_name_lbl)
        ctrl_layout.addWidget(self._img_preview, stretch=1)
        ctrl_layout.addWidget(self._infer_progress)

        # 결과 카드 / Result card
        self._grp_result = QGroupBox(t("grp_result"))
        res_layout = QVBoxLayout(self._grp_result)
        res_layout.setContentsMargins(8, 8, 8, 8)
        res_layout.setSpacing(4)

        self._level_badge = QLabel("—")
        self._level_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._level_badge.setMinimumHeight(56)
        self._level_badge.setStyleSheet(
            "font-size: 36pt; font-weight: 700; color: #3b82f6; background: transparent;"
        )

        self._conf_bar_lbl = QLabel(t("lbl_conf_val").replace("{v}", "—"))
        self._conf_bar_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._conf_bar_lbl.setStyleSheet("font-size: 13pt; background: transparent;")

        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setMaximumHeight(12)

        self._top3_lbl = QLabel("Top-3: —")
        self._top3_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._top3_lbl.setStyleSheet("font-size: 10pt; background: transparent;")

        # 앙상블 채널별 결과 레이블
        self._per_ch_lbl = QLabel("")
        self._per_ch_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._per_ch_lbl.setStyleSheet("font-size: 9pt; background: transparent;")
        self._per_ch_lbl.setWordWrap(True)

        self._infer_log = LogPanel()
        self._infer_log.setMaximumHeight(90)

        res_layout.addWidget(self._level_badge)
        res_layout.addWidget(self._conf_bar_lbl)
        res_layout.addWidget(self._conf_bar)
        res_layout.addWidget(self._top3_lbl)
        res_layout.addWidget(self._per_ch_lbl)
        res_layout.addWidget(self._infer_log)

        # GradCAM 패널 / GradCAM visualization panel
        self._grp_gradcam = QGroupBox("🔥 GradCAM 시각화 / Activation Map")
        gcam_v = QVBoxLayout(self._grp_gradcam)
        gcam_row = QHBoxLayout()
        self._run_gradcam_btn = QPushButton("🔥  GradCAM 실행")
        self._run_gradcam_btn.setEnabled(False)
        self._run_gradcam_btn.clicked.connect(self._start_gradcam)
        gcam_row.addWidget(self._run_gradcam_btn)
        gcam_row.addStretch()

        self._gradcam_preview = QLabel("GradCAM 히트맵이 여기에 표시됩니다.")
        self._gradcam_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gradcam_preview.setMinimumHeight(140)
        self._gradcam_preview.setStyleSheet(
            "border: 1px dashed #4a6080; border-radius: 4px; background: transparent;"
        )
        gcam_v.addLayout(gcam_row)
        gcam_v.addWidget(self._gradcam_preview)

        layout.addWidget(self._grp_single, stretch=3)
        layout.addWidget(self._grp_result, stretch=2)
        layout.addWidget(self._grp_gradcam)

        self._browse_img_btn.clicked.connect(self._browse_image)
        self._run_infer_btn.clicked.connect(self.start_single_inference)
        self._stop_infer_btn.clicked.connect(self._stop_single)
        return panel

    # ── 배치 패널 ─────────────────────────────────────────────────────────────

    def _build_batch_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(6)

        self._grp_batch = QGroupBox(t("grp_batch_infer"))
        batch_layout = QVBoxLayout(self._grp_batch)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        batch_layout.setSpacing(6)

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(t("lbl_no_folder"))
        self._folder_edit.setReadOnly(True)
        self._browse_folder_btn = QPushButton(t("btn_browse_folder"))
        self._browse_folder_btn.setFixedWidth(150)
        folder_row.addWidget(self._folder_edit, stretch=1)
        folder_row.addWidget(self._browse_folder_btn)

        ctrl_row = QHBoxLayout()
        self._run_batch_btn  = QPushButton(t("btn_run_batch"))
        self._stop_batch_btn = QPushButton(t("btn_stop"))
        self._export_csv_btn = QPushButton(t("btn_export_csv"))
        self._stop_batch_btn.setEnabled(False)
        self._export_csv_btn.setEnabled(False)
        ctrl_row.addWidget(self._run_batch_btn)
        ctrl_row.addWidget(self._stop_batch_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self._export_csv_btn)

        self._batch_status_lbl = QLabel("")
        self._batch_status_lbl.setStyleSheet("background: transparent;")
        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setValue(0)
        self._batch_progress.setMaximumHeight(16)

        self._result_table = QTableWidget(0, 4)
        self._result_table.setHorizontalHeaderLabels([
            t("col_filename"), t("col_pred_level"), t("col_confidence"), "Top-3",
        ])
        self._result_table.horizontalHeader().setStretchLastSection(True)
        self._result_table.setAlternatingRowColors(True)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.setColumnWidth(0, 200)
        self._result_table.setColumnWidth(1, 90)
        self._result_table.setColumnWidth(2, 90)

        self._batch_log = LogPanel()
        self._batch_log.setMaximumHeight(80)

        # 레벨별 정확도 테이블 / Per-level accuracy table
        num_levels = self.cfg.get("data", {}).get("num_levels", 6)
        self._level_acc_table = LevelAccuracyTable(num_levels)

        batch_layout.addLayout(folder_row)
        batch_layout.addLayout(ctrl_row)
        batch_layout.addWidget(self._batch_status_lbl)
        batch_layout.addWidget(self._batch_progress)
        batch_layout.addWidget(self._result_table, stretch=1)
        batch_layout.addWidget(self._level_acc_table)
        batch_layout.addWidget(self._batch_log)

        layout.addWidget(self._grp_batch, stretch=1)

        self._browse_folder_btn.clicked.connect(self._browse_folder)
        self._run_batch_btn.clicked.connect(self.start_batch_inference)
        self._stop_batch_btn.clicked.connect(self._stop_batch)
        self._export_csv_btn.clicked.connect(self._export_csv)
        return panel

    # ══════════════════════════════════════════════════════════════════════════
    # 체크포인트 + 채널
    # ══════════════════════════════════════════════════════════════════════════

    def _current_channel(self) -> str:
        return self._channel_combo.currentData() or "Y"

    def _on_channel_changed(self) -> None:
        ch = self._current_channel()
        if ch == "all":
            self._ckpt_edit.setPlaceholderText("전체 앙상블 — 자동 탐지 (Auto-detect all 4 best_*.pt)")
            self._ckpt_edit.clear()
        else:
            self._ckpt_edit.setPlaceholderText(t("lbl_no_ckpt"))

    def _auto_detect_checkpoint(self) -> None:
        """models_dir 에서 채널에 맞는 best_*.pt를 자동 탐지한다."""
        ch = self._current_channel()
        if ch == "all":
            paths = auto_find_all_checkpoints(self.cfg)
            found = {c: p for c, p in paths.items() if p}
            missing = [c for c, p in paths.items() if not p]
            if found:
                summary = "  |  ".join(f"{c}: {Path(p).name}" for c, p in found.items())
                self._ckpt_edit.setText(summary)
                self._infer_log.append(f"✅ 자동 탐지 / Found: {summary}")
            if missing:
                self._infer_log.append(f"⚠️  미발견 / Not found: {missing}")
        else:
            path = auto_find_checkpoint(self.cfg, ch)
            if path:
                self._checkpoint_path = path
                self._ckpt_edit.setText(path)
                self._save_checkpoint(path)
                self._infer_log.append(f"✅ 자동 탐지 / Found: {Path(path).name}")
            else:
                self._infer_log.append(
                    f"⚠️  {ch} 채널 체크포인트를 찾을 수 없습니다 / Not found for channel {ch}"
                )

    def _on_source_changed(self) -> None:
        """데이터 소스 변경 시 기본 폴더를 folder_edit에 채운다."""
        src_key   = self._src_combo.currentData() or "labeled"
        base_path = _DATA_SOURCES.get(src_key, "data_set/labeled")
        self._batch_log.append(f"데이터 소스: {src_key} → {base_path}")

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "", _CKPT_FILTER
        )
        if not path:
            return
        self._checkpoint_path = path
        self._ckpt_edit.setText(path)
        self._save_checkpoint(path)

    # ══════════════════════════════════════════════════════════════════════════
    # 단일 이미지 추론
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_image(self) -> None:
        src_key   = self._src_combo.currentData() or "labeled"
        start_dir = _DATA_SOURCES.get(src_key, "data_set/labeled")
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", start_dir, _IMG_FILTER)
        if not path:
            return
        self._selected_image = path
        self._img_name_lbl.setText(f"📄 {Path(path).name}")
        self._reset_result()
        self._infer_log.append(f"Selected: {Path(path).name}")

        px = QPixmap(path)
        if not px.isNull():
            self._img_preview.setPixmap(
                px.scaled(
                    self._img_preview.width() or 400,
                    self._img_preview.height() or 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._img_preview.setText(f"[Preview unavailable]\n{Path(path).name}")

    def start_single_inference(self) -> None:
        if not self._selected_image:
            self._infer_log.append("⚠️  이미지를 먼저 선택하세요 / Select an image first")
            return
        if self.infer_worker is not None and self.infer_worker.isRunning():
            self._infer_log.append("⚠️  추론 중입니다 / Already running")
            return

        ch   = self._current_channel()
        ckpt = "" if ch == "all" else self._checkpoint_path

        self._reset_result()
        self._run_infer_btn.setEnabled(False)
        self._stop_infer_btn.setEnabled(True)
        self._infer_progress.setValue(0)

        self.infer_worker = InferenceWorker(self.cfg, self._selected_image, ckpt, channel=ch)
        self.infer_worker.progress_updated.connect(self._infer_progress.setValue)
        self.infer_worker.log_emitted.connect(self._infer_log.append)
        self.infer_worker.finished.connect(self._on_single_finished)
        self.infer_worker.error_occurred.connect(self._on_single_error)
        self.infer_worker.start()

    def _stop_single(self) -> None:
        if self.infer_worker and self.infer_worker.isRunning():
            self.infer_worker.cancel()
            self._infer_log.append("⏹  중지됨 / Stopped")
        self._run_infer_btn.setEnabled(True)
        self._stop_infer_btn.setEnabled(False)

    def _on_single_finished(self, result: dict) -> None:
        pred = result.get("pred_level", "?")
        conf = result.get("confidence", 0.0)
        top3 = result.get("top3", [])

        self._level_badge.setText(f"Level {pred}")
        self._conf_bar.setValue(int(conf * 100))
        self._conf_bar_lbl.setText(t("lbl_conf_val").replace("{v}", f"{conf:.1%}"))
        if top3:
            self._top3_lbl.setText(
                "Top-3: " + "  |  ".join(f"L{lvl}: {p:.1%}" for lvl, p in top3)
            )

        # 앙상블이면 채널별 결과 표시
        per_ch = result.get("per_channel")
        if per_ch:
            ch_str = "  ".join(f"[{c}] L{v['pred']} {v['conf']:.0%}" for c, v in per_ch.items())
            self._per_ch_lbl.setText(ch_str)
        else:
            ckpt_name = result.get("checkpoint", "")
            self._per_ch_lbl.setText(f"Channel: {result.get('channel', '')}  |  {ckpt_name}")

        colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444", "#7c3aed"]
        color  = colors[pred % len(colors)] if isinstance(pred, int) else "#3b82f6"
        self._level_badge.setStyleSheet(
            f"font-size: 36pt; font-weight: 700; color: {color}; background: transparent;"
        )

        self._infer_log.append(
            f"✅ Level {pred}  conf={conf:.3f}  {Path(result.get('image_path', '')).name}"
        )
        self._run_infer_btn.setEnabled(True)
        self._stop_infer_btn.setEnabled(False)
        # GradCAM 버튼 활성화 (단일 채널일 때만)
        if result.get("channel") not in (None, "all") and self._selected_image:
            self._run_gradcam_btn.setEnabled(True)

    def _on_single_error(self, msg: str) -> None:
        self._infer_log.append(f"❌  {msg}")
        self._run_infer_btn.setEnabled(True)
        self._stop_infer_btn.setEnabled(False)

    def _reset_result(self) -> None:
        self._level_badge.setText("—")
        self._level_badge.setStyleSheet(
            "font-size: 36pt; font-weight: 700; color: #3b82f6; background: transparent;"
        )
        self._conf_bar.setValue(0)
        self._conf_bar_lbl.setText(t("lbl_conf_val").replace("{v}", "—"))
        self._top3_lbl.setText("Top-3: —")
        self._per_ch_lbl.setText("")

    # ══════════════════════════════════════════════════════════════════════════
    # GradCAM
    # ══════════════════════════════════════════════════════════════════════════

    def _start_gradcam(self) -> None:
        if not self._selected_image:
            return
        if self.gradcam_worker and self.gradcam_worker.isRunning():
            return
        ch   = self._current_channel()
        ckpt = "" if ch == "all" else self._checkpoint_path
        self._run_gradcam_btn.setEnabled(False)
        self._gradcam_preview.setText("GradCAM 계산 중…")
        self.gradcam_worker = GradCAMWorker(self.cfg, self._selected_image, ckpt, channel=ch)
        self.gradcam_worker.progress_updated.connect(self._infer_progress.setValue)
        self.gradcam_worker.log_emitted.connect(self._infer_log.append)
        self.gradcam_worker.finished.connect(self._on_gradcam_finished)
        self.gradcam_worker.error_occurred.connect(
            lambda msg: (self._infer_log.append(f"❌ GradCAM: {msg.splitlines()[0]}"),
                         self._run_gradcam_btn.setEnabled(True))
        )
        self.gradcam_worker.start()

    def _on_gradcam_finished(self, result: dict) -> None:
        import numpy as np
        overlay = result.get("overlay")
        if overlay is None:
            self._infer_log.append("⚠️  GradCAM 결과 없음")
            self._run_gradcam_btn.setEnabled(True)
            return
        try:
            h, w = overlay.shape[:2]
            from PyQt6.QtGui import QImage
            # overlay is RGB np.ndarray
            img_bytes = overlay.astype(np.uint8).tobytes()
            qimg = QImage(img_bytes, w, h, w * 3, QImage.Format.Format_RGB888)
            px   = QPixmap.fromImage(qimg)
            self._gradcam_preview.setPixmap(
                px.scaled(
                    self._gradcam_preview.width() or 300,
                    self._gradcam_preview.height() or 200,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._infer_log.append(
                f"🔥 GradCAM 완료 — L{result.get('pred_level')} ({result.get('confidence', 0):.1%})"
            )
        except Exception as exc:
            self._infer_log.append(f"❌ GradCAM render: {exc}")
        finally:
            self._run_gradcam_btn.setEnabled(True)

    # ══════════════════════════════════════════════════════════════════════════
    # 배치 추론
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_folder(self) -> None:
        src_key   = self._src_combo.currentData() or "labeled"
        start_dir = _DATA_SOURCES.get(src_key, "data_set/labeled")
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start_dir)
        if not folder:
            return
        self._selected_folder = folder
        self._folder_edit.setText(folder)
        self._batch_log.append(f"📁  {folder}")
        self._result_table.setRowCount(0)
        self._batch_results = []
        self._export_csv_btn.setEnabled(False)

    def start_batch_inference(self) -> None:
        if not self._selected_folder:
            self._batch_log.append("⚠️  폴더를 먼저 선택하세요 / Select a folder first")
            return
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self._batch_log.append("⚠️  배치 추론 중입니다 / Already running")
            return

        ch   = self._current_channel()
        ckpt = "" if ch == "all" else self._checkpoint_path

        self._result_table.setRowCount(0)
        self._batch_results = []
        self._export_csv_btn.setEnabled(False)
        self._run_batch_btn.setEnabled(False)
        self._stop_batch_btn.setEnabled(True)
        self._batch_progress.setValue(0)

        self.batch_worker = BatchInferenceWorker(
            self.cfg, self._selected_folder, ckpt, channel=ch
        )
        self.batch_worker.progress_updated.connect(self._batch_progress.setValue)
        self.batch_worker.log_emitted.connect(self._on_batch_log)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.error_occurred.connect(self._on_batch_error)
        self.batch_worker.start()

    def _stop_batch(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self._batch_log.append("⏹  중지됨 / Stopped")
        self._run_batch_btn.setEnabled(True)
        self._stop_batch_btn.setEnabled(False)
        if self._batch_results:
            self._export_csv_btn.setEnabled(True)

    def _on_batch_log(self, msg: str) -> None:
        if msg.startswith("__ROW__"):
            try:
                data = json.loads(msg[len("__ROW__"):])
                self._add_table_row(data)
            except Exception:
                pass
        else:
            self._batch_log.append(msg)
            self._batch_status_lbl.setText(msg)

    def _add_table_row(self, data: dict) -> None:
        row        = self._result_table.rowCount()
        self._result_table.insertRow(row)
        pred_level = data.get("pred_level", -1)
        confidence = data.get("confidence", 0.0)
        top3       = data.get("top3", [])
        top3_str   = "  ".join(f"L{lvl}:{p:.0%}" for lvl, p in top3)

        self._result_table.setItem(row, 0, QTableWidgetItem(data.get("filename", "")))
        lv = QTableWidgetItem(str(pred_level))
        lv.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_table.setItem(row, 1, lv)
        cf = QTableWidgetItem(f"{confidence:.1%}")
        cf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_table.setItem(row, 2, cf)
        self._result_table.setItem(row, 3, QTableWidgetItem(top3_str))
        self._result_table.scrollToBottom()

    def _on_batch_finished(self, result: dict) -> None:
        self._batch_results = result.get("results", [])
        total     = result.get("total", 0)
        succeeded = result.get("succeeded", 0)
        failed    = result.get("failed", 0)
        summary   = (
            f"✅ 완료 / Done — {succeeded}/{total} 성공"
            f"{f',  {failed} 실패' if failed else ''}"
        )
        self._batch_log.append(summary)
        self._batch_status_lbl.setText(summary)
        self._run_batch_btn.setEnabled(True)
        self._stop_batch_btn.setEnabled(False)
        if self._batch_results:
            self._export_csv_btn.setEnabled(True)
            ch = self._current_channel()
            self._level_acc_table.update_from_results(
                self._batch_results, channel=ch if ch != "all" else "Y"
            )

    def _on_batch_error(self, msg: str) -> None:
        self._batch_log.append(f"❌  {msg}")
        self._run_batch_btn.setEnabled(True)
        self._stop_batch_btn.setEnabled(False)

    def _export_csv(self) -> None:
        if not self._batch_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "inference_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["filename", "path", "pred_level", "confidence", "error"]
                )
                writer.writeheader()
                for row in self._batch_results:
                    writer.writerow({
                        "filename":   row.get("filename", ""),
                        "path":       row.get("path", ""),
                        "pred_level": row.get("pred_level", -1),
                        "confidence": f"{row.get('confidence', 0):.4f}",
                        "error":      row.get("error") or "",
                    })
            self._batch_log.append(f"💾  저장 완료 / Saved: {Path(path).name}")
        except Exception as exc:
            self._batch_log.append(f"❌  저장 실패 / Failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # 체크포인트 영속 / Checkpoint persistence
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _load_saved_checkpoint() -> str:
        cfg_path = Path(__file__).resolve().parents[1] / "assets" / "config.json"
        try:
            if cfg_path.exists():
                return json.loads(cfg_path.read_text(encoding="utf-8")).get("checkpoint_path", "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _save_checkpoint(path: str) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / "assets" / "config.json"
        try:
            data = {}
            if cfg_path.exists():
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
            data["checkpoint_path"] = path
            cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # BaseTab interface
    # ══════════════════════════════════════════════════════════════════════════

    def refresh(self) -> None:
        saved = self._load_saved_checkpoint()
        if saved and not self._ckpt_edit.text():
            self._checkpoint_path = saved
            self._ckpt_edit.setText(saved)

    def retranslate_ui(self, lang: str) -> None:
        self._grp_ckpt.setTitle(t("grp_ckpt"))
        self._grp_single.setTitle(t("grp_single_infer"))
        self._grp_result.setTitle(t("grp_result"))
        self._grp_batch.setTitle(t("grp_batch_infer"))
        self._ckpt_edit.setPlaceholderText(t("lbl_no_ckpt"))
        self._folder_edit.setPlaceholderText(t("lbl_no_folder"))
        self._browse_img_btn.setText(t("btn_browse_img"))
        self._run_infer_btn.setText(t("btn_run_infer"))
        self._stop_infer_btn.setText(t("btn_stop"))
        self._browse_folder_btn.setText(t("btn_browse_folder"))
        self._run_batch_btn.setText(t("btn_run_batch"))
        self._stop_batch_btn.setText(t("btn_stop"))
        self._export_csv_btn.setText(t("btn_export_csv"))
        self._result_table.setHorizontalHeaderLabels([
            t("col_filename"), t("col_pred_level"), t("col_confidence"), "Top-3",
        ])

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        pass
