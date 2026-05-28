"""InferenceTab — 학습된 모델로 이미지 추론 (단일 / 배치).
Test images with a trained model: single-image and batch-folder modes.

Contract: Contract_gui.md §3.6  /  SSOT_GUI.md §6.6
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
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
from gui.i18n import t
from gui.tabs.base_tab import BaseTab
from gui.workers.batch_inference_worker import BatchInferenceWorker
from gui.workers.inference_worker import InferenceWorker


# ── 공통 상수 ──────────────────────────────────────────────────────────────────
_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
_CKPT_FILTER = "Checkpoint (*.pt *.pth)"


class InferenceTab(BaseTab):
    """단일 이미지 + 배치 폴더 추론 탭.
    Provides single-image and batch-folder inference with a trained model.

    Left  panel : single image — browse → preview → run → result badge + top-K
    Right panel : batch folder — select folder → run → results table → export CSV
    Both panels share a checkpoint browser at the top.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self._selected_image: str = ""
        self._selected_folder: str = ""
        self._checkpoint_path: str = self._load_saved_checkpoint()
        self._batch_results: list[dict] = []

        self.infer_worker: InferenceWorker | None = None
        self.batch_worker: BatchInferenceWorker | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # ── 체크포인트 행 / Checkpoint row (shared) ────────────────────────────
        root_layout.addWidget(self._build_checkpoint_group())

        # ── 메인 QSplitter: 단일(좌) / 배치(우) ───────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_single_panel())
        splitter.addWidget(self._build_batch_panel())
        splitter.setSizes([480, 560])

        root_layout.addWidget(splitter, stretch=1)

    # ══════════════════════════════════════════════════════════════════════════
    # Build helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _build_checkpoint_group(self) -> QGroupBox:
        self._grp_ckpt = QGroupBox(t("grp_ckpt"))
        row = QHBoxLayout(self._grp_ckpt)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText(t("lbl_no_ckpt"))
        self._ckpt_edit.setReadOnly(True)
        if self._checkpoint_path:
            self._ckpt_edit.setText(self._checkpoint_path)

        browse_btn = QPushButton(t("btn_browse_ckpt"))
        browse_btn.setFixedWidth(180)
        browse_btn.clicked.connect(self._browse_checkpoint)

        row.addWidget(QLabel("Checkpoint:"))
        row.addWidget(self._ckpt_edit, stretch=1)
        row.addWidget(browse_btn)
        return self._grp_ckpt

    # ── Single image panel ────────────────────────────────────────────────────

    def _build_single_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # Controls
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

        # Image preview
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

        # Progress
        self._infer_progress = QProgressBar()
        self._infer_progress.setRange(0, 100)
        self._infer_progress.setValue(0)
        self._infer_progress.setTextVisible(True)
        self._infer_progress.setMaximumHeight(16)

        ctrl_layout.addLayout(btn_row)
        ctrl_layout.addWidget(self._img_name_lbl)
        ctrl_layout.addWidget(self._img_preview, stretch=1)
        ctrl_layout.addWidget(self._infer_progress)

        # Result card
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

        self._infer_log = LogPanel()
        self._infer_log.setMaximumHeight(90)

        res_layout.addWidget(self._level_badge)
        res_layout.addWidget(self._conf_bar_lbl)
        res_layout.addWidget(self._conf_bar)
        res_layout.addWidget(self._top3_lbl)
        res_layout.addWidget(self._infer_log)

        layout.addWidget(self._grp_single, stretch=3)
        layout.addWidget(self._grp_result, stretch=2)

        # Connect signals
        self._browse_img_btn.clicked.connect(self._browse_image)
        self._run_infer_btn.clicked.connect(self.start_single_inference)
        self._stop_infer_btn.clicked.connect(self._stop_single)

        return panel

    # ── Batch panel ───────────────────────────────────────────────────────────

    def _build_batch_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(6)

        self._grp_batch = QGroupBox(t("grp_batch_infer"))
        batch_layout = QVBoxLayout(self._grp_batch)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        batch_layout.setSpacing(6)

        # Folder row
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(t("lbl_no_folder"))
        self._folder_edit.setReadOnly(True)
        self._browse_folder_btn = QPushButton(t("btn_browse_folder"))
        self._browse_folder_btn.setFixedWidth(150)
        folder_row.addWidget(self._folder_edit, stretch=1)
        folder_row.addWidget(self._browse_folder_btn)

        # Batch controls
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

        # Batch progress
        self._batch_status_lbl = QLabel("")
        self._batch_status_lbl.setStyleSheet("background: transparent;")
        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setValue(0)
        self._batch_progress.setTextVisible(True)
        self._batch_progress.setMaximumHeight(16)

        # Results table
        self._result_table = QTableWidget(0, 4)
        self._result_table.setHorizontalHeaderLabels([
            t("col_filename"), t("col_pred_level"), t("col_confidence"), "Top-3",
        ])
        self._result_table.horizontalHeader().setStretchLastSection(True)
        self._result_table.setAlternatingRowColors(True)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._result_table.verticalHeader().setVisible(False)
        # Column widths
        self._result_table.setColumnWidth(0, 200)
        self._result_table.setColumnWidth(1, 90)
        self._result_table.setColumnWidth(2, 90)

        # Summary log
        self._batch_log = LogPanel()
        self._batch_log.setMaximumHeight(80)

        batch_layout.addLayout(folder_row)
        batch_layout.addLayout(ctrl_row)
        batch_layout.addWidget(self._batch_status_lbl)
        batch_layout.addWidget(self._batch_progress)
        batch_layout.addWidget(self._result_table, stretch=1)
        batch_layout.addWidget(self._batch_log)

        layout.addWidget(self._grp_batch, stretch=1)

        # Connect
        self._browse_folder_btn.clicked.connect(self._browse_folder)
        self._run_batch_btn.clicked.connect(self.start_batch_inference)
        self._stop_batch_btn.clicked.connect(self._stop_batch)
        self._export_csv_btn.clicked.connect(self._export_csv)

        return panel

    # ══════════════════════════════════════════════════════════════════════════
    # Single-image inference
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", _IMG_FILTER
        )
        if not path:
            return
        self._selected_image = path
        self._img_name_lbl.setText(f"📄 {Path(path).name}")
        self._reset_result()
        self._infer_log.append(f"Selected: {Path(path).name}")

        # Preview
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
            self._infer_log.append("⚠️  먼저 이미지를 선택하세요 / Select an image first")
            return
        if self.infer_worker is not None and self.infer_worker.isRunning():
            self._infer_log.append("⚠️  추론 중입니다 / Already running")
            return

        ckpt = self._ckpt_edit.text().strip()
        if not ckpt:
            self._infer_log.append("⚠️  체크포인트를 먼저 선택하세요 / Select a checkpoint first")
            return

        self._reset_result()
        self._run_infer_btn.setEnabled(False)
        self._stop_infer_btn.setEnabled(True)
        self._infer_progress.setValue(0)

        self.infer_worker = InferenceWorker(self.cfg, self._selected_image, ckpt)
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
        pred   = result.get("pred_level", "?")
        conf   = result.get("confidence", 0.0)
        top3   = result.get("top3", [])

        self._level_badge.setText(f"Level {pred}")
        pct = int(conf * 100)
        self._conf_bar.setValue(pct)
        self._conf_bar_lbl.setText(
            t("lbl_conf_val").replace("{v}", f"{conf:.1%}")
        )
        if top3:
            top3_str = "  |  ".join(f"L{lvl}: {p:.1%}" for lvl, p in top3)
            self._top3_lbl.setText(f"Top-3: {top3_str}")

        # Accent color based on level
        colors = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444", "#7c3aed"]
        color = colors[pred % len(colors)] if isinstance(pred, int) else "#3b82f6"
        self._level_badge.setStyleSheet(
            f"font-size: 36pt; font-weight: 700; color: {color}; background: transparent;"
        )

        self._infer_log.append(
            f"✅ Level {pred}  conf={conf:.3f}  {Path(result.get('image_path', '')).name}"
        )
        self._run_infer_btn.setEnabled(True)
        self._stop_infer_btn.setEnabled(False)

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

    # ══════════════════════════════════════════════════════════════════════════
    # Batch inference
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self._selected_folder = folder
        self._folder_edit.setText(folder)
        self._batch_log.append(f"📁  {folder}")
        # Clear previous table
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

        ckpt = self._ckpt_edit.text().strip()
        if not ckpt:
            self._batch_log.append("⚠️  체크포인트를 먼저 선택하세요 / Select a checkpoint first")
            return

        # Reset table
        self._result_table.setRowCount(0)
        self._batch_results = []
        self._export_csv_btn.setEnabled(False)
        self._run_batch_btn.setEnabled(False)
        self._stop_batch_btn.setEnabled(True)
        self._batch_progress.setValue(0)

        self.batch_worker = BatchInferenceWorker(self.cfg, self._selected_folder, ckpt)
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
        """log_emitted 처리 — __ROW__ 접두사면 테이블 행으로 추가."""
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
        row = self._result_table.rowCount()
        self._result_table.insertRow(row)

        filename   = data.get("filename", "")
        pred_level = data.get("pred_level", -1)
        confidence = data.get("confidence", 0.0)
        top3       = data.get("top3", [])

        top3_str = "  ".join(f"L{lvl}:{p:.0%}" for lvl, p in top3)

        self._result_table.setItem(row, 0, QTableWidgetItem(filename))
        lv_item = QTableWidgetItem(str(pred_level))
        lv_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_table.setItem(row, 1, lv_item)
        conf_item = QTableWidgetItem(f"{confidence:.1%}")
        conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_table.setItem(row, 2, conf_item)
        self._result_table.setItem(row, 3, QTableWidgetItem(top3_str))

        self._result_table.scrollToBottom()

    def _on_batch_finished(self, result: dict) -> None:
        self._batch_results = result.get("results", [])
        total     = result.get("total", 0)
        succeeded = result.get("succeeded", 0)
        failed    = result.get("failed", 0)

        summary = (
            f"✅ 완료 / Done — "
            f"{succeeded}/{total} 성공 / succeeded"
            f"{f',  {failed} 실패 / failed' if failed else ''}"
        )
        self._batch_log.append(summary)
        self._batch_status_lbl.setText(summary)

        self._run_batch_btn.setEnabled(True)
        self._stop_batch_btn.setEnabled(False)
        if self._batch_results:
            self._export_csv_btn.setEnabled(True)

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
            self._batch_log.append(f"💾  CSV 저장 완료 / Saved: {Path(path).name}")
        except Exception as exc:
            self._batch_log.append(f"❌  CSV 저장 실패 / Failed: {exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # Checkpoint
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "", _CKPT_FILTER
        )
        if not path:
            return
        self._checkpoint_path = path
        self._ckpt_edit.setText(path)
        self._save_checkpoint(path)

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
        """탭 활성화 시 체크포인트 경로 갱신."""
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
