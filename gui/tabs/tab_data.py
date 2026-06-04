"""Data tab — 데이터셋 현황 + 전처리 파라미터 편집.
Dataset overview and preprocessing parameter editor.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.components.image_viewer import ImageViewer
from gui.components.log_panel import LogPanel
from gui.i18n import t
from gui.tabs.base_tab import BaseTab

_ROOT = Path(__file__).resolve().parents[2]
_SRC_CONFIG = _ROOT / "src" / "config" / "config.json"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DataTab(BaseTab):
    """Dataset sample count table + preprocessing parameter editor."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)

        num_levels = self.cfg.get("data", {}).get("num_levels", 6)
        self._num_levels = num_levels

        # ── 스캔 결과 테이블 / Sample-count table ─────────────────────────
        headers = ["Channel"] + [f"L{i}" for i in range(num_levels)] + ["Total"]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self._on_cell_clicked)

        self._scan_btn = QPushButton(t("btn_scan"))
        self._browse_btn = QPushButton(t("btn_browse_img"))
        self._scan_btn.clicked.connect(self.refresh)
        self._browse_btn.clicked.connect(self._browse_image)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)

        scan_row = QHBoxLayout()
        scan_row.addWidget(self._scan_btn)
        scan_row.addWidget(self._browse_btn)
        scan_row.addStretch()

        # ── 이미지 미리보기 / Image preview ───────────────────────────────
        self.image_viewer = ImageViewer()
        self.image_viewer.setFixedHeight(180)

        # ── 전처리 파라미터 편집 그룹 / Preprocessing params group ───────
        self._log = LogPanel()
        self._log.setMaximumHeight(60)

        # ── 스크롤 레이아웃 / Scroll layout ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        v = QVBoxLayout(content)
        v.setSpacing(10)
        self._overview_lbl = QLabel(f"<b>{t('lbl_ds_overview')}</b>")
        self._overview_lbl.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self._overview_lbl)
        v.addLayout(scan_row)
        v.addWidget(self._status_lbl)
        v.addWidget(self.table)
        v.addWidget(self._build_preprocess_group())
        v.addWidget(self._build_pipeline_group())
        v.addWidget(QLabel("<b>이미지 미리보기</b>"))
        v.addWidget(self.image_viewer)
        v.addWidget(self._log)
        v.addStretch()
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(scroll)

        self.refresh()

    # ── BaseTab interface ─────────────────────────────────────────────────────

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        pass

    def retranslate_ui(self, lang: str) -> None:
        self._overview_lbl.setText(f"<b>{t('lbl_ds_overview')}</b>")
        self._save_btn.setText(t("btn_save_parameters"))
        self._scan_btn.setText(t("btn_scan"))
        self._browse_btn.setText(t("btn_browse_img"))
        self._holdout_btn.setText(t("btn_holdout"))
        self._syn_btn.setText(t("btn_generate_synthetic"))
        self._grp_pipeline.setTitle(t("grp_data_pipeline"))
        self._grp_preprocess.setTitle(t("grp_preprocess_params"))

    def refresh(self) -> None:
        """채널×레벨 샘플 수 재스캔."""
        channels = self.cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
        data_root = Path(
            self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled")
        )
        num_levels = self._num_levels

        self.table.setRowCount(len(channels))
        for row, channel in enumerate(channels):
            self.table.setItem(row, 0, QTableWidgetItem(channel))
            total = 0
            for level in range(num_levels):
                folder = data_root / channel / str(level)
                count = (
                    len([p for p in folder.glob("*") if p.suffix.lower() in _IMG_EXTS])
                    if folder.exists()
                    else 0
                )
                total += count
                item = QTableWidgetItem(str(count))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, level + 1, item)

            total_item = QTableWidgetItem(str(total))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, num_levels + 1, total_item)

        self.table.resizeColumnsToContents()

        grand_total = sum(
            int(self.table.item(r, num_levels + 1).text())
            for r in range(self.table.rowCount())
            if self.table.item(r, num_levels + 1)
        )
        exists_str = "✅ 존재" if data_root.exists() else "❌ 없음"
        ts = datetime.now().strftime("%H:%M:%S")
        self._status_lbl.setText(
            f"✅ 스캔 완료 — 총 {grand_total:,}개 이미지 | {exists_str} | {data_root} | {ts}"
        )

    # ── Private — build UI ────────────────────────────────────────────────────

    def _build_preprocess_group(self) -> QGroupBox:
        self._grp_preprocess = QGroupBox(t("grp_preprocess_params"))
        f = QFormLayout(self._grp_preprocess)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)

        d = self.cfg.get("data", {})
        norm = d.get("normalization", {})
        split = d.get("split_ratios", {})

        # image_size
        self._img_size = QSpinBox()
        self._img_size.setRange(32, 512)
        self._img_size.setSingleStep(32)
        self._img_size.setValue(d.get("image_size", 128))
        self._img_size.setMaximumWidth(120)

        # normalization mean / std (comma-separated)
        mean_vals = norm.get("mean", [0.485, 0.456, 0.406])
        std_vals = norm.get("std", [0.229, 0.224, 0.225])
        self._mean_edit = QLineEdit(", ".join(f"{v:.3f}" for v in mean_vals))
        self._std_edit = QLineEdit(", ".join(f"{v:.3f}" for v in std_vals))
        self._mean_edit.setMaximumWidth(200)
        self._std_edit.setMaximumWidth(200)
        self._mean_edit.setPlaceholderText("e.g. 0.485, 0.456, 0.406")
        self._std_edit.setPlaceholderText("e.g. 0.229, 0.224, 0.225")

        # split ratios
        self._train_split = QDoubleSpinBox()
        self._val_split = QDoubleSpinBox()
        self._test_split = QDoubleSpinBox()
        for sb, key, default in [
            (self._train_split, "train", 0.7),
            (self._val_split, "val", 0.15),
            (self._test_split, "test", 0.15),
        ]:
            sb.setRange(0.01, 0.98)
            sb.setSingleStep(0.05)
            sb.setDecimals(2)
            sb.setValue(split.get(key, default))
            sb.setMaximumWidth(100)

        # labeled_dir
        labeled_dir = self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled")
        self._labeled_dir_edit = QLineEdit(labeled_dir)
        browse_dir_btn = QPushButton("📂")
        browse_dir_btn.setMaximumWidth(36)
        browse_dir_btn.clicked.connect(self._browse_labeled_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._labeled_dir_edit)
        dir_row.addWidget(browse_dir_btn)

        self._save_btn = QPushButton(t("btn_save_parameters"))
        self._save_btn.clicked.connect(self._save_preprocess)

        f.addRow(t("lbl_train_data_location"), dir_row)
        f.addRow(t("lbl_image_size_px"), self._img_size)
        f.addRow(t("lbl_normalize_mean"), self._mean_edit)
        f.addRow(t("lbl_normalize_std"), self._std_edit)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel(t("lbl_train")))
        split_row.addWidget(self._train_split)
        split_row.addWidget(QLabel(t("lbl_val")))
        split_row.addWidget(self._val_split)
        split_row.addWidget(QLabel(t("lbl_test")))
        split_row.addWidget(self._test_split)
        split_row.addStretch()
        f.addRow(t("lbl_split_ratio"), split_row)
        f.addRow(self._save_btn)

        return self._grp_preprocess

    def _build_pipeline_group(self) -> QGroupBox:
        """Holdout 분리 + 합성 데이터 생성 버튼 그룹."""
        from PyQt6.QtWidgets import QComboBox as _QComboBox
        from PyQt6.QtWidgets import QSpinBox as _QSpinBox

        self._grp_pipeline = QGroupBox(t("grp_data_pipeline"))
        v = QVBoxLayout(self._grp_pipeline)
        v.setSpacing(6)
        v.setContentsMargins(8, 10, 8, 8)

        # ── Holdout 분리 ──────────────────────────────────────────────
        holdout_lbl = QLabel(
            "<b>① Holdout 분리</b> — 학습 전 한 번만 실행 / Run ONCE before training"
        )
        holdout_lbl.setStyleSheet("color:#f38ba8;")
        self._holdout_btn = QPushButton(t("btn_holdout"))
        self._holdout_btn.setToolTip(
            "prepare_holdout.py --dry-run 실행 — 실제 파일을 이동하지 않습니다.\n"
            "CLI에서 --no-dry-run 플래그로 실제 실행하세요."
        )
        self._holdout_btn.clicked.connect(self._run_prepare_holdout)

        # ── 합성 데이터 생성 ──────────────────────────────────────────
        syn_lbl = QLabel(
            "<b>② 합성 데이터 생성</b> — Holdout 분리 이후 실행 / Run AFTER holdout split"
        )
        syn_lbl.setStyleSheet("color:#a6e3a1;")

        syn_row = QHBoxLayout()
        self._syn_channel = _QComboBox()
        self._syn_channel.addItems(["all", "Y", "M", "C", "K"])
        self._syn_channel.setMaximumWidth(80)
        self._syn_count = _QSpinBox()
        self._syn_count.setRange(10, 500)
        self._syn_count.setValue(100)
        self._syn_count.setMaximumWidth(80)
        self._syn_btn = QPushButton(t("btn_generate_synthetic"))
        self._syn_btn.clicked.connect(self._run_generate_synthetic)

        syn_row.addWidget(QLabel(t("lbl_channel")))
        syn_row.addWidget(self._syn_channel)
        syn_row.addWidget(QLabel(t("lbl_count")))
        syn_row.addWidget(self._syn_count)
        syn_row.addWidget(self._syn_btn)
        syn_row.addStretch()

        v.addWidget(holdout_lbl)
        v.addWidget(self._holdout_btn)
        v.addWidget(syn_lbl)
        v.addLayout(syn_row)
        return self._grp_pipeline

    # ── Private — actions ─────────────────────────────────────────────────────

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if col == 0 or col > self._num_levels:
            return
        channels = self.cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
        data_root = Path(
            self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled")
        )
        channel = channels[row] if row < len(channels) else None
        if channel is None:
            return
        level = col - 1
        folder = data_root / channel / str(level)
        images = [p for p in folder.glob("*") if p.suffix.lower() in _IMG_EXTS]
        if images:
            self.image_viewer.load_image(images[0])

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "data_set/labeled",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if path:
            self.image_viewer.load_image(path)

    def _browse_labeled_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Labeled Dir", "data_set")
        if d:
            self._labeled_dir_edit.setText(d)

    def _run_prepare_holdout(self) -> None:
        """prepare_holdout.py --dry-run 실행."""
        import subprocess
        import sys

        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.scripts.prepare_holdout", "--dry-run"],
                capture_output=True,
                text=True,
                cwd=str(_ROOT),
            )
            out = (result.stdout + result.stderr).strip()
            self._log.append(
                "📋 prepare_holdout --dry-run:\n"
                + (out[:600] if out else "(no output)")
            )
            self._log.append(
                "⚠️  실제 분리는 CLI에서: python -m src.scripts.prepare_holdout\n"
                "    (파일이 이동됩니다 / files will be MOVED)"
            )
        except Exception as exc:
            self._log.append(f"❌ {exc}")

    def _run_generate_synthetic(self) -> None:
        """generate_synthetic.py 실행 (dry-run)."""
        import subprocess
        import sys

        channel = self._syn_channel.currentText()
        count = str(self._syn_count.value())
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.scripts.generate_synthetic",
                    "--channel",
                    channel,
                    "--count",
                    count,
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(_ROOT),
            )
            out = (result.stdout + result.stderr).strip()
            self._log.append(
                f"🧪 generate_synthetic (dry-run) ch={channel} count={count}:\n"
                + (out[:600] if out else "(no output)")
            )
            self._log.append(
                "⚠️  실제 생성은 CLI에서: python -m src.scripts.generate_synthetic "
                f"--channel {channel} --count {count}"
            )
        except Exception as exc:
            self._log.append(f"❌ {exc}")

    def _save_preprocess(self) -> None:
        try:
            mean = [float(x.strip()) for x in self._mean_edit.text().split(",")]
            std = [float(x.strip()) for x in self._std_edit.text().split(",")]
            assert len(mean) == 3 and len(std) == 3, "mean/std must each have 3 values"

            src_cfg: dict = json.loads(_SRC_CONFIG.read_text(encoding="utf-8"))
            src_cfg.setdefault("data", {}).update(
                {
                    "image_size": self._img_size.value(),
                    "split_ratios": {
                        "train": round(self._train_split.value(), 2),
                        "val": round(self._val_split.value(), 2),
                        "test": round(self._test_split.value(), 2),
                    },
                    "normalization": {"mean": mean, "std": std},
                }
            )
            src_cfg.setdefault("storage", {})[
                "labeled_dir"
            ] = self._labeled_dir_edit.text()
            _SRC_CONFIG.write_text(
                json.dumps(src_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.cfg.update(src_cfg)
            self._log.append("✅ 전처리 파라미터 저장 완료 → src/config/config.json")
            self.refresh()
        except Exception as exc:
            self._log.append(f"❌ 저장 실패: {exc}")
