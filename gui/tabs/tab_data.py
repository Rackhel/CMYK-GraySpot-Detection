"""Data tab — 데이터셋 채널×레벨 샘플 수 표시 및 이미지 미리보기.
Dataset channel×level sample count display and image preview.

Contract: Contract_gui.md §3  /  SSOT_GUI.md §6.1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from gui.components.image_viewer import ImageViewer
from gui.tabs.base_tab import BaseTab


class DataTab(BaseTab):
    """Scan configured dataset folders and display counts by CMYK channel × level.

    레벨 폴더는 0-based (0~N-1) 로 저장된다.
    Level folders are 0-based (0 ~ num_levels-1).
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)

        num_levels = self.cfg.get("data", {}).get("num_levels", 6)
        self._num_levels = num_levels

        # 헤더: Channel + L0~L(N-1) + Total
        headers = ["Channel"] + [f"L{i}" for i in range(num_levels)] + ["Total"]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self._on_cell_clicked)

        # 이미지 미리보기
        self.image_viewer = ImageViewer()

        # 버튼
        scan_button = QPushButton("Scan Dataset")
        scan_button.clicked.connect(self.refresh)
        browse_button = QPushButton("Browse Image…")
        browse_button.clicked.connect(self._browse_image)

        # 레이아웃
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Dataset Overview — Channel × Level Sample Count"))

        btn_row = QHBoxLayout()
        btn_row.addWidget(scan_button)
        btn_row.addWidget(browse_button)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(self.table)
        layout.addWidget(QLabel("Image Preview"))
        layout.addWidget(self.image_viewer)

        self.refresh()

    # ── BaseTab interface ─────────────────────────────────────────────────────

    def on_worker_finished(self, result: dict[str, Any]) -> None:
        """Data 탭은 Worker를 사용하지 않으므로 no-op."""

    def refresh(self) -> None:
        """채널×레벨 샘플 수를 다시 스캔한다. / Re-scan sample counts."""
        channels  = self.cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
        data_root = Path(self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
        num_levels = self._num_levels

        self.table.setRowCount(len(channels))
        for row, channel in enumerate(channels):
            self.table.setItem(row, 0, QTableWidgetItem(channel))
            total = 0
            for level in range(num_levels):           # 0-based 폴더 / 0-based folders
                folder = data_root / channel / str(level)
                count = (
                    len([p for p in folder.glob("*") if p.is_file()])
                    if folder.exists()
                    else 0
                )
                total += count
                item = QTableWidgetItem(str(count))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, level + 1, item)   # col 0 = Channel

            total_item = QTableWidgetItem(str(total))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, num_levels + 1, total_item)

        self.table.resizeColumnsToContents()

        # 경로 존재 여부 표시
        status = f"labeled_dir: {data_root}  ({'존재' if data_root.exists() else '없음 / NOT FOUND'})"
        self.table.setToolTip(status)

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_cell_clicked(self, row: int, col: int) -> None:
        """테이블 셀 클릭 시 첫 번째 이미지 미리보기 / Preview first image on cell click."""
        if col == 0 or col > self._num_levels:
            return
        channels  = self.cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
        data_root = Path(self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
        channel = channels[row] if row < len(channels) else None
        level   = col - 1       # 0-based
        if channel is None:
            return
        folder = data_root / channel / str(level)
        images = [p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
        if images:
            self.image_viewer.load_image(images[0])

    def _browse_image(self) -> None:
        """파일 다이얼로그로 이미지를 직접 선택해 미리본다."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "data_set/labeled", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.image_viewer.load_image(path)
