"""Data tab for dataset summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout

from gui.tabs.base_tab import BaseTab


class DataTab(BaseTab):
    """Scan configured dataset folders and display counts by CMYK channel."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Channel", "L1", "L2", "L3", "L4", "L5", "L6"])
        scan_button = QPushButton("Scan Dataset")
        scan_button.clicked.connect(self.refresh)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Dataset Overview"))
        layout.addWidget(scan_button)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self) -> None:
        """Refresh channel and level sample counts."""

        channels = self.cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
        data_root = Path(self.cfg.get("storage", {}).get("labeled_dir", "data_set/labeled"))
        self.table.setRowCount(len(channels))
        for row, channel in enumerate(channels):
            self.table.setItem(row, 0, QTableWidgetItem(channel))
            for level in range(1, 7):
                folder = data_root / channel / str(level)
                count = len([p for p in folder.glob("*") if p.is_file()]) if folder.exists() else 0
                self.table.setItem(row, level, QTableWidgetItem(str(count)))
