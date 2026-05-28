"""LevelAccuracyTable — 레벨별 / 채널별 정확도 요약 테이블 위젯.
Displays per-level accuracy broken down by channel after batch inference.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LevelAccuracyTable(QWidget):
    """Table showing: Level | Y acc | M acc | C acc | K acc | Overall avg."""

    def __init__(self, num_levels: int = 6, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._num_levels = num_levels
        self._channels: list[str] = []

        self._title = QLabel("레벨별 정확도 / Per-Level Accuracy")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        headers = ["Level"] + ["Y", "M", "C", "K"] + ["Overall"]
        self._table = QTableWidget(num_levels, len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setMaximumHeight(240)

        # Populate level column
        for lvl in range(num_levels):
            item = QTableWidgetItem(f"L{lvl}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(lvl, 0, item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._table)

    def update_from_results(self, results: list[dict], channel: str = "?") -> None:
        """Update table from batch-inference results list.

        Each result dict must have: pred_level (int), true_level (int, optional).
        If true_level absent, shows raw count only.

        For accuracy to be meaningful, each image filename should encode the true
        level (e.g., the parent folder name). We derive true_level from path when
        present, otherwise mark as '—'.
        """
        from pathlib import Path

        # Group by true level (derived from folder name)
        per_level: dict[int, dict] = {i: {"total": 0, "correct": 0} for i in range(self._num_levels)}

        for r in results:
            if r.get("error"):
                continue
            path = r.get("path", "")
            try:
                true_level = int(Path(path).parent.name)
            except (ValueError, TypeError):
                continue  # can't derive level
            pred_level = r.get("pred_level", -1)
            if 0 <= true_level < self._num_levels:
                per_level[true_level]["total"] += 1
                if pred_level == true_level:
                    per_level[true_level]["correct"] += 1

        ch_col = {"Y": 1, "M": 2, "C": 3, "K": 4}.get(channel, None)

        for lvl in range(self._num_levels):
            d = per_level[lvl]
            if d["total"] > 0:
                acc = d["correct"] / d["total"]
                text = f"{acc:.1%} ({d['correct']}/{d['total']})"
                color = self._acc_color(acc)
            else:
                text = "—"
                color = None

            if ch_col is not None:
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(Qt.GlobalColor.green if acc >= 0.8 else Qt.GlobalColor.yellow if acc >= 0.5 else Qt.GlobalColor.red)
                self._table.setItem(lvl, ch_col, item)

            # Overall column
            overall_item = QTableWidgetItem(text)
            overall_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(lvl, 5, overall_item)

        self._table.resizeColumnsToContents()

    def reset(self) -> None:
        for r in range(self._num_levels):
            for c in range(1, 6):
                self._table.setItem(r, c, QTableWidgetItem("—"))

    @staticmethod
    def _acc_color(acc: float) -> str | None:
        if acc >= 0.8:
            return "green"
        if acc >= 0.5:
            return "yellow"
        return "red"
