"""Reusable metric card widget."""

from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class MetricCard(QFrame):
    """Display a title and value for engineering metrics."""

    def __init__(self, title: str, value: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        """Update the displayed value."""

        self.value_label.setText(value)
