"""Reusable image preview widget."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ImageViewer(QWidget):
    """Display a selected image without embedding prediction logic."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(220)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

    def load_image(self, path: str | Path) -> None:
        """Load an image from disk into the preview area."""

        image_path = Path(path)
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText(f"Unable to load image: {image_path}")
            return
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
