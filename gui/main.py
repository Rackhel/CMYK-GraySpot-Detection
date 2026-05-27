"""QApplication entry point for the CMYK PyQt6 GUI."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> None:
    """Start the desktop application."""

    app = QApplication.instance() or QApplication(sys.argv)
    theme_path = Path(__file__).resolve().parent / "styles" / "dark_theme.qss"
    if theme_path.exists():
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
