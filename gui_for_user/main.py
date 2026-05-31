"""CMYK Level Inspector — entry point.

실행 방법 / Run:
    python -m gui_for_user.main
    python gui_for_user/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── sys.path 설정 / path setup ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── WebEngine는 QApplication 생성 전에 import 해야 함 (macOS/Linux 필수) ───────
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView as _QWE  # noqa: F401

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
except Exception:
    pass

from PyQt6.QtGui import QFont, QFontDatabase

_DARK_QSS = """
QWidget            { background:#1e1e2e; color:#cdd6f4; font-size:12px; }
QGroupBox          { border:1px solid #45475a; border-radius:6px; margin-top:10px;
                     padding-top:8px; }
QGroupBox::title   { color:#89b4fa; subcontrol-position:top left; padding:0 4px; }
QPushButton        { background:#313244; border:1px solid #45475a; border-radius:5px;
                     padding:4px 12px; min-height:26px; }
QPushButton:hover  { background:#45475a; }
QPushButton:pressed{ background:#585b70; }
QPushButton:disabled{ color:#585b70; border-color:#313244; }
QLineEdit          { background:#313244; border:1px solid #45475a; border-radius:4px;
                     padding:3px 6px; }
QLineEdit:focus    { border-color:#89b4fa; }
QComboBox          { background:#313244; border:1px solid #45475a; border-radius:4px;
                     padding:3px 6px; min-height:24px; }
QComboBox::drop-down  { border:none; width:20px; }
QComboBox QAbstractItemView { background:#313244; border:1px solid #45475a;
                               selection-background-color:#45475a; }
QTableWidget       { background:#181825; gridline-color:#313244; border:none; }
QTableWidget::item { padding:4px; }
QTableWidget::item:selected { background:#45475a; }
QHeaderView::section { background:#313244; border:1px solid #45475a; padding:5px; }
QScrollBar:vertical   { background:#1e1e2e; width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:#45475a; border-radius:4px; min-height:20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QProgressBar       { background:#313244; border:1px solid #45475a; border-radius:4px;
                     text-align:center; min-height:8px; }
QProgressBar::chunk{ background:#89b4fa; border-radius:3px; }
QSplitter::handle  { background:#45475a; }
QLabel             { background:transparent; }
"""


def _detect_font() -> str:
    families = set(QFontDatabase.families())
    for name in [".AppleSystemUIFont", "Helvetica Neue", "Segoe UI", "Ubuntu", "Arial"]:
        if name in families:
            return name
    return "Arial"


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont(_detect_font(), 10))
    app.setStyleSheet(_DARK_QSS)

    from gui_for_user.app_window import AppWindow

    window = AppWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
