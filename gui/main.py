"""QApplication entry point for the CMYK PyQt6 GUI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

# QWebEngineView는 QApplication 생성 전에 import되어야 한다 (macOS/Linux 필수).
# Must be imported before QApplication is instantiated on macOS/Linux.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView as _QWebEngineView  # noqa: F401

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
except Exception:
    pass

from gui.i18n import set_lang
from gui.main_window import MainWindow

_ASSETS = Path(__file__).resolve().parent / "assets"
_STYLES = Path(__file__).resolve().parent / "styles"
_GUI_CFG = Path(__file__).resolve().parent / "assets" / "config.json"

# 플랫폼별 폰트 후보 (앞에서부터 실제 존재하는 것을 선택)
_PLATFORM_FONTS = {
    "darwin": [".AppleSystemUIFont", "Helvetica Neue", "Helvetica", "Arial"],
    "win32": ["Segoe UI", "Arial"],
    "linux": ["Ubuntu", "DejaVu Sans", "Arial"],
}


def _detect_font() -> str:
    """실제 Qt 폰트 DB에 존재하는 첫 번째 폰트 이름을 반환한다."""
    from PyQt6.QtGui import QFontDatabase

    candidates = _PLATFORM_FONTS.get(sys.platform, ["Arial"])
    available = set(QFontDatabase.families())
    for name in candidates:
        if name in available:
            return name
    return "Arial"


def _load_gui_cfg() -> dict:
    try:
        if _GUI_CFG.exists():
            return json.loads(_GUI_CFG.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_qss(name: str, font: str = "Arial") -> str:
    path = _STYLES / name
    if not path.exists():
        return ""
    return (
        path.read_text(encoding="utf-8")
        .replace("%ASSETS%", _ASSETS.as_posix())
        .replace("%FONT%", font)
    )


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)

    # ── 실제 존재하는 폰트 감지 후 QApplication에 설정 ────────────────────
    font_name = _detect_font()
    app.setFont(QFont(font_name, 10))

    # ── 저장된 설정 로드 / Load persisted appearance settings ──────────────
    gui_cfg = _load_gui_cfg()
    theme = gui_cfg.get("theme", "dark")
    lang = gui_cfg.get("lang", "ko")

    # 언어 전역 설정 (MainWindow 생성 전에 적용해야 탭 라벨이 올바르게 뜸)
    set_lang(lang)

    # 테마 QSS 적용 — %FONT% 에 감지된 폰트명 주입
    qss_file = "dark_theme.qss" if theme == "dark" else "light_theme.qss"
    qss = _load_qss(qss_file, font=font_name)
    if qss:
        app.setStyleSheet(qss)

    # ── 메인 윈도우 ────────────────────────────────────────────────────────
    window = MainWindow()

    # Settings 탭 콤보박스를 저장된 값으로 초기화
    st = window.settings_tab
    idx = st._theme_combo.findData(theme)
    if idx >= 0:
        st._theme_combo.blockSignals(True)
        st._theme_combo.setCurrentIndex(idx)
        st._theme_combo.blockSignals(False)

    idx = st._lang_combo.findData(lang)
    if idx >= 0:
        st._lang_combo.blockSignals(True)
        st._lang_combo.setCurrentIndex(idx)
        st._lang_combo.blockSignals(False)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
