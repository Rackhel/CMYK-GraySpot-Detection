"""AppWindow — 메인 윈도우: 헤더 바 + 사이드바(숨기기 가능) + 추론 뷰.
Main window with a toggleable sidebar and full-screen inference panel.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui_for_user.i18n import get_lang, set_lang, t
from gui_for_user.inference_view import InferenceView
from gui_for_user.sidebar import SidebarWidget

_SIDEBAR_W = 280  # 사이드바 기본 너비 (px)


class AppWindow(QMainWindow):
    """헤더 + 토글 가능한 사이드바 + 추론 패널."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(t("app_title"))
        self.resize(1160, 740)
        self.setMinimumSize(720, 500)
        self._sidebar_open = True

        central = QWidget()
        self.setCentralWidget(central)
        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        root_v.addWidget(self._build_header())

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(4)
        self._splitter.setChildrenCollapsible(True)

        self.sidebar = SidebarWidget()
        self.inference_view = InferenceView()

        self._splitter.addWidget(self.sidebar)
        self._splitter.addWidget(self.inference_view)
        self._splitter.setSizes([_SIDEBAR_W, 880])
        self._splitter.setCollapsible(0, True)

        self.sidebar.settings_applied.connect(self.inference_view.apply_settings)

        root_v.addWidget(self._splitter, stretch=1)

        # 앱 시작 시 사이드바 설정을 바로 추론 뷰에 반영
        self.inference_view.apply_settings(self.sidebar.collect_settings())

    # ── Header bar ────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(
            "QWidget { background:#181825; border-bottom:1px solid #313244; }"
        )
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 0, 12, 0)
        h.setSpacing(10)

        # 사이드바 토글 버튼
        self._toggle_btn = QPushButton("☰")
        self._toggle_btn.setFixedSize(36, 32)
        self._toggle_btn.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; font-size:16px;"
            " border-radius:5px; border:none; }"
            "QPushButton:hover { background:#45475a; }"
        )
        self._toggle_btn.setToolTip(t("tooltip_toggle"))
        self._toggle_btn.clicked.connect(self._toggle_sidebar)

        # 앱 타이틀
        self._title_lbl = QLabel(t("app_title"))
        self._title_lbl.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#89b4fa; background:transparent;"
        )

        # 모드 배지
        self._mode_badge = QLabel("")
        self._mode_badge.setStyleSheet(
            "background:#313244; color:#a6e3a1; font-size:11px;"
            " border-radius:4px; padding:2px 8px;"
        )

        # 스페이서
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # 도움말 레이블
        self._hint_lbl = QLabel(t("header_hint"))
        self._hint_lbl.setStyleSheet(
            "font-size:11px; color:#585b70; background:transparent;"
        )

        # 언어 전환 버튼
        self._lang_btn = QPushButton(t("btn_lang"))
        self._lang_btn.setFixedSize(72, 28)
        self._lang_btn.setStyleSheet(
            "QPushButton { background:#313244; color:#cdd6f4; font-size:11px;"
            " border-radius:5px; border:1px solid #45475a; }"
            "QPushButton:hover { background:#45475a; }"
        )
        self._lang_btn.clicked.connect(self._toggle_lang)

        h.addWidget(self._toggle_btn)
        h.addSpacing(6)
        h.addWidget(self._title_lbl)
        h.addSpacing(12)
        h.addWidget(self._mode_badge)
        h.addWidget(spacer)
        h.addWidget(self._hint_lbl)
        h.addSpacing(8)
        h.addWidget(self._lang_btn)
        return bar

    # ── Toggle sidebar ────────────────────────────────────────────────────────

    def _toggle_sidebar(self) -> None:
        total = self._splitter.width()
        if self._sidebar_open:
            self._splitter.setSizes([0, total])
            self._toggle_btn.setStyleSheet(
                "QPushButton { background:#45475a; color:#89b4fa; font-size:16px;"
                " border-radius:5px; border:none; }"
                "QPushButton:hover { background:#585b70; }"
            )
            self._sidebar_open = False
        else:
            self._splitter.setSizes([_SIDEBAR_W, total - _SIDEBAR_W])
            self._toggle_btn.setStyleSheet(
                "QPushButton { background:#313244; color:#cdd6f4; font-size:16px;"
                " border-radius:5px; border:none; }"
                "QPushButton:hover { background:#45475a; }"
            )
            self._sidebar_open = True

    # ── Toggle language ───────────────────────────────────────────────────────

    def _toggle_lang(self) -> None:
        new_lang = "en" if get_lang() == "ko" else "ko"
        set_lang(new_lang)
        self._retranslate_all()

    def _retranslate_all(self) -> None:
        self.setWindowTitle(t("app_title"))
        self._title_lbl.setText(t("app_title"))
        self._hint_lbl.setText(t("header_hint"))
        self._lang_btn.setText(t("btn_lang"))
        self._toggle_btn.setToolTip(t("tooltip_toggle"))
        self.sidebar.retranslate()
        self.inference_view.retranslate()

    # ── Close event ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        w = self.inference_view._infer_worker
        if w is not None and w.isRunning():
            w.cancel()
            w.wait(2000)
        event.accept()
