"""Main PyQt6 window for the CMYK engineering GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar, QTabWidget

from gui.i18n import set_lang, t
from gui.tabs import (
    DataTab,
    EmbeddingTab,
    EvaluationTab,
    InferenceTab,
    OptunaTab,
    SettingsTab,
    TrainingTab,
)

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ASSETS = Path(__file__).resolve().parent / "assets"
_STYLES = Path(__file__).resolve().parent / "styles"


def _load_cfg() -> dict[str, Any]:
    try:
        from src.utils.utils_config import load_config

        return load_config()
    except Exception:
        return {
            "data": {"channels": ["Y", "M", "C", "K"], "num_levels": 6},
            "storage": {
                "labeled_dir": "data_set/labeled",
                "data_root": "data_set",
                "models_dir": "data_set/models",
                "reports_dir": "outputs/reports",
            },
            "phase2": {"epochs": 50},
        }


def _load_qss(name: str) -> str:
    """QSS 파일 로드 후 %ASSETS% 치환 / Load QSS and substitute %ASSETS%."""
    path = _STYLES / name
    if not path.exists():
        return ""
    qss = path.read_text(encoding="utf-8")
    return qss.replace("%ASSETS%", _ASSETS.as_posix())


class MainWindow(QMainWindow):
    """Main window — tab orchestration, theme & language switching."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = _load_cfg()
        self.setWindowTitle("CMYK AI - Engineering GUI")
        self.resize(1280, 820)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tab_widget)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._add_tabs()
        self._connect_appearance_signals()
        self.status.showMessage("Ready")

    # ── Tab setup ──────────────────────────────────────────────────────────────

    def _add_tabs(self) -> None:
        self.settings_tab = SettingsTab(self.cfg)

        self.data_tab = DataTab(self.cfg)
        self.training_tab = TrainingTab(self.cfg)
        self.evaluation_tab = EvaluationTab(self.cfg, settings_tab=self.settings_tab)
        self.optuna_tab = OptunaTab(self.cfg)
        self.embedding_tab = EmbeddingTab(
            self.cfg,
            labels_dir=Path(self.cfg.get("storage", {}).get("data_root", "data_set")),
            settings_tab=self.settings_tab,
        )
        self.inference_tab = InferenceTab(self.cfg)

        self.tab_widget.addTab(self.data_tab, t("tab_data"))
        self.tab_widget.addTab(self.training_tab, t("tab_training"))
        self.tab_widget.addTab(self.evaluation_tab, t("tab_evaluation"))
        self.tab_widget.addTab(self.settings_tab, t("tab_settings"))
        self.tab_widget.addTab(self.optuna_tab, t("tab_optuna"))
        self.tab_widget.addTab(self.embedding_tab, t("tab_embedding"))
        self.tab_widget.addTab(self.inference_tab, t("tab_inference"))

    def _connect_appearance_signals(self) -> None:
        """Settings 탭의 테마·언어 콤보에 즉시 적용 슬롯 연결."""
        self.settings_tab._theme_combo.currentIndexChanged.connect(
            lambda _: self._apply_theme(self.settings_tab._theme_combo.currentData())
        )
        self.settings_tab._lang_combo.currentIndexChanged.connect(
            lambda _: self._apply_lang(self.settings_tab._lang_combo.currentData())
        )

    # ── Theme ──────────────────────────────────────────────────────────────────

    def apply_theme(self, name: str) -> None:
        """외부(main.py)에서 초기 테마 적용 시 사용."""
        self._apply_theme(name)

    def _apply_theme(self, name: str) -> None:
        """'dark' | 'light' 테마를 즉시 적용한다."""
        from gui.main import _detect_font
        from gui.main import _load_qss as _mload_qss

        qss_file = "dark_theme.qss" if name == "dark" else "light_theme.qss"
        font_name = _detect_font()
        qss = _mload_qss(qss_file, font=font_name)
        if qss:
            QApplication.instance().setStyleSheet(qss)
        self.status.showMessage(f"Theme: {name}")

    # ── Language ───────────────────────────────────────────────────────────────

    def apply_lang(self, lang: str) -> None:
        """외부(main.py)에서 초기 언어 적용 시 사용."""
        self._apply_lang(lang)

    def _apply_lang(self, lang: str) -> None:
        """언어를 즉시 적용한다 — 탭 라벨 + 각 탭 retranslate."""
        set_lang(lang)
        self._retranslate_tab_labels()
        self._retranslate_all(lang)
        self.status.showMessage(f"Language: {'한국어' if lang == 'ko' else 'English'}")

    def _retranslate_tab_labels(self) -> None:
        keys = [
            "tab_data",
            "tab_training",
            "tab_evaluation",
            "tab_settings",
            "tab_optuna",
            "tab_embedding",
            "tab_inference",
        ]
        for i, key in enumerate(keys):
            self.tab_widget.setTabText(i, t(key))

    def _retranslate_all(self, lang: str) -> None:
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            if hasattr(tab, "retranslate_ui"):
                tab.retranslate_ui(lang)

    # ── Tab change ─────────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tab_widget.widget(index)
        if hasattr(widget, "refresh"):
            widget.refresh()
        self.status.showMessage(self.tab_widget.tabText(index))

    # ── Close ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        worker_attrs = ("worker", "eval_worker", "infer_worker", "batch_worker")
        for tab_index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(tab_index)
            for attr in worker_attrs:
                worker = getattr(tab, attr, None)
                if worker is not None and worker.isRunning():
                    worker.cancel()
                    worker.wait(2000)
        event.accept()
