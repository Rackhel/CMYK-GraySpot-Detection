"""Main PyQt6 window for the CMYK engineering GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from gui.tabs import DataTab, EmbeddingTab, EvaluationTab, OptunaTab, SettingsTab, TrainingTab

# src/ 경로 등록 — load_config() 등 src 모듈 사용을 위해
# Register src/ path for load_config() and other src module imports
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_cfg() -> dict[str, Any]:
    """src/config/config.json 을 로드한다. 실패 시 최소 fallback 반환.
    Loads src/config/config.json. Returns minimal fallback on failure."""
    try:
        from src.utils.utils_config import load_config
        return load_config()
    except Exception:
        # 테스트 환경 또는 패키지 미설치 시 안전 fallback
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


class MainWindow(QMainWindow):
    """Main window responsible only for tab orchestration."""

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
        self.status.showMessage("Ready")

    def _add_tabs(self) -> None:
        """Create the six contract-defined tabs (SSOT_GUI.md §2 순서 / order)."""

        # Settings 먼저 생성 — checkpoint 경로를 다른 탭에 주입하기 위해
        # Create Settings first so checkpoint path can be injected into other tabs
        self.settings_tab = SettingsTab(self.cfg)

        self.data_tab       = DataTab(self.cfg)
        self.training_tab   = TrainingTab(self.cfg)
        self.evaluation_tab = EvaluationTab(self.cfg, settings_tab=self.settings_tab)
        self.optuna_tab     = OptunaTab(self.cfg)
        self.embedding_tab  = EmbeddingTab(
            self.cfg,
            labels_dir=Path(self.cfg.get("storage", {}).get("data_root", "data_set")),
            settings_tab=self.settings_tab,   # checkpoint 경로 공유
        )

        self.tab_widget.addTab(self.data_tab,       "Data")
        self.tab_widget.addTab(self.training_tab,   "Training")
        self.tab_widget.addTab(self.evaluation_tab, "Evaluation")
        self.tab_widget.addTab(self.settings_tab,   "Settings")
        self.tab_widget.addTab(self.optuna_tab,     "Optuna HPO")
        self.tab_widget.addTab(self.embedding_tab,  "Embedding")

    def _on_tab_changed(self, index: int) -> None:
        """Refresh the activated tab and update status text."""

        widget = self.tab_widget.widget(index)
        if hasattr(widget, "refresh"):
            widget.refresh()
        self.status.showMessage(f"{self.tab_widget.tabText(index)} ready")

    def closeEvent(self, event) -> None:
        """Cancel active worker threads before closing."""

        # 각 탭에서 모든 worker 속성을 찾아 안전하게 종료
        # Find all worker attributes in each tab and cancel them safely
        worker_attrs = ("worker", "eval_worker", "infer_worker")
        for tab_index in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(tab_index)
            for attr in worker_attrs:
                worker = getattr(tab, attr, None)
                if worker is not None and worker.isRunning():
                    worker.cancel()
                    worker.wait(2000)
        event.accept()
