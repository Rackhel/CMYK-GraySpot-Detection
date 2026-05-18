"""
test_gui_tabs.py
Tests for PyQt6 tab widget classes (gui/tabs/).
Status: FAILING — gui/tabs/ not yet implemented.
Ref: doc/TDD/TDD_GUI.md
"""

import pytest

# Skip if PyQt6 not available
PyQt6 = pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")  # requires pytest-qt

# Will raise ImportError until implemented
from gui.tabs.tab_data import DataTab
from gui.tabs.tab_training import TrainingTab
from gui.tabs.tab_evaluation import EvaluationTab
from gui.tabs.tab_settings import SettingsTab
from gui.tabs.tab_optuna import OptunaTab
from gui.tabs.tab_embedding import EmbeddingTab

ALL_TABS = [DataTab, TrainingTab, EvaluationTab, SettingsTab, OptunaTab, EmbeddingTab]


class TestTabInterface:
    """T-GUI-20 ~ T-GUI-22: 모든 탭이 BaseTab 인터페이스를 구현"""

    @pytest.mark.parametrize("TabClass", ALL_TABS)
    def test_tab_has_refresh_method(self, TabClass):
        """T-GUI-20: refresh() 메서드 존재"""
        assert hasattr(TabClass, "refresh"), f"{TabClass.__name__} missing refresh()"

    @pytest.mark.parametrize("TabClass", ALL_TABS)
    def test_tab_has_on_worker_finished(self, TabClass):
        """T-GUI-21: on_worker_finished() 메서드 존재"""
        assert hasattr(TabClass, "on_worker_finished"), (
            f"{TabClass.__name__} missing on_worker_finished()"
        )

    def test_main_window_has_six_tabs(self, qtbot):
        """T-GUI-22: MainWindow의 탭 위젯이 6개 탭 포함"""
        from gui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.tab_widget.count() == 6


class TestEmbeddingTabLabelSave:
    """T-GUI-30 ~ T-GUI-31: Tab 6 라벨 수정 저장"""

    def test_save_label_correction_creates_csv(self, qtbot, tmp_path):
        """T-GUI-30: save_label_correction() → CSV 파일 생성"""
        tab = EmbeddingTab(cfg={}, labels_dir=str(tmp_path))
        qtbot.addWidget(tab)
        tab.save_label_correction("img1.png", new_level=3)
        csv_files = list(tmp_path.glob("labels_v*.csv"))
        assert len(csv_files) > 0

    def test_save_label_correction_correct_level(self, qtbot, tmp_path):
        """T-GUI-31: 저장된 CSV에서 해당 경로 level == new_level"""
        import pandas as pd
        tab = EmbeddingTab(cfg={}, labels_dir=str(tmp_path))
        qtbot.addWidget(tab)
        tab.save_label_correction("img1.png", new_level=3)
        csv_files = list(tmp_path.glob("labels_v*.csv"))
        assert len(csv_files) > 0
        df = pd.read_csv(csv_files[0])
        row = df[df["path"] == "img1.png"]
        assert len(row) > 0
        assert row["level"].iloc[0] == 3
