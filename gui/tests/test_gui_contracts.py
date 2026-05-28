"""PyQt6 GUI contract tests."""

from __future__ import annotations

import inspect


def test_all_workers_have_required_signals() -> None:
    """Workers expose the SSOT-defined signal interface."""

    from gui.workers.embedding_worker import EmbeddingWorker
    from gui.workers.evaluation_worker import EvaluationWorker
    from gui.workers.training_worker import TrainingWorker
    from gui.workers.tuning_worker import TuningWorker

    for worker_class in [
        TrainingWorker,
        EvaluationWorker,
        TuningWorker,
        EmbeddingWorker,
    ]:
        for signal_name in [
            "progress_updated",
            "log_emitted",
            "finished",
            "error_occurred",
        ]:
            assert hasattr(worker_class, signal_name)


def test_training_worker_does_not_touch_ui_directly() -> None:
    """Workers must not directly manipulate QWidget instances."""

    from gui.workers.training_worker import TrainingWorker

    source = inspect.getsource(TrainingWorker.run)
    for token in [
        "QWidget",
        "QLabel",
        "QProgressBar",
        "QLineEdit",
        "setText",
        "setValue",
    ]:
        assert token not in source


def test_main_window_has_six_tabs(qtbot) -> None:
    """MainWindow follows the six-tab SSOT contract."""

    from gui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.tab_widget.count() == 6


def test_all_tabs_implement_interface(qtbot) -> None:
    """All tab widgets expose refresh and completion hooks."""

    from gui.tabs import (
        DataTab,
        EmbeddingTab,
        EvaluationTab,
        OptunaTab,
        SettingsTab,
        TrainingTab,
    )

    for tab_class in [
        DataTab,
        TrainingTab,
        EvaluationTab,
        SettingsTab,
        OptunaTab,
        EmbeddingTab,
    ]:
        tab = tab_class()
        qtbot.addWidget(tab)
        assert hasattr(tab, "refresh")
        assert hasattr(tab, "on_worker_finished")


def test_embedding_tab_save_label(qtbot, tmp_path) -> None:
    """Embedding corrections are saved to version-incremented CSV files."""

    from gui.tabs.tab_embedding import EmbeddingTab

    tab = EmbeddingTab(cfg={}, labels_dir=tmp_path)
    qtbot.addWidget(tab)
    tab.save_label_correction("img1.png", 3)
    csv_files = list(tmp_path.glob("labels_v*.csv"))
    assert csv_files
    assert "img1.png,3" in csv_files[0].read_text(encoding="utf-8")
