"""
test_gui_workers.py
Tests for PyQt6 Worker thread classes (gui/workers/).
Status: FAILING — gui/workers/ not yet implemented.
Ref: doc/TDD/TDD_GUI.md
"""

import inspect
import pytest

# Skip if PyQt6 not available
PyQt6 = pytest.importorskip("PyQt6")

# Will raise ImportError until implemented
from gui.workers.training_worker import TrainingWorker
from gui.workers.evaluation_worker import EvaluationWorker
from gui.workers.tuning_worker import TuningWorker
from gui.workers.embedding_worker import EmbeddingWorker

ALL_WORKERS = [TrainingWorker, EvaluationWorker, TuningWorker, EmbeddingWorker]
REQUIRED_SIGNALS = ["progress_updated", "log_emitted", "finished", "error_occurred"]


class TestWorkerSignals:
    """T-GUI-01 ~ T-GUI-07: 모든 Worker가 필수 시그널을 보유하는지 확인"""

    @pytest.mark.parametrize("WorkerClass", ALL_WORKERS)
    @pytest.mark.parametrize("signal_name", REQUIRED_SIGNALS)
    def test_worker_has_signal(self, WorkerClass, signal_name):
        """각 Worker 클래스가 필수 시그널을 보유"""
        assert hasattr(WorkerClass, signal_name), (
            f"{WorkerClass.__name__} is missing signal: {signal_name}"
        )

    def test_training_worker_inherits_qthread(self):
        """TrainingWorker는 QThread를 상속"""
        from PyQt6.QtCore import QThread
        assert issubclass(TrainingWorker, QThread)

    def test_evaluation_worker_inherits_qthread(self):
        """EvaluationWorker는 QThread를 상속"""
        from PyQt6.QtCore import QThread
        assert issubclass(EvaluationWorker, QThread)

    def test_tuning_worker_inherits_qthread(self):
        """TuningWorker는 QThread를 상속"""
        from PyQt6.QtCore import QThread
        assert issubclass(TuningWorker, QThread)

    def test_embedding_worker_inherits_qthread(self):
        """EmbeddingWorker는 QThread를 상속"""
        from PyQt6.QtCore import QThread
        assert issubclass(EmbeddingWorker, QThread)


class TestWorkerUiIsolation:
    """T-GUI-10 ~ T-GUI-11: Worker에서 UI 직접 접근 금지"""

    FORBIDDEN_UI_TOKENS = [
        "QWidget", "QLabel", "QProgressBar", "QLineEdit",
        "setText", "setValue", "QTextEdit", "QPushButton",
    ]

    @pytest.mark.parametrize("WorkerClass", ALL_WORKERS)
    def test_run_method_has_no_direct_ui_access(self, WorkerClass):
        """run() 메서드에 UI 위젯 직접 참조 없음"""
        assert hasattr(WorkerClass, "run"), f"{WorkerClass.__name__} has no run()"
        source = inspect.getsource(WorkerClass.run)
        for token in self.FORBIDDEN_UI_TOKENS:
            assert token not in source, (
                f"{WorkerClass.__name__}.run() directly references UI widget: {token}"
            )

    @pytest.mark.parametrize("WorkerClass", ALL_WORKERS)
    def test_constructor_does_not_accept_widget(self, WorkerClass):
        """Worker 생성자가 QWidget을 인수로 받지 않음"""
        sig = inspect.signature(WorkerClass.__init__)
        param_names = list(sig.parameters.keys())
        widget_params = [p for p in param_names if "widget" in p.lower()]
        assert len(widget_params) == 0, (
            f"{WorkerClass.__name__}.__init__() accepts widget params: {widget_params}"
        )
