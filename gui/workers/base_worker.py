from PyQt6.QtCore import QThread, pyqtSignal


class BaseWorker(QThread):
    """Base QThread worker that defines the required signals for all workers.

    Signals:
        progress_updated: int (0-100)
        log_emitted: str
        finished: dict
        error_occurred: str
    """

    progress_updated = pyqtSignal(int)
    log_emitted = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""

        self._cancel_requested = True
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        """Return whether the worker has been asked to stop."""

        return self._cancel_requested or self.isInterruptionRequested()

    def emit_progress(self, value: int, message: str) -> None:
        """Emit progress and a log message using the common contract."""

        self.progress_updated.emit(max(0, min(100, int(value))))
        if message:
            self.log_emitted.emit(message)

    def run(self) -> None:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError()
