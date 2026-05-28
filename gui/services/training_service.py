"""Training service adapter for GUI orchestration."""

from __future__ import annotations

from typing import Any

from gui.workers.training_worker import TrainingWorker


class TrainingService:
    """Create and manage training workers behind a stable GUI contract."""

    def __init__(self) -> None:
        self._worker: TrainingWorker | None = None

    def start_training(self, cfg: dict[str, Any], phase: int, channel: str) -> TrainingWorker:
        """Create a worker for a training request without blocking the UI."""

        self._worker = TrainingWorker(cfg, phase, channel)
        return self._worker

    def stop_training(self) -> None:
        """Cancel the active training worker if one exists."""

        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._worker = None
