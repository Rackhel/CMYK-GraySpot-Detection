"""Evaluation service adapter for GUI orchestration."""

from __future__ import annotations

from typing import Any

from gui.workers.evaluation_worker import EvaluationWorker


class EvaluationService:
    """Create and manage evaluation workers through a stable boundary."""

    def __init__(self) -> None:
        self._worker: EvaluationWorker | None = None

    def start_evaluation(
        self,
        cfg: dict[str, Any],
        channel: str,
        checkpoint_path: str,
    ) -> EvaluationWorker:
        """Create an evaluation worker."""

        self._worker = EvaluationWorker(cfg, channel, checkpoint_path)
        return self._worker

    def stop_evaluation(self) -> None:
        """Cancel the active evaluation worker."""

        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._worker = None
