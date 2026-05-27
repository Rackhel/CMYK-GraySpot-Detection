"""Tuning service adapter for GUI orchestration."""

from __future__ import annotations

from typing import Any

from gui.workers.tuning_worker import TuningWorker


class TuningService:
    """Create and manage HPO workers through a stable boundary."""

    def __init__(self) -> None:
        self._worker: TuningWorker | None = None

    def start_tuning(
        self,
        cfg: dict[str, Any],
        channel: str,
        n_trials: int,
        phase: int = 2,
    ) -> TuningWorker:
        """Create a tuning worker."""

        self._worker = TuningWorker(cfg, channel, n_trials, phase)
        return self._worker

    def stop_tuning(self) -> None:
        """Cancel the active tuning worker."""

        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._worker = None
