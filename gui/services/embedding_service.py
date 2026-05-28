"""Embedding service adapter for GUI orchestration."""

from __future__ import annotations

from typing import Any

from gui.workers.embedding_worker import EmbeddingWorker


class EmbeddingService:
    """Create and manage embedding workers through a stable boundary."""

    def __init__(self) -> None:
        self._worker: EmbeddingWorker | None = None

    def start_embedding(
        self,
        cfg: dict[str, Any],
        channel: str,
        checkpoint_path: str,
    ) -> EmbeddingWorker:
        """Create an embedding worker."""

        self._worker = EmbeddingWorker(cfg, channel, checkpoint_path)
        return self._worker

    def stop_embedding(self) -> None:
        """Cancel the active embedding worker."""

        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self._worker = None
