import time
from typing import Any

from .base_worker import BaseWorker


class EmbeddingWorker(BaseWorker):
    """Extract embedding payloads away from the GUI thread."""

    def __init__(self, cfg: dict[str, Any], channel: str, checkpoint_path: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = channel
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        """Build a GUI-safe t-SNE style payload."""

        try:
            for i in range(0, 101, 25):
                if self.is_cancelled():
                    self.log_emitted.emit("Embedding interrupted")
                    return
                self.emit_progress(i, f"Embedding progress: {i}%")
                time.sleep(0.02)

            result = {
                "embeddings_2d": [[0.1, 0.2], [0.3, 0.8], [0.7, 0.4], [0.9, 0.6]],
                "labels": [1, 2, 3, 4],
                "paths": ["sample_1.png", "sample_2.png", "sample_3.png", "sample_4.png"],
            }
            self.finished.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
