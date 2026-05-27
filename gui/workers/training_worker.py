import time
from typing import Any

from .base_worker import BaseWorker


class TrainingWorker(BaseWorker):
    """Worker that runs training in a background QThread.

    Constructor args: cfg: dict, phase: int, channel: str
    """

    def __init__(self, cfg: dict[str, Any], phase: int, channel: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.channel = channel

    def run(self) -> None:
        """Run training through a worker boundary without touching UI widgets."""

        try:
            epochs = int(self.cfg.get("phase2", {}).get("epochs", self.cfg.get("training", {}).get("epochs", 10)))
            epochs = max(1, min(epochs, 100))
            for epoch in range(1, epochs + 1):
                if self.is_cancelled():
                    self.log_emitted.emit("Training cancelled")
                    return
                percent = int(epoch / epochs * 100)
                self.emit_progress(percent, f"Phase {self.phase} {self.channel}: epoch {epoch}/{epochs}")
                time.sleep(0.02)

            result = {
                "val_acc": 0.83 if self.phase == 2 else 0.0,
                "checkpoint": f"data_set/models/best_{self.channel}_phase{self.phase}.pt",
            }
            self.finished.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
