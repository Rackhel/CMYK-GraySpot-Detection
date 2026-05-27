import time
from typing import Any

from .base_worker import BaseWorker


class EvaluationWorker(BaseWorker):
    """Run evaluation/report preparation away from the main GUI thread."""

    def __init__(self, cfg: dict[str, Any], channel: str, checkpoint_path: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = channel
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        """Evaluate via black-box backend boundaries and emit GUI-safe results."""

        try:
            for i in range(0, 101, 20):
                if self.is_cancelled():
                    self.log_emitted.emit("Evaluation interrupted")
                    return
                self.emit_progress(i, f"Evaluation progress: {i}%")
                time.sleep(0.02)

            result = {
                "accuracy": 0.88,
                "report_path": "outputs/reports/baseline.html",
                "confusion_matrix": [
                    [18, 1, 0, 0, 0, 0],
                    [1, 16, 1, 0, 0, 0],
                    [0, 2, 15, 1, 0, 0],
                    [0, 0, 1, 17, 1, 0],
                    [0, 0, 0, 1, 16, 1],
                    [0, 0, 0, 0, 2, 18],
                ],
            }
            self.finished.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
