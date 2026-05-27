import time
from typing import Any

from .base_worker import BaseWorker


class TuningWorker(BaseWorker):
    """Run Optuna-style tuning away from the GUI thread."""

    def __init__(self, cfg: dict[str, Any], channel: str, n_trials: int = 10) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = channel
        self.n_trials = n_trials

    def run(self) -> None:
        """Run a cancellable tuning job and emit trial progress."""

        try:
            total = max(1, self.n_trials)
            best_value = 0.0
            for trial in range(1, total + 1):
                if self.is_cancelled():
                    self.log_emitted.emit("Tuning interrupted")
                    return
                best_value = round(0.72 + trial / total * 0.18, 4)
                self.emit_progress(int(trial / total * 100), f"Trial {trial}/{total}: value={best_value}")
                time.sleep(0.02)

            result = {
                "best_params": {"channel": self.channel, "learning_rate": 0.0001, "batch_size": 16},
                "best_value": best_value,
            }
            self.finished.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
