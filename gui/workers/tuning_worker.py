"""TuningWorker — Optuna HPO를 백그라운드 QThread에서 실행.
Contract: Contract_gui.md §2.4  /  SSOT_GUI.md §3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base_worker import BaseWorker

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TuningWorker(BaseWorker):
    """Run Optuna HPO away from the GUI thread.

    Args (Contract §2.4):
        cfg      : dict — load_config() 반환값
        channel  : str  — "Y" | "M" | "C" | "K" | "all"
        n_trials : int  — trial 수
        phase    : int  — 0 (SimCLR) or 2 (Supervised)
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        channel: str,
        n_trials: int = 10,
        phase: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = channel
        self.n_trials = n_trials
        self.phase = phase

    def run(self) -> None:
        try:
            from src.tuning.optuna_tuner import run_optuna

            ch_lower = self.channel.lower()
            self.emit_progress(
                0,
                f"[{self.channel}] Phase {self.phase} Optuna HPO 시작 ({self.n_trials} trials)",
            )

            if self.is_cancelled():
                self.log_emitted.emit("Tuning cancelled before start")
                return

            run_optuna(n_trials=self.n_trials, channel=ch_lower)

            if self.is_cancelled():
                self.log_emitted.emit("Tuning interrupted")
                return

            self.emit_progress(90, f"[{self.channel}] best params 로드 중")

            # load_best_params 실제 시그니처: (channel, output_dir)
            best_params: dict = {}
            best_value: float = 0.0
            try:
                from src.utils.optuna_utils import load_best_params
                output_dir = Path("outputs/optuna")
                best_params = load_best_params(channel=ch_lower, output_dir=output_dir)
                best_value = float(best_params.pop("_best_value", 0.0))
            except Exception as e:
                self.log_emitted.emit(f"[WARN] best params 로드 실패: {e}")

            self.emit_progress(100, f"[{self.channel}] HPO 완료")
            self.finished.emit({
                "best_params": best_params,
                "best_value":  best_value,
                "channel":     self.channel,
            })

        except Exception as exc:
            import traceback
            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")
