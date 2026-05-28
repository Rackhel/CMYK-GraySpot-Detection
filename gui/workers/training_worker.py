"""TrainingWorker — Phase 0 / Phase 2 학습을 백그라운드 QThread에서 실행.
Contract: Contract_gui.md §2.2  /  SSOT_GUI.md §3
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


class TrainingWorker(BaseWorker):
    """Phase 0 or Phase 2 training in a background QThread.

    Args (Contract §2.2):
        cfg     : dict  — load_config() 반환값
        phase   : int   — 0 (SimCLR) 또는 2 (Supervised)
        channel : str   — "Y" | "M" | "C" | "K"
    """

    def __init__(self, cfg: dict[str, Any], phase: int, channel: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.phase = phase
        self.channel = channel

    def run(self) -> None:
        try:
            import torch

            device = self._resolve_device()
            self.emit_progress(0, f"[{self.channel}] Phase {self.phase} 학습 시작")

            if self.is_cancelled():
                self.log_emitted.emit("Training cancelled before start")
                return

            result = (
                self._run_phase0(device)
                if self.phase == 0
                else self._run_phase2(device)
            )
            self.emit_progress(100, f"[{self.channel}] 학습 완료")
            self.finished.emit(result)

        except Exception as exc:
            import traceback

            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")

    def _resolve_device(self):
        import torch

        d = self.cfg.get("system", {}).get("device", "cpu")
        if d == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        return torch.device(d)

    def _run_phase0(self, device) -> dict[str, Any]:
        from src.scripts.run_phase0 import run_phase0

        self.emit_progress(10, f"[{self.channel}] Phase 0 SimCLR 시작")
        # run_phase0 시그니처: (cfg, channel, device, optuna_trial=None)
        result = run_phase0(cfg=self.cfg, channel=self.channel, device=device)
        return {
            "val_acc": float(result.get("final_loss", 0.0)),  # Phase 0은 loss
            "checkpoint": str(result.get("backbone_path", "")),
            "phase": 0,
            "channel": self.channel,
            "skipped": result.get("skipped", False),
        }

    def _run_phase2(self, device) -> dict[str, Any]:
        from src.scripts.run_phase2 import run_phase2

        storage = self.cfg.get("storage", {})
        phase0_dir = Path(storage.get("models_dir", "data_set/models"))
        ckpt_dir = Path("outputs/checkpoints")
        self.emit_progress(10, f"[{self.channel}] Phase 2 Supervised 시작")
        result = run_phase2(
            cfg=self.cfg,
            channel=self.channel,
            device=device,
            phase0_dir=phase0_dir,
            ckpt_dir=ckpt_dir,
        )
        return {
            "val_acc": float(result.get("best_val_acc", 0.0)),
            "test_acc": float(result.get("test_acc", 0.0)),
            "mae": float(result.get("mae", 0.0)),
            "checkpoint": str(result.get("checkpoint_path", "")),
            "phase": 2,
            "channel": self.channel,
        }
