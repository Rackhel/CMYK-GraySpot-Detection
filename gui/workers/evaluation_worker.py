"""EvaluationWorker — 평가를 백그라운드 QThread에서 실행.
Runs model evaluation in a background QThread.

Contract: Contract_gui.md §2.3
SSOT:     SSOT_GUI.md §3
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


class EvaluationWorker(BaseWorker):
    """Run evaluation/report preparation away from the main GUI thread.

    Constructor args (Contract_gui.md §2.3):
        cfg             : dict — load_config() 반환값
        channel         : str  — "Y" | "M" | "C" | "K"
        checkpoint_path : str  — .pt 파일 경로 (비어있으면 models_dir/best_{channel}.pt 자동 탐색)
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        channel: str,
        checkpoint_path: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel = channel
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        """run_evaluate()를 호출하고 결과를 시그널로 전달.
        Calls run_evaluate() and emits results via signals.
        """
        try:
            from src.scripts.evaluate import run_evaluate

            self.emit_progress(0, f"[{self.channel}] 평가 시작 / Evaluation started")

            if self.is_cancelled():
                self.log_emitted.emit("Evaluation cancelled before start")
                return

            ckpt = Path(self.checkpoint_path) if self.checkpoint_path else None
            storage = self.cfg.get("storage", {})
            reports_dir = Path(storage.get("reports_dir", "outputs/reports"))

            self.emit_progress(20, f"[{self.channel}] 모델 로드 중 / Loading model...")
            report_path = run_evaluate(
                channel=self.channel,
                cfg=self.cfg,
                checkpoint=ckpt,
                output_dir=reports_dir,
            )

            self.emit_progress(100, f"[{self.channel}] 평가 완료 / Evaluation finished → {report_path}")

            # JSON 리포트에서 메트릭 읽기 / Read metrics from JSON report
            accuracy = macro_f1 = mae = 0.0
            n_samples = 0
            try:
                import json
                data = json.loads(Path(report_path).read_text(encoding="utf-8"))
                accuracy  = float(data.get("accuracy", 0.0))
                macro_f1  = float(data.get("macro_f1", 0.0))
                mae       = float(data.get("mae", 0.0))
                n_samples = int(data.get("n_samples", 0))
            except Exception:
                pass

            self.finished.emit({
                "accuracy":    accuracy,
                "macro_f1":    macro_f1,
                "mae":         mae,
                "n_samples":   n_samples,
                "report_path": str(report_path),
                "channel":     self.channel,
            })

        except Exception as exc:
            import traceback
            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")
