"""InferenceWorker — 단일 이미지 추론 (단일 채널 / 앙상블).
Single-image inference: single-channel or 4-channel ensemble.

Contract: Contract_gui.md §2.6  /  SSOT_GUI.md §3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ._ckpt_utils import auto_find_all_checkpoints, auto_find_checkpoint, run_ensemble
from .base_worker import BaseWorker

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class InferenceWorker(BaseWorker):
    """단일 이미지를 지정 채널 모델 또는 4채널 앙상블로 추론한다.
    Runs single-image inference using a channel model or 4-channel ensemble.

    Args (Contract §2.6):
        cfg             : dict  — load_config() 반환값
        image_path      : str   — 추론할 이미지 경로
        checkpoint_path : str   — .pt 경로 (빈 문자열이면 자동 탐지)
        channel         : str   — "Y"|"M"|"C"|"K"|"all"
                                  "all" = 4채널 앙상블 / 4-channel ensemble
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        image_path: str,
        checkpoint_path: str,
        channel: str = "Y",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.channel = channel  # "Y"|"M"|"C"|"K"|"all"

    def run(self) -> None:
        try:
            import cv2
            import numpy as np
            import torch
            import torch.nn.functional as F
            from torchvision import transforms as T

            from src.data.normalize import _IMAGENET_NORMALIZE
            from src.utils.utils_model import build_model

            self.emit_progress(10, "이미지 로드 / Loading image…")

            # ── 이미지 전처리 / Preprocess ────────────────────────────────────
            image_size = self.cfg.get("data", {}).get("image_size", 128)
            img = cv2.imread(self.image_path)
            if img is None:
                raise FileNotFoundError(f"Cannot open image: {self.image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))
            tensor = T.ToTensor()(img)
            tensor = _IMAGENET_NORMALIZE(tensor)  # SSOT-NM01
            tensor = tensor.unsqueeze(0)  # (1, 3, H, W)

            # ── 디바이스 / Device ─────────────────────────────────────────────
            d = self.cfg.get("system", {}).get("device", "cpu")
            device = _resolve_device(d)

            self.emit_progress(30, "모델 로드 / Loading model…")

            # ── 앙상블 / Ensemble ─────────────────────────────────────────────
            if self.channel == "all":
                ckpt_paths = auto_find_all_checkpoints(self.cfg)
                # 수동 지정 checkpoint_path 는 무시 (all 모드)
                missing = [ch for ch, p in ckpt_paths.items() if not p]
                if missing:
                    self.emit_progress(
                        35, f"⚠️  체크포인트 미발견 / Not found: {missing}"
                    )
                result = run_ensemble(self.cfg, tensor, ckpt_paths, device)
                result["image_path"] = self.image_path
            else:
                # ── 단일 채널 / Single channel ────────────────────────────────
                ckpt = self.checkpoint_path
                if not ckpt:
                    ckpt = auto_find_checkpoint(self.cfg, self.channel)
                    if ckpt:
                        self.emit_progress(
                            35, f"자동 탐지 / Auto-found: {Path(ckpt).name}"
                        )
                    else:
                        raise FileNotFoundError(
                            f"체크포인트를 찾을 수 없습니다 / Checkpoint not found for channel {self.channel}"
                        )

                ckpt_path = Path(ckpt)
                model = build_model(self.cfg, ckpt_path, device)
                model.eval()

                self.emit_progress(70, "추론 중 / Running inference…")

                with torch.no_grad():
                    logits = model(tensor.to(device))
                    probs = F.softmax(logits, dim=1)[0]

                probs_list = probs.cpu().tolist()
                pred_level = int(torch.argmax(probs).item())
                confidence = float(probs[pred_level])
                sorted_idx = sorted(
                    range(len(probs_list)), key=lambda i: probs_list[i], reverse=True
                )
                top3 = [(i, probs_list[i]) for i in sorted_idx[:3]]

                result = {
                    "pred_level": pred_level,
                    "confidence": confidence,
                    "probs": probs_list,
                    "top3": top3,
                    "image_path": self.image_path,
                    "channel": self.channel,
                    "checkpoint": str(ckpt_path.name),
                }

            self.emit_progress(
                100,
                f"완료 / Done — Level {result['pred_level']} ({result['confidence']:.1%})",
            )
            self.finished.emit(result)

        except Exception as exc:
            import traceback

            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")


def _resolve_device(d: str):
    import torch

    if d == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    return torch.device(d)
