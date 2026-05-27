"""InferenceWorker — 단일 이미지 추론을 백그라운드 QThread에서 실행.
Single-image inference in a background QThread.

Contract: Contract_gui.md §2.5  /  SSOT_GUI.md §3
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


class InferenceWorker(BaseWorker):
    """단일 이미지를 로드된 모델로 추론한다.
    Runs single-image inference using the loaded model.

    Args (Contract §2.5):
        cfg             : dict — load_config() 반환값
        image_path      : str  — 추론할 이미지 경로
        checkpoint_path : str  — .pt 체크포인트 경로
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        image_path: str,
        checkpoint_path: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path

    def run(self) -> None:
        try:
            import cv2
            import torch
            import torch.nn.functional as F

            from src.data.normalize import _IMAGENET_NORMALIZE
            from src.utils.utils_model import build_model

            self.emit_progress(10, "이미지 로드 / Loading image…")

            # ── 이미지 전처리 / Preprocess image ─────────────────────────────
            image_size = self.cfg.get("data", {}).get("image_size", 128)
            img = cv2.imread(self.image_path)
            if img is None:
                raise FileNotFoundError(f"Cannot open image: {self.image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size))

            import numpy as np
            from torchvision import transforms as T

            tensor = T.ToTensor()(img)                   # (3, H, W) float32 in [0,1]
            tensor = _IMAGENET_NORMALIZE(tensor)          # SSOT-NM01
            tensor = tensor.unsqueeze(0)                  # (1, 3, H, W)

            self.emit_progress(30, "모델 로드 / Loading model…")

            # ── 디바이스 결정 / Resolve device ───────────────────────────────
            d = self.cfg.get("system", {}).get("device", "cpu")
            if d == "auto":
                device = torch.device(
                    "cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu"
                )
            else:
                device = torch.device(d)

            # ── 모델 빌드 ─────────────────────────────────────────────────────
            ckpt_path = Path(self.checkpoint_path) if self.checkpoint_path else None
            model = build_model(self.cfg, ckpt_path, device)
            model.eval()

            self.emit_progress(70, "추론 중 / Running inference…")

            # ── 추론 / Inference ──────────────────────────────────────────────
            with torch.no_grad():
                tensor = tensor.to(device)
                logits = model(tensor)                    # (1, num_classes)
                probs  = F.softmax(logits, dim=1)[0]      # (num_classes,)

            probs_list = probs.cpu().tolist()
            pred_level = int(torch.argmax(probs).item())
            confidence = float(probs[pred_level])

            # Top-3
            sorted_idx = sorted(range(len(probs_list)), key=lambda i: probs_list[i], reverse=True)
            top3 = [(i, probs_list[i]) for i in sorted_idx[:3]]

            self.emit_progress(100, f"추론 완료 / Done — Level {pred_level} ({confidence:.1%})")
            self.finished.emit({
                "pred_level": pred_level,
                "confidence": confidence,
                "probs":      probs_list,
                "top3":       top3,
                "image_path": self.image_path,
            })

        except Exception as exc:
            import traceback
            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")
