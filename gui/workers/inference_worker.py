"""InferenceWorker — 단일 이미지 추론 (단일 채널 / 앙상블).
Single-image inference: single-channel or 4-channel ensemble.

src/inference/predictor.py 의 GrayspotPredictor를 사용한다.
Uses GrayspotPredictor from src/inference/predictor.py.

정규화는 체크포인트별 .meta.json에서 자동 로드된다 (src 레벨 로직).
Normalization is auto-loaded from per-checkpoint .meta.json (src-level logic).

Contract: Contract_gui.md §2.6  /  SSOT_GUI.md §3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ._ckpt_utils import auto_find_all_checkpoints, auto_find_checkpoint
from .base_worker import BaseWorker

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CHANNELS = ["Y", "M", "C", "K"]


class InferenceWorker(BaseWorker):
    """단일 이미지를 GrayspotPredictor(src)를 통해 추론한다.
    Runs single-image inference via GrayspotPredictor from src.

    Args (Contract §2.6):
        cfg             : dict  — load_config() 반환값
        image_path      : str   — 추론할 이미지 경로
        checkpoint_path : str   — .pt 경로 (빈 문자열이면 자동 탐지)
        channel         : str   — "Y"|"M"|"C"|"K"|"all"
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
        self.channel = channel

    def run(self) -> None:
        try:
            import cv2
            import numpy as np

            from inference.predictor import GrayspotPredictor

            self.emit_progress(10, "이미지 로드 / Loading image…")

            # ── 이미지 로드 (BGR, uint8) ─────────────────────────────────────
            # GrayspotPredictor._preprocess_images 가 BGR→float32→정규화를 담당한다.
            # GrayspotPredictor._preprocess_images handles BGR→float32→normalize.
            image_size = self.cfg.get("data", {}).get("image_size", 128)
            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot open image: {self.image_path}")

            # BGR → RGB (predictor는 RGB 입력을 기대함)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (image_size, image_size))
            # (H, W, 3) → (1, H, W, 3) batch
            images_np = img_resized[np.newaxis, ...]

            self.emit_progress(30, "모델 로드 / Loading model…")

            # ── GrayspotPredictor 초기화 ──────────────────────────────────────
            predictor = GrayspotPredictor()

            if self.channel == "all":
                # ── 앙상블: 4채널 각각 로드 후 softmax 평균 ──────────────────
                ckpt_paths = auto_find_all_checkpoints(self.cfg)
                missing = [ch for ch, p in ckpt_paths.items() if not p]
                if missing:
                    self.emit_progress(35, f"⚠️  체크포인트 미발견: {missing}")

                per_channel: dict[str, dict] = {}
                all_probs = []

                for i, ch in enumerate(_CHANNELS):
                    ckpt = ckpt_paths.get(ch, "")
                    if not ckpt:
                        continue
                    try:
                        # load_model이 .meta.json에서 normalizer를 로드함
                        # load_model loads normalizer from .meta.json
                        predictor.load_model(ch, model_path=ckpt)
                        out = predictor.predict(images_np, channel=ch)
                        probs = out["probabilities"][0]  # (num_levels,)
                        pred = int(out["predictions"][0])
                        conf = float(out["confidences"][0])
                        per_channel[ch] = {"pred": pred, "conf": conf}
                        all_probs.append(probs)
                        self.emit_progress(
                            30 + (i + 1) * 10,
                            f"[{ch}] Lv {pred} ({conf:.1%}) 로드 완료",
                        )
                    except Exception as e:
                        self.emit_progress(35, f"⚠️  [{ch}] 로드 실패: {e}")

                if not all_probs:
                    raise RuntimeError("앙상블: 로드된 채널 모델이 없습니다.")

                import numpy as _np

                avg_probs = _np.mean(all_probs, axis=0)
                pred_level = int(_np.argmax(avg_probs))
                confidence = float(avg_probs[pred_level])
                sorted_idx = _np.argsort(avg_probs)[::-1]
                top3 = [(int(i), float(avg_probs[i])) for i in sorted_idx[:3]]

                result = {
                    "pred_level": pred_level,
                    "confidence": confidence,
                    "probs": avg_probs.tolist(),
                    "top3": top3,
                    "per_channel": per_channel,
                    "image_path": self.image_path,
                    "channel": "all",
                }

            else:
                # ── 단일 채널 ─────────────────────────────────────────────────
                ckpt = self.checkpoint_path
                if not ckpt:
                    ckpt = auto_find_checkpoint(self.cfg, self.channel)
                    if ckpt:
                        self.emit_progress(35, f"자동 탐지: {Path(ckpt).name}")
                    else:
                        raise FileNotFoundError(
                            f"체크포인트를 찾을 수 없습니다: channel {self.channel}"
                        )

                # load_model이 .meta.json 로드 → 학습 시 정규화 자동 적용
                # load_model reads .meta.json → applies training-time normalization
                predictor.load_model(self.channel, model_path=ckpt)

                self.emit_progress(70, "추론 중 / Running inference…")
                out = predictor.predict(images_np, channel=self.channel)

                probs_list = out["probabilities"][0].tolist()
                pred_level = int(out["predictions"][0])
                confidence = float(out["confidences"][0])
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
                    "checkpoint": str(Path(ckpt).name),
                }

            self.emit_progress(
                100,
                f"완료 / Done — Level {result['pred_level']} ({result['confidence']:.1%})",
            )
            self.finished.emit(result)

        except Exception as exc:
            import traceback

            self.error_occurred.emit(f"{exc}\n{traceback.format_exc()}")
