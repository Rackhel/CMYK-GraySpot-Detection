"""
Grayspot — 추론 파이프라인 / Inference Pipeline
inference/predictor.py

입력: 스캔 이미지 경로 / Input: scan image path
출력: {Y_Level, M_Level, C_Level, K_Level, confidence, status}
      Output: per-channel Grayspot level with confidence and fallback status
"""

import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from data.preprocessing import preprocess, CHANNELS
from data.augmentation import get_supervised_transforms
from models.grayspot_model import GrayspotModel


class GrayspotPredictor:
    """
    스캔 이미지 1장을 입력받아 CMYK 4색상의 Grayspot Level을 추론한다.
    Takes a single scan image and infers Grayspot Level for all 4 CMYK channels.

    Fallback 전략 (Section 5.5.4) / Confidence-based Fallback Strategy:
        max(softmax) ≥ 0.80 → 자동 판정 확정 / Confirmed auto prediction
        0.50 ≤ max(softmax) < 0.80 → 자동 판정 + 경고 / Auto prediction with warning
        max(softmax) < 0.50 → 수동 검수 요청 / Manual review requested
    """

    # 상태 출력 문자열 / Status display strings
    STATUS = {
        "confirmed":     " 자동 판정 / Auto confirmed",
        "warning":       "  자동 판정 (검토 권장) / Auto (review recommended)",
        "manual_review": "🔴 수동 검수 요청 / Manual review required",
    }

    def __init__(self, cfg: dict):
        self.cfg       = cfg
        self.device    = torch.device("cpu")  # GPU 불필요 / No GPU required
        self.models: dict[str, GrayspotModel] = {}
        self.transform = get_supervised_transforms(cfg, augment=False)  # 추론 시 증강 없음 / No augmentation for inference
        self._load_models()

    def _load_models(self) -> None:
        """채널별 학습된 모델을 로드한다. / Loads trained models for each channel."""
        model_dir = Path(self.cfg["inference"]["model_dir"])
        for ch in CHANNELS:
            model_path = model_dir / f"best_{ch}.pt"
            if model_path.exists():
                model = GrayspotModel(self.cfg, phase=2)
                model.load(model_path)
                model.eval()
                self.models[ch] = model
                print(f"    [{ch}] 모델 로드 / Model loaded: {model_path}")
            else:
                print(f"    [{ch}] 모델 없음 / Model not found: {model_path}")

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """
        스캔 이미지 1장에 대해 CMYK 4색상의 Grayspot Level을 추론한다.
        Infers Grayspot Level for all 4 CMYK channels from a single scan image.

        Returns:
            {
              "image":         "scan_001.png",
              "Y_Level":       2,              # 예측 레벨 / Predicted level
              "M_Level":       0,
              "C_Level":       3,
              "K_Level":       1,
              "confidence":    {"Y": 0.92, "M": 0.97, ...},  # 최대 softmax 값 / Max softmax value
              "status":        {"Y": "confirmed", ...},       # 판정 상태 / Decision status
              "probabilities": {"Y": [0.01, 0.03, 0.92, ...], ...},  # 전체 확률 분포 / Full probability distribution
              "timestamp":     "2026-03-16T14:30:22",
              "elapsed_ms":    120,            # 추론 소요 시간 (ms) / Inference time in ms
            }
        """
        t0        = time.time()
        processed = preprocess(image_path, self.cfg, return_feature=False)

        result = {
            "image":         Path(image_path).name,
            "timestamp":     datetime.now().isoformat(timespec="seconds"),
            "confidence":    {},
            "status":        {},
            "probabilities": {},
        }

        inf_cfg = self.cfg["inference"]

        for ch in CHANNELS:
            # 모델이 없는 채널 처리 / Handle missing model for channel
            if ch not in self.models:
                result[f"{ch}_Level"]        = -1
                result["confidence"][ch]     = 0.0
                result["status"][ch]         = "no_model"
                result["probabilities"][ch]  = []
                continue

            roi    = processed[ch]                     # (H, W, 3) float32
            tensor = self.transform(roi).unsqueeze(0)  # (1, 3, H, W)
            logits = self.models[ch](tensor)           # (1, 6) — raw logits
            probs  = F.softmax(logits, dim=1)[0]       # (6,) — 확률 분포 / Probability distribution

            level      = int(probs.argmax().item())    # 최고 확률 레벨 / Highest probability level
            confidence = float(probs.max().item())     # 최대 확률값 / Max probability value

            # Fallback 판단 / Fallback decision
            if confidence >= inf_cfg["confidence_auto"]:
                status = "confirmed"
            elif confidence >= inf_cfg["confidence_warn"]:
                status = "warning"
            else:
                status = "manual_review"
                level  = -1  # 수동 검수 전 판정 보류 / Withhold prediction until manual review

            result[f"{ch}_Level"]           = level
            result["confidence"][ch]        = round(confidence, 4)
            result["status"][ch]            = status
            result["probabilities"][ch]     = [round(p, 4) for p in probs.tolist()]

        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    def predict_and_save(self, image_path: str | Path) -> dict:
        """
        추론 후 결과를 analyzed/ 폴더에 JSON으로 저장한다.
        Runs inference and saves the result as a JSON file in the analyzed/ folder.
        """
        result       = self.predict(image_path)
        analyzed_dir = Path(self.cfg["storage"]["analyzed_dir"])
        analyzed_dir.mkdir(parents=True, exist_ok=True)

        # 타임스탬프 기반 파일명 / Timestamp-based filename
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = analyzed_dir / f"result_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 결과 터미널 출력 / Print results to terminal
        print(f"\n📊  추론 결과 / Inference Results")
        for ch in CHANNELS:
            lvl    = result.get(f"{ch}_Level", -1)
            conf   = result["confidence"].get(ch, 0)
            status = result["status"].get(ch, "")
            print(f"  [{ch}] Level {lvl} | Conf: {conf:.3f} | {self.STATUS.get(status, status)}")
        print(f"\n    저장 / Saved: {out_path}")
        print(f"  ⏱   {result['elapsed_ms']}ms\n")

        return result