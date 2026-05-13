"""
inference/__init__.py

추론 패키지 / Inference package.

Grayspot 탐지 모델 추론 파이프라인을 내보낸다.
Exports the Grayspot detection model inference pipeline.

주요 클래스 / Key classes:
    GrayspotPredictor : 채널별 모델 캐싱 + 배치 추론 엔진
                        Per-channel model caching + batch inference engine

사용법 / Usage:
    from inference import GrayspotPredictor

    predictor = GrayspotPredictor()
    predictor.load_model(channel="Y", model_path="models/best_Y.pt")
    results   = predictor.predict(images, channel="Y")
"""

from .predictor import GrayspotPredictor

__all__ = [
    "GrayspotPredictor",
]
