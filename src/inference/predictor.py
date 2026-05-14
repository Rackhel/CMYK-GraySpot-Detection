"""
inference/predictor.py

Grayspot 탐지 모델 추론 파이프라인 — Orchestrator
Grayspot detection model inference pipeline — Orchestrator

이 파일은 세 Mixin을 조합한 조율자(Orchestrator)다.
This file is an Orchestrator composing three Mixins.

```
inference/
├── predictor_device.py    — DeviceMixin      (장치 감지·설정 / Device detection)
├── predictor_loader.py    — ModelLoaderMixin (모델 로딩·캐시 / Model loading & cache)
├── predictor_inference.py — InferenceMixin   (추론 실행 / Inference execution)
└── predictor.py           — GrayspotPredictor (Orchestrator)
```

사용법 / Usage:
    from inference.predictor import GrayspotPredictor

    predictor = GrayspotPredictor()
    predictor.load_model(channel="Y", model_path="outputs/models/best_Y.pt")
    result = predictor.predict(images, channel="Y")

SSOT 근거 / SSOT Reference:
    - SSOT_Evaluation_Reporting.md §3 — 신뢰도 임계값
    - SSOT_Data_Pipeline.md §3 — 추론 전처리 (ImageNet 정규화)
    - SSOT_Core.md §5 — SOLID 원칙 (SRP/ISP 준수)
    - Contract.md §10 — GrayspotPredictor 공개 API 계약

Python 3.11+ | PyTorch 2.x compatible
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from inference.predictor_device import DeviceMixin
from inference.predictor_inference import InferenceMixin
from inference.predictor_loader import ModelLoaderMixin
from utils.utils_config import load_config


class GrayspotPredictor(DeviceMixin, ModelLoaderMixin, InferenceMixin):
    """
    Grayspot 추론 엔진 — Orchestrator / Inference engine — Orchestrator.

    세 Mixin의 공개 인터페이스를 단일 클래스로 노출한다.
    Exposes the public interfaces of three Mixins through a single class.

    책임 분리 / Responsibility separation:
        - DeviceMixin       : 장치 감지 및 설정
        - ModelLoaderMixin  : 채널별 모델 로딩·캐시 관리
        - InferenceMixin    : 단일·멀티 채널 추론 실행

    Attributes:
        cfg         : config dict (config.json 로드 결과)
        device      : torch.device — 추론 장치
        models      : Dict[str, nn.Module] — 채널별 캐시된 모델
        model_paths : Dict[str, Path] — 채널별 모델 파일 경로
        channels    : List[str] — 지원 채널 목록
        image_size  : int — 입력 이미지 크기
        num_levels  : int — 분류 클래스 수 (NUM_LEVELS = 6)
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """
        추론기 초기화 / Initialize predictor.

        DIP 준수: config 로딩은 utils_config.load_config 에 위임 (fallback 없음).
        DIP compliant: config loading delegated to utils_config.load_config (no fallback).

        Args:
            config_path: config.json 경로. None 이면 기본 경로 사용.
                         Path to config.json. None uses the default path.

        Raises:
            FileNotFoundError: config.json 없음 — Fail-Fast
        """
        self.logger.info("[Predictor] Initializing GrayspotPredictor...")

        # Config 로딩 — Fail-Fast (fallback 없음 / no fallback)
        self.cfg: dict = (
            load_config(config_path=config_path) if config_path else load_config()
        )

        # 장치 설정 — DeviceMixin._setup_device()
        self.device = self._setup_device()
        self.logger.info(f"  Device: {self.device}")

        # 모델 캐시 초기화 — ModelLoaderMixin에서 사용
        self.models: Dict[str, Any] = {}
        self.model_paths: Dict[str, Path] = {}

        # SSOT 상수 (config 우선, SSOT 기본값 fallback)
        self.channels = self.cfg.get("data", {}).get("channels") or ["Y", "M", "C", "K"]
        self.image_size = self.cfg.get("data", {}).get("image_size") or 128
        self.num_levels = self.cfg.get("data", {}).get("num_levels") or 6

        self.logger.info(f"  Channels  : {self.channels}")
        self.logger.info(f"  Image size: {self.image_size}x{self.image_size}")
        self.logger.info(f"  Num levels: {self.num_levels}")
        self.logger.info("[Predictor] Initialization complete ✓")
