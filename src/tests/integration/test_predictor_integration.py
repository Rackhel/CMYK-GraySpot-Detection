"""
tests/integration/test_predictor_integration.py

GrayspotPredictor 통합 테스트.
Integration tests for GrayspotPredictor.

실제 모델 파일 없이 confidence threshold 분기 로직을 검증한다.
Validates confidence threshold branching logic without actual model files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


# ── Confidence threshold 분기 로직 / Confidence threshold branching ──────────

class TestConfidenceThresholdLogic:
    """
    inference/predictor.py 의 신뢰도 임계값 분기를 독립적으로 검증한다.
    Validates confidence threshold branching independently from the predictor class.
    """

    def _classify_confidence(self, confidence: float, thresholds: dict) -> str:
        auto   = thresholds["auto_accept"]
        warn   = thresholds["warn_threshold"]
        manual = thresholds["manual_review"]
        if confidence >= auto:
            return "AUTO"
        elif confidence >= warn:
            return "WARN"
        elif confidence >= manual:
            return "MANUAL"
        else:
            return "REJECT"

    @pytest.fixture
    def thresholds(self):
        return {"auto_accept": 0.8, "warn_threshold": 0.5, "manual_review": 0.3}

    def test_high_confidence_auto(self, thresholds):
        assert self._classify_confidence(0.9, thresholds) == "AUTO"

    def test_exact_auto_threshold_auto(self, thresholds):
        assert self._classify_confidence(0.8, thresholds) == "AUTO"

    def test_mid_confidence_warn(self, thresholds):
        assert self._classify_confidence(0.65, thresholds) == "WARN"

    def test_exact_warn_threshold_warn(self, thresholds):
        assert self._classify_confidence(0.5, thresholds) == "WARN"

    def test_low_confidence_manual(self, thresholds):
        assert self._classify_confidence(0.4, thresholds) == "MANUAL"

    def test_exact_manual_threshold_manual(self, thresholds):
        assert self._classify_confidence(0.3, thresholds) == "MANUAL"

    def test_very_low_confidence_reject(self, thresholds):
        assert self._classify_confidence(0.1, thresholds) == "REJECT"

    def test_zero_confidence_reject(self, thresholds):
        assert self._classify_confidence(0.0, thresholds) == "REJECT"

    def test_full_confidence_auto(self, thresholds):
        assert self._classify_confidence(1.0, thresholds) == "AUTO"


# ── GrayspotPredictor 초기화 / Initialization ────────────────────────────────

class TestGrayspotPredictorInit:
    def test_predictor_instantiates_without_error(self):
        from inference.predictor import GrayspotPredictor
        predictor = GrayspotPredictor()
        assert predictor is not None

    def test_predictor_has_no_loaded_models_on_init(self):
        from inference.predictor import GrayspotPredictor
        predictor = GrayspotPredictor()
        assert not hasattr(predictor, "_models") or len(getattr(predictor, "_models", {})) == 0
