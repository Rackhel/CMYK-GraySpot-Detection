"""
tests/unit/test_predictor.py

inference/ 모듈 단위 테스트.
Unit tests for the inference/ module.

BDD 근거 / BDD Reference:
    - BDD.md §9 미구현 목록: 시나리오 1.1–1.3, 8.3
    - 시나리오 1.1–1.3: 신뢰도 임계값 플래그 분류 (AUTO_ACCEPT / WARN / MANUAL_REVIEW)
    - 시나리오 8.3: SSOT-NM01 — 추론 시 ImageNet 정규화 적용 검증

TDD 근거 / TDD Reference:
    - TDD.md §3.6: test_predictor.py 필수 테스트 목록

SSOT 근거 / SSOT Reference:
    - SSOT_Data_Pipeline.md §3 — BGR float32, ImageNet 정규화
    - SSOT_Evaluation_Reporting.md §3 — 신뢰도 임계값
    - SSOT_Core.md §6 — SSOT-FF01 Fail-Fast (모델 파일 누락 시 즉시 실패)
    - Contract.md §10 — GrayspotPredictor 공개 API 계약

Python 3.11.5
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data.normalize import _IMAGENET_NORMALIZE
from inference.predictor_inference import InferenceMixin

# ── 신뢰도 임계값 상수 / Confidence threshold constants (Contract.md §10) ──
AUTO_ACCEPT_THRESHOLD: float = 0.8  # 자동 승인 하한 / Lower bound for auto accept
WARN_THRESHOLD: float = 0.5  # 경고 구간 하한 / Lower bound for warn zone
MANUAL_REVIEW_THRESHOLD: float = 0.3  # 수동 검토 상한 / Upper bound for manual review


# ── 테스트용 최소 구체 클래스 / Minimal concrete class for testing ──────────

class _ConcreteInferenceMixin(InferenceMixin):
    """
    InferenceMixin을 단독으로 인스턴스화하기 위한 최소 구체 클래스.
    Minimal concrete class to instantiate InferenceMixin in isolation.
    """

    def __init__(self, cfg: dict, device: torch.device = torch.device("cpu")) -> None:
        # LoggerMixin.logger는 @property — _logger로 직접 주입한다
        # LoggerMixin.logger is a @property — inject via _logger directly
        import logging
        self._logger = logging.getLogger(self.__class__.__name__)
        self.cfg = cfg
        self.device = device
        self.models: Dict[str, nn.Module] = {}
        self.model_paths: Dict[str, Path] = {}
        self.channels = cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
        self.image_size = cfg.get("data", {}).get("image_size", 128)
        self.num_levels = cfg.get("data", {}).get("num_levels", 6)


# ── 신뢰도 임계값 분류 헬퍼 / Confidence threshold classifier helper ─────

def _classify_confidence(conf: float, thresholds: dict) -> str:
    """
    신뢰도 값을 임계값 기반으로 플래그로 분류한다.
    Classifies a confidence value into a flag based on thresholds.

    Contract.md §10 / SSOT_Evaluation_Reporting.md §3 준수
    Compliant with Contract.md §10 / SSOT_Evaluation_Reporting.md §3.

    Args:
        conf      : max-softmax 신뢰도 값 [0, 1] / max-softmax confidence value
        thresholds: dict — auto_accept, warn_threshold, manual_review 키 포함

    Returns:
        "AUTO_ACCEPT" | "WARN" | "MANUAL_REVIEW" | "UNCERTAIN"
    """
    if conf >= thresholds["auto_accept"]:
        return "AUTO_ACCEPT"
    if conf >= thresholds["warn_threshold"]:
        return "WARN"
    if conf < thresholds["manual_review"]:
        return "MANUAL_REVIEW"
    return "UNCERTAIN"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg(minimal_cfg) -> dict:
    """단위 테스트용 config dict — minimal_cfg 재사용 / Reuse minimal_cfg."""
    return minimal_cfg


@pytest.fixture
def mixin(cfg) -> _ConcreteInferenceMixin:
    """테스트용 InferenceMixin 인스턴스 / InferenceMixin instance for tests."""
    return _ConcreteInferenceMixin(cfg)


@pytest.fixture
def confidence_thresholds(cfg) -> dict:
    """config에서 신뢰도 임계값 추출 / Extract confidence thresholds from config."""
    return cfg["inference"]["confidence_thresholds"]


@pytest.fixture
def dummy_model() -> nn.Module:
    """
    6-클래스 분류기를 흉내내는 최소 더미 모델.
    Minimal dummy model mimicking a 6-class classifier.
    """
    model = nn.Linear(3 * 128 * 128, 6)
    model.eval()
    return model


# ── BDD 시나리오 1.1 / BDD Scenario 1.1 ──────────────────────────────────
# Given: 신뢰도 임계값 설정 (auto_accept=0.8)
# When : 신뢰도 0.9 입력
# Then : "AUTO_ACCEPT" 플래그 반환


class TestConfidenceAutoAcceptFlag:
    """BDD §9 시나리오 1.1 — AUTO_ACCEPT 플래그 분류."""

    def test_confidence_above_auto_accept_threshold(self, confidence_thresholds):
        """신뢰도 ≥ 0.8 이면 AUTO_ACCEPT 플래그를 반환한다."""
        # Confidence ≥ 0.8 must return AUTO_ACCEPT flag
        result = _classify_confidence(0.9, confidence_thresholds)
        assert result == "AUTO_ACCEPT"

    def test_confidence_exactly_at_auto_accept_boundary(self, confidence_thresholds):
        """신뢰도 = 0.8 (경계값) 이면 AUTO_ACCEPT 플래그를 반환한다."""
        # Exactly at boundary must still be AUTO_ACCEPT
        result = _classify_confidence(0.8, confidence_thresholds)
        assert result == "AUTO_ACCEPT"

    def test_auto_accept_threshold_value_in_config(self, confidence_thresholds):
        """config의 auto_accept 임계값은 0.8이다 (Contract.md §10)."""
        # Config auto_accept must equal 0.8 per contract
        assert confidence_thresholds["auto_accept"] == pytest.approx(0.8)

    def test_auto_accept_confidence_is_1_0(self, confidence_thresholds):
        """신뢰도 1.0 (완전 확신)도 AUTO_ACCEPT이다."""
        result = _classify_confidence(1.0, confidence_thresholds)
        assert result == "AUTO_ACCEPT"


# ── BDD 시나리오 1.2 / BDD Scenario 1.2 ──────────────────────────────────
# Given: 신뢰도 임계값 설정 (warn_threshold=0.5, auto_accept=0.8)
# When : 0.5 ≤ 신뢰도 < 0.8
# Then : "WARN" 플래그 반환


class TestConfidenceWarnFlag:
    """BDD §9 시나리오 1.2 — WARN 플래그 분류."""

    def test_confidence_in_warn_zone(self, confidence_thresholds):
        """0.5 ≤ 신뢰도 < 0.8 이면 WARN 플래그를 반환한다."""
        result = _classify_confidence(0.65, confidence_thresholds)
        assert result == "WARN"

    def test_confidence_exactly_at_warn_lower_boundary(self, confidence_thresholds):
        """신뢰도 = 0.5 (경계값)이면 WARN 플래그를 반환한다."""
        result = _classify_confidence(0.5, confidence_thresholds)
        assert result == "WARN"

    def test_confidence_just_below_auto_accept(self, confidence_thresholds):
        """신뢰도 = 0.799 (AUTO_ACCEPT 하한 직전)이면 WARN이다."""
        result = _classify_confidence(0.799, confidence_thresholds)
        assert result == "WARN"

    def test_warn_threshold_value_in_config(self, confidence_thresholds):
        """config의 warn_threshold는 0.5이다 (Contract.md §10)."""
        assert confidence_thresholds["warn_threshold"] == pytest.approx(0.5)


# ── BDD 시나리오 1.3 / BDD Scenario 1.3 ──────────────────────────────────
# Given: 신뢰도 임계값 설정 (manual_review=0.3)
# When : 신뢰도 < 0.3
# Then : "MANUAL_REVIEW" 플래그 반환


class TestConfidenceManualReviewFlag:
    """BDD §9 시나리오 1.3 — MANUAL_REVIEW 플래그 분류."""

    def test_confidence_below_manual_review_threshold(self, confidence_thresholds):
        """신뢰도 < 0.3 이면 MANUAL_REVIEW 플래그를 반환한다."""
        result = _classify_confidence(0.2, confidence_thresholds)
        assert result == "MANUAL_REVIEW"

    def test_confidence_exactly_0_is_manual_review(self, confidence_thresholds):
        """신뢰도 0.0 (최소값)이면 MANUAL_REVIEW이다."""
        result = _classify_confidence(0.0, confidence_thresholds)
        assert result == "MANUAL_REVIEW"

    def test_confidence_just_below_manual_review_boundary(self, confidence_thresholds):
        """신뢰도 = 0.29 (경계 직전)이면 MANUAL_REVIEW이다."""
        result = _classify_confidence(0.29, confidence_thresholds)
        assert result == "MANUAL_REVIEW"

    def test_manual_review_threshold_value_in_config(self, confidence_thresholds):
        """config의 manual_review는 0.3이다 (Contract.md §10)."""
        assert confidence_thresholds["manual_review"] == pytest.approx(0.3)

    def test_three_flags_partition_confidence_range(self, confidence_thresholds):
        """
        신뢰도 구간이 셋으로 분할됨을 확인한다.
        AUTO_ACCEPT ∪ WARN ∪ MANUAL_REVIEW (with UNCERTAIN buffer) covers [0,1].
        """
        samples = {
            0.9: "AUTO_ACCEPT",
            0.8: "AUTO_ACCEPT",
            0.65: "WARN",
            0.5: "WARN",
            0.2: "MANUAL_REVIEW",
            0.0: "MANUAL_REVIEW",
        }
        for conf, expected_flag in samples.items():
            assert _classify_confidence(conf, confidence_thresholds) == expected_flag, (
                f"conf={conf} expected {expected_flag}"
            )


# ── BDD 시나리오 8.3 / BDD Scenario 8.3 — SSOT-NM01 ─────────────────────
# Given: uint8 BGR 이미지 배열
# When : _preprocess_images() 호출
# Then : ImageNet mean/std 로 정규화된 텐서 반환


class TestPreprocessImagesImagenetNormalized:
    """BDD §9 시나리오 8.3 — SSOT-NM01 ImageNet 정규화 검증."""

    def test_output_is_tensor(self, mixin, dummy_image_np):
        """_preprocess_images()는 torch.Tensor를 반환한다."""
        images = dummy_image_np[np.newaxis]  # (1, H, W, 3)
        result = mixin._preprocess_images(images)
        assert isinstance(result, torch.Tensor)

    def test_output_shape_is_nchw(self, mixin, dummy_image_np):
        """출력 형상은 (N, C, H, W) 이다."""
        # Output shape must be (N, C, H, W)
        batch = np.stack([dummy_image_np, dummy_image_np])  # (2, H, W, 3)
        result = mixin._preprocess_images(batch)
        n, c, h, w = result.shape
        assert n == 2
        assert c == 3
        assert h == dummy_image_np.shape[0]
        assert w == dummy_image_np.shape[1]

    def test_imagenet_mean_applied(self, mixin):
        """
        ImageNet mean=[0.485, 0.456, 0.406] 이 적용되었는지 검증한다.
        픽셀값 0인 이미지(검정)를 입력하면 정규화 후 -mean/std 값이 나와야 한다.
        Verifies ImageNet mean is applied: zero-image → -mean/std after normalization.
        """
        # 픽셀 0으로 채운 이미지 (float32로 직접 주입)
        zeros = np.zeros((1, 8, 8, 3), dtype=np.float32)
        result = mixin._preprocess_images(zeros)

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        expected = -imagenet_mean / imagenet_std  # shape: (3,)

        # 각 채널의 평균값이 expected에 수렴해야 한다
        for c in range(3):
            actual_mean = result[0, c].mean().item()
            assert actual_mean == pytest.approx(expected[c].item(), abs=1e-4), (
                f"Channel {c}: expected {expected[c].item():.4f}, got {actual_mean:.4f}"
            )

    def test_imagenet_std_applied(self, mixin):
        """
        ImageNet std=[0.229, 0.224, 0.225] 이 적용되었는지 검증한다.
        픽셀값 255인 이미지(흰색)를 정규화하면 (1-mean)/std 값이 나와야 한다.
        Verifies ImageNet std applied: white image → (1-mean)/std after normalization.
        """
        whites = np.full((1, 8, 8, 3), 255, dtype=np.uint8)
        result = mixin._preprocess_images(whites)

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        expected = (1.0 - imagenet_mean) / imagenet_std  # shape: (3,)

        for c in range(3):
            actual_mean = result[0, c].mean().item()
            assert actual_mean == pytest.approx(expected[c].item(), abs=1e-4), (
                f"Channel {c}: expected {expected[c].item():.4f}, got {actual_mean:.4f}"
            )

    def test_uint8_input_scaled_to_0_1(self, mixin, dummy_image_np):
        """
        uint8 [0, 255] 입력이 [0, 1] 로 스케일링된 뒤 정규화된다.
        Verifies uint8 input is scaled to [0,1] before normalization.
        """
        images = dummy_image_np[np.newaxis]
        result = mixin._preprocess_images(images)
        # ImageNet 정규화 후 값 범위가 0-255 그대로가 아님을 확인
        # After normalization the range should not be [0, 255]
        assert result.max().item() < 10.0

    def test_greyscale_input_expanded_to_3_channels(self, mixin):
        """
        (N, H, W) 단일채널 입력이 (N, 3, H, W) 로 확장된다.
        Greyscale (N,H,W) input is expanded to (N,3,H,W).
        """
        grey = np.zeros((2, 8, 8), dtype=np.uint8)
        result = mixin._preprocess_images(grey)
        assert result.shape == (2, 3, 8, 8)

    def test_invalid_ndim_raises_value_error(self, mixin):
        """형상이 잘못된 입력은 ValueError를 발생시킨다."""
        # Invalid ndim must raise ValueError
        bad_input = np.zeros((2, 8), dtype=np.float32)  # 2D: invalid
        with pytest.raises(ValueError):
            mixin._preprocess_images(bad_input)

    def test_imagenet_normalize_import_is_singleton(self):
        """
        SSOT-NM01: _IMAGENET_NORMALIZE 는 data.normalize 에서 임포트되어야 한다.
        SSOT-NM01: _IMAGENET_NORMALIZE must be imported from data.normalize.
        """
        from data.normalize import _IMAGENET_NORMALIZE as norm_ref
        assert _IMAGENET_NORMALIZE is norm_ref


# ── TDD §3.6 — predict() 반환 구조 / predict() return structure ──────────


class TestPredictReturnsRequiredKeys:
    """TDD §3.6 — predict() 반환 딕셔너리 키 검증."""

    def _setup_mixin_with_mock_model(self, cfg) -> _ConcreteInferenceMixin:
        """더미 모델이 로드된 InferenceMixin 인스턴스를 반환한다."""
        mixin = _ConcreteInferenceMixin(cfg)

        # 6-클래스 더미 선형 모델을 채널 "Y"에 등록
        # Register a 6-class dummy linear model for channel "Y"
        class _DummyClassifier(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch = x.size(0)
                return torch.zeros(batch, 6)

        mixin.models["Y"] = _DummyClassifier().eval()
        return mixin

    def test_returns_dict(self, cfg, dummy_image_np):
        """predict()는 dict를 반환한다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        assert isinstance(result, dict)

    def test_predictions_key_present(self, cfg, dummy_image_np):
        """반환 딕셔너리에 'predictions' 키가 있다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        assert "predictions" in result

    def test_logits_key_present(self, cfg, dummy_image_np):
        """반환 딕셔너리에 'logits' 키가 있다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        assert "logits" in result

    def test_probabilities_key_present(self, cfg, dummy_image_np):
        """반환 딕셔너리에 'probabilities' 키가 있다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        assert "probabilities" in result

    def test_confidences_key_present_when_flag_true(self, cfg, dummy_image_np):
        """return_confidences=True 이면 'confidences' 키가 있다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y", return_confidences=True)
        assert "confidences" in result

    def test_confidences_key_absent_when_flag_false(self, cfg, dummy_image_np):
        """return_confidences=False 이면 'confidences' 키가 없다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y", return_confidences=False)
        assert "confidences" not in result

    def test_predictions_shape_matches_input_n(self, cfg, dummy_image_np):
        """predictions 배열의 길이는 입력 이미지 수와 같다."""
        # Length of predictions must equal number of input images
        mixin = self._setup_mixin_with_mock_model(cfg)
        n = 5
        images = np.stack([dummy_image_np] * n)  # (5, H, W, 3)
        result = mixin.predict(images, "Y")
        assert len(result["predictions"]) == n

    def test_predictions_are_valid_class_indices(self, cfg, dummy_image_np):
        """predictions 값은 0 이상 5 이하의 정수다 (6-level 분류)."""
        # Prediction values must be in [0, num_levels-1]
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        preds = result["predictions"]
        assert preds.min() >= 0
        assert preds.max() <= 5

    def test_confidences_in_zero_one_range(self, cfg, dummy_image_np):
        """confidences 값은 [0, 1] 범위에 있다 (max-softmax)."""
        # Confidence values must be in [0, 1] (max-softmax)
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        confs = result["confidences"]
        assert confs.min() >= 0.0
        assert confs.max() <= 1.0

    def test_probabilities_rows_sum_to_one(self, cfg, dummy_image_np):
        """probabilities 각 행의 합은 1.0이다 (소프트맥스 확률)."""
        # Each row of probabilities must sum to 1.0
        mixin = self._setup_mixin_with_mock_model(cfg)
        images = dummy_image_np[np.newaxis]
        result = mixin.predict(images, "Y")
        row_sums = result["probabilities"].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-5)

    def test_logits_shape_is_n_by_num_levels(self, cfg, dummy_image_np):
        """logits 형상은 (N, num_levels) 이다."""
        mixin = self._setup_mixin_with_mock_model(cfg)
        n = 3
        images = np.stack([dummy_image_np] * n)
        result = mixin.predict(images, "Y")
        assert result["logits"].shape == (n, 6)


# ── SSOT-FF01 Fail-Fast — 모델 미로드 / Model not loaded ─────────────────


class TestPredictRaisesWhenModelNotLoaded:
    """SSOT-FF01 — predict() 호출 전 모델 미로드 시 RuntimeError 발생."""

    def test_runtime_error_when_no_model_for_channel(self, mixin, dummy_image_np):
        """모델이 로드되지 않은 채널에 predict() 호출 시 RuntimeError가 발생한다."""
        # predict() without prior load_model() must raise RuntimeError
        images = dummy_image_np[np.newaxis]
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            mixin.predict(images, "Y")

    def test_runtime_error_message_contains_channel(self, mixin, dummy_image_np):
        """RuntimeError 메시지에 채널명이 포함된다."""
        images = dummy_image_np[np.newaxis]
        with pytest.raises(RuntimeError, match="Y"):
            mixin.predict(images, "Y")

    def test_value_error_for_non_numpy_input(self, mixin, dummy_image_np):
        """numpy가 아닌 입력은 ValueError를 발생시킨다 (Contract.md §10)."""
        import torch

        # 더미 모델 등록 후 잘못된 타입 전달
        # Register dummy model then pass wrong type
        mixin.models["Y"] = nn.Linear(1, 6).eval()
        with pytest.raises(ValueError, match="numpy.ndarray"):
            mixin.predict(torch.zeros(1, 3, 8, 8), "Y")


# ── SSOT-FF01 — load_model() FileNotFoundError ───────────────────────────


class TestLoadModelFileNotFoundError:
    """SSOT-FF01 — load_model() 모델 파일 누락 시 즉시 FileNotFoundError 발생."""

    def _make_predictor(self, cfg):
        """load_config를 mock하여 GrayspotPredictor를 생성한다."""
        from inference.predictor import GrayspotPredictor

        with patch("inference.predictor.load_config", return_value=cfg):
            predictor = GrayspotPredictor.__new__(GrayspotPredictor)
            import logging
            predictor._logger = logging.getLogger("TestPredictor")
            predictor.cfg = cfg
            predictor.device = torch.device("cpu")
            predictor.models = {}
            predictor.model_paths = {}
            predictor.channels = cfg.get("data", {}).get("channels", ["Y", "M", "C", "K"])
            predictor.image_size = cfg.get("data", {}).get("image_size", 128)
            predictor.num_levels = cfg.get("data", {}).get("num_levels", 6)
        return predictor

    def test_file_not_found_on_missing_model(self, cfg, tmp_path):
        """존재하지 않는 모델 경로를 전달하면 FileNotFoundError가 발생한다."""
        # Passing a non-existent path must raise FileNotFoundError (SSOT-FF01)
        predictor = self._make_predictor(cfg)
        missing_path = tmp_path / "nonexistent" / "best_Y.pt"
        with pytest.raises(FileNotFoundError):
            predictor.load_model("Y", model_path=str(missing_path))

    def test_file_not_found_message_contains_artifact_info(self, cfg, tmp_path):
        """FileNotFoundError 메시지에 파일 경로 관련 정보가 포함된다."""
        predictor = self._make_predictor(cfg)
        missing_path = tmp_path / "missing.pt"
        with pytest.raises(FileNotFoundError, match=r"best_Y\.pt|missing\.pt|artifact|not found"):
            predictor.load_model("Y", model_path=str(missing_path))

    def test_unsupported_channel_raises_value_error(self, cfg):
        """지원하지 않는 채널명 전달 시 ValueError가 발생한다."""
        predictor = self._make_predictor(cfg)
        with pytest.raises(ValueError, match="Unsupported channel"):
            predictor.load_model("Z", model_path="/any/path.pt")


# ── 장치 설정 / Device setup ──────────────────────────────────────────────


class TestDeviceSetup:
    """DeviceMixin — config 기반 장치 선택 검증."""

    def _make_predictor_with_device_cfg(self, cfg: dict, device_str: str):
        """system.device 값을 재정의한 predictor 인스턴스를 반환한다."""
        from inference.predictor_device import DeviceMixin

        class _ConcreteDevice(DeviceMixin):
            def __init__(self, cfg):
                import logging
                self._logger = logging.getLogger(self.__class__.__name__)
                self.cfg = cfg

        modified_cfg = {**cfg, "system": {"device": device_str}}
        return _ConcreteDevice(modified_cfg)

    def test_cpu_device_config_returns_cpu(self, cfg):
        """config system.device='cpu' 이면 torch.device('cpu')를 반환한다."""
        dev_mixin = self._make_predictor_with_device_cfg(cfg, "cpu")
        result = dev_mixin._setup_device()
        assert result.type == "cpu"

    def test_auto_device_returns_torch_device(self, cfg):
        """config system.device='auto' 이면 torch.device 인스턴스를 반환한다."""
        dev_mixin = self._make_predictor_with_device_cfg(cfg, "auto")
        result = dev_mixin._setup_device()
        assert isinstance(result, torch.device)

    def test_unknown_device_falls_back_to_cpu(self, cfg):
        """알 수 없는 device 값은 CPU로 폴백된다."""
        # Unknown device value must fall back to CPU
        dev_mixin = self._make_predictor_with_device_cfg(cfg, "unknown_device")
        result = dev_mixin._setup_device()
        assert result.type == "cpu"
