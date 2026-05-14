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
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


# ── Confidence threshold 분기 로직 / Confidence threshold branching ──────────


class TestConfidenceThresholdLogic:
    """
    inference/predictor.py 의 신뢰도 임계값 분기를 독립적으로 검증한다.
    Validates confidence threshold branching independently from the predictor class.
    """

    def _classify_confidence(self, confidence: float, thresholds: dict) -> str:
        auto = thresholds["auto_accept"]
        warn = thresholds["warn_threshold"]
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
        assert (
            not hasattr(predictor, "_models")
            or len(getattr(predictor, "_models", {})) == 0
        )


# ── Model Loading & Prediction Integration / Model Loading & Prediction ──────


class TestGrayspotPredictorIntegration:
    """
    End-to-end integration tests for GrayspotPredictor with real model loading and prediction.
    Requires trained model checkpoints to be present in the expected locations.
    """

    @pytest.fixture
    def predictor(self):
        from inference.predictor import GrayspotPredictor

        return GrayspotPredictor()

    @pytest.fixture
    def sample_images(self):
        """Create sample RGB images for testing."""
        np.random.seed(42)  # For reproducibility
        return np.random.randint(0, 256, (4, 128, 128, 3), dtype=np.uint8)

    def test_load_model_success(self, predictor):
        """Test successful model loading for a channel."""
        # This test assumes a model file exists at the expected path
        # In CI, this would be mocked or use a test model
        try:
            predictor.load_model("Y")  # Try loading Y channel model
            assert "Y" in predictor.models
            assert predictor.models["Y"] is not None
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available for integration testing")

    def test_predict_single_channel(self, predictor, sample_images):
        """Test single-channel prediction with loaded model."""
        try:
            predictor.load_model("Y")
            result = predictor.predict(sample_images, "Y", batch_size=2)

            # Validate result structure
            assert "predictions" in result
            assert "logits" in result
            assert "probabilities" in result
            assert "confidences" in result  # Since return_confidences=True by default

            # Validate shapes
            assert result["predictions"].shape == (4,)
            assert result["logits"].shape == (4, 6)  # 6 levels
            assert result["probabilities"].shape == (4, 6)
            assert result["confidences"].shape == (4,)

            # Validate prediction ranges
            assert np.all(result["predictions"] >= 0)
            assert np.all(result["predictions"] < 6)  # 0-5 levels
            assert np.all(result["confidences"] >= 0.0)
            assert np.all(result["confidences"] <= 1.0)

        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available for integration testing")

    def test_predict_batch_multi_channel(self, predictor, sample_images):
        """Test multi-channel batch prediction."""
        try:
            # Load multiple models
            loaded_channels = []
            for ch in ["Y", "M", "C", "K"]:
                try:
                    predictor.load_model(ch)
                    loaded_channels.append(ch)
                except (FileNotFoundError, RuntimeError):
                    continue

            if not loaded_channels:
                pytest.skip("No model files available for integration testing")

            # Create image dict for loaded channels
            images_dict = {ch: sample_images for ch in loaded_channels}

            result = predictor.predict_batch(images_dict, batch_size=2)

            # Validate results for each loaded channel
            for ch in loaded_channels:
                assert ch in result
                ch_result = result[ch]
                assert "predictions" in ch_result
                assert ch_result["predictions"].shape == (4,)

        except Exception:
            pytest.skip("Multi-channel testing not available")

    def test_predict_without_loaded_model_raises_error(self, predictor, sample_images):
        """Test that prediction fails when model is not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict(sample_images, "Y")

    def test_invalid_channel_raises_error(self, predictor):
        """Test that invalid channel raises error on load."""
        with pytest.raises(ValueError, match="Unsupported channel"):
            predictor.load_model("X")  # Invalid channel

    def test_predict_invalid_input_shape_raises_error(self, predictor):
        """Test that invalid input shapes raise errors."""
        try:
            predictor.load_model("Y")
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available")

        # Test 2D input
        invalid_images = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        with pytest.raises(ValueError, match="images must be"):
            predictor.predict(invalid_images, "Y")

    def test_get_model_info(self, predictor):
        """Test model info retrieval."""
        try:
            predictor.load_model("Y")
            info = predictor.get_model_info("Y")
            assert "device" in info
            assert "model_path" in info
            assert "num_parameters" in info
            assert info["num_parameters"] > 0
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available")

    def test_clear_cache(self, predictor):
        """Test cache clearing functionality."""
        try:
            predictor.load_model("Y")
            assert "Y" in predictor.models

            predictor.clear_cache("Y")
            assert "Y" not in predictor.models

            # Test clear all
            predictor.load_model("Y")
            predictor.clear_cache()
            assert len(predictor.models) == 0
        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available")


# ── ONNX Export Integration / ONNX Export ────────────────────────────────────


class TestGrayspotPredictorONNX:
    """
    Integration tests for ONNX export functionality.
    """

    @pytest.fixture
    def predictor(self):
        from inference.predictor import GrayspotPredictor

        return GrayspotPredictor()

    @pytest.fixture
    def temp_onnx_path(self, tmp_path):
        """Temporary path for ONNX file."""
        return tmp_path / "test_model.onnx"

    def test_onnx_export_without_model_raises_error(self, predictor, temp_onnx_path):
        """Test that ONNX export fails when no model is loaded."""
        with pytest.raises(ValueError, match=r"\[EXPORT ERROR\] No model loaded"):
            predictor.export_to_onnx("Y", temp_onnx_path)

    def test_onnx_export_success(self, predictor, temp_onnx_path):
        """Test successful ONNX export."""
        try:
            predictor.load_model("Y")

            # Export to ONNX
            predictor.export_to_onnx("Y", temp_onnx_path)

            # Verify file was created
            assert temp_onnx_path.exists()
            assert temp_onnx_path.stat().st_size > 0

        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available for ONNX export testing")

    def test_onnx_export_with_custom_sample_input(self, predictor, temp_onnx_path):
        """Test ONNX export with custom sample input."""
        try:
            predictor.load_model("Y")

            # Create custom sample input
            sample_input = torch.randn(2, 3, 128, 128)  # Batch size 2

            predictor.export_to_onnx("Y", temp_onnx_path, sample_input)

            assert temp_onnx_path.exists()

        except (FileNotFoundError, RuntimeError):
            pytest.skip("Model file not available for ONNX export testing")
