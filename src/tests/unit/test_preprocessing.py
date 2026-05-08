"""
tests/unit/test_preprocessing.py

data/preprocessing.py 단위 테스트.
Unit tests for data/preprocessing.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data.preprocessing import preprocess


# ── 출력 shape / dtype / range ──────────────────────────────────────────────

class TestPreprocessOutputSpec:
    def test_output_shape_default_size(self, dummy_image_np):
        result = preprocess(dummy_image_np, image_size=128)
        assert result.shape == (128, 128, 3)

    def test_output_shape_custom_size(self, dummy_image_np):
        result = preprocess(dummy_image_np, image_size=64)
        assert result.shape == (64, 64, 3)

    def test_output_dtype_float32(self, dummy_image_np):
        result = preprocess(dummy_image_np, image_size=128)
        assert result.dtype == np.float32

    def test_output_range_min_gte_zero(self, dummy_image_np):
        result = preprocess(dummy_image_np, image_size=128)
        assert float(result.min()) >= 0.0

    def test_output_range_max_lte_one(self, dummy_image_np):
        result = preprocess(dummy_image_np, image_size=128)
        assert float(result.max()) <= 1.0


# ── 엣지케이스 / Edge cases ──────────────────────────────────────────────────

class TestPreprocessEdgeCases:
    def test_all_black_image_outputs_zeros(self):
        black = np.zeros((128, 128, 3), dtype=np.uint8)
        result = preprocess(black, image_size=128)
        assert result.max() == pytest.approx(0.0)

    def test_all_white_image_outputs_ones(self):
        white = np.full((128, 128, 3), 255, dtype=np.uint8)
        result = preprocess(white, image_size=128)
        assert result.min() == pytest.approx(1.0)

    def test_different_input_size_resized_correctly(self):
        img = np.random.randint(0, 256, (64, 32, 3), dtype=np.uint8)
        result = preprocess(img, image_size=128)
        assert result.shape == (128, 128, 3)

    def test_returns_numpy_array(self, dummy_image_np):
        result = preprocess(dummy_image_np)
        assert isinstance(result, np.ndarray)

    def test_default_size_is_128(self, dummy_image_np):
        result = preprocess(dummy_image_np)
        assert result.shape[0] == 128
        assert result.shape[1] == 128
