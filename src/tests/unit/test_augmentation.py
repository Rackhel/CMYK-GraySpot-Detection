"""
tests/unit/test_augmentation.py

data/augmentation.py 단위 테스트.
Unit tests for data/augmentation.py.

주의 / Note:
    augment_contrastive()는 단일 이미지 반환 (pair가 아님).
    ContrastiveDataset이 두 번 호출하여 pair를 만든다.
    augment_contrastive() returns a single image (not a pair).
    ContrastiveDataset calls it twice to create a pair.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data.augmentation import augment_supervised, augment_contrastive


# ── augment_supervised ───────────────────────────────────────────────────────

class TestAugmentSupervised:
    def test_output_shape_preserved(self, dummy_image_float):
        result = augment_supervised(dummy_image_float)
        assert result.shape == dummy_image_float.shape

    def test_output_dtype_float32(self, dummy_image_float):
        result = augment_supervised(dummy_image_float)
        assert result.dtype == np.float32

    def test_output_range_min_gte_zero(self, dummy_image_float):
        result = augment_supervised(dummy_image_float)
        assert float(result.min()) >= 0.0

    def test_output_range_max_lte_one(self, dummy_image_float):
        result = augment_supervised(dummy_image_float)
        assert float(result.max()) <= 1.0

    def test_custom_aug_cfg_accepted(self, dummy_image_float):
        aug_cfg = {"flip_prob": 0.0, "brightness_prob": 0.0, "noise_prob": 0.0}
        result = augment_supervised(dummy_image_float, aug_cfg=aug_cfg)
        assert result.shape == dummy_image_float.shape

    def test_no_augmentation_preserves_image(self, dummy_image_float):
        aug_cfg = {"flip_prob": 0.0, "brightness_prob": 0.0, "noise_prob": 0.0}
        result = augment_supervised(dummy_image_float, aug_cfg=aug_cfg)
        np.testing.assert_array_almost_equal(result, dummy_image_float)

    def test_returns_numpy_array(self, dummy_image_float):
        assert isinstance(augment_supervised(dummy_image_float), np.ndarray)


# ── augment_contrastive ──────────────────────────────────────────────────────

class TestAugmentContrastive:
    def test_output_shape_preserved(self, dummy_image_float):
        result = augment_contrastive(dummy_image_float, image_size=128)
        assert result.shape == dummy_image_float.shape

    def test_output_dtype_float32(self, dummy_image_float):
        result = augment_contrastive(dummy_image_float, image_size=128)
        assert result.dtype == np.float32

    def test_output_range_min_gte_zero(self, dummy_image_float):
        result = augment_contrastive(dummy_image_float, image_size=128)
        assert float(result.min()) >= 0.0

    def test_output_range_max_lte_one(self, dummy_image_float):
        result = augment_contrastive(dummy_image_float, image_size=128)
        assert float(result.max()) <= 1.0

    def test_two_calls_produce_different_views(self, dummy_image_float):
        # 두 번 호출하면 확률적으로 다른 결과 (aug_cfg 기본값: prob=0.5)
        results = [augment_contrastive(dummy_image_float, image_size=128) for _ in range(10)]
        # 10번 중 적어도 2개는 달라야 함
        all_same = all(np.allclose(results[0], r) for r in results[1:])
        assert not all_same, "모든 augmentation 결과가 동일함 — 확률적 다양성 필요"

    def test_custom_aug_cfg_no_augmentation(self, dummy_image_float):
        aug_cfg = {
            "flip_prob": 0.0,
            "crop_prob": 0.0,
            "blur_prob": 0.0,
        }
        result = augment_contrastive(dummy_image_float, image_size=128, aug_cfg=aug_cfg)
        np.testing.assert_array_almost_equal(result, dummy_image_float)

    def test_returns_numpy_array(self, dummy_image_float):
        result = augment_contrastive(dummy_image_float, image_size=128)
        assert isinstance(result, np.ndarray)

    def test_custom_image_size_applied_with_crop(self, dummy_image_float):
        aug_cfg = {"crop_prob": 1.0, "flip_prob": 0.0, "blur_prob": 0.0}
        result = augment_contrastive(dummy_image_float, image_size=64, aug_cfg=aug_cfg)
        assert result.shape == (64, 64, 3)
