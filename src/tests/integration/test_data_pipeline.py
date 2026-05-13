"""
tests/integration/test_data_pipeline.py

데이터 파이프라인 통합 테스트.
Integration tests for the data pipeline.

preprocess → CMYKDataset/ContrastiveDataset → DataLoader 흐름을 검증한다.
Validates the flow: preprocess → CMYKDataset/ContrastiveDataset → DataLoader.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data.augmentation import augment_contrastive, augment_supervised
from data.dataset import CMYKDataset, ContrastiveDataset
from data.preprocessing import preprocess

# ── preprocess → tensor 변환 흐름 / preprocess → tensor conversion ───────────


class TestPreprocessToTensorFlow:
    def test_preprocess_output_compatible_with_unsqueeze(self):
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        arr = preprocess(img, image_size=128)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        assert tensor.shape == (3, 128, 128)

    def test_preprocess_then_augment_supervised_compatible(self):
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        arr = preprocess(img, image_size=128)
        aug = augment_supervised(
            arr, aug_cfg={"flip_prob": 0.0, "brightness_prob": 0.0, "noise_prob": 0.0}
        )
        assert aug.shape == (128, 128, 3)
        assert aug.dtype == np.float32

    def test_preprocess_then_augment_contrastive_produces_same_shape(self):
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        arr = preprocess(img, image_size=128)
        view = augment_contrastive(arr, image_size=128)
        assert view.shape == (128, 128, 3)


# ── CMYKDataset ───────────────────────────────────────────────────────────────


class TestCMYKDataset:
    def test_dataset_length_positive(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = CMYKDataset(
            cfg, channel="Y", split="train", augment=False, oversample=False
        )
        assert len(ds) > 0

    def test_item_shape(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = CMYKDataset(
            cfg, channel="Y", split="train", augment=False, oversample=False
        )
        x, y = ds[0]
        assert x.shape == (3, 128, 128)

    def test_item_dtype_float(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = CMYKDataset(
            cfg, channel="Y", split="train", augment=False, oversample=False
        )
        x, y = ds[0]
        assert x.dtype == torch.float32

    def test_label_in_valid_range(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = CMYKDataset(
            cfg, channel="Y", split="train", augment=False, oversample=False
        )
        for i in range(len(ds)):
            _, y = ds[i]
            assert 0 <= int(y) < cfg["data"]["num_levels"]

    def test_tensor_is_imagenet_normalized(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = CMYKDataset(
            cfg, channel="Y", split="train", augment=False, oversample=False
        )
        x, _ = ds[0]
        # Values must be finite; ImageNet normalization produces values outside [0, 1]
        assert torch.isfinite(x).all()
        assert x.dtype == torch.float32
        # After mean subtraction at least one channel's dark pixels go negative
        assert (
            float(x.min()) < 0.0
        ), "ImageNet normalization not applied — min should be negative"

    def test_dataloader_batch_shape(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = CMYKDataset(
            cfg, channel="Y", split="train", augment=False, oversample=False
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        x, y = next(iter(loader))
        assert x.shape[1:] == (3, 128, 128)
        assert x.shape[0] <= 2


# ── ContrastiveDataset ────────────────────────────────────────────────────────


class TestContrastiveDataset:
    def test_dataset_length_positive(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = ContrastiveDataset(cfg, channel="Y")
        assert len(ds) > 0

    def test_item_returns_pair(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = ContrastiveDataset(cfg, channel="Y")
        v1, v2 = ds[0]
        assert isinstance(v1, torch.Tensor)
        assert isinstance(v2, torch.Tensor)

    def test_pair_shapes_equal(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = ContrastiveDataset(cfg, channel="Y")
        v1, v2 = ds[0]
        assert v1.shape == v2.shape

    def test_pair_shape_correct(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = ContrastiveDataset(cfg, channel="Y")
        v1, v2 = ds[0]
        assert v1.shape == (3, 128, 128)

    def test_pair_values_differ(self, labeled_data_dir, minimal_cfg):
        cfg = copy.deepcopy(minimal_cfg)
        cfg["storage"]["labeled_dir"] = str(labeled_data_dir / "labeled")
        ds = ContrastiveDataset(cfg, channel="Y")
        # 여러 번 시도하여 적어도 한 번은 다른 augmentation 확인
        found_different = False
        for _ in range(10):
            v1, v2 = ds[0]
            if not torch.allclose(v1, v2):
                found_different = True
                break
        assert found_different, "Positive pair가 항상 동일 — augmentation 다양성 필요"
