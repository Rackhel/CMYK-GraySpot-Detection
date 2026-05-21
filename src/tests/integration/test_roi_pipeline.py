"""
tests/integration/test_roi_pipeline.py

ROI 파이프라인 통합 테스트.
Integration tests for the ROI pipeline.

ROIExtractor → 패치 추출 → ContrastiveDataset 로드 → 모델 forward pass 전체 흐름 검증.
Validates end-to-end: ROIExtractor → patch extraction → ContrastiveDataset → model forward.

테스트 ID / Test IDs:
    T-ROI-INT-01: ROI 이미지 → extract_patches_from_roi() → 디스크에 PNG 저장 확인
    T-ROI-INT-02: 저장된 패치를 ContrastiveDataset이 로드 → Tensor (3,128,128)
    T-ROI-INT-03: 로드된 텐서를 GrayspotModel forward → 출력 shape (B, 6)

TDD Reference: doc/TDD/TDD_ROI_Pipeline.md §3
BDD Reference: doc/BDD/BDD_ROI_Pipeline.md Scenario R.1

Python 3.11.5
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from data.dataset import ContrastiveDataset
from data.roi_extractor import ROIExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def roi_cfg():
    """ROIExtractor 최소 cfg (auto 모드 — 크롭 없이 전체 이미지)."""
    return {
        "roi": {"mode": "auto"},
        "data": {"image_size": 128},
    }


@pytest.fixture
def roi_image_path(tmp_path):
    """384×128 BGR PNG 이미지 (슬라이딩 윈도우로 3패치 생성 가능)."""
    img = np.random.randint(10, 245, (384, 128, 3), dtype=np.uint8)
    path = tmp_path / "lvl3_test_scan_Y.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def labeled_dir_from_roi(tmp_path, roi_cfg, roi_image_path):
    """
    ROIExtractor로 패치를 추출하여 labeled/{ch}/{lv}/ 에 저장하고
    labeled_dir (tmp_path) 을 반환한다.
    """
    extractor = ROIExtractor(cfg=roi_cfg)
    patches = extractor.extract_patches_from_roi(roi_image_path)

    ch, lv = "Y", 3
    dst_dir = tmp_path / "labeled" / ch / str(lv)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for idx, patch in enumerate(patches, start=1):
        fname = f"lvl3_test_scan_Y_{idx:04d}.png"
        cv2.imwrite(str(dst_dir / fname), patch)

    return tmp_path


# ── T-ROI-INT-01 ──────────────────────────────────────────────────────────────


class TestROIExtractorToFile:
    """T-ROI-INT-01: ROI 이미지 → extract_patches_from_roi → 파일 저장."""

    def test_patches_saved_as_png(self, labeled_dir_from_roi):
        """T-ROI-INT-01a: 추출된 패치가 PNG 파일로 디스크에 저장된다."""
        saved = list((labeled_dir_from_roi / "labeled" / "Y" / "3").glob("*.png"))
        assert len(saved) > 0, "패치 PNG 파일이 저장되지 않음"

    def test_saved_patch_shape(self, labeled_dir_from_roi):
        """T-ROI-INT-01b: 저장된 PNG를 읽으면 (128,128,3) BGR uint8 이다."""
        saved = sorted((labeled_dir_from_roi / "labeled" / "Y" / "3").glob("*.png"))
        img = cv2.imread(str(saved[0]))
        assert img is not None
        assert img.shape == (128, 128, 3)
        assert img.dtype == np.uint8

    def test_saved_filename_follows_naming_convention(self, labeled_dir_from_roi):
        """T-ROI-INT-01c: 파일명이 {roi_stem}_{idx:04d}.png 형식이다."""
        saved = sorted((labeled_dir_from_roi / "labeled" / "Y" / "3").glob("*.png"))
        for f in saved:
            name = f.stem
            # 마지막 _XXXX 가 4자리 숫자인지 확인
            parts = name.rsplit("_", 1)
            assert len(parts) == 2, f"파일명 규칙 불일치: {name}"
            assert parts[1].isdigit() and len(parts[1]) == 4, f"idx 형식 불일치: {parts[1]}"


# ── T-ROI-INT-02 ──────────────────────────────────────────────────────────────


class TestContrastiveDatasetLoad:
    """T-ROI-INT-02: 저장된 패치를 ContrastiveDataset이 로드 → Tensor (3,128,128)."""

    @pytest.fixture
    def minimal_cfg(self, labeled_dir_from_roi):
        return {
            "system": {"device": "cpu"},
            "data": {
                "channels": ["Y"],
                "num_levels": 6,
                "image_size": 128,
                "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            },
            "model": {"backbone": "efficientnet_b0", "frozen_backbone": False},
            "phase0": {
                "projection_dim": 128,
                "hidden_dim": 256,
                "temperature": 0.1,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "augmentation": {
                    "flip_prob": 0.5,
                    "blur_prob": 0.3,
                    "crop_prob": 0.3,
                    "color_jitter": 0.2,
                    "crop_scale_min": 0.8,
                    "crop_scale_max": 1.0,
                    "contrast_scale_min": 0.9,
                    "contrast_scale_max": 1.1,
                    "blur_kernels": [3],
                },
            },
            "storage": {
                "labeled_dir": str(labeled_dir_from_roi / "labeled"),
            },
            "train": {"seed": 42},
        }

    def test_dataset_loads_without_error(self, minimal_cfg):
        """T-ROI-INT-02a: ContrastiveDataset 생성 시 에러 없음."""
        ds = ContrastiveDataset(minimal_cfg, channel="Y")
        assert len(ds) > 0

    def test_item_returns_two_tensors(self, minimal_cfg):
        """T-ROI-INT-02b: __getitem__ 이 (view1, view2) 텐서 쌍을 반환한다."""
        ds = ContrastiveDataset(minimal_cfg, channel="Y")
        v1, v2 = ds[0]
        assert isinstance(v1, torch.Tensor)
        assert isinstance(v2, torch.Tensor)

    def test_tensor_shape_is_3_128_128(self, minimal_cfg):
        """T-ROI-INT-02c: 각 텐서 shape 이 (3, 128, 128) 이다."""
        ds = ContrastiveDataset(minimal_cfg, channel="Y")
        v1, v2 = ds[0]
        assert v1.shape == (3, 128, 128)
        assert v2.shape == (3, 128, 128)

    def test_tensor_dtype_float32(self, minimal_cfg):
        """T-ROI-INT-02d: 텐서 dtype 이 float32 이다."""
        ds = ContrastiveDataset(minimal_cfg, channel="Y")
        v1, _ = ds[0]
        assert v1.dtype == torch.float32


# ── T-ROI-INT-03 ──────────────────────────────────────────────────────────────


class TestModelForwardPass:
    """T-ROI-INT-03: 로드된 텐서를 GrayspotModel forward → 출력 shape (B, 6)."""

    @pytest.fixture
    def model_cfg(self):
        return {
            "system": {"device": "cpu"},
            "data": {"num_levels": 6, "image_size": 128},
            "model": {"backbone": "efficientnet_b0", "frozen_backbone": False},
            "phase0": {
                "projection_dim": 128,
                "hidden_dim": 256,
            },
            "phase2": {
                "dropout": 0.3,
                "hidden_dim": 256,
                "heads": {
                    "efficientnet_b0": {
                        "mid_dim": None,
                        "hidden_dim": 256,
                        "dropout": 0.2,
                    }
                },
            },
            "train": {"seed": 42},
        }

    def test_phase0_forward_output_shape(self, model_cfg):
        """T-ROI-INT-03a: Phase 0 임베딩 출력 shape 이 (B, projection_dim) 이다."""
        from models.grayspot_model import GrayspotModel

        model = GrayspotModel(model_cfg, phase=0).eval()
        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 128), f"Phase 0 출력 shape 불일치: {out.shape}"

    def test_phase2_forward_output_shape(self, model_cfg):
        """T-ROI-INT-03b: Phase 2 분류 출력 shape 이 (B, num_levels) 이다."""
        from models.grayspot_model import GrayspotModel

        model = GrayspotModel(model_cfg, phase=2).eval()
        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6), f"Phase 2 출력 shape 불일치: {out.shape}"
