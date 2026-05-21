"""
test_roi_extractor.py
Tests for ROIExtractor class (data/roi_extractor.py).
Status: FAILING — ROIExtractor not yet implemented.
Ref: doc/TDD/TDD_ROI_Pipeline.md
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# ── sys.path 설정 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

pytest.importorskip("data.roi_extractor", reason="ROIExtractor not implemented yet")
# Will raise ImportError until implemented — correct failing behavior

from data.roi_extractor import ROIExtractor  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def roi_cfg():
    """ROIExtractor 전용 최소 cfg — conftest.minimal_cfg 와 별개로 정의."""
    return {
        "roi": {"mode": "fixed", "fixed_coords": [0, 0, 128, 128]},
        "data": {"image_size": 128},
    }


@pytest.fixture
def tmp_image_path(tmp_path):
    """256×256 랜덤 BGR PNG 이미지를 tmp_path에 저장하고 경로를 반환."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / "test_scan.png"
    cv2.imwrite(str(path), img)
    return path


# ── split_cmyk() ──────────────────────────────────────────────────────────────
# T-ROI-01 ~ T-ROI-05


class TestSplitCmyk:
    """T-ROI-01 ~ T-ROI-05: split_cmyk() 기본 동작 검증."""

    def test_white_image_cmyk_all_zero(self, roi_cfg):
        """T-ROI-01: 흰색 이미지(255,255,255) → C=M=Y=K=0.0"""
        extractor = ROIExtractor(cfg=roi_cfg)
        white = np.ones((128, 128, 3), dtype=np.uint8) * 255
        result = extractor.split_cmyk(white)
        assert set(result.keys()) == {
            "C",
            "M",
            "Y",
            "K",
        }, "결과 dict 키는 정확히 {'C','M','Y','K'} 여야 함"
        assert np.allclose(result["C"], 0.0, atol=1e-5), "흰색 → C=0.0 기대"
        assert np.allclose(result["M"], 0.0, atol=1e-5), "흰색 → M=0.0 기대"
        assert np.allclose(result["Y"], 0.0, atol=1e-5), "흰색 → Y=0.0 기대"
        assert np.allclose(result["K"], 0.0, atol=1e-5), "흰색 → K=0.0 기대"

    def test_black_image_cmyk_all_one(self, roi_cfg):
        """T-ROI-02: 검정 이미지(0,0,0) → C=M=Y=K=1.0"""
        extractor = ROIExtractor(cfg=roi_cfg)
        black = np.zeros((128, 128, 3), dtype=np.uint8)
        result = extractor.split_cmyk(black)
        assert np.allclose(result["C"], 1.0, atol=1e-5), "검정 → C=1.0 기대"
        assert np.allclose(result["M"], 1.0, atol=1e-5), "검정 → M=1.0 기대"
        assert np.allclose(result["Y"], 1.0, atol=1e-5), "검정 → Y=1.0 기대"
        assert np.allclose(result["K"], 1.0, atol=1e-5), "검정 → K=1.0 기대"

    def test_pure_red_channel_values(self, roi_cfg):
        """T-ROI-03: 순수 빨강 BGR=(0,0,255) → C=0.0, M=1.0, Y=1.0"""
        extractor = ROIExtractor(cfg=roi_cfg)
        red_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255  # BGR 포맷에서 R 채널
        result = extractor.split_cmyk(red_bgr)
        assert np.allclose(result["C"], 0.0, atol=1e-5), "순수 빨강 → C=0.0 기대"
        assert np.allclose(result["M"], 1.0, atol=1e-5), "순수 빨강 → M=1.0 기대"
        assert np.allclose(result["Y"], 1.0, atol=1e-5), "순수 빨강 → Y=1.0 기대"

    def test_random_image_values_in_range(self, roi_cfg):
        """T-ROI-04: 랜덤 BGR 이미지 → 모든 채널 값 [0.0, 1.0] 범위"""
        extractor = ROIExtractor(cfg=roi_cfg)
        rng = np.random.default_rng(0)
        random_img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        result = extractor.split_cmyk(random_img)
        for ch in ["C", "M", "Y", "K"]:
            assert (result[ch] >= 0.0).all(), f"{ch} 채널에 0 미만 값 존재"
            assert (result[ch] <= 1.0).all(), f"{ch} 채널에 1 초과 값 존재"

    def test_output_dtype_is_float32(self, roi_cfg):
        """T-ROI-05: 입력 uint8 → 반환 dtype은 반드시 float32"""
        extractor = ROIExtractor(cfg=roi_cfg)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = extractor.split_cmyk(img)
        for ch in ["C", "M", "Y", "K"]:
            assert (
                result[ch].dtype == np.float32
            ), f"{ch} dtype={result[ch].dtype}, float32 기대"


# ── extract_patches() ─────────────────────────────────────────────────────────
# T-ROI-10 ~ T-ROI-14


class TestExtractPatches:
    """T-ROI-10 ~ T-ROI-14: extract_patches() 기본 동작 검증."""

    def test_returns_correct_shape(self, roi_cfg, tmp_image_path):
        """T-ROI-10: 유효 이미지 경로 → 각 패치 shape (128,128,3), dtype uint8"""
        extractor = ROIExtractor(cfg=roi_cfg)
        patches = extractor.extract_patches(tmp_image_path, channel="Y", level=3)
        assert len(patches) > 0, "패치 리스트가 비어 있음"
        for p in patches:
            assert p.shape == (128, 128, 3), f"패치 shape={p.shape}, (128,128,3) 기대"
            assert p.dtype == np.uint8, f"패치 dtype={p.dtype}, uint8 기대"

    def test_invalid_path_raises_file_not_found(self, roi_cfg):
        """T-ROI-11: 존재하지 않는 경로 → FileNotFoundError"""
        extractor = ROIExtractor(cfg=roi_cfg)
        with pytest.raises(FileNotFoundError):
            extractor.extract_patches(
                "/nonexistent/path/image.png", channel="Y", level=1
            )

    def test_three_channel_values_equal_for_gray_channel(self, roi_cfg, tmp_image_path):
        """T-ROI-12: Y채널 패치 → 3채널이지만 세 채널 값이 모두 동일 (grayscale 복제)"""
        extractor = ROIExtractor(cfg=roi_cfg)
        patches = extractor.extract_patches(tmp_image_path, channel="Y", level=3)
        assert len(patches) > 0, "패치 리스트가 비어 있음"
        for idx, p in enumerate(patches):
            assert np.array_equal(p[:, :, 0], p[:, :, 1]), f"패치[{idx}] Ch0 != Ch1"
            assert np.array_equal(p[:, :, 1], p[:, :, 2]), f"패치[{idx}] Ch1 != Ch2"

    def test_result_not_empty_for_valid_input(self, roi_cfg, tmp_image_path):
        """T-ROI-13: 유효 입력(level=3) → 비어있지 않은 List 반환"""
        extractor = ROIExtractor(cfg=roi_cfg)
        patches = extractor.extract_patches(tmp_image_path, channel="Y", level=3)
        assert isinstance(patches, list), "반환값이 list 타입이어야 함"
        assert len(patches) > 0, "패치 리스트가 비어 있음"

    def test_missing_cfg_raises_type_error(self):
        """T-ROI-14: cfg 없이 ROIExtractor() 생성 → TypeError"""
        with pytest.raises(TypeError):
            ROIExtractor()  # type: ignore[call-arg]
