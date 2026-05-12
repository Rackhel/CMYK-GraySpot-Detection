"""
tests/unit/conftest.py

단위 테스트 공용 Fixture.
Shared fixtures for unit tests.

모든 Fixture는 외부 파일 I/O 없이 동작한다.
All fixtures work without external file I/O.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── sys.path 설정 / sys.path setup ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


# ── Config Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def minimal_cfg():
    """실제 파일 없이 동작하는 최소 config dict (device=cpu 고정)."""
    return {
        "system": {"device": "cpu"},
        "data": {
            "channels": ["Y", "M", "C", "K"],
            "num_levels": 6,
            "image_size": 128,
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "model": {
            "backbone": "efficientnet_b0",
            "frozen_backbone": False,
        },
        "phase0": {
            "projection_dim": 128,
            "hidden_dim": 256,
            "temperature": 0.1,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "augmentation": {
                "color_jitter": 0.4,
                "blur_prob": 0.5,
                "flip_prob": 0.5,
                "crop_prob": 0.5,
                "crop_scale_min": 0.6,
                "crop_scale_max": 1.0,
                "contrast_scale_min": 0.8,
                "contrast_scale_max": 1.2,
                "blur_kernels": [3, 5],
            },
        },
        "phase2": {
            "dropout": 0.3,
            "hidden_dim": 256,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "oversample": False,
            "early_stopping": {"enabled": False, "patience": 5, "min_delta": 1e-4},
            "augmentation": {
                "flip_prob": 0.5,
                "brightness_prob": 0.5,
                "brightness_range": 30,
                "noise_prob": 0.5,
                "noise_range": 10,
            },
            "heads": {
                "efficientnet_b0": {"mid_dim": None, "hidden_dim": 256, "dropout": 0.2},
                "resnet50": {"mid_dim": 512, "hidden_dim": 256, "dropout": 0.4},
            },
        },
        "evaluation": {
            "targets": {
                "overall_accuracy": 0.90,
                "per_color_accuracy": 0.85,
                "per_class_f1": 0.80,
                "mae": 0.50,
            },
            "swing_thresholds": {
                "acc_retry": 0.80,
                "f1_retry": 0.70,
                "mae_retry": 0.80,
            },
        },
        "storage": {
            "data_root": "data_set",
            "labeled_dir": "data_set/labeled",
            "models_dir": "data_set/models",
            "reports_dir": "data_set/reports",
            "logs_dir": "outputs/logs",
        },
        "train": {"seed": 42},
        "inference": {
            "confidence_thresholds": {
                "auto_accept": 0.8,
                "warn_threshold": 0.5,
                "manual_review": 0.3,
            },
        },
        "cuda": {"deterministic": True, "benchmark": False},
    }


@pytest.fixture
def minimal_config_file(tmp_path, minimal_cfg):
    """tmp_path에 최소 config.json 파일을 생성하고 경로를 반환한다."""
    cfg_for_file = {k: v for k, v in minimal_cfg.items()}
    # storage 경로는 상대 경로로 저장 (load_config가 절대경로로 변환)
    cfg_for_file["storage"] = {
        "data_root": "data_set",
        "labeled_dir": "data_set/labeled",
        "models_dir": "data_set/models",
        "reports_dir": "data_set/reports",
        "logs_dir": "outputs/logs",
    }
    cfg_for_file["system"] = {"device": "cpu"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg_for_file), encoding="utf-8")
    return path


# ── Tensor / Array Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def dummy_image_np():
    """(128, 128, 3) uint8 NumPy BGR 이미지."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_float():
    """(128, 128, 3) float32 [0, 1] NumPy 이미지 (전처리 완료 상태)."""
    rng = np.random.default_rng(42)
    return rng.random((128, 128, 3)).astype(np.float32)


@pytest.fixture
def dummy_batch():
    """(4, 3, 128, 128) float32 배치 텐서 (CPU)."""
    torch.manual_seed(42)
    return torch.rand(4, 3, 128, 128)


# ── Prediction Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def perfect_predictions():
    """완전 정확한 예측 — accuracy=1.0, mae=0.0."""
    labels = list(range(6)) * 4  # 24개
    return {
        "y_true": np.array(labels),
        "y_pred": np.array(labels),
    }


@pytest.fixture
def worst_predictions():
    """최대 오차 예측 — Level 0 → 5 로 전부 예측."""
    y_true = np.array([0] * 6)
    y_pred = np.array([5] * 6)
    return {"y_true": y_true, "y_pred": y_pred}


@pytest.fixture
def multi_channel_results():
    """compute_all_channels / build_evaluation_summary 용 멀티채널 결과."""
    rng = np.random.default_rng(0)
    result = {}
    for ch in ["Y", "M", "C", "K"]:
        n = 30
        y_true = rng.integers(0, 6, n)
        y_pred = rng.integers(0, 6, n)
        result[ch] = {"y_true": y_true, "y_pred": y_pred}
    return result
