"""
tests/integration/conftest.py

통합 테스트 공용 Fixture.
Shared fixtures for integration tests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def minimal_cfg():
    return {
        "system": {"device": "cpu"},
        "data": {
            "channels": ["Y", "M", "C", "K"],
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
                "blur_prob": 0.5,
                "crop_prob": 0.5,
                "color_jitter": 0.4,
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
                "flip_prob": 0.0,
                "brightness_prob": 0.0,
                "noise_prob": 0.0,
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
def labeled_data_dir(tmp_path, minimal_cfg):
    """
    tmp_path 에 labeled/{channel}/{level}/*.png 구조를 생성한다.
    Creates labeled/{channel}/{level}/*.png structure under tmp_path.
    """
    import cv2
    import numpy as np

    image_size = minimal_cfg["data"]["image_size"]
    channels = ["Y"]  # 빠른 테스트를 위해 Y채널만 / Y only for speed
    n_per_level = 4

    for ch in channels:
        for level in range(minimal_cfg["data"]["num_levels"]):
            level_dir = tmp_path / "labeled" / ch / str(level)
            level_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_level):
                img = np.random.randint(
                    0, 256, (image_size, image_size, 3), dtype=np.uint8
                )
                path = level_dir / f"img_{i:04d}.png"
                cv2.imwrite(str(path), img)

    return tmp_path
