"""
tests/unit/test_utils_model.py

utils/utils_model.py 단위 테스트.
Unit tests for utils/utils_model.py.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from utils.utils_model import backbone_tag, set_seed

# ── backbone_tag ────────────────────────────────────────────────────────────


class TestBackboneTag:
    def test_efficientnet_b0_returns_effb0(self):
        assert backbone_tag("efficientnet_b0") == "effb0"

    def test_resnet50_returns_res50(self):
        assert backbone_tag("resnet50") == "res50"

    def test_unknown_backbone_returns_truncated_name(self):
        tag = backbone_tag("some_custom_backbone")
        assert isinstance(tag, str)
        assert len(tag) <= 8

    def test_unknown_backbone_removes_underscores(self):
        tag = backbone_tag("my_model")
        assert "_" not in tag

    def test_returns_string(self):
        assert isinstance(backbone_tag("efficientnet_b0"), str)


# ── set_seed ────────────────────────────────────────────────────────────────


class TestSetSeed:
    def test_same_seed_produces_same_torch_random(self):
        set_seed(42)
        a = torch.rand(5).tolist()
        set_seed(42)
        b = torch.rand(5).tolist()
        assert a == b

    def test_same_seed_produces_same_numpy_random(self):
        set_seed(0)
        a = np.random.rand(5).tolist()
        set_seed(0)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_same_seed_produces_same_python_random(self):
        set_seed(7)
        a = [random.random() for _ in range(5)]
        set_seed(7)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_produce_different_results(self):
        set_seed(1)
        a = torch.rand(10).tolist()
        set_seed(2)
        b = torch.rand(10).tolist()
        assert a != b

    def test_with_cfg_does_not_raise(self, minimal_cfg):
        set_seed(42, cfg=minimal_cfg)

    def test_without_cfg_does_not_raise(self):
        set_seed(42)
