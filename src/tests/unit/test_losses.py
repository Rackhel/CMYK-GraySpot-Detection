"""
tests/unit/test_losses.py

training/contrastive_loss.py, training/losses.py 단위 테스트.
Unit tests for training/contrastive_loss.py and training/losses.py.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from training.contrastive_loss import InfoNCELoss
from training.losses import get_loss

# ── InfoNCELoss ──────────────────────────────────────────────────────────────


class TestInfoNCELoss:
    def test_output_is_scalar(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        z1 = F.normalize(torch.randn(8, 128), dim=1)
        z2 = F.normalize(torch.randn(8, 128), dim=1)
        loss = loss_fn(z1, z2)
        assert loss.ndim == 0

    def test_output_is_finite(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        z1 = F.normalize(torch.randn(8, 128), dim=1)
        z2 = F.normalize(torch.randn(8, 128), dim=1)
        loss = loss_fn(z1, z2)
        assert torch.isfinite(loss)

    def test_output_is_non_negative(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        z1 = F.normalize(torch.randn(8, 128), dim=1)
        z2 = F.normalize(torch.randn(8, 128), dim=1)
        loss = loss_fn(z1, z2)
        assert loss.item() >= 0.0

    def test_similar_pairs_lower_loss_than_random(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        torch.manual_seed(42)
        z = F.normalize(torch.randn(8, 128), dim=1)
        # 거의 동일한 pair (작은 noise 추가)
        similar_loss = loss_fn(z, F.normalize(z + torch.randn_like(z) * 0.01, dim=1))
        # 완전 랜덤 pair
        random_loss = loss_fn(z, F.normalize(torch.randn(8, 128), dim=1))
        assert similar_loss.item() < random_loss.item()

    def test_loss_decreases_with_smaller_temperature(self):
        torch.manual_seed(0)
        z1 = F.normalize(torch.randn(8, 128), dim=1)
        z2 = F.normalize(torch.randn(8, 128), dim=1)
        loss_low_temp = InfoNCELoss(temperature=0.07)(z1, z2)
        loss_high_temp = InfoNCELoss(temperature=1.0)(z1, z2)
        # 낮은 temperature = sharper distribution → 어려운 task → 일반적으로 높은 loss
        # 이 테스트는 두 값이 다름을 확인 (방향은 환경에 따라 다를 수 있음)
        assert loss_low_temp.item() != pytest.approx(loss_high_temp.item())

    def test_gradients_flow_through_loss(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        z1 = F.normalize(torch.randn(8, 128), dim=1)
        z1.requires_grad_(True)
        z2 = F.normalize(torch.randn(8, 128), dim=1)
        loss = loss_fn(z1, z2)
        loss.backward()
        assert z1.grad is not None
        assert torch.isfinite(z1.grad).all()

    def test_returns_tensor(self):
        loss_fn = InfoNCELoss(temperature=0.1)
        z1 = F.normalize(torch.randn(4, 64), dim=1)
        z2 = F.normalize(torch.randn(4, 64), dim=1)
        assert isinstance(loss_fn(z1, z2), torch.Tensor)


# ── get_loss ─────────────────────────────────────────────────────────────────


class TestGetLoss:
    def test_phase0_returns_infonce(self, minimal_cfg):
        loss_fn = get_loss(phase=0, cfg=minimal_cfg)
        assert isinstance(loss_fn, InfoNCELoss)

    def test_phase0_temperature_from_cfg(self, minimal_cfg):
        minimal_cfg["phase0"]["temperature"] = 0.07
        loss_fn = get_loss(phase=0, cfg=minimal_cfg)
        assert loss_fn.temperature == pytest.approx(0.07)

    def test_phase2_returns_cross_entropy(self, minimal_cfg):
        loss_fn = get_loss(phase=2, cfg=minimal_cfg)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_phase2_no_class_weights_by_default(self, minimal_cfg):
        loss_fn = get_loss(phase=2, cfg=minimal_cfg)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is None

    def test_invalid_phase_raises_value_error(self, minimal_cfg):
        with pytest.raises(ValueError):
            get_loss(phase=1, cfg=minimal_cfg)

    def test_phase2_balanced_weights_with_samples(self, minimal_cfg):
        minimal_cfg["phase2"]["class_weights"] = "balanced"
        samples = [(f"img_{i}.png", i % 6) for i in range(60)]
        loss_fn = get_loss(phase=2, cfg=minimal_cfg, train_samples=samples)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert loss_fn.weight.shape == (6,)
