"""
tests/unit/test_models.py

models/ 단위 테스트.
Unit tests for models/.

ClassifierHead / ProjectionHead 은 파일 I/O 없이 테스트.
GrayspotModel 은 pretrained weight 다운로드가 필요하므로 별도 마킹.
ClassifierHead / ProjectionHead tested without file I/O.
GrayspotModel requires pretrained weight download — marked separately.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from models.classifier import ClassifierHead
from models.projection_head import ProjectionHead

# ── ClassifierHead ───────────────────────────────────────────────────────────


class TestClassifierHead:
    def test_output_shape_default(self):
        head = ClassifierHead(in_dim=1280)
        x = torch.randn(4, 1280)
        out = head(x)
        assert out.shape == (4, 6)

    def test_output_shape_custom_num_classes(self):
        head = ClassifierHead(in_dim=1280, num_classes=10)
        x = torch.randn(2, 1280)
        assert head(x).shape == (2, 10)

    def test_output_shape_resnet_feature_dim(self):
        head = ClassifierHead(in_dim=2048, num_classes=6)
        x = torch.randn(8, 2048)
        assert head(x).shape == (8, 6)

    def test_output_is_logits_not_probability(self):
        head = ClassifierHead(in_dim=1280)
        x = torch.randn(4, 1280)
        out = head(x)
        # logit 이면 [0,1] 범위를 벗어날 수 있음 / logits can exceed [0, 1]
        assert not (
            out.min() >= 0 and out.max() <= 1
        ), "출력이 logit이 아닌 확률처럼 보임 — Softmax가 포함됐는지 확인"

    def test_batch_size_1(self):
        head = ClassifierHead(in_dim=1280)
        x = torch.randn(1, 1280)
        # BatchNorm1d는 batch_size=1에서 eval 모드에서만 동작
        head.eval()
        with torch.no_grad():
            out = head(x)
        assert out.shape == (1, 6)

    def test_returns_tensor(self):
        head = ClassifierHead(in_dim=1280)
        x = torch.randn(2, 1280)
        assert isinstance(head(x), torch.Tensor)

    def test_custom_dropout_accepted(self):
        head = ClassifierHead(in_dim=1280, dropout=0.5)
        x = torch.randn(4, 1280)
        assert head(x).shape == (4, 6)

    def test_custom_hidden_dim_accepted(self):
        head = ClassifierHead(in_dim=1280, hidden_dim=512)
        x = torch.randn(4, 1280)
        assert head(x).shape == (4, 6)

    # ── mid_dim 파라미터 (backbone-specific) ──────────────────────────────

    def test_effb0_direct_compression_mid_dim_none(self):
        """EfficientNet-B0 경로: mid_dim=None → 1280 → hidden_dim → 6."""
        head = ClassifierHead(
            in_dim=1280, hidden_dim=256, num_classes=6, dropout=0.2, mid_dim=None
        )
        x = torch.randn(4, 1280)
        assert head(x).shape == (4, 6)

    def test_resnet50_staged_compression_mid_dim_512(self):
        """ResNet-50 경로: mid_dim=512 → 2048 → 512 → hidden_dim → 6."""
        head = ClassifierHead(
            in_dim=2048, hidden_dim=256, num_classes=6, dropout=0.4, mid_dim=512
        )
        x = torch.randn(4, 2048)
        assert head(x).shape == (4, 6)

    def test_effb0_head_layer_count_is_2_linear(self):
        """EfficientNet-B0 head(mid_dim=None)는 Linear 레이어가 2개여야 한다."""
        import torch.nn as nn

        head = ClassifierHead(in_dim=1280, hidden_dim=256, num_classes=6, mid_dim=None)
        linears = [m for m in head.net if isinstance(m, nn.Linear)]
        assert len(linears) == 2

    def test_resnet50_head_layer_count_is_3_linear(self):
        """ResNet-50 head(mid_dim=512)는 Linear 레이어가 3개여야 한다."""
        import torch.nn as nn

        head = ClassifierHead(in_dim=2048, hidden_dim=256, num_classes=6, mid_dim=512)
        linears = [m for m in head.net if isinstance(m, nn.Linear)]
        assert len(linears) == 3


# ── ProjectionHead ───────────────────────────────────────────────────────────


class TestProjectionHead:
    def test_output_shape_default(self):
        head = ProjectionHead(in_dim=1280)
        x = torch.randn(4, 1280)
        out = head(x)
        assert out.shape == (4, 128)

    def test_output_shape_custom_out_dim(self):
        head = ProjectionHead(in_dim=1280, out_dim=64)
        x = torch.randn(2, 1280)
        assert head(x).shape == (2, 64)

    def test_output_shape_resnet_feature_dim(self):
        head = ProjectionHead(in_dim=2048, out_dim=128)
        x = torch.randn(4, 2048)
        assert head(x).shape == (4, 128)

    def test_batch_size_1_in_eval_mode(self):
        head = ProjectionHead(in_dim=1280)
        head.eval()
        x = torch.randn(1, 1280)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (1, 128)

    def test_returns_tensor(self):
        head = ProjectionHead(in_dim=1280)
        x = torch.randn(2, 1280)
        assert isinstance(head(x), torch.Tensor)

    def test_custom_hidden_dim_accepted(self):
        head = ProjectionHead(in_dim=1280, hidden_dim=512, out_dim=128)
        x = torch.randn(4, 1280)
        assert head(x).shape == (4, 128)


# ── GrayspotModel (pretrained weight 필요 / requires pretrained weights) ────


@pytest.mark.slow
class TestGrayspotModel:
    """pretrained weights 다운로드가 필요 — `pytest -m slow` 로 별도 실행."""

    def test_phase0_output_shape(self, minimal_cfg):
        from models.grayspot_model import GrayspotModel

        model = GrayspotModel(minimal_cfg, phase=0)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, minimal_cfg["phase0"]["projection_dim"])

    def test_phase2_output_shape(self, minimal_cfg):
        from models.grayspot_model import GrayspotModel

        model = GrayspotModel(minimal_cfg, phase=2)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, minimal_cfg["data"]["num_levels"])

    def test_invalid_phase_raises_value_error(self, minimal_cfg):
        from models.grayspot_model import GrayspotModel

        with pytest.raises(ValueError):
            GrayspotModel(minimal_cfg, phase=1)

    def test_phase_attribute_set_correctly(self, minimal_cfg):
        from models.grayspot_model import GrayspotModel

        model = GrayspotModel(minimal_cfg, phase=2)
        assert model.phase == 2

    def test_phase2_effb0_head_has_no_mid_dim(self, minimal_cfg):
        """EfficientNet-B0 Phase 2 모델의 head는 mid_dim이 없어야 한다."""
        import torch.nn as nn
        from models.grayspot_model import GrayspotModel

        minimal_cfg["model"]["backbone"] = "efficientnet_b0"
        model = GrayspotModel(minimal_cfg, phase=2)
        linears = [m for m in model.head.net if isinstance(m, nn.Linear)]
        assert len(linears) == 2

    def test_phase2_resnet50_head_has_mid_dim(self, minimal_cfg):
        """ResNet-50 Phase 2 모델의 head는 mid_dim이 있어야 한다(Linear 3개)."""
        import torch.nn as nn
        from models.grayspot_model import GrayspotModel

        minimal_cfg["model"]["backbone"] = "resnet50"
        model = GrayspotModel(minimal_cfg, phase=2)
        linears = [m for m in model.head.net if isinstance(m, nn.Linear)]
        assert len(linears) == 3

    def test_phase2_effb0_output_shape(self, minimal_cfg):
        from models.grayspot_model import GrayspotModel

        minimal_cfg["model"]["backbone"] = "efficientnet_b0"
        model = GrayspotModel(minimal_cfg, phase=2)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, minimal_cfg["data"]["num_levels"])

    def test_phase2_resnet50_output_shape(self, minimal_cfg):
        from models.grayspot_model import GrayspotModel

        minimal_cfg["model"]["backbone"] = "resnet50"
        model = GrayspotModel(minimal_cfg, phase=2)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, minimal_cfg["data"]["num_levels"])
