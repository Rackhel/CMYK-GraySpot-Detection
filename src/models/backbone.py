"""
models/backbone.py

Backbone 모델 로드 및 feature_dim 반환.
Loads backbone model and returns feature_dim.

지원 backbone / Supported backbones:
  - efficientnet_b0 (feature_dim: 1280)
  - resnet50        (feature_dim: 2048)
"""

import torch.nn as nn
from torchvision import models


def build_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    """
    Backbone 모델을 로드하고 Classifier Head를 제거하여 반환한다.
    Loads backbone model, removes classifier head, and returns it.

    Args:
        backbone_name: "efficientnet_b0" | "resnet50"

    Returns:
        (backbone, feature_dim)
        - backbone:    Head가 제거된 feature extractor / Feature extractor with head removed
        - feature_dim: backbone 출력 차원 / Backbone output dimension
    """
    if backbone_name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights

        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        feature_dim = backbone.classifier[1].in_features  # 1280
        # Classifier Head 제거 → GAP 출력만 사용 / Remove classifier head → use GAP output only
        backbone.classifier = nn.Identity()

    elif backbone_name == "resnet50":
        from torchvision.models import ResNet50_Weights

        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features  # 2048
        # FC Layer 제거 / Remove FC layer
        backbone.fc = nn.Identity()

    else:
        raise ValueError(
            f"지원하지 않는 backbone / Unsupported backbone: {backbone_name}. "
            f"선택 가능 / Available: ['efficientnet_b0', 'resnet50']"
        )

    return backbone, feature_dim
