"""
models/classifier.py

Phase 2 Supervised Classification 전용 Classifier Head.
Classifier Head exclusively for Phase 2 Supervised Classification.

구조 / Structure:
    EfficientNet-B0 특화 (mid_dim=None):
      Backbone 출력 → Linear(hidden_dim) → BN → ReLU → Dropout → Linear(6)
    ResNet-50 특화 (mid_dim=512):
      Backbone 출력 → Linear(mid_dim) → BN → ReLU → Dropout
                    → Linear(hidden_dim) → BN → ReLU → Dropout → Linear(6)

주의 / Note:
    Softmax 없음 — CrossEntropyLoss가 내부적으로 처리한다.
    No Softmax — CrossEntropyLoss handles it internally.
    추론 시에만 F.softmax(logits, dim=1) 를 별도 적용한다.
    Apply F.softmax(logits, dim=1) separately only during inference.
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    Phase 2 Supervised Classification 전용 Classifier Head.
    Phase 2 Supervised Classification exclusive Classifier Head.

    Args:
        in_dim:      Backbone 출력 차원 / Backbone output dimension (e.g. 1280, 2048)
        hidden_dim:  최종 은닉층 차원 / Final hidden layer dimension (default: 256)
        num_classes: 분류 클래스 수 / Number of classes (default: 6, Level 0~5)
        dropout:     Dropout 비율 / Dropout rate (default: 0.3)
        mid_dim:     중간 압축 차원 — ResNet-50 전용, None이면 단일 레이어 구조
                     Intermediate compression dim — ResNet-50 only, None = single-layer head
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 num_classes: int = 6, dropout: float = 0.3,
                 mid_dim: int | None = None):
        super().__init__()

        if mid_dim is not None:
            # ResNet-50 특화: 단계적 압축 (in_dim → mid_dim → hidden_dim → num_classes)
            # ResNet-50 specialized: staged compression
            self.net = nn.Sequential(
                nn.Linear(in_dim, mid_dim),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # EfficientNet-B0 특화: 직접 압축 (in_dim → hidden_dim → num_classes)
            # EfficientNet-B0 specialized: direct compression
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) backbone feature vector

        Returns:
            (B, num_classes) logits (Softmax 적용 전 / before Softmax)
        """
        return self.net(x)
