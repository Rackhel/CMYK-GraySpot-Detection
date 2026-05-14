"""
models/classifier.py

Phase 2 Supervised Classification 전용 Classifier Head.
Classifier Head exclusively for Phase 2 Supervised Classification.

구조 / Structure:
    Backbone 출력 → Linear(256) → BN → ReLU → Dropout → Linear(6)
    Backbone output → Linear(256) → BN → ReLU → Dropout → Linear(6)

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
        hidden_dim:  은닉층 차원 / Hidden layer dimension (default: 256)
        num_classes: 분류 클래스 수 / Number of classes (default: 6, Level 0~5)
        dropout:     Dropout 비율 / Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 배치 정규화 / Batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 과적합 방지 / Prevent overfitting
            nn.Linear(hidden_dim, num_classes),
            # Softmax 없음 — CrossEntropyLoss가 내부적으로 포함
            # No Softmax — included internally in CrossEntropyLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) backbone feature vector

        Returns:
            (B, num_classes) logits (Softmax 적용 전 / before Softmax)
        """
        return self.net(x)
