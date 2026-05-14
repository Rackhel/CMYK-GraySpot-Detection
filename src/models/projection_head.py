"""
models/projection_head.py

Phase 0 Contrastive Learning 전용 Projection Head.
Projection Head exclusively for Phase 0 Contrastive Learning.

구조 / Structure:
    Backbone 출력 → Linear(256) → BN → ReLU → Linear(128)
    Backbone output → Linear(256) → BN → ReLU → Linear(128)

주의 / Note:
    Phase 2 진입 시 이 Head 전체를 제거하고 ClassifierHead로 교체한다.
    Remove this entire Head and replace with ClassifierHead upon entering Phase 2.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Phase 0 Contrastive Learning 전용 Projection Head.
    Phase 0 Contrastive Learning exclusive Projection Head.

    Args:
        in_dim:     Backbone 출력 차원 / Backbone output dimension (e.g. 1280, 2048)
        hidden_dim: 은닉층 차원 / Hidden layer dimension (default: 256)
        out_dim:    출력 차원 / Output dimension — projection vector (default: 128)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 배치 정규화 / Batch normalization
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            # Softmax 없음 — InfoNCE Loss에서 L2 정규화 후 처리
            # No Softmax — handled by L2 normalization in InfoNCE Loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) backbone feature vector

        Returns:
            (B, out_dim) projection vector (정규화 전 / before normalization)
        """
        return self.net(x)
