"""
training/contrastive_loss.py

SimCLR 기반 InfoNCE Loss.
SimCLR-based InfoNCE Loss.

핵심 원리 / Core principle:
    - Positive Pair (같은 이미지의 두 augmentation) → 가깝게 / pull together
    - Negative Pair (다른 이미지)                   → 멀게   / push apart

주의 / Note:
    - L2 정규화를 반드시 적용해야 함 — 누락 시 유사도 행렬 값이 폭발하여 loss가 NaN이 됨
    - L2 normalization must be applied — omitting causes similarity matrix explosion → NaN loss
    - 배치 크기가 클수록 Negative Pair가 많아져 더 discriminative한 표현 학습 가능
    - Larger batch size = more Negative Pairs = more discriminative representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    SimCLR 기반 InfoNCE Loss.
    SimCLR-based InfoNCE Loss.

    Args:
        temperature: τ — 유사도 분포의 sharpness 제어 / Controls sharpness
                     낮을수록 sharp, 높을수록 smooth / Lower = sharper, higher = smoother
                     권장값 / Recommended: 0.1
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) view1 projection vectors
            z2: (B, D) view2 projection vectors

        Returns:
            scalar loss
        """
        B      = z1.size(0)
        device = z1.device

        # L2 정규화 — 코사인 유사도를 위해 단위 벡터로 변환
        # L2 normalization — convert to unit vectors for cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # (2B, D) 결합 / Concatenate (2B, D)
        z = torch.cat([z1, z2], dim=0)

        # 유사도 행렬 (2B, 2B) — temperature 스케일링 / Similarity matrix scaled by temperature
        sim = torch.matmul(z, z.T) / self.temperature

        # 자기 자신과의 유사도 마스킹 / Mask self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, -1e9)

        # Positive pair 인덱스: view1[i] ↔ view2[i] (i+B)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=device),
            torch.arange(0, B,     device=device),
        ])

        # CrossEntropyLoss — Positive pair 유사도 최대화 / Maximize positive pair similarity
        return F.cross_entropy(sim, labels)
