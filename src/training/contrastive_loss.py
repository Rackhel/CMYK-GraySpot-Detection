"""
Grayspot — Contrastive Loss (SimCLR / InfoNCE)
training/contrastive_loss.py

Phase 0 Contrastive Learning 전용 Loss 함수.
같은 이미지의 두 augmentation(view1, view2)을 Positive Pair로 학습한다.

Loss function exclusively for Phase 0 Contrastive Learning.
Trains using two augmented views (view1, view2) of the same image as a Positive Pair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    SimCLR 기반 InfoNCE Loss.
    같은 이미지의 두 augmentation(view1, view2)을 Positive Pair로 학습한다.
    배치 크기가 클수록 더 많은 Negative Pair를 확보하여 학습 효과가 높아진다.

    SimCLR-based InfoNCE Loss.
    Trains using two augmented views (view1, view2) of the same image as a Positive Pair.
    Larger batch sizes provide more Negative Pairs, improving training effectiveness.

    Args:
        temperature: τ — 유사도 분포의 sharpness 제어 / Controls sharpness of similarity distribution (default: 0.1)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) — view1 projection vectors
            z2: (B, D) — view2 projection vectors

        Returns:
            scalar loss
        """
        B      = z1.size(0)
        device = z1.device

        # L2 정규화 — 코사인 유사도 계산을 위해 단위 벡터로 변환
        # L2 normalization — convert to unit vectors for cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # (2B, D) 결합 — view1과 view2를 하나의 배치로 합침
        # Concatenate (2B, D) — merge view1 and view2 into a single batch
        z = torch.cat([z1, z2], dim=0)

        # 유사도 행렬 (2B, 2B) — temperature로 스케일링
        # Similarity matrix (2B, 2B) — scaled by temperature
        sim = torch.matmul(z, z.T) / self.temperature

        # 자기 자신과의 유사도 제거 (대각선 마스킹)
        # Mask out self-similarity (diagonal masking)
        mask = torch.eye(2*B, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, -1e9)

        # Positive pair 인덱스 정의: i ↔ i+B (view1[i]의 Positive는 view2[i])
        # Define positive pair indices: i ↔ i+B (Positive of view1[i] is view2[i])
        labels = torch.cat([
            torch.arange(B, 2*B, device=device),  # view1[i]의 Positive: view2[i] / Positive of view1[i]: view2[i]
            torch.arange(0, B,   device=device),  # view2[i]의 Positive: view1[i] / Positive of view2[i]: view1[i]
        ])

        # CrossEntropyLoss로 Positive pair를 최대화, Negative pair를 최소화
        # Maximize positive pair similarity, minimize negative pair similarity via CrossEntropyLoss
        loss = F.cross_entropy(sim, labels)
        return loss