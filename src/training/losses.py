"""
training/losses.py

Phase별 Loss 함수 선택 및 반환.
Loss function selection and return per phase.

Phase 0: InfoNCELoss (Contrastive Learning)
Phase 2: CrossEntropyLoss (Supervised Classification)
"""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.contrastive_loss import InfoNCELoss


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        gather_target = targets.unsqueeze(1)
        target_log_probs = log_probs.gather(1, gather_target).squeeze(1)
        target_probs = probs.gather(1, gather_target).squeeze(1)

        if self.weight is not None:
            weights = self.weight.to(logits.device).gather(0, targets)
        else:
            weights = 1.0

        loss = -weights * ((1.0 - target_probs) ** self.gamma) * target_log_probs
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


def get_loss(phase: int, cfg: dict, train_samples: list = None) -> nn.Module:
    """
    Phase에 맞는 Loss 함수를 반환한다.
    Returns the appropriate loss function for the given phase.

    Args:
        phase:         0 (Contrastive) | 2 (Supervised)
        cfg:           config.json dict
        train_samples: Phase 2에서 class_weights 계산 시 필요한 샘플 리스트
                       [(image_path, level), ...] — required for class_weights in Phase 2

    Returns:
        nn.Module loss function
    """
    if phase == 0:
        # Phase 0: InfoNCE Loss
        temperature = cfg["phase0"].get("temperature", 0.1)
        return InfoNCELoss(temperature=temperature)

    elif phase == 2:
        # Phase 2: supervised classification loss
        loss_type = cfg["phase2"].get("loss", "cross_entropy").lower()
        class_weights_mode = cfg["phase2"].get("class_weights", "none")
        label_smoothing = float(cfg["phase2"].get("label_smoothing", 0.0))
        focal_gamma = float(cfg["phase2"].get("focal_gamma", 2.0))

        weights = None
        if class_weights_mode == "balanced" and train_samples is not None:
            num_levels = cfg["data"]["num_levels"]
            level_counts = Counter([lv for _, lv in train_samples])
            total = sum(level_counts.values())
            weights = torch.tensor(
                [
                    total / (level_counts.get(lv, 1) * num_levels)
                    for lv in range(num_levels)
                ],
                dtype=torch.float32,
            )
            weights = weights / weights.sum()

        if loss_type == "focal":
            return FocalLoss(gamma=focal_gamma, weight=weights)
        else:
            # Softmax 포함 — Head에 Softmax 추가 금지 / Includes Softmax — do NOT add Softmax to Head
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    else:
        raise ValueError(
            f"지원하지 않는 phase / Unsupported phase: {phase}. "
            f"선택 가능 / Available: [0, 2]"
        )
