"""
training/losses.py

Phase별 Loss 함수 선택 및 반환.
Loss function selection and return per phase.

Phase 0: InfoNCELoss (Contrastive Learning)
Phase 2: CrossEntropyLoss (Supervised Classification)
"""

import torch
import torch.nn as nn
from collections import Counter

from training.contrastive_loss import InfoNCELoss


def get_loss(phase: int, cfg: dict,
             train_samples: list = None) -> nn.Module:
    """
    Phase에 맞는 Loss 함수를 반환한다.
    Returns the appropriate loss function for the given phase.

    Args:
        phase:         0 (Contrastive) | 2 (Supervised)
        cfg:           config.yaml dict
        train_samples: Phase 2에서 class_weights 계산 시 필요한 샘플 리스트
                       [(image_path, level), ...] — required for class_weights in Phase 2

    Returns:
        nn.Module loss function
    """
    if phase == 0:
        # Phase 0: InfoNCE Loss
        temperature = cfg["phase0"]["temperature"]  # 0.1
        return InfoNCELoss(temperature=temperature)

    elif phase == 2:
        # Phase 2: CrossEntropyLoss
        # Softmax 포함 — Head에 Softmax 추가 금지 / Includes Softmax — do NOT add Softmax to Head
        class_weights_mode = cfg["phase2"].get("class_weights", "none")

        if class_weights_mode == "balanced" and train_samples is not None:
            # 클래스 가중치 계산 (데이터 불균형 보정) / Compute class weights (imbalance correction)
            num_levels   = cfg["data"]["num_levels"]
            level_counts = Counter([lv for _, lv in train_samples])
            total        = sum(level_counts.values())
            weights      = torch.tensor([
                total / (level_counts.get(lv, 1) * num_levels)
                for lv in range(num_levels)
            ], dtype=torch.float32)
            # 정규화 / Normalize
            weights = weights / weights.sum()
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()

    else:
        raise ValueError(
            f"지원하지 않는 phase / Unsupported phase: {phase}. "
            f"선택 가능 / Available: [0, 2]"
        )
