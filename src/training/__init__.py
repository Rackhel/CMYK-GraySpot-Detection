"""
training/__init__.py

학습 패키지 / Training package.

Phase 0 (Contrastive) 및 Phase 2 (Supervised) 학습 구성 요소를 내보낸다.
Exports Phase 0 (Contrastive) and Phase 2 (Supervised) training components.

주요 클래스 / Key classes:
    Phase0Trainer : SimCLR InfoNCE Loss 기반 Contrastive Learning 학습기
    Phase2Trainer : CrossEntropyLoss 기반 Supervised Classification 학습기
    InfoNCELoss   : SimCLR 기반 Contrastive Loss

주요 함수 / Key functions:
    get_loss      : Phase에 맞는 Loss 함수 반환 팩토리
    backbone_tag  : backbone 이름 → 파일명 약어 변환

사용법 / Usage:
    from training import Phase0Trainer, Phase2Trainer, backbone_tag
    from training import InfoNCELoss, get_loss
"""

from .trainer import Phase0Trainer, Phase2Trainer, backbone_tag
from .contrastive_loss import InfoNCELoss
from .losses import get_loss

__all__ = [
    "Phase0Trainer",
    "Phase2Trainer",
    "backbone_tag",
    "InfoNCELoss",
    "get_loss",
]
