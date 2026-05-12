"""
data/__init__.py

데이터 패키지 / Data package.

데이터셋, 전처리, 증강 모듈을 내보낸다.
Exports dataset, preprocessing, and augmentation modules.

주요 클래스 / Key classes:
    CMYKDataset         : Phase 2 Supervised 학습용 폴더 기반 Dataset
    ContrastiveDataset  : Phase 0 Contrastive Learning Positive Pair Dataset

주요 함수 / Key functions:
    preprocess          : BGR 이미지 → float32 [0, 1] 정규화
    augment_supervised  : Phase 2용 증강 (flip, brightness, noise)
    augment_contrastive : Phase 0용 증강 (flip, crop, contrast, blur)

사용법 / Usage:
    from data import CMYKDataset, ContrastiveDataset, preprocess
"""

from .dataset import CMYKDataset, ContrastiveDataset
from .preprocessing import preprocess
from .augmentation import augment_supervised, augment_contrastive

__all__ = [
    "CMYKDataset",
    "ContrastiveDataset",
    "preprocess",
    "augment_supervised",
    "augment_contrastive",
]
