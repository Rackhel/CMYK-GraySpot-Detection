"""
data/normalize.py

ImageNet 정규화 상수 단일 출처 / Single source for ImageNet normalization constant.

SSOT-NM01: 학습·추론·평가 전 단계에서 이 모듈의 _IMAGENET_NORMALIZE 를 import하여 사용한다.
SSOT-NM01: All stages (training, inference, evaluation) must import _IMAGENET_NORMALIZE from here.

mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] — ImageNet 채널별 통계값.
SSOT 근거: ADR_ImageNet_Normalization.md
"""

from torchvision import transforms as T

_IMAGENET_NORMALIZE = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
