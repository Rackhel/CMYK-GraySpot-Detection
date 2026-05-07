"""
data/preprocessing.py

CMYK 패치 이미지 표준 전처리 — SSOT (Section 6.5~6.9).
Standard preprocessing for CMYK patch images — SSOT (Section 6.5~6.9).

학습·추론 공통 적용 / Applied to both training and inference:
    - Resize → image_size × image_size
    - Normalize → [0, 1]

주의 / Note:
    - Augmentation은 별도 augmentation.py에서 적용 (학습 전용)
    - Augmentation is applied separately in augmentation.py (training only)
"""

import cv2
import numpy as np


def preprocess(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    """
    표준 전처리를 적용한다.
    Applies standard preprocessing.

    Args:
        image:      BGR uint8 이미지 (H, W, 3) / BGR uint8 image
        image_size: 리사이즈 목표 크기 / Target resize size

    Returns:
        float32 정규화 이미지 (image_size, image_size, 3), 범위 [0, 1]
        float32 normalized image, range [0, 1]
    """
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype(np.float32) / 255.0
    return image
