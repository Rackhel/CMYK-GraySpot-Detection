"""
data/augmentation.py

학습 전용 데이터 증강 함수 모음.
Data augmentation functions — training only.

두 가지 정책 / Two augmentation policies:
    - augment_supervised   : Supervised 학습(Phase 2)용
    - augment_contrastive  : Contrastive Learning(Phase 0, SimCLR)용

주의 / Note:
    - 입력은 float32 [0, 1] 정규화 이미지여야 함 (preprocessing.py 적용 후)
    - Input must be float32 normalized image in [0, 1] (after preprocessing.py)
    - Inference 시에는 호출하지 않는다 / Do NOT call during inference
"""

import random
import numpy as np

import cv2
import numpy as np


def augment_supervised(image: np.ndarray) -> np.ndarray:
    """
    Supervised 학습(Phase 2)용 증강.
    Augmentation for Supervised training (Phase 2).

    적용 변환 / Applied transforms:
        - Random horizontal flip
        - Brightness jitter  (±30/255)
        - Additive noise     (0~10/255)

    Args:
        image: float32 [0, 1] 정규화 이미지 (H, W, 3) / Normalized image

    Returns:
        증강된 float32 [0, 1] 이미지 / Augmented float32 [0, 1] image
    """
    # 수평 뒤집기 / Horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 밝기 조절 / Brightness jitter
    if random.random() > 0.5:
        image = np.clip(image + random.randint(-30, 30) / 255.0, 0, 1)

    # 노이즈 / Additive noise
    if random.random() > 0.5:
        image = np.clip(image + random.randint(0, 10) / 255.0, 0, 1)

    return image


def augment_contrastive(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    """
    Contrastive Learning(Phase 0, SimCLR)용 증강.
    Augmentation for Contrastive Learning (Phase 0, SimCLR).

    적용 변환 / Applied transforms:
        - Random horizontal flip
        - Random crop + resize  (scale 0.6~1.0)
        - Brightness jitter     (±0.2)
        - Contrast jitter       (×0.8~1.2)
        - Gaussian blur         (kernel 3 or 5)

    Args:
        image:      float32 [0, 1] 정규화 이미지 (H, W, 3) / Normalized image
        image_size: crop 후 리사이즈 목표 크기 / Target size after crop

    Returns:
        증강된 float32 [0, 1] 이미지 / Augmented float32 [0, 1] image
    """
    # 수평 뒤집기 / Horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 랜덤 크롭 + 리사이즈 / Random crop + resize
    if random.random() > 0.5:
        h, w   = image.shape[:2]
        scale  = random.uniform(0.6, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0     = random.randint(0, h - ch)
        x0     = random.randint(0, w - cw)
        image  = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (image_size, image_size))

    # 밝기 조절 / Brightness jitter
    if random.random() > 0.5:
        image = np.clip(image + random.uniform(-0.2, 0.2), 0, 1)

    # 대비 조절 / Contrast jitter
    if random.random() > 0.5:
        image = np.clip((image - 0.5) * random.uniform(0.8, 1.2) + 0.5, 0, 1)

    # 가우시안 블러 / Gaussian blur
    if random.random() > 0.5:
        k     = random.choice([3, 5])
        image = cv2.GaussianBlur(
            (image * 255).astype(np.uint8), (k, k), 0
        ).astype(np.float32) / 255.0

    return image
