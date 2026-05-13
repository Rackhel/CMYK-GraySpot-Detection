"""
data/augmentation.py

학습 과정에서 사용하는 이미지 증강 함수 모음.
Image augmentation utilities used during training.

지원하는 증강 방식 / Available augmentation pipelines:
    - augment_supervised   : 지도학습용 증강
    - augment_contrastive  : Contrastive Learning(SimCLR)용 증강

주의사항 / Important:
    - 입력 이미지는 float32 형식의 [0, 1] 범위여야 함
    - preprocessing 이후에 사용하는 것을 가정함
    - 추론(inference) 단계에서는 사용하지 않음
"""

import random

import cv2
import numpy as np


def augment_supervised(image: np.ndarray) -> np.ndarray:
    """
    지도학습 단계에서 사용하는 증강 함수.
    Augmentation pipeline for supervised learning.

    적용 가능한 변환 / Possible transforms:
        - 좌우 반전
        - 밝기 변화
        - 약한 랜덤 노이즈 추가

    Args:
        image: float32 정규화 이미지 (H, W, 3)

    Returns:
        증강이 적용된 float32 이미지
    """

    # 랜덤 좌우 반전 / Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 밝기 랜덤 조절 / Random brightness adjustment
    if random.random() > 0.5:
        image = np.clip(image + random.randint(-30, 30) / 255.0, 0, 1)

    # 작은 노이즈 추가 / Add light random noise
    if random.random() > 0.5:
        image = np.clip(image + random.randint(0, 10) / 255.0, 0, 1)

    return image


def augment_contrastive(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    """
    Contrastive Learning(SimCLR)용 증강 함수.
    Augmentation pipeline for contrastive learning.

    적용 가능한 변환 / Possible transforms:
        - 좌우 반전
        - 랜덤 크롭 후 리사이즈
        - 밝기 변화
        - 대비 조절
        - Gaussian Blur

    Args:
        image: float32 정규화 이미지
        image_size: 출력 이미지 크기

    Returns:
        증강된 float32 이미지
    """

    # 좌우 반전 적용 / Apply horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 랜덤 크롭 + 리사이즈 / Random crop and resize
    if random.random() > 0.5:
        h, w = image.shape[:2]

        scale = random.uniform(0.6, 1.0)
        ch, cw = int(h * scale), int(w * scale)

        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)

        cropped = image[y0 : y0 + ch, x0 : x0 + cw]
        image = cv2.resize(cropped, (image_size, image_size))

    # 밝기 랜덤 변경 / Random brightness shift
    if random.random() > 0.5:
        image = np.clip(image + random.uniform(-0.2, 0.2), 0, 1)

    # 대비 랜덤 조절 / Random contrast scaling
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        image = np.clip((image - 0.5) * factor + 0.5, 0, 1)

    # Gaussian Blur 적용 / Apply gaussian blur
    if random.random() > 0.5:
        kernel = random.choice([3, 5])

        image = (
            cv2.GaussianBlur(
                (image * 255).astype(np.uint8), (kernel, kernel), 0
            ).astype(np.float32)
            / 255.0
        )

    return image
