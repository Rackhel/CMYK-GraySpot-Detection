"""
data/preprocessing.py

CMYK 패치 이미지 전처리 함수.
Preprocessing utilities for CMYK patch images.

학습 및 추론 단계에서 공통으로 사용된다.
Used for both training and inference stages.

처리 과정 / Processing steps:
    - 이미지 리사이즈
    - float32 변환
    - [0, 1] 범위 정규화

참고 / Note:
    - 데이터 증강은 augmentation.py에서 별도로 처리됨
    - augmentation is handled separately during training
"""

import cv2
import numpy as np


def preprocess(image: np.ndarray, image_size: int = 128) -> np.ndarray:
    """
    입력 이미지에 기본 전처리를 수행한다.
    Apply basic preprocessing to input image.

    Args:
        image:
            BGR 형식의 uint8 이미지
            BGR uint8 image

        image_size:
            출력 이미지 크기
            target resize dimension

    Returns:
        float32 타입의 정규화된 이미지
        normalized float32 image in range [0, 1]
    """

    # 입력 이미지를 고정 크기로 변환
    # Resize image to target resolution
    image = cv2.resize(image, (image_size, image_size))

    # float32 변환 후 정규화
    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0

    return image