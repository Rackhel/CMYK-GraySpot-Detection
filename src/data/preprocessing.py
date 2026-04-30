"""
data/preprocessing.py

Preprocessing functions for Grayspot detection pipeline.
Grayspot 탐지 파이프라인을 위한 전처리 함수들.

Features / 기능:
    - Image resizing and normalization / 이미지 크기 조정 및 정규화
    - Batch preprocessing support / 배치 전처리 지원
"""

import cv2
import numpy as np
from typing import Union, Tuple


def preprocess(image: Union[np.ndarray, str]) -> np.ndarray:
    """
    Basic preprocessing: resize and normalize image.
    기본 전처리: 이미지 크기 조정 및 정규화.
    
    Args:
        image: Input image (numpy array or file path)
        
    Returns:
        Preprocessed image [0, 1] float32 normalized
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image from {image}")
    
    image = cv2.resize(image, (128, 128))
    image = np.clip(image / 255.0, 0, 1)
    return image.astype(np.float32)


def preprocess_image(image: Union[np.ndarray, str], 
                     size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Preprocess image with customizable size.
    사용자 정의 크기로 이미지 전처리.
    
    Args:
        image: Input image (numpy array or file path)
        size : Target size (height, width)
        
    Returns:
        Preprocessed image [0, 1] float32 normalized
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image from {image}")
    
    image = cv2.resize(image, size)
    image = np.clip(image / 255.0, 0, 1)
    return image.astype(np.float32)


def preprocess_batch(images: np.ndarray, 
                     size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Batch preprocessing for multiple images.
    여러 이미지의 배치 전처리.
    
    Args:
        images: Batch of images (N, H, W, C)
        size  : Target size (height, width)
        
    Returns:
        Preprocessed batch [0, 1] float32 normalized
    """
    batch_size = images.shape[0]
    processed = np.zeros((batch_size, size[0], size[1], images.shape[3]), 
                         dtype=np.float32)
    
    for i in range(batch_size):
        processed[i] = preprocess_image(images[i], size)
    
    return processed