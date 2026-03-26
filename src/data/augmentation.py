"""
Grayspot — 데이터 증강 / Data Augmentation
data/augmentation.py

Phase 0: Contrastive Learning 전용 강한 증강
Phase 2: Supervised Learning 전용 약한 증강 (Training 전용)

Phase 0: Strong augmentation for Contrastive Learning
Phase 2: Mild augmentation for Supervised Learning (Training only)
"""

import numpy as np
import torch
from torchvision import transforms


def get_contrastive_transforms(cfg: dict):
    """
    Phase 0 Contrastive Learning 전용 Augmentation.
    같은 이미지의 두 augmentation을 Positive Pair로 만들기 위해 강한 변환을 적용한다.

    Phase 0 Contrastive Learning augmentation.
    Applies strong transformations to generate two different views of the same image as a Positive Pair.
    """
    aug  = cfg["phase0"]["augmentation"]
    size = cfg["data"]["image_size"]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size, scale=aug["random_crop_scale"]),   # 랜덤 크롭 / Random crop
        transforms.RandomHorizontalFlip() if aug["horizontal_flip"] else transforms.Lambda(lambda x: x),  # 수평 뒤집기 / Horizontal flip
        transforms.ColorJitter(
            brightness=aug["brightness_jitter"],   # 밝기 변동 / Brightness jitter
            contrast=aug["contrast_jitter"],       # 대비 변동 / Contrast jitter
        ),
        transforms.GaussianBlur(
            kernel_size=aug["gaussian_blur_kernel"][1],  # 블러 커널 크기 / Blur kernel size
            sigma=aug["gaussian_blur_sigma"],            # 블러 시그마 범위 / Blur sigma range
        ),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # [-1, 1] 범위로 정규화 / Normalize to [-1, 1]
    ])


def get_supervised_transforms(cfg: dict, augment: bool = True):
    """
    Phase 2 지도학습 Augmentation.
    augment=True (train 전용), augment=False (val/test 전용).

    Phase 2 Supervised Learning augmentation.
    augment=True for training only, augment=False for val/test.
    """
    size = cfg["data"]["image_size"]

    # 기본 변환 (val/test 공통) / Base transforms (shared for val/test)
    base = [
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ]

    if not augment:
        return transforms.Compose(base)  # val/test: 증강 없음 / No augmentation for val/test

    # Training 전용 증강 추가 / Additional augmentation for training only
    aug = cfg["phase2"]["augmentation"]
    train_transforms = [
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),    # 랜덤 회전 / Random rotation
        transforms.ColorJitter(
            brightness=aug["brightness_jitter"],  # 밝기 변동 / Brightness jitter
            contrast=aug["contrast_jitter"],      # 대비 변동 / Contrast jitter
        ),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ]
    return transforms.Compose(train_transforms)