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
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Named constants — Soft SSOT (성능에만 영향 / performance-only impact)
# config 키가 없는 파라미터는 여기서 명시적으로 정의한다.
# Parameters without config keys are explicitly defined here.
# ---------------------------------------------------------------------------

# Supervised augmentation (Phase 2)
_SUP_FLIP_PROB       = 0.5    # 수평 뒤집기 확률 / Horizontal flip probability
_SUP_BRIGHTNESS_PROB = 0.5    # 밝기 조절 적용 확률 / Brightness jitter apply probability
_SUP_BRIGHTNESS_RANGE= 30     # 밝기 변화량 (±pixels/255) / Brightness delta (±pixels/255)
_SUP_NOISE_PROB      = 0.5    # 노이즈 추가 확률 / Additive noise probability
_SUP_NOISE_RANGE     = 10     # 노이즈 최대값 (pixels/255) / Max noise (pixels/255)

# Contrastive augmentation (Phase 0)
_CON_FLIP_PROB       = 0.5    # 수평 뒤집기 확률
_CON_CROP_PROB       = 0.5    # 랜덤 크롭 적용 확률 / Random crop apply probability
_CON_CROP_SCALE_MIN  = 0.6    # 크롭 최소 스케일 / Crop minimum scale
_CON_CROP_SCALE_MAX  = 1.0    # 크롭 최대 스케일 / Crop maximum scale
_CON_CONTRAST_SCALE_MIN = 0.8 # 대비 스케일 하한 / Contrast scale lower bound
_CON_CONTRAST_SCALE_MAX = 1.2 # 대비 스케일 상한 / Contrast scale upper bound
_CON_BLUR_KERNELS    = [3, 5] # 블러 커널 크기 선택지 / Gaussian blur kernel choices


def augment_supervised(image: np.ndarray, aug_cfg: Optional[dict] = None) -> np.ndarray:
    """
    Supervised 학습(Phase 2)용 증강.
    Augmentation for Supervised training (Phase 2).

    적용 변환 / Applied transforms:
        - Random horizontal flip        (prob: flip_prob)
        - Brightness jitter             (prob: brightness_prob, ±brightness_range/255)
        - Additive noise                (prob: noise_prob, 0~noise_range/255)

    Args:
        image:   float32 [0, 1] 정규화 이미지 (H, W, 3) / Normalized image
        aug_cfg: cfg["phase2"]["augmentation"] dict (선택 / optional)

    Returns:
        증강된 float32 [0, 1] 이미지 / Augmented float32 [0, 1] image
    """
    if aug_cfg is None:
        aug_cfg = {}

    flip_prob        = float(aug_cfg.get("flip_prob",        _SUP_FLIP_PROB))
    brightness_prob  = float(aug_cfg.get("brightness_prob",  _SUP_BRIGHTNESS_PROB))
    brightness_range = int(aug_cfg.get("brightness_range",   _SUP_BRIGHTNESS_RANGE))
    noise_prob       = float(aug_cfg.get("noise_prob",       _SUP_NOISE_PROB))
    noise_range      = int(aug_cfg.get("noise_range",        _SUP_NOISE_RANGE))

    if random.random() < flip_prob:
        image = cv2.flip(image, 1)

    if random.random() < brightness_prob:
        image = np.clip(image + random.randint(-brightness_range, brightness_range) / 255.0, 0, 1)

    if random.random() < noise_prob:
        image = np.clip(image + random.randint(0, noise_range) / 255.0, 0, 1)

    return image


def augment_contrastive(
    image: np.ndarray,
    image_size: int,
    aug_cfg: Optional[dict] = None,
) -> np.ndarray:
    """
    Contrastive Learning(Phase 0, SimCLR)용 증강.
    Augmentation for Contrastive Learning (Phase 0, SimCLR).

    적용 변환 / Applied transforms:
        - Random horizontal flip        (prob: flip_prob)
        - Random crop + resize          (prob: crop_prob, scale crop_scale_min ~ crop_scale_max)
        - Brightness jitter             (prob: blur_prob, ±color_jitter)
        - Contrast jitter               (prob: blur_prob, ×contrast_scale_min ~ contrast_scale_max)
        - Gaussian blur                 (prob: blur_prob, kernel blur_kernels)

    모든 파라미터는 aug_cfg (= cfg["phase0"]["augmentation"]) 에서 읽으며,
    키가 없는 경우 모듈 상수가 fallback으로 사용된다.
    All parameters are read from aug_cfg (= cfg["phase0"]["augmentation"]);
    module constants serve as fallback defaults when keys are absent.

    Args:
        image:      float32 [0, 1] 정규화 이미지 (H, W, 3) / Normalized image
        image_size: crop 후 리사이즈 목표 크기 / Target size after crop
        aug_cfg:    cfg["phase0"]["augmentation"] dict (선택 / optional)

    Returns:
        증강된 float32 [0, 1] 이미지 / Augmented float32 [0, 1] image
    """
    if aug_cfg is None:
        aug_cfg = {}

    color_jitter        = float(aug_cfg.get("color_jitter",       0.2))
    blur_prob           = float(aug_cfg.get("blur_prob",          _CON_FLIP_PROB))
    flip_prob           = float(aug_cfg.get("flip_prob",          _CON_FLIP_PROB))
    crop_prob           = float(aug_cfg.get("crop_prob",          _CON_CROP_PROB))
    crop_scale_min      = float(aug_cfg.get("crop_scale_min",     _CON_CROP_SCALE_MIN))
    crop_scale_max      = float(aug_cfg.get("crop_scale_max",     _CON_CROP_SCALE_MAX))
    contrast_scale_min  = float(aug_cfg.get("contrast_scale_min", _CON_CONTRAST_SCALE_MIN))
    contrast_scale_max  = float(aug_cfg.get("contrast_scale_max", _CON_CONTRAST_SCALE_MAX))
    blur_kernels        = list(aug_cfg.get("blur_kernels",        _CON_BLUR_KERNELS))

    # 수평 뒤집기 / Horizontal flip
    if random.random() < flip_prob:
        image = cv2.flip(image, 1)

    # 랜덤 크롭 + 리사이즈 / Random crop + resize
    if random.random() < crop_prob:
        h, w   = image.shape[:2]
        scale  = random.uniform(crop_scale_min, crop_scale_max)
        ch, cw = int(h * scale), int(w * scale)
        y0     = random.randint(0, h - ch)
        x0     = random.randint(0, w - cw)
        image  = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (image_size, image_size))

    # 밝기 조절 / Brightness jitter (config: color_jitter)
    if random.random() < blur_prob:
        image = np.clip(image + random.uniform(-color_jitter, color_jitter), 0, 1)

    # 대비 조절 / Contrast jitter
    if random.random() < blur_prob:
        image = np.clip(
            (image - 0.5) * random.uniform(contrast_scale_min, contrast_scale_max) + 0.5,
            0, 1,
        )

    # 가우시안 블러 / Gaussian blur (config: blur_prob)
    if random.random() < blur_prob:
        k     = random.choice(blur_kernels)
        image = cv2.GaussianBlur(
            (image * 255).astype(np.uint8), (k, k), 0
        ).astype(np.float32) / 255.0

    return image
