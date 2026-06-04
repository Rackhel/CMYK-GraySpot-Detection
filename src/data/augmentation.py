"""
data/augmentation.py

학습 전용 데이터 증강 함수 모음.
Data augmentation functions — training only.

두 가지 정책 / Two augmentation policies:
    - augment_supervised   : Supervised 학습(Phase 2)용
    - augment_contrastive  : Contrastive Learning(Phase 0, SimCLR)용

추가 유틸리티 / Extra utilities:
    - mixup_batch   : MixUp (배치 레벨 / batch-level)
    - cutmix_batch  : CutMix (배치 레벨 / batch-level)

주의 / Note:
    - 입력은 float32 [0, 1] 정규화 이미지 (preprocessing.py + BGR→RGB 완료 후)
    - Input must be float32 [0, 1] after preprocessing (BGR→RGB already applied)
    - Inference 시에는 호출하지 않는다 / Do NOT call during inference
"""

import random
from typing import Optional

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────
# Named constants — Soft SSOT
# ──────────────────────────────────────────────────────────────

# Supervised augmentation (Phase 2)
_SUP_FLIP_PROB = 0.5
_SUP_VFLIP_PROB = 0.0  # 수직 뒤집기 기본 꺼짐 / vertical flip off by default
_SUP_BRIGHTNESS_PROB = 0.5
_SUP_BRIGHTNESS_RANGE = 30
_SUP_NOISE_PROB = 0.5
_SUP_NOISE_RANGE = 10
_SUP_ROTATION_PROB = 0.0
_SUP_ROTATION_MAX = 15  # degrees

# Contrastive augmentation (Phase 0)
_CON_FLIP_PROB = 0.5
_CON_CROP_PROB = 0.5
_CON_CROP_SCALE_MIN = 0.6
_CON_CROP_SCALE_MAX = 1.0
_CON_CONTRAST_SCALE_MIN = 0.8
_CON_CONTRAST_SCALE_MAX = 1.2
_CON_BLUR_KERNELS = [3, 5]

# MixUp / CutMix defaults
_MIXUP_ALPHA = 0.2
_CUTMIX_ALPHA = 1.0


# ──────────────────────────────────────────────────────────────
# Phase 2 — Supervised augmentation
# ──────────────────────────────────────────────────────────────


def augment_supervised(image: np.ndarray, aug_cfg: Optional[dict] = None) -> np.ndarray:
    """
    Supervised 학습(Phase 2)용 증강.
    Augmentation for Supervised training (Phase 2).

    적용 변환 / Applied transforms:
        - Random horizontal flip      (flip_prob)
        - Random vertical flip        (vflip_prob, default 0)
        - Random rotation             (rotation_prob, ±rotation_max degrees)
        - Brightness jitter           (brightness_prob, ±brightness_range/255)
        - Additive noise              (noise_prob, 0~noise_range/255)

    policy="strong" 설정 시 확률이 더 높아집니다 (Y/K 채널 권장).
    Set policy="strong" for higher probability augmentation (recommended for Y/K channels).
    """
    if aug_cfg is None:
        aug_cfg = {}

    policy = aug_cfg.get("policy", "light")
    prob_scale = 1.3 if policy == "strong" else 1.0

    def _p(key, default):
        return min(1.0, float(aug_cfg.get(key, default)) * prob_scale)

    flip_prob = _p("flip_prob", _SUP_FLIP_PROB)
    vflip_prob = _p("vflip_prob", _SUP_VFLIP_PROB)
    brightness_prob = _p("brightness_prob", _SUP_BRIGHTNESS_PROB)
    brightness_range = int(aug_cfg.get("brightness_range", _SUP_BRIGHTNESS_RANGE))
    noise_prob = _p("noise_prob", _SUP_NOISE_PROB)
    noise_range = int(aug_cfg.get("noise_range", _SUP_NOISE_RANGE))
    rotation_prob = _p("rotation_prob", _SUP_ROTATION_PROB)
    rotation_max = float(aug_cfg.get("rotation_max", _SUP_ROTATION_MAX))

    # If all augmentation probabilities are zero, return the image unchanged
    # (preserve dtype and avoid accidental in-place changes).
    if all(
        float(p) == 0.0
        for p in (
            flip_prob,
            vflip_prob,
            brightness_prob,
            noise_prob,
            rotation_prob,
        )
    ):
        return image.copy().astype(np.float32)

    if random.random() < flip_prob:
        image = cv2.flip(image, 1)

    if random.random() < vflip_prob:
        image = cv2.flip(image, 0)

    if random.random() < rotation_prob:
        h, w = image.shape[:2]
        angle = random.uniform(-rotation_max, rotation_max)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    if random.random() < brightness_prob:
        image = np.clip(
            image + random.randint(-brightness_range, brightness_range) / 255.0, 0, 1
        )

    if random.random() < noise_prob:
        image = np.clip(image + random.randint(0, noise_range) / 255.0, 0, 1)

    return image


# ──────────────────────────────────────────────────────────────
# Phase 0 — Contrastive augmentation
# ──────────────────────────────────────────────────────────────


def augment_contrastive(
    image: np.ndarray,
    image_size: int,
    aug_cfg: Optional[dict] = None,
) -> np.ndarray:
    """
    Contrastive Learning(Phase 0, SimCLR)용 증강.
    Augmentation for Contrastive Learning (Phase 0, SimCLR).
    """
    if aug_cfg is None:
        aug_cfg = {}

    color_jitter = float(aug_cfg.get("color_jitter", 0.2))
    blur_prob = float(aug_cfg.get("blur_prob", _CON_FLIP_PROB))
    flip_prob = float(aug_cfg.get("flip_prob", _CON_FLIP_PROB))
    crop_prob = float(aug_cfg.get("crop_prob", _CON_CROP_PROB))
    crop_scale_min = float(aug_cfg.get("crop_scale_min", _CON_CROP_SCALE_MIN))
    crop_scale_max = float(aug_cfg.get("crop_scale_max", _CON_CROP_SCALE_MAX))
    contrast_scale_min = float(
        aug_cfg.get("contrast_scale_min", _CON_CONTRAST_SCALE_MIN)
    )
    contrast_scale_max = float(
        aug_cfg.get("contrast_scale_max", _CON_CONTRAST_SCALE_MAX)
    )
    blur_kernels = list(aug_cfg.get("blur_kernels", _CON_BLUR_KERNELS))

    if random.random() < flip_prob:
        image = cv2.flip(image, 1)

    if random.random() < crop_prob:
        h, w = image.shape[:2]
        scale = random.uniform(crop_scale_min, crop_scale_max)
        ch, cw = int(h * scale), int(w * scale)
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        image = cv2.resize(image[y0 : y0 + ch, x0 : x0 + cw], (image_size, image_size))

    if random.random() < blur_prob:
        image = np.clip(image + random.uniform(-color_jitter, color_jitter), 0, 1)

    if random.random() < blur_prob:
        image = np.clip(
            (image - 0.5) * random.uniform(contrast_scale_min, contrast_scale_max)
            + 0.5,
            0,
            1,
        )

    if random.random() < blur_prob:
        k = random.choice(blur_kernels)
        image = (
            cv2.GaussianBlur((image * 255).astype(np.uint8), (k, k), 0).astype(
                np.float32
            )
            / 255.0
        )

    return image


# ──────────────────────────────────────────────────────────────
# Batch-level MixUp / CutMix (DataLoader 이후에 적용)
# Apply after DataLoader — operate on batched tensors
# ──────────────────────────────────────────────────────────────


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = _MIXUP_ALPHA,
    num_classes: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp augmentation for a batch of tensors.

    두 이미지를 lambda 비율로 혼합, 레이블도 동일 비율로 혼합 (soft label).
    Blends two images at ratio lambda; labels are also blended (soft labels).

    Args:
        x:           (B, C, H, W) float tensor
        y:           (B,) int tensor — class indices
        alpha:       Beta distribution parameter (higher → more mixing)
        num_classes: number of classes for one-hot encoding

    Returns:
        mixed_x: (B, C, H, W)
        mixed_y: (B, num_classes) soft labels
    """
    if alpha <= 0:
        y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        return x, y_onehot

    import numpy as _np

    lam = float(_np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[idx]

    y_onehot = torch.zeros(batch_size, num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]

    return mixed_x, mixed_y


def cutmix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = _CUTMIX_ALPHA,
    num_classes: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix augmentation for a batch of tensors.

    한 이미지의 직사각형 영역을 다른 이미지 패치로 교체.
    Replaces a rectangular region of one image with a patch from another.

    Args:
        x:           (B, C, H, W) float tensor
        y:           (B,) int tensor
        alpha:       Beta distribution parameter
        num_classes: number of classes

    Returns:
        mixed_x: (B, C, H, W)
        mixed_y: (B, num_classes) soft labels
    """
    import numpy as _np

    if alpha <= 0:
        y_onehot = torch.zeros(x.size(0), num_classes, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        return x, y_onehot

    lam = float(_np.random.beta(alpha, alpha))
    batch_size, _, H, W = x.shape
    idx = torch.randperm(batch_size, device=x.device)

    # 잘라낼 박스 계산 / Compute cut box
    cut_ratio = _np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = _np.random.randint(W)
    cy = _np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]

    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    y_onehot = torch.zeros(batch_size, num_classes, device=x.device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    mixed_y = lam_actual * y_onehot + (1 - lam_actual) * y_onehot[idx]

    return mixed_x, mixed_y
