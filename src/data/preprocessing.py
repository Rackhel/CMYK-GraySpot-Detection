"""
Grayspot — 전처리 파이프라인 (SSOT)
data/preprocessing.py

Section 6.5~6.9 전처리 표준 구현.
Training / Inference 동일 전처리 적용. Augmentation만 Training 전용.

Implements the preprocessing standard from Section 6.5~6.9.
Same preprocessing applied for both Training and Inference. Augmentation is Training-only.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


CHANNELS = ["Y", "M", "C", "K"]


# ──────────────────────────────────────────────
# 6.5.1  RGB → CMYK 변환 / RGB to CMYK Conversion
# ──────────────────────────────────────────────
def rgb_to_cmyk(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """
    RGB 이미지를 CMYK 4채널로 변환한다.
    Converts an RGB image into 4 CMYK channels.

    Args:
        rgb: (H, W, 3) uint8 이미지 / uint8 image

    Returns:
        {"Y": ..., "M": ..., "C": ..., "K": ...}
        각각 (H, W) float32 [0,1] / each (H, W) float32 [0,1]
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    R, G, B = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    K = 1.0 - np.maximum(np.maximum(R, G), B)
    denom = 1.0 - K
    denom_safe = np.where(denom < 1e-6, 1e-6, denom)  # 0 나눗셈 방지 / Prevent division by zero

    C = (1.0 - R - K) / denom_safe
    M = (1.0 - G - K) / denom_safe
    Y = (1.0 - B - K) / denom_safe

    return {
        "Y": np.clip(Y, 0, 1),
        "M": np.clip(M, 0, 1),
        "C": np.clip(C, 0, 1),
        "K": np.clip(K, 0, 1),
    }


# ──────────────────────────────────────────────
# 6.5.3  ROI 분할 / ROI Extraction
# ──────────────────────────────────────────────
def extract_roi_fixed(
    cmyk: dict[str, np.ndarray],
    coords: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """
    고정 좌표(x, y, w, h)로 CMYK 채널별 ROI를 분리한다.
    Extracts per-channel ROI using fixed coordinates (x, y, w, h).

    Args:
        cmyk:   rgb_to_cmyk() 결과 / Result of rgb_to_cmyk()
        coords: {"Y": [x,y,w,h], "M": [...], ...}

    Returns:
        {"Y": roi_array, ...}
    """
    rois = {}
    for ch in CHANNELS:
        x, y, w, h = coords[ch]
        rois[ch] = cmyk[ch][y:y+h, x:x+w].copy()
    return rois


def extract_roi_auto(
    cmyk: dict[str, np.ndarray],
    min_confidence: float = 0.7,
    fallback_coords: Optional[dict] = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """
    수직 프로파일 분석으로 CMYK 스트립 경계를 자동 검출한다.
    신뢰도 미달 시 fixed fallback을 사용한다.
    Automatically detects CMYK strip boundaries via vertical profile analysis.
    Falls back to fixed coordinates if confidence is below threshold.

    스트립 배치 순서 (상→하) / Strip layout order (top→bottom): Y → M → C → K

    Returns:
        (rois_dict, used_fallback)
    """
    # K 채널 수직 프로파일로 경계 검출 (대비가 명확)
    # Use K channel vertical profile for boundary detection (highest contrast)
    h, w = cmyk["K"].shape

    # 각 채널의 수직 평균을 합산하여 경계 에너지 계산
    # Sum vertical means across all channels to compute boundary energy
    profile = np.zeros(h, dtype=np.float32)
    for ch in CHANNELS:
        profile += np.mean(cmyk[ch], axis=1)
    profile /= len(CHANNELS)

    # 1차 미분으로 변화 급격한 지점 검출
    # Detect sharp transitions via first-order derivative
    diff = np.abs(np.diff(profile))
    smoothed = np.convolve(diff, np.ones(5)/5, mode="same")

    # 상위 3개 경계점 선택 (4구역 → 3 경계)
    # Select top 3 boundary points (4 regions → 3 boundaries)
    threshold = np.percentile(smoothed, 90)
    candidates = np.where(smoothed > threshold)[0]

    boundaries = []
    if len(candidates) >= 3:
        # 클러스터링: 근접한 후보들을 묶어 대표점 선택
        # Clustering: group nearby candidates and pick representative points
        groups, group = [[candidates[0]]], [candidates[0]]
        for c in candidates[1:]:
            if c - group[-1] < 10:
                group.append(c)
            else:
                groups.append(group)
                group = [c]
        groups.append(group)
        boundaries = sorted([int(np.mean(g)) for g in groups])[:3]

    # 신뢰도 계산: 경계가 3개이고 균등한지 확인
    # Confidence check: verify 3 boundaries exist and are evenly spaced
    confidence = 0.0
    if len(boundaries) == 3:
        gaps = [boundaries[0], boundaries[1]-boundaries[0],
                boundaries[2]-boundaries[1], h-boundaries[2]]
        min_gap, max_gap = min(gaps), max(gaps)
        confidence = min_gap / max_gap if max_gap > 0 else 0.0

    if confidence < min_confidence:
        if fallback_coords:
            rois = extract_roi_fixed(cmyk, fallback_coords)
            return rois, True
        # fallback 없으면 균등 4분할 / If no fallback, divide into 4 equal parts
        step = h // 4
        boundaries = [step, step*2, step*3]

    # ROI 분리 / Split into individual ROIs
    b0, b1, b2 = boundaries[0], boundaries[1], boundaries[2]
    rois = {
        "Y": cmyk["Y"][0:b0,  :].copy(),
        "M": cmyk["M"][b0:b1, :].copy(),
        "C": cmyk["C"][b1:b2, :].copy(),
        "K": cmyk["K"][b2:h,  :].copy(),
    }
    return rois, False


# ──────────────────────────────────────────────
# 6.6  배경 톤 보정 / Background Normalization
# ──────────────────────────────────────────────
def background_normalization(roi: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """
    큰 커널 GaussianBlur로 저주파(배경) 성분을 추정하고 제거한다.
    잔차(residual) 이미지에서 Grayspot 패턴을 부각시킨다.
    Estimates and removes low-frequency (background) components via large-kernel GaussianBlur.
    The resulting residual image highlights Grayspot patterns.

    Args:
        roi:         (H, W) float32 [0,1]
        kernel_size: 배경 추정 커널 크기 (홀수) / Background estimation kernel size (must be odd)

    Returns:
        (H, W) float32 잔차 이미지 / residual image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # 홀수 보정 / Ensure odd kernel size

    roi_u8 = (roi * 255).astype(np.uint8)
    background = cv2.GaussianBlur(roi_u8, (kernel_size, kernel_size), 0)
    residual = roi_u8.astype(np.int16) - background.astype(np.int16)

    # 잔차를 [0,1]로 정규화 (중심 0.5) / Normalize residual to [0,1] (centered at 0.5)
    residual_f = (residual.astype(np.float32) + 255) / 510.0
    return np.clip(residual_f, 0, 1)


# ──────────────────────────────────────────────
# 6.7  색상별 독립 정규화 / Per-channel Independent Normalization
# ──────────────────────────────────────────────
def normalize_channel(
    roi: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    색상별 독립 정규화.
    Applies per-channel independent normalization.

    Args:
        roi:  (H, W) float32 [0,1]
        mean: 채널별 평균 / Per-channel mean
        std:  채널별 표준편차 / Per-channel standard deviation

    Returns:
        (H, W) float32 정규화된 이미지 / Normalized image
    """
    return (roi - mean) / (std + 1e-8)  # 0 나눗셈 방지 / Prevent division by zero


# ──────────────────────────────────────────────
# 6.8  Grayspot 강조 / Feature Enhancement
# ──────────────────────────────────────────────
def local_std_map(roi: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    국소 표준편차 맵 — 텍스처 불균일 강조.
    Local standard deviation map — highlights texture non-uniformity.
    """
    roi_u8 = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    mean_sq  = cv2.filter2D(roi_u8.astype(np.float32)**2, -1, kernel)
    mean_    = cv2.filter2D(roi_u8.astype(np.float32),    -1, kernel)
    variance = np.maximum(mean_sq - mean_**2, 0)
    std_map  = np.sqrt(variance)
    return std_map / (std_map.max() + 1e-8)  # 최대값으로 정규화 / Normalize by max value


def bandpass_filter(roi: np.ndarray, low_k: int = 3, high_k: int = 15) -> np.ndarray:
    """
    대역 통과 필터 — 특정 크기 대역의 불균일만 강조.
    Bandpass filter — highlights non-uniformity within a specific spatial frequency band.
    """
    roi_f = roi.astype(np.float32)
    low  = cv2.GaussianBlur(roi_f, (low_k*2+1,  low_k*2+1),  0)
    high = cv2.GaussianBlur(roi_f, (high_k*2+1, high_k*2+1), 0)
    bp = low - high  # 저주파 - 고주파 = 중간 대역 / Low-freq minus high-freq = mid-band
    bp = (bp - bp.min()) / (bp.max() - bp.min() + 1e-8)
    return bp.astype(np.float32)


def directional_filter(roi: np.ndarray) -> np.ndarray:
    """
    방향성 필터 — 수평/수직 Streak 강조 (Cyan 전용).
    Directional filter — highlights horizontal/vertical streaks (Cyan channel only).
    """
    roi_u8 = (np.clip(roi, 0, 1) * 255).astype(np.uint8)
    # Sobel 수평 + 수직 / Sobel horizontal + vertical
    sobel_h   = cv2.Sobel(roi_u8, cv2.CV_32F, 1, 0, ksize=3)
    sobel_v   = cv2.Sobel(roi_u8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    return magnitude / (magnitude.max() + 1e-8)


def feature_enhancement(
    roi: np.ndarray,
    channel: str,
    cfg: dict,
) -> np.ndarray:
    """
    Grayspot 강조 특징 이미지를 생성한다.
    Generates a Grayspot-enhanced feature image.

    Returns:
        (H, W) float32 강조 특징 맵 / Enhanced feature map
    """
    p       = cfg["preprocessing"]
    std_map = local_std_map(roi, p["local_std_kernel"])
    bp      = bandpass_filter(roi, p["bandpass_low"], p["bandpass_high"])

    if channel == "C" and p.get("directional_filter", True):
        # Cyan은 방향성 streak가 많아 방향성 필터 추가 적용
        # Cyan has many directional streaks — apply directional filter as well
        df      = directional_filter(roi)
        feature = (std_map + bp + df) / 3.0
    else:
        feature = (std_map + bp) / 2.0

    return feature.astype(np.float32)


# ──────────────────────────────────────────────
# 6.8.4  단일채널 → 3채널 / Single-channel to 3-channel
#         Pretrained Backbone 입력 호환 / Pretrained backbone input compatibility
# ──────────────────────────────────────────────
def to_3channel(img: np.ndarray) -> np.ndarray:
    """
    (H, W) 단일채널 → (H, W, 3) 3채널 복제.
    Replicates a single-channel image to 3 channels (H, W) → (H, W, 3).
    Pretrained EfficientNet/ResNet 입력 호환 / Compatible with pretrained EfficientNet/ResNet input.
    """
    return np.stack([img, img, img], axis=-1)


# ──────────────────────────────────────────────
# 전체 전처리 파이프라인 (SSOT)
# Full Preprocessing Pipeline (Single Source of Truth)
# ──────────────────────────────────────────────
def preprocess(
    image_path: str | Path,
    cfg: dict,
    return_feature: bool = True,
) -> dict[str, np.ndarray]:
    """
    스캔 이미지 1장에 대해 전체 전처리 파이프라인을 실행한다.
    Training / Inference 동일 전처리 (Augmentation 제외).
    Runs the full preprocessing pipeline on a single scan image.
    Same preprocessing for both Training and Inference (excluding Augmentation).

    Args:
        image_path:     스캔 이미지 경로 / Scan image path
        cfg:            config.yaml 딕셔너리 / config.yaml dictionary
        return_feature: True면 강조 특징도 함께 반환 / If True, also returns enhanced feature maps

    Returns:
        {
          "Y": (H, W, 3) float32,
          "M": ...,
          "C": ...,
          "K": ...,
          "Y_feature": (H, W, 3) float32,  # return_feature=True 시 / when return_feature=True
          ...
        }

    Pipeline:
        RGB 로드 / Load RGB
        → RGB→CMYK 변환 / RGB to CMYK conversion
        → ROI 분리 (auto / fixed) / ROI extraction
        → Background Normalization / 배경 보정
        → 색상별 독립 정규화 / Per-channel normalization
        → Feature Enhancement / Grayspot 강조
        → Resize & 3채널 변환 / Resize & 3-channel replication
        → Tensor-ready (H,W,3) float32
    """
    p    = cfg["preprocessing"]
    d    = cfg["data"]
    size = d["image_size"]

    # 1. RGB 로드 / Load RGB image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다 / Cannot load image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. RGB → CMYK 변환 / Convert RGB to CMYK
    cmyk = rgb_to_cmyk(rgb)

    # 3. ROI 분리 / Extract ROI per channel
    if p["roi_mode"] == "auto":
        rois, used_fallback = extract_roi_auto(
            cmyk,
            min_confidence=p["roi_auto_min_confidence"],
            fallback_coords=p.get("roi_fixed_coords"),
        )
        if used_fallback:
            print("  ROI 자동 검출 실패 → 고정 좌표 fallback 사용 / Auto ROI detection failed → using fixed coordinate fallback")
    else:
        rois = extract_roi_fixed(cmyk, p["roi_fixed_coords"])

    result = {}
    for ch in CHANNELS:
        roi = rois[ch]

        # 4. 배경 보정 / Background normalization
        roi_bg = background_normalization(roi, p["bg_kernel_size"])

        # 5. 색상별 독립 정규화 / Per-channel independent normalization
        norm_params = p["normalization"][ch]
        roi_norm    = normalize_channel(roi_bg, norm_params["mean"], norm_params["std"])

        # 6. Feature Enhancement — Grayspot 강조 / Grayspot defect emphasis
        feature = feature_enhancement(roi_bg, ch, cfg)

        # 7. Resize & 3채널 변환 / Resize and replicate to 3 channels
        roi_resized = cv2.resize(roi_norm, (size, size))
        roi_3ch     = to_3channel(roi_resized.astype(np.float32))
        result[ch]  = roi_3ch

        if return_feature:
            feat_resized        = cv2.resize(feature, (size, size))
            feat_3ch            = to_3channel(feat_resized)
            result[f"{ch}_feature"] = feat_3ch

    return result