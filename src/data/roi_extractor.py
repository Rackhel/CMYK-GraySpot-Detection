"""
data/roi_extractor.py

CMYK 채널 분리 및 ROI 패치 추출.
CMYK channel splitting and ROI patch extraction.

인터페이스 / Interface:
    ROIExtractor(cfg)
        .split_cmyk(image)              → dict[str, np.ndarray]  (float32 [0,1])
        .extract_patches(path, ch, lv)  → list[np.ndarray]       (128×128×3 uint8)
        .extract_patches_from_roi(path) → list[np.ndarray]       (128×128×3 uint8)

SSOT 근거 / SSOT Reference:
    - SSOT_ROI_Pipeline.md §2 — CMYK 채널 분리 수식
    - SSOT_Data_Pipeline.md §0 — 데이터 생산 파이프라인 흐름
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CHANNELS = frozenset({"C", "M", "Y", "K"})


class ROIExtractor:
    """
    원본 스캔 이미지에서 CMYK 채널을 분리하고 128×128 패치를 추출한다.

    Splits CMYK channels from a raw scan image and extracts 128×128 patches.

    Args:
        cfg: 설정 딕셔너리 / Configuration dictionary.
            Required keys:
                cfg["data"]["image_size"]          — 패치 크기 / Patch size (int)
                cfg["roi"]["mode"]                 — "fixed" | "auto"
                cfg["roi"]["fixed_coords"]         — [x0, y0, x1, y1] (mode="fixed" 시 필수)
            Optional keys:
                cfg["roi"].get("min_std", 5.0)     — 저분산 패치 제거 임계값
    """

    def __init__(self, cfg: dict) -> None:
        roi_cfg = cfg["roi"]
        self._patch_size: int = cfg["data"]["image_size"]
        self._mode: str = roi_cfg["mode"]
        self._fixed_coords: list[int] | None = roi_cfg.get("fixed_coords")
        self._min_std: float = float(roi_cfg.get("min_std", 5.0))

    # ── 공개 API / Public API ──────────────────────────────────────────────────

    def split_cmyk(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        BGR uint8 이미지를 CMYK 채널로 분리한다.
        Splits a BGR uint8 image into CMYK channels.

        수식 (SSOT_ROI_Pipeline.md §2) / Formula:
            C = 1 - R,  M = 1 - G,  Y = 1 - B,  K = min(C, M, Y)

        Args:
            image: BGR uint8 ndarray (H, W, 3)

        Returns:
            dict with keys "C","M","Y","K", each float32 ndarray [0,1] of shape (H,W)
        """
        img_f = image.astype(np.float32) / 255.0
        R = img_f[:, :, 2]  # BGR 포맷: index 2 = R
        G = img_f[:, :, 1]  # BGR 포맷: index 1 = G
        B = img_f[:, :, 0]  # BGR 포맷: index 0 = B

        C = (1.0 - R).astype(np.float32)
        M = (1.0 - G).astype(np.float32)
        Y = (1.0 - B).astype(np.float32)
        K = np.minimum(np.minimum(C, M), Y).astype(np.float32)

        return {"C": C, "M": M, "Y": Y, "K": K}

    def extract_patches(
        self,
        image_path: str | Path,
        channel: str,
        level: int,  # noqa: ARG002  (향후 레벨별 필터링 확장용 / reserved for per-level filtering)
    ) -> list[np.ndarray]:
        """
        전체 스캔 이미지에서 지정 CMYK 채널의 패치를 추출한다.
        Extracts patches for the specified CMYK channel from a full scan image.

        처리 순서 / Processing order:
            1. 이미지 로드 (BGR uint8)
            2. CMYK 채널 분리
            3. 지정 채널 → grayscale uint8 → 3채널 복제 (BGR 호환)
            4. ROI 크롭 적용 (mode="fixed")
            5. 슬라이딩 윈도우 패치 추출

        Args:
            image_path: 이미지 파일 경로 / Path to image file
            channel:    "C" | "M" | "Y" | "K"
            level:      Grayspot 결함 수준 (0-5) — 현재 미사용, 확장 예약

        Returns:
            list of (patch_size, patch_size, 3) uint8 ndarrays

        Raises:
            FileNotFoundError: 이미지 파일이 존재하지 않을 때
            ValueError: 알 수 없는 채널명
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if channel not in _CHANNELS:
            raise ValueError(f"Unknown channel '{channel}'. Must be one of {_CHANNELS}.")

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cv2 cannot read image: {image_path}")

        cmyk = self.split_cmyk(img)
        ch_float = cmyk[channel]  # (H, W) float32

        # grayscale → 3채널 BGR (세 채널 동일값)
        ch_u8 = (ch_float * 255.0).clip(0, 255).astype(np.uint8)
        ch_bgr = np.stack([ch_u8, ch_u8, ch_u8], axis=2)

        ch_bgr = self._apply_roi_crop(ch_bgr)
        return self._slide_patches(ch_bgr)

    def extract_patches_from_roi(self, image_path: str | Path) -> list[np.ndarray]:
        """
        이미 CMYK 분리된 ROI 이미지에서 패치를 추출한다.
        Extracts patches from a pre-split per-channel ROI image.

        prepare_dataset.py 에서 lvlX_..._CH.png 파일에 사용한다.
        Used by prepare_dataset.py for lvlX_..._CH.png files.

        Args:
            image_path: 채널 ROI 이미지 경로 / Path to channel ROI image

        Returns:
            list of (patch_size, patch_size, 3) uint8 ndarrays

        Raises:
            FileNotFoundError: 이미지 파일이 존재하지 않을 때
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"cv2 cannot read image: {image_path}")

        return self._slide_patches(img)

    # ── 내부 헬퍼 / Internal helpers ───────────────────────────────────────────

    def _apply_roi_crop(self, img: np.ndarray) -> np.ndarray:
        """fixed_coords 가 설정된 경우 이미지를 크롭한다."""
        if self._mode == "fixed" and self._fixed_coords is not None:
            x0, y0, x1, y1 = self._fixed_coords
            img = img[y0:y1, x0:x1]
        return img

    def _slide_patches(self, img: np.ndarray) -> list[np.ndarray]:
        """
        이미지에서 patch_size × patch_size 패치를 stride=patch_size 로 추출한다.

        가로 정규화 / Width normalization:
            - 가로 ≥ patch_size: 중앙 크롭
            - 가로 < patch_size: reflect 패딩

        저분산 패치 제거 / Low-variance filtering:
            - 표준편차 < min_std 인 패치는 비인쇄 영역으로 판단하여 제거
        """
        ps = self._patch_size
        h, w = img.shape[:2]

        if w >= ps:
            x0 = (w - ps) // 2
            strip = img[:, x0 : x0 + ps]
        else:
            pad = ps - w
            pad_l = pad // 2
            strip = cv2.copyMakeBorder(
                img, 0, 0, pad_l, pad - pad_l, cv2.BORDER_REFLECT
            )

        patches: list[np.ndarray] = []
        for y in range(0, h - ps + 1, ps):
            patch = strip[y : y + ps, :ps].copy()
            if patch.std() >= self._min_std:
                patches.append(patch)

        return patches
