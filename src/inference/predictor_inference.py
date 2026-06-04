"""
inference/predictor_inference.py

책임 / Responsibility: 단일 채널 및 멀티 채널 배치 추론
Responsibility: Single-channel and multi-channel batch inference

SRP 준수: 이 모듈은 "추론 실행"만 담당한다.
SRP compliant: this module handles only "inference execution".

SSOT 근거 / SSOT Reference:
    - SSOT_Data_Pipeline.md §3 — 추론 입력은 BGR float32, ImageNet-normalized (학습과 일치 필수)
    - SSOT_Evaluation_Reporting.md §3 — 신뢰도 임계값
    - SSOT-NM01 해소: predict() 내부에서 ImageNet 정규화 적용
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.normalize import _IMAGENET_NORMALIZE  # SSOT-NM01: fallback용 기본값
from utils.logger import LoggerMixin


class InferenceMixin(LoggerMixin):
    """
    추론 실행 Mixin / Inference execution Mixin.

    SSOT-NM01 준수: predict()에서 [0,1] 정규화 후 반드시 ImageNet mean/std 를 적용한다.
    SSOT-NM01 compliant: after [0,1] normalization, ImageNet mean/std is always applied.

    ISP 준수: 추론 클라이언트는 이 Mixin 메서드만 의존한다.
    ISP compliant: inference clients depend only on this Mixin's methods.
    """

    def predict(
        self,
        images: np.ndarray,
        channel: str,
        batch_size: int = 32,
        return_confidences: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        단일 채널 이미지 배치 추론 / Single-channel batch inference.

        Args:
            images            : (N, H, W, 3) or (N, H, W) uint8 or float32 numpy array
                                BGR 형식 / BGR format
            channel           : CMYK 채널명 (Y/M/C/K)
            batch_size        : DataLoader 배치 크기 / Batch size
            return_confidences: True 이면 confidences 키 포함 / Include confidences key

        Returns:
            dict:
                predictions   : (N,) int64 — 예측 클래스 / Predicted classes [0-5]
                logits        : (N, num_levels) float32
                probabilities : (N, num_levels) float32
                confidences   : (N,) float32 — max-softmax (return_confidences=True 시)

        Raises:
            RuntimeError: 모델 미로드 / Model not loaded for channel
            ValueError  : 잘못된 images 형상 / Invalid images shape
        """
        channel = channel.upper()

        if channel not in self.models:
            raise RuntimeError(
                f"[Inference] Model not loaded for [{channel}]. "
                f"Call load_model() first."
            )

        if not isinstance(images, np.ndarray):
            raise ValueError(
                f"[Inference] images must be numpy.ndarray, got {type(images)}"
            )

        # 채널별 normalizer: load_model() 시 .meta.json에서 로드, 없으면 ImageNet 기본값
        # Per-channel normalizer loaded from .meta.json; fallback to ImageNet defaults
        normalizer = getattr(self, "normalizers", {}).get(channel, _IMAGENET_NORMALIZE)
        img_tensor = self._preprocess_images(images, normalizer=normalizer)
        loader = DataLoader(
            TensorDataset(img_tensor), batch_size=batch_size, shuffle=False
        )

        model = self.models[channel]
        all_logits, all_preds, all_confs = [], [], []

        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)

                all_logits.append(logits.cpu().numpy())
                all_preds.append(pred.cpu().numpy())
                all_confs.append(conf.cpu().numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        preds_arr = np.concatenate(all_preds, axis=0)
        confs_arr = np.concatenate(all_confs, axis=0)
        probs_arr = F.softmax(torch.from_numpy(logits_arr), dim=1).numpy()

        result: Dict[str, np.ndarray] = {
            "predictions": preds_arr,
            "logits": logits_arr,
            "probabilities": probs_arr,
        }
        if return_confidences:
            result["confidences"] = confs_arr

        return result

    def predict_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        멀티 채널 배치 추론 / Multi-channel batch inference.

        Args:
            images_dict: {channel: image_array} — BGR numpy array per channel
            batch_size : DataLoader 배치 크기

        Returns:
            {channel: predict() 반환값} — 로드된 채널만 포함
        """
        self.logger.info("[Inference] Starting multi-channel batch inference...")
        results: Dict[str, Dict[str, np.ndarray]] = {}

        for channel, images in images_dict.items():
            ch = channel.upper()
            if ch not in self.models:
                self.logger.warning(f"  [{ch}] Model not loaded — skipping")
                continue
            results[ch] = self.predict(images, ch, batch_size)

        self.logger.info(f"  ✓ Batch inference complete for {len(results)} channel(s)")
        return results

    # ------------------------------------------------------------------
    # 내부 헬퍼 / Internal helper
    # ------------------------------------------------------------------

    def _preprocess_images(self, images: np.ndarray, normalizer=None) -> torch.Tensor:
        """
        numpy 이미지 배열을 정규화된 텐서로 변환한다.
        Converts numpy image array to a normalized tensor.

        SSOT_Data_Pipeline.md §3 처리 순서 / Processing order:
            1. (N,H,W) → (N,H,W,3) 채널 확장
            2. float32 변환 및 [0, 1] 정규화
            3. (N,H,W,C) → (N,C,H,W) 축 전치
            4. 채널별 normalizer 적용 (학습 시 저장된 mean/std 사용)
               Per-channel normalizer from .meta.json; fallback = ImageNet defaults
        """
        if normalizer is None:
            normalizer = _IMAGENET_NORMALIZE

        # 1. 형상 보정: (N,H,W) → (N,H,W,3)
        if images.ndim == 3:
            images = np.stack([images, images, images], axis=-1)
        elif images.ndim != 4:
            raise ValueError(
                f"[Inference] images must be (N,H,W) or (N,H,W,3), "
                f"got shape {images.shape}"
            )

        # 2. float32 변환 및 [0, 1] 정규화
        img_float = images.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        # 3. (N,H,W,C) → (N,C,H,W)
        img_tensor = torch.from_numpy(img_float).permute(0, 3, 1, 2)

        # 4. 채널별 정규화 적용 (학습당시 사용한 mean/std)
        #    Apply per-channel normalization (mean/std from training time)
        img_tensor = torch.stack(
            [normalizer(img_tensor[i]) for i in range(img_tensor.size(0))]
        )

        return img_tensor
