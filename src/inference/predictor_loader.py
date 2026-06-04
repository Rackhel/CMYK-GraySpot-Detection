"""
inference/predictor_loader.py

책임 / Responsibility: 채널별 모델 로딩 및 캐시 관리
Responsibility: Per-channel model loading and cache management

SRP 준수: 이 모듈은 "모델 파일 로딩과 캐시"만 담당한다.
SRP compliant: this module handles only "model file loading and caching".

SSOT 근거 / SSOT Reference:
    - SSOT_Artifacts.md — best_{channel}.pt 아티팩트 경로
    - Contract.md §6 — 아티팩트 경계 계약
    - SSOT_Core.md §6 — Fail-Fast: 아티팩트 누락 시 즉시 실패
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.logger import LoggerMixin
from utils.utils_model import build_model


def _load_normalizer_for_checkpoint(model_path: Path):
    """체크포인트 옆 .meta.json에서 정규화 변환을 로드한다.
    Loads normalization transform from .meta.json beside the checkpoint.

    .meta.json이 없으면 ImageNet 기본값을 반환한다.
    Returns ImageNet defaults if .meta.json is not found.
    """
    import json

    from torchvision import transforms as T

    from data.normalize import _IMAGENET_NORMALIZE

    meta_path = model_path.parent / (model_path.stem + ".meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return T.Normalize(
                mean=meta["normalize_mean"],
                std=meta["normalize_std"],
            )
        except Exception:
            pass
    return _IMAGENET_NORMALIZE


class ModelLoaderMixin(LoggerMixin):
    """
    모델 로딩 및 캐시 관리 Mixin / Model loading and cache management Mixin.

    DIP 준수: GrayspotModel을 직접 임포트 (fallback try/except 없음 — Fail-Fast).
    DIP compliant: GrayspotModel imported directly — no fallback try/except (Fail-Fast).
    """

    def load_model(
        self,
        channel: str,
        model_path: Optional[str | Path] = None,
    ) -> None:
        """
        채널별 모델을 로드하여 캐시에 저장한다.
        Loads and caches the model for a given channel.

        Args:
            channel   : CMYK 채널명 (Y/M/C/K) / Channel name
            model_path: 모델 파일 경로. None 이면 config의 storage.models_dir 에서 자동 탐색.
                        Model file path. None uses storage.models_dir from config.

        Raises:
            ValueError      : 지원하지 않는 채널 / Unsupported channel
            FileNotFoundError: 모델 파일 없음 — SSOT-FF01 / Model file missing
        """
        channel = channel.upper()

        if channel not in self.channels:
            raise ValueError(
                f"[Loader] Unsupported channel: {channel}. "
                f"Available: {self.channels}"
            )

        if channel in self.models:
            self.logger.debug(f"  [{channel}] Model already cached — skipping load")
            return

        resolved_path = self._resolve_model_path(channel, model_path)

        # SSOT-FF01: 아티팩트 누락 즉시 실패 / Fail immediately on missing artifact
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"[SSOT-FF01] Model artifact not found: {resolved_path}. "
                f"Run Phase 2 training for channel [{channel}] first."
            )

        self.logger.info(f"[Loader] Loading [{channel}] from {resolved_path}")

        # build_model: 체크포인트 weight shape에서 architecture 자동감지 (Optuna 튜닝 후에도 안전)
        # build_model: auto-detects architecture from weight shapes (safe after Optuna tuning)
        model = build_model(self.cfg, resolved_path, self.device)

        self.models[channel] = model
        self.model_paths[channel] = resolved_path

        # 체크포인트 옆 .meta.json에서 정규화 변환 로드 (없으면 ImageNet 기본값)
        # Load normalization from .meta.json beside checkpoint (fallback: ImageNet defaults)
        normalizer = _load_normalizer_for_checkpoint(resolved_path)
        if not hasattr(self, "normalizers"):
            self.normalizers: Dict[str, Any] = {}
        self.normalizers[channel] = normalizer
        meta_exists = (
            resolved_path.parent / (resolved_path.stem + ".meta.json")
        ).exists()
        self.logger.info(
            f"  ✓ [{channel}] loaded | normalizer: "
            f"{'meta.json' if meta_exists else 'ImageNet fallback'}"
        )

    def clear_cache(self, channel: Optional[str] = None) -> None:
        """
        모델 캐시를 비운다 / Clears model cache.

        Args:
            channel: None 이면 전체 비움 / None clears all channels
        """
        if channel is None:
            self.models.clear()
            self.model_paths.clear()
            self.logger.debug("[Loader] All model caches cleared")
        else:
            ch = channel.upper()
            self.models.pop(ch, None)
            self.model_paths.pop(ch, None)
            self.logger.debug(f"[Loader] Cache cleared for [{ch}]")

    def get_model_info(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """
        로드된 모델 정보를 반환한다 / Returns loaded model information.

        Args:
            channel: None 이면 전체 채널 반환 / None returns all channels

        Returns:
            dict — device, model_path, num_parameters
        """
        if channel is None:
            return {ch: self._channel_info(ch) for ch in self.models}
        ch = channel.upper()
        if ch not in self.models:
            return {"error": f"Model not loaded for [{ch}]"}
        return self._channel_info(ch)

    # ------------------------------------------------------------------
    # 내부 헬퍼 / Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(
        self, channel: str, model_path: Optional[str | Path]
    ) -> Path:
        """config 또는 인자에서 모델 경로를 결정한다."""
        if model_path is not None:
            return Path(model_path)
        models_dir = Path(self.cfg["storage"]["models_dir"])
        return models_dir / f"best_{channel}.pt"

    def _channel_info(self, ch: str) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "model_path": str(self.model_paths.get(ch, "N/A")),
            "num_parameters": sum(p.numel() for p in self.models[ch].parameters()),
        }
