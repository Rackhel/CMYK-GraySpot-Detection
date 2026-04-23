"""
inference/predictor.py

최적화된 추론 파이프라인 / Optimized Inference Pipeline
Grayspot 탐지 모델 추론 시스템

프로젝트 전체에서 단일 채널 또는 멀티 채널 추론을 수행합니다.
Perform single-channel or multi-channel inference across the project.

기능 / Features:
  - 배치 추론 지원 / Batch inference support
  - 장치 자동 감지 (CUDA/MPS/CPU) / Auto device detection
  - 신뢰도 점수 계산 / Confidence score computation
  - 캐시된 모델 로딩 / Cached model loading
  - 상세 로깅 / Detailed logging

사용법 / Usage:
    from inference.predictor import GrayspotPredictor

    predictor = GrayspotPredictor()
    predictor.load_model(channel="Y", model_path="models/best_Y.pt")
    predictions = predictor.predict(images, channel="Y", batch_size=32)

Python 3.11.5 | PyTorch 2.x compatible
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Logger setup — works regardless of whether the full utils package is loaded
# ---------------------------------------------------------------------------
try:
    from utils.logger import get_logger, LoggerMixin
    _module_logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    _module_logger = logging.getLogger(__name__)

    class LoggerMixin:  # type: ignore[no-redef]
        @property
        def logger(self) -> logging.Logger:
            if not hasattr(self, "_logger"):
                self._logger = logging.getLogger(self.__class__.__name__)
            return self._logger


# ---------------------------------------------------------------------------
# Lazy config / model imports to avoid circular dependency issues
# ---------------------------------------------------------------------------

def _get_config(config_path=None):
    """Load ConfigManager, handling different sys.path scenarios."""
    try:
        from config import get_config as _gc
    except ImportError:
        from src.config import get_config as _gc  # type: ignore
    return _gc(config_path) if config_path else _gc()


def _get_model_class():
    """Return GrayspotModel class."""
    try:
        from models.grayspot_model import GrayspotModel
    except ImportError:
        from src.models.grayspot_model import GrayspotModel  # type: ignore
    return GrayspotModel


class GrayspotPredictor(LoggerMixin):
    """
    최적화된 추론 엔진 / Optimized inference engine for Grayspot model.

    배치 처리, 장치 관리, 모델 캐싱을 지원합니다.
    Supports batch processing, device management, and model caching.

    Attributes:
        config     : 설정 관리자 / Configuration manager
        device     : 추론 장치 (cuda/mps/cpu) / Inference device
        models     : 채널별 로드된 모델 캐시 / Per-channel loaded models cache
        model_paths: 채널별 모델 경로 / Per-channel model paths
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        """
        초기화 / Initialize predictor.

        Args:
            config_path: 설정 파일 경로 (None 이면 기본값 사용)
                         Path to config.yaml (None uses default)
        """
        self.logger.info("[Predictor] Initializing GrayspotPredictor...")

        self.config = _get_config(config_path)
        self.device = self._setup_device()
        self.logger.info(f"  Device: {self.device}")

        self.models: Dict[str, Any] = {}
        self.model_paths: Dict[str, Path] = {}

        self.channels    = self.config.get("data.channels") or ["Y", "M", "C", "K"]
        self.image_size  = self.config.get("data.image_size") or 128
        self.num_levels  = self.config.get("data.num_levels") or 6

        self.logger.info(f"  Channels: {self.channels}")
        self.logger.info(f"  Image size: {self.image_size}x{self.image_size}")
        self.logger.info("[Predictor] Initialization complete ✓")

    def _setup_device(self) -> torch.device:
        """장치 자동 설정 / Auto-detect and set compute device."""
        device_cfg = (self.config.get("system.device") or "auto").lower()

        if device_cfg == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if device_cfg == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            # Fallback to MPS or CPU if CUDA requested but unavailable
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.logger.warning("CUDA not available; falling back to MPS.")
                return torch.device("mps")
            self.logger.warning("CUDA not available; falling back to CPU.")
            return torch.device("cpu")

        if device_cfg == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            self.logger.warning("MPS not available; falling back to CPU.")
            return torch.device("cpu")

        return torch.device("cpu")

    def load_model(
        self,
        channel: str,
        model_path: Optional[str | Path] = None,
    ) -> None:
        """
        채널별 모델 로드 / Load model for a specific channel.

        Args:
            channel   : 채널 (Y/M/C/K)
            model_path: 모델 파일 경로 (None 이면 config 에서 자동 탐색)
        """
        channel = channel.upper()

        if channel not in self.channels:
            raise ValueError(
                f"Unsupported channel: {channel}. Available: {self.channels}"
            )

        if channel in self.models:
            self.logger.debug(f"  [{channel}] Model already cached")
            return

        if model_path is None:
            models_dir = self.config.get_path("storage.models_dir")
            model_path = models_dir / f"best_{channel}.pt"
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.logger.info(f"[Predictor] Loading model [{channel}] from {model_path}")

        GrayspotModel = _get_model_class()
        model = GrayspotModel(self.config.config, phase=2)

        checkpoint = torch.load(str(model_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device).eval()
        self.models[channel]      = model
        self.model_paths[channel] = model_path
        self.logger.info(f"  ✓ Model [{channel}] loaded successfully")

    def predict(
        self,
        images: np.ndarray,
        channel: str,
        batch_size: int = 32,
        return_confidences: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        이미지에 대한 추론 수행 / Perform inference on images.

        Args:
            images            : (N, H, W, 3) or (N, H, W) float32 [0-255] numpy array
            channel           : 채널명 (Y/M/C/K)
            batch_size        : 배치 크기
            return_confidences: 신뢰도 반환 여부

        Returns:
            dict with:
                predictions  : (N,) predicted classes
                confidences  : (N,) max-softmax confidence (if return_confidences)
                logits        : (N, num_levels)
                probabilities : (N, num_levels)
        """
        channel = channel.upper()

        if channel not in self.models:
            raise RuntimeError(
                f"Model not loaded for [{channel}]. Call load_model() first."
            )

        if not isinstance(images, np.ndarray):
            raise ValueError(f"images must be numpy array, got {type(images)}")

        # Ensure (N, H, W, 3) shape
        if images.ndim == 3:
            images = np.stack([images, images, images], axis=-1)
        elif images.ndim != 4:
            raise ValueError(
                f"images must be (N, H, W) or (N, H, W, 3), got shape {images.shape}"
            )

        # Normalise to [0, 1]
        img_float = images.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        # (N, H, W, C) -> (N, C, H, W)
        img_tensor = torch.from_numpy(img_float).permute(0, 3, 1, 2).float()
        dataset    = TensorDataset(img_tensor)
        loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = self.models[channel]
        all_logits, all_preds, all_confs = [], [], []

        with torch.no_grad():
            for (batch,) in loader:
                batch   = batch.to(self.device)
                logits  = model(batch)
                probs   = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)

                all_logits.append(logits.cpu().numpy())
                all_preds.append(pred.cpu().numpy())
                all_confs.append(conf.cpu().numpy())

        logits_arr = np.concatenate(all_logits, axis=0)
        preds_arr  = np.concatenate(all_preds,  axis=0)
        confs_arr  = np.concatenate(all_confs,  axis=0)
        probs_arr  = F.softmax(torch.from_numpy(logits_arr), dim=1).numpy()

        result: Dict[str, np.ndarray] = {
            "predictions"  : preds_arr,
            "logits"       : logits_arr,
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
        멀티 채널 배치 추론 / Batch inference on multiple channels.

        Args:
            images_dict: {channel: image_array, ...}
            batch_size : 배치 크기
        """
        self.logger.info("[Predict] Starting multi-channel batch inference...")
        results = {}
        for channel, images in images_dict.items():
            ch = channel.upper()
            if ch not in self.models:
                self.logger.warning(f"  Skipping [{ch}] — model not loaded")
                continue
            results[ch] = self.predict(images, ch, batch_size)

        self.logger.info(
            f"  ✓ Batch inference complete for {len(results)} channels"
        )
        return results

    def clear_cache(self, channel: Optional[str] = None) -> None:
        """모델 캐시 비우기 / Clear model cache."""
        if channel is None:
            self.models.clear()
            self.model_paths.clear()
            self.logger.debug("[Predictor] All model caches cleared")
        else:
            ch = channel.upper()
            if ch in self.models:
                del self.models[ch]
                del self.model_paths[ch]
                self.logger.debug(f"[Predictor] Cache cleared for [{ch}]")

    def get_model_info(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """로드된 모델 정보 조회 / Get loaded model information."""
        if channel is None:
            return {
                ch: {
                    "device"        : str(self.device),
                    "model_path"    : str(self.model_paths.get(ch, "N/A")),
                    "num_parameters": sum(
                        p.numel() for p in self.models[ch].parameters()
                    ),
                }
                for ch in self.models
            }

        ch = channel.upper()
        if ch not in self.models:
            return {"error": f"Model not loaded for [{ch}]"}

        return {
            "device"        : str(self.device),
            "model_path"    : str(self.model_paths[ch]),
            "num_parameters": sum(
                p.numel() for p in self.models[ch].parameters()
            ),
        }