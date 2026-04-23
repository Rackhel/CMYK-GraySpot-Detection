"""
inference/predictor.py

최적화된 추론 파이프라인 / Optimized Inference Pipeline
Grayspot 탐지 모델 추론 시스템

프로젝트 전체에서 단일 채널 또는 멀티 채널 추론을 수행합니다.
Perform single-channel or multi-channel inference across the project.

기능 / Features:
  - 배치 추론 지원 / Batch inference support
  - 장치 자동 감지 (GPU/CPU) / Auto device detection
  - 신뢰도 점수 계산 / Confidence score computation
  - 캐시된 모델 로딩 / Cached model loading
  - 상세 로깅 / Detailed logging

사용법 / Usage:
    from src.inference.predictor import GrayspotPredictor
    
    predictor = GrayspotPredictor(config_path="src/config/config.yaml")
    predictor.load_model(channel="Y", model_path="models/best_Y.pt")
    
    predictions = predictor.predict(images, channel="Y", batch_size=32)
    
Python 3.11.5 | PyTorch 2.2.2+ compatible
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Setup path for imports / 임포트 경로 설정 ────────────────────────
_SRC_DIR = Path(__file__).parent.parent.resolve()  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Logger setup / 로거 설정 ────────────────────────────────────────
try:
    from utils.logger import get_logger, LoggerMixin
    _logger = get_logger(__name__)
except ImportError:
    _logger = logging.getLogger(__name__)

# ── Project imports / 프로젝트 임포트 ────────────────────────────────
from config import get_config
from models.grayspot_model import GrayspotModel


class GrayspotPredictor(LoggerMixin):
    """
    최적화된 추론 엔진 / Optimized inference engine for Grayspot model.
    
    배치 처리, 장치 관리, 모델 캐싱을 지원합니다.
    Supports batch processing, device management, and model caching.
    
    Attributes:
        config: 설정 관리자 / Configuration manager
        device: 추론 장치 (cuda/cpu) / Inference device
        models: 채널별 로드된 모델 캐시 / Per-channel loaded models cache
        model_paths: 채널별 모델 경로 / Per-channel model paths
    """
    
    def __init__(self, config_path: Optional[str | Path] = None):
        """
        초기화 / Initialize predictor.
        
        Args:
            config_path: 설정 파일 경로 / Path to config.yaml
                        None이면 기본값 사용 / Uses default if None
        """
        self.logger.info("[Predictor] Initializing GrayspotPredictor...")
        
        # 설정 로드 / Load configuration
        self.config = get_config(config_path) if config_path else get_config()
        self.logger.debug(f"  Config loaded: {self.config}")
        
        # 장치 설정 / Setup device
        self.device = self._setup_device()
        self.logger.info(f"  Device: {self.device}")
        
        # 모델 캐시 / Model cache
        self.models: Dict[str, GrayspotModel] = {}
        self.model_paths: Dict[str, Path] = {}
        
        # 채널 및 설정 / Channels and settings
        self.channels = self.config.get("data.channels", ["Y", "M", "C", "K"])
        self.image_size = self.config.get("data.image_size", 128)
        self.num_levels = self.config.get("data.num_levels", 6)
        
        self.logger.info(f"  Channels: {self.channels}")
        self.logger.info(f"  Image size: {self.image_size}x{self.image_size}")
        self.logger.info(f"  Num levels: {self.num_levels}")
        self.logger.info("[Predictor] Initialization complete ✓")
    
    def _setup_device(self) -> torch.device:
        """
        장치 자동 설정 / Auto setup device.
        
        Returns:
            torch.device: CUDA 가능하면 cuda, 아니면 cpu
        """
        device_config = self.config.get("system.device", "auto")
        
        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_config.lower() == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        if device.type == "cuda":
            self.logger.debug(f"  GPU: {torch.cuda.get_device_name(0)}")
            self.logger.debug(f"  CUDA version: {torch.version.cuda}")
        
        return device
    
    def load_model(
        self,
        channel: str,
        model_path: Optional[str | Path] = None,
    ) -> None:
        """
        채널별 모델 로드 / Load model for a specific channel.
        
        Args:
            channel: 채널 (Y/M/C/K) / Channel name
            model_path: 모델 파일 경로 / Path to model checkpoint
                       None이면 config에서 자동 탐색 / Auto-search if None
        
        Raises:
            FileNotFoundError: 모델을 찾을 수 없으면 / If model not found
            ValueError: 채널이 지원되지 않으면 / If channel not supported
        """
        channel = channel.upper()
        
        if channel not in self.channels:
            raise ValueError(f"Unsupported channel: {channel}. Available: {self.channels}")
        
        # 캐시 확인 / Check cache
        if channel in self.models:
            self.logger.debug(f"  [{channel}] Model already cached")
            return
        
        # 모델 경로 결정 / Determine model path
        if model_path is None:
            models_dir = self.config.get_path("storage.models_dir")
            model_path = models_dir / f"best_{channel}.pt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"[Predictor] Loading model [{channel}] from {model_path}")
        
        try:
            # 모델 생성 / Create model
            model = GrayspotModel(self.config.config, phase=2)
            
            # 가중치 로드 / Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 호환성: checkpoint가 dict일 수 있음 / Compatibility: checkpoint might be dict
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            # 평가 모드 / Evaluation mode
            model = model.to(self.device)
            model.eval()
            
            # 캐시에 저장 / Store in cache
            self.models[channel] = model
            self.model_paths[channel] = model_path
            
            self.logger.info(f"  ✓ Model [{channel}] loaded successfully")
            
        except Exception as e:
            self.logger.error(f"  ✗ Failed to load model [{channel}]: {e}")
            raise
    
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
            images: 입력 이미지 배열 (N, H, W) 또는 (N, H, W, 3)
                   Input image array
            channel: 채널명 (Y/M/C/K) / Channel name
            batch_size: 배치 크기 / Batch size for inference
            return_confidences: True이면 신뢰도 반환 / Return confidence scores
        
        Returns:
            Dict with keys:
                - "predictions": 예측 클래스 (N,) / Predicted classes
                - "confidences": 신뢰도 점수 (N,) / Confidence scores (if return_confidences=True)
                - "logits": 모델 로짓 (N, num_classes) / Model logits
                - "probabilities": 소프트맥스 확률 (N, num_classes) / Softmax probabilities
        
        Raises:
            ValueError: 입력 형태가 잘못되면 / If input shape is invalid
            RuntimeError: 모델이 로드되지 않았으면 / If model not loaded
        """
        channel = channel.upper()
        
        # 모델 확인 / Check model
        if channel not in self.models:
            raise RuntimeError(f"Model not loaded for channel [{channel}]. Call load_model() first.")
        
        self.logger.debug(f"[Predict] Starting inference for [{channel}]")
        self.logger.debug(f"  Input shape: {images.shape}, batch_size: {batch_size}")
        
        # 입력 검증 / Validate input
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                # (N, H, W) → (N, H, W, 3) 변환
                if images.shape[1:] == (self.image_size, self.image_size):
                    images = np.stack([images, images, images], axis=-1)
                else:
                    raise ValueError(
                        f"Image size mismatch. Expected {self.image_size}x{self.image_size}, "
                        f"got {images.shape[1]}x{images.shape[2]}"
                    )
            elif images.ndim != 4:
                raise ValueError(f"Input must be (N, H, W) or (N, H, W, 3), got {images.shape}")
        else:
            raise ValueError(f"Input must be numpy array, got {type(images)}")
        
        # 데이터 준비 / Prepare data
        model = self.models[channel]
        images_tensor = torch.from_numpy(images).float().to(self.device)
        
        # 전처리 (정규화) / Preprocess (normalize)
        # ImageNet 정규화 / ImageNet normalization
        images_tensor = images_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 1, 1, 3)
        images_tensor = (images_tensor - mean) / std
        
        # 채널 순서 변경: (N, H, W, C) → (N, C, H, W)
        images_tensor = images_tensor.permute(0, 3, 1, 2)
        
        # 데이터셋 및 로더 / Dataset and loader
        dataset = TensorDataset(images_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 추론 수행 / Perform inference
        all_logits = []
        all_predictions = []
        all_confidences = []
        
        self.logger.debug(f"  Processing {len(images)} images in {len(dataloader)} batches")
        
        with torch.no_grad():
            for batch_idx, (batch,) in enumerate(dataloader):
                # 포워드 패스 / Forward pass
                logits = model(batch)
                probabilities = F.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                # CPU로 이동 및 numpy 변환 / Move to CPU and convert to numpy
                all_logits.append(logits.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_confidences.append(confidences.cpu().numpy())
        
        # 배열 연결 / Concatenate arrays
        logits_array = np.concatenate(all_logits, axis=0)
        predictions_array = np.concatenate(all_predictions, axis=0)
        confidences_array = np.concatenate(all_confidences, axis=0)
        probabilities_array = F.softmax(torch.from_numpy(logits_array), dim=1).numpy()
        
        self.logger.debug(f"  ✓ Inference complete")
        self.logger.debug(
            f"    Predictions shape: {predictions_array.shape}, "
            f"Confidence range: [{confidences_array.min():.4f}, {confidences_array.max():.4f}]"
        )
        
        # 결과 반환 / Return results
        result = {
            "predictions": predictions_array,
            "logits": logits_array,
            "probabilities": probabilities_array,
        }
        
        if return_confidences:
            result["confidences"] = confidences_array
        
        return result
    
    def predict_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        멀티 채널 배치 추론 / Perform batch inference on multiple channels.
        
        Args:
            images_dict: 채널별 이미지 딕셔너리
                        예: {"Y": array, "M": array, ...}
                        Dict of channel -> image array
            batch_size: 배치 크기 / Batch size
        
        Returns:
            Dict[channel, Dict[str, np.ndarray]]: 채널별 추론 결과
        """
        self.logger.info(f"[Predict] Starting multi-channel batch inference...")
        
        results = {}
        for channel, images in images_dict.items():
            if channel not in self.models:
                self.logger.warning(f"  Skipping [{channel}] - model not loaded")
                continue
            
            self.logger.debug(f"  Processing channel [{channel}]...")
            results[channel] = self.predict(images, channel, batch_size)
        
        self.logger.info(f"  ✓ Batch inference complete for {len(results)} channels")
        return results
    
    def clear_cache(self, channel: Optional[str] = None) -> None:
        """
        모델 캐시 비우기 / Clear model cache.
        
        Args:
            channel: 특정 채널만 비우기 / Clear specific channel
                    None이면 전체 비우기 / Clear all if None
        """
        if channel is None:
            self.logger.debug(f"[Predictor] Clearing all model caches")
            self.models.clear()
            self.model_paths.clear()
        else:
            channel = channel.upper()
            if channel in self.models:
                del self.models[channel]
                del self.model_paths[channel]
                self.logger.debug(f"[Predictor] Cleared cache for [{channel}]")
    
    def get_model_info(self, channel: Optional[str] = None) -> Dict[str, Any]:
        """
        로드된 모델 정보 조회 / Get loaded model information.
        
        Args:
            channel: 특정 채널 / Specific channel
                    None이면 모든 채널 정보 / All channels if None
        
        Returns:
            Dict with model metadata
        """
        if channel is None:
            return {
                ch: {
                    "device": str(self.device),
                    "model_path": str(self.model_paths.get(ch, "N/A")),
                    "num_parameters": sum(p.numel() for p in self.models[ch].parameters()),
                }
                for ch in self.models
            }
        else:
            channel = channel.upper()
            if channel not in self.models:
                return {"error": f"Model not loaded for [{channel}]"}
            
            return {
                "device": str(self.device),
                "model_path": str(self.model_paths[channel]),
                "num_parameters": sum(p.numel() for p in self.models[channel].parameters()),
            }