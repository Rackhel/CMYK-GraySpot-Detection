"""
utils/logger.py

로깅 시스템 / Logging system for CMYK Printer Project.

프로젝트 전체에서 사용하는 중앙 로거를 제공합니다.
Provides centralized logger for use throughout the project.

사용법 / Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started...")
    logger.warning("Low validation accuracy detected")
"""
import json
import logging
import logging.handlers
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    컬러 포맷팅 로거 / Colored log formatter for console output.
    
    자동으로 로그 레벨에 따라 색상을 다릅니다.
    Automatically colors log output based on level.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan / 청록색
        'INFO': '\033[32m',       # Green / 녹색
        'WARNING': '\033[33m',    # Yellow / 노란색
        'ERROR': '\033[31m',      # Red / 빨강
        'CRITICAL': '\033[35m',   # Magenta / 자주색
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """로그 기록 포맷 / Format log record."""
        if sys.platform == 'win32':
            if platform.architecture()[0] == '64bit':
                # Windows 64bit
                log_color = self.COLORS.get(record.levelname, self.RESET)
                record.levelname_colored = f"{log_color}{record.levelname}{self.RESET}"
                return super().format(record)
            else:
                # Windows 32bit → Can't Supporting ANSI 
                return super().format(record)
        
        else:
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname_colored = f"{log_color}{record.levelname}{self.RESET}"
            return super().format(record)


class LoggerConfig:
    """로거 설정 / Logger configuration."""
    
    # 콘솔 포맷 / Console format (사람이 읽기 좋은 형식)
    CONSOLE_FORMAT = (
        "[%(asctime)s] %(levelname_colored)s %(name)s - %(message)s"
    )
    
    # 파일 포맷 / File format (상세 정보 포함)
    FILE_FORMAT = (
        "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    
    # 날짜/시간 포맷 / Date/time format (한국/영어 공통)
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # 기본 레벨 / Default level
    DEFAULT_LEVEL = logging.INFO
    
    # 로그 파일 최대 크기 / Max log file size (10MB)
    MAX_BYTES = 10 * 1024 * 1024
    
    # 로그 파일 백업 개수 / Number of backup log files
    BACKUP_COUNT = 5


_loggers = {}  # 생성된 로거 캐시 / Cache for created loggers


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    format_style: str = "detailed",
    console: bool = True,
    file: bool = True,
) -> None:
    """
    프로젝트 전체 로깅 설정 / Setup logging for entire project.
    
    한 번만 호출하면 됩니다 (일반적으로 main 함수에서).
    Call once during program initialization (typically in main function).
    
    Args:
        log_dir: 로그 디렉토리 / Directory to save log files
        level: 로그 레벨 / Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        format_style: 포맷 스타일 / Format style ("simple", "detailed", "json")
        console: 콘솔 출력 활성화 / Enable console output
        file: 파일 출력 활성화 / Enable file output
        
    Example:
        setup_logging(
            log_dir=Path("outputs/logs"),
            level="INFO",
            console=True,
            file=True
        )
    """
    # 파일 로그 디렉토리 생성 / Create log directory
    if file and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # 루트 로거 설정 / Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 기존 핸들러 제거 / Remove existing handlers
    root_logger.handlers.clear()
    
    # 로그 레벨 매핑 / Map format style to logging config
    if format_style == "simple":
        console_fmt = "[%(levelname)s] %(message)s"
        file_fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
    elif format_style == "json":
        console_fmt = "%(message)s"  # JSON 포맷은 별도로 처리
        file_fmt = "%(message)s"
    else:  # detailed
        console_fmt = LoggerConfig.CONSOLE_FORMAT
        file_fmt = LoggerConfig.FILE_FORMAT
    
    # 콘솔 핸들러 / Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        if format_style == "detailed":
            console_formatter = ColoredFormatter(
                console_fmt,
                datefmt=LoggerConfig.DATETIME_FORMAT
            )
        else:
            console_formatter = logging.Formatter(
                console_fmt,
                datefmt=LoggerConfig.DATETIME_FORMAT
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 파일 핸들러 / File handler
    if file and log_dir:
        log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LoggerConfig.MAX_BYTES,
            backupCount=LoggerConfig.BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            file_fmt,
            datefmt=LoggerConfig.DATETIME_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    로거 인스턴스 반환 / Get logger instance.
    
    캐되어 있으므로 같은 이름으로 여러 번 호출해도 같은 인스턴스를 반환합니다.
    Cached, so multiple calls with same name return same instance.
    
    Args:
        name: 로거 이름 (보통 __name__ 전달) / Logger name (typically __name__)
              None이면 루트 로거 반환 / If None, returns root logger
    
    Returns:
        Logger 인스턴스 / Logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Starting training...")
        logger.debug(f"Config: {config}")
        logger.warning("Missing data for channel K")
        logger.error(f"Training failed: {error}")
    """
    if name is None:
        return logging.getLogger()
    
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


class LoggerMixin:
    """
    로거 믹스인 / Logger mixin for classes.
    
    이 클래스를 상속받은 클래스는 self.logger를 자동으로 가집니다.
    Classes inheriting from this have self.logger automatically.
    
    Example:
        class MyTrainer(LoggerMixin):
            def train(self):
                self.logger.info("Training started")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """로거 프로퍼티 / Logger property."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


# ─────────────────────────────────────────────────────────────────────
# 편의 함수들 / Convenience functions
# ─────────────────────────────────────────────────────────────────────

def log_training_config(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    훈련 설정 로깅 / Log training configuration.
    
    Args:
        config: 설정 딕셔너리 / Configuration dictionary
        logger: 로거 인스턴스 / Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION / 훈련 설정")
    logger.info("=" * 70)
    
    if "model" in config:
        logger.info(f"  [Model] Backbone: {config['model'].get('backbone', 'N/A')}")
    
    if "data" in config:
        logger.info(f"  [Data] Channels: {config['data'].get('channels', [])}")
        logger.info(f"  [Data] Image Size: {config['data'].get('image_size', 'N/A')}")
    
    if "phase2" in config:
        phase2 = config["phase2"]
        logger.info(f"  [Phase2] Epochs: {phase2.get('epochs', 'N/A')}")
        logger.info(f"  [Phase2] Batch Size: {phase2.get('batch_size', 'N/A')}")
        logger.info(f"  [Phase2] Learning Rate: {phase2.get('learning_rate', 'N/A')}")
    
    if "system" in config:
        logger.info(f"  [System] Device: {config['system'].get('device_name', 'N/A')}")
    
    logger.info("=" * 70)


def log_epoch_summary(
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    에폭 요약 로깅 / Log epoch summary.
    
    Args:
        epoch: 에폭 번호 / Epoch number
        train_loss: 훈련 손실 / Training loss
        val_loss: 검증 손실 / Validation loss
        metrics: 메트릭 딕셔너리 / Metrics dictionary
        logger: 로거 인스턴스 / Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    msg = f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    
    if metrics:
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        msg += f" | Acc: {acc:.4f} | F1: {f1:.4f}"
    
    logger.info(msg)


def log_inference_summary(
    channel: str,
    num_samples: int,
    batch_size: int,
    device: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    추론 요약 로깅 / Log inference summary.
    
    추론 파이프라인 시작 시 호출합니다.
    Call at the start of inference pipeline.
    
    Args:
        channel: CMYK 채널 / Channel name
        num_samples: 샘플 수 / Number of samples
        batch_size: 배치 크기 / Batch size
        device: 추론 장치 / Device (cuda/cpu)
        logger: 로거 인스턴스 / Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 70)
    logger.info(f"INFERENCE SUMMARY / 추론 요약 - [{channel}]")
    logger.info("=" * 70)
    logger.info(f"  Samples: {num_samples:,}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Total Batches: {(num_samples + batch_size - 1) // batch_size}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 70)


def log_prediction_stats(
    channel: str,
    predictions: dict,
    targets: Optional[dict] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    예측 통계 로깅 / Log prediction statistics.
    
    예측 결과의 기본 통계를 출력합니다.
    Print basic statistics of predictions.
    
    Args:
        channel: CMYK 채널 / Channel name
        predictions: 예측 결과 딕셔너리 / Prediction results dict
                    'predictions': class predictions
                    'confidences': confidence scores
        targets: 선택사항 목표값 / Optional ground truth
        logger: 로거 인스턴스 / Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    preds = predictions.get("predictions", [])
    confs = predictions.get("confidences", [])
    
    import numpy as np
    
    logger.info(f"[{channel}] Prediction Statistics / 예측 통계")
    logger.info(f"  Num Predictions: {len(preds)}")
    
    if len(confs) > 0:
        confs_arr = np.array(confs)
        logger.info(f"  Confidence - Mean: {confs_arr.mean():.4f}, "
                   f"Std: {confs_arr.std():.4f}, "
                   f"Min: {confs_arr.min():.4f}, Max: {confs_arr.max():.4f}")
    
    if targets is not None and "targets" in targets:
        targets_arr = np.array(targets["targets"])
        preds_arr = np.array(preds)
        accuracy = (preds_arr == targets_arr).mean()
        logger.info(f"  Accuracy (vs targets): {accuracy:.4f}")


def log_report_generation(
    report_path: str | Path,
    num_channels: int,
    num_samples: int,
    generation_time: float,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    보고서 생성 요약 로깅 / Log report generation summary.
    
    HTML 보고서 생성 완료 후 호출합니다.
    Call after HTML report generation completes.
    
    Args:
        report_path: 보고서 파일 경로 / Report file path
        num_channels: 채널 수 / Number of channels
        num_samples: 샘플 수 / Number of samples
        generation_time: 생성 소요 시간 (초) / Generation time in seconds
        logger: 로거 인스턴스 / Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 70)
    logger.info("REPORT GENERATION COMPLETE / 보고서 생성 완료")
    logger.info("=" * 70)
    logger.info(f"  Report Path: {report_path}")
    logger.info(f"  Channels: {num_channels}")
    logger.info(f"  Samples: {num_samples:,}")
    logger.info(f"  Generation Time: {generation_time:.2f}s")
    logger.info("=" * 70)


def log_pipeline_error(
    pipeline_name: str,
    stage: str,
    error: Exception,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    파이프라인 오류 로깅 / Log pipeline error with context.
    
    Args:
        pipeline_name: 파이프라인 이름 / Pipeline name
        stage: 오류 발생 단계 / Stage where error occurred
        error: 예외 객체 / Exception object
        logger: 로거 인스턴스 / Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    logger.error("=" * 70)
    logger.error(f"PIPELINE ERROR / 파이프라인 오류")
    logger.error("=" * 70)
    logger.error(f"  Pipeline: {pipeline_name}")
    logger.error(f"  Stage: {stage}")
    logger.error(f"  Error Type: {type(error).__name__}")
    logger.error(f"  Message: {str(error)}")
    logger.error("=" * 70)
    
    logger.debug(f"Stack trace:", exc_info=error)


def log_snapshot(
    config: dict,
    snapshot_dir: Optional[Path] = None,
    tag: str = "",
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    학습 시점 스냅샷 저장 / Save a point-in-time training snapshot to JSON.

    학습 시작 시 설정·환경·코드(git) 상태를 JSON 파일로 저장한다.
    Saves config dict, environment info, and git state as a JSON file at training start.

    Args:
        config:       저장할 설정 딕셔너리 / Configuration dictionary to snapshot
        snapshot_dir: 저장 디렉토리 / Directory for snapshot files
                      (기본값 / default: outputs/snapshots)
        tag:          식별 태그 (채널명, 페이즈 등) / Identifier tag (channel, phase, etc.)
        logger:       로거 인스턴스 / Logger instance

    Returns:
        저장된 스냅샷 파일 경로 / Path of saved snapshot file

    Example:
        snap = log_snapshot(cfg, snapshot_dir=Path("outputs/snapshots"), tag="Y_phase0")
    """
    if logger is None:
        logger = get_logger()

    if snapshot_dir is None:
        snapshot_dir = Path("outputs/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_snapshot_{tag}_{ts}.json" if tag else f"config_snapshot_{ts}.json"
    snapshot_path = snapshot_dir / filename

    # 환경 정보 수집 / Collect environment info
    torch_version = "N/A"
    cuda_available = False
    cuda_version: Optional[str] = None
    try:
        import torch as _torch
        torch_version = _torch.__version__
        cuda_available = _torch.cuda.is_available()
        cuda_version = _torch.version.cuda if cuda_available else None
    except ImportError:
        pass

    # Git 상태 수집 / Collect git state (non-fatal if git unavailable)
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_dirty = bool(porcelain)
    except Exception:
        pass

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "tag": tag,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
        },
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
        },
        "config": config,
    }

    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

    logger.info("=" * 70)
    logger.info("SNAPSHOT SAVED / 스냅샷 저장 완료")
    logger.info("=" * 70)
    logger.info(f"  Path   : {snapshot_path}")
    logger.info(f"  Tag    : {tag or '(none)'}")
    logger.info(f"  Python : {sys.version.split()[0]}")
    logger.info(f"  Torch  : {torch_version}")
    logger.info(f"  Git    : {git_commit or 'N/A'} (dirty={git_dirty})")
    logger.info("=" * 70)

    return snapshot_path


# 프로젝트 시작 시 자동 설정 (선택사항)
# Auto-setup on import (optional - can be overridden by user)
# _default_setup_done = False

# def _auto_setup():
#     """자동 설정 (한 번만) / Auto-setup (once only)."""
#     global _default_setup_done
#     if not _default_setup_done:
#         try:
#             setup_logging(level="INFO")
#             _default_setup_done = True
#         except Exception:
#             pass
#
# _auto_setup()
