"""
Grayspot -- 로깅 유틸리티 / Logging Utility
utils/logger.py

학습, 평가, 추론 전 과정에서 일관된 로그를 기록한다.
Records consistent logs throughout training, evaluation, and inference.

로그 출력 대상 / Log output targets:
    - 터미널 (콘솔) / Terminal (console)
    - 파일 (data/reports/logs/) / File (data/reports/logs/)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# 전역 logger 인스턴스 캐시 / Global logger instance cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, cfg: dict = None) -> logging.Logger:
    """
    이름 기반으로 logger를 생성하거나 기존 것을 반환한다.
    Creates or returns an existing logger by name.

    같은 name으로 호출하면 동일한 logger 인스턴스를 반환한다 (중복 핸들러 방지).
    Calling with the same name returns the same logger instance (prevents duplicate handlers).

    Args:
        name: logger 이름 / Logger name (예: "train", "evaluate", "inference")
        cfg:  config.yaml 딕셔너리 (없으면 기본값 사용) / config.yaml dict (uses defaults if None)

    Returns:
        logging.Logger

    Example:
        >>> logger = get_logger("train", cfg)
        >>> logger.info("Phase 2 학습 시작")
        >>> logger.warning("클래스 불균형 감지")
        >>> logger.error("모델 로드 실패")
    """
    if name in _loggers:
        return _loggers[name]

    # 로그 레벨 결정 / Determine log level
    level_str = "INFO"
    log_dir   = Path("data/reports/logs")

    if cfg:
        level_str = cfg.get("logging", {}).get("level", "INFO")
        log_dir   = Path(cfg.get("logging", {}).get("log_dir", "data/reports/logs"))

    level = getattr(logging, level_str.upper(), logging.INFO)

    # logger 생성 / Create logger
    logger = logging.getLogger(f"grayspot.{name}")
    logger.setLevel(level)
    logger.propagate = False  # 상위 logger로 전파 방지 / Prevent propagation to root logger

    # 포맷 정의 / Define format
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 콘솔 핸들러 / Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 파일 핸들러 / File handler
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file  = log_dir / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_train_logger(cfg: dict) -> logging.Logger:
    """
    학습 전용 logger를 반환한다. / Returns the training-specific logger.

    Example:
        >>> logger = get_train_logger(cfg)
        >>> logger.info("Phase 0 학습 시작 / Phase 0 training started")
    """
    return get_logger("train", cfg)


def get_eval_logger(cfg: dict) -> logging.Logger:
    """
    평가 전용 logger를 반환한다. / Returns the evaluation-specific logger.

    Example:
        >>> logger = get_eval_logger(cfg)
        >>> logger.info("Phase 3 평가 시작 / Phase 3 evaluation started")
    """
    return get_logger("evaluate", cfg)


def get_inference_logger(cfg: dict) -> logging.Logger:
    """
    추론 전용 logger를 반환한다. / Returns the inference-specific logger.

    Example:
        >>> logger = get_inference_logger(cfg)
        >>> logger.warning("신뢰도 낮음 -- 수동 검수 필요 / Low confidence -- manual review required")
    """
    return get_logger("inference", cfg)


def log_phase_start(logger: logging.Logger, phase: int, channel: str, cfg: dict) -> None:
    """
    Phase 학습 시작 정보를 로그에 기록한다.
    Logs Phase training start information.

    Args:
        logger:  logger 인스턴스 / Logger instance
        phase:   0 또는 2 / 0 or 2
        channel: "Y" | "M" | "C" | "K"
        cfg:     config.yaml 딕셔너리 / config.yaml dictionary
    """
    logger.info("=" * 50)
    logger.info(f"Phase {phase} 학습 시작 / Training started -- Channel: {channel}")
    logger.info("=" * 50)

    if phase == 0:
        p = cfg["phase0"]
        logger.info(f"Epochs: {p['epochs']} | Batch: {p['batch_size']} | LR: {p['learning_rate']} | tau: {p['temperature']}")
    else:
        p = cfg["phase2"]
        logger.info(f"Stage1: {p['stage1_epochs']}ep | Stage2: {p['stage2_epochs']}ep | "
                    f"Batch: {p['batch_size']} | LR: {p['learning_rate']}")


def log_epoch(
    logger: logging.Logger,
    phase: int,
    channel: str,
    epoch: int,
    total_epochs: int,
    metrics: dict,
) -> None:
    """
    에폭 단위 학습 결과를 로그에 기록한다.
    Logs per-epoch training results.

    Args:
        logger:       logger 인스턴스 / Logger instance
        phase:        0 또는 2 / 0 or 2
        channel:      "Y" | "M" | "C" | "K"
        epoch:        현재 에폭 번호 / Current epoch number
        total_epochs: 전체 에폭 수 / Total epochs
        metrics:      기록할 메트릭 딕셔너리 / Metrics dictionary to log
                      Phase 0: {"loss": 0.42, "lr": 1e-3}
                      Phase 2: {"train_loss": 0.3, "train_acc": 0.85, "val_loss": 0.35, "val_acc": 0.82}
    """
    parts = [f"Epoch {epoch:>4}/{total_epochs} [{channel}]"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}: {v:.4f}")
        else:
            parts.append(f"{k}: {v}")
    logger.info(" | ".join(parts))


def log_best_model(logger: logging.Logger, channel: str, val_acc: float, epoch: int) -> None:
    """
    Best 모델 저장 이벤트를 로그에 기록한다.
    Logs best model save event.
    """
    logger.info(f"[{channel}] Best model saved -- Val Acc: {val_acc:.4f} (Epoch {epoch})")


def log_swing_decision(logger: logging.Logger, swing_decision: dict) -> None:
    """
    Phase 3 Swing 피드백 루프 판단 결과를 로그에 기록한다.
    Logs Phase 3 Swing feedback loop decisions.

    Args:
        swing_decision: {"Y": "pass", "M": "phase1", "C": "phase0", "K": "pass"}
    """
    logger.info("Phase 3 Swing 판단 결과 / Swing decision results:")
    for ch, action in swing_decision.items():
        if action == "pass":
            logger.info(f"  [{ch}] PASS -- 목표 달성 / Target met")
        elif action == "phase1":
            logger.warning(f"  [{ch}] --> Phase 1 복귀 / Return -- Level 경계 재검토 필요 / Level boundary review needed")
        elif action == "phase0":
            logger.warning(f"  [{ch}] --> Phase 0 복귀 / Return -- 표현 재학습 필요 / Representation retraining needed")


def log_inference_result(logger: logging.Logger, result: dict) -> None:
    """
    추론 결과를 로그에 기록한다. / Logs inference results.

    Args:
        result: predictor.predict() 반환값 / Return value of predictor.predict()
    """
    logger.info(f"Inference -- {result['image']} ({result['elapsed_ms']}ms)")
    for ch in ["Y", "M", "C", "K"]:
        level  = result.get(f"{ch}_Level", -1)
        conf   = result["confidence"].get(ch, 0.0)
        status = result["status"].get(ch, "")
        logger.info(f"  [{ch}] Level {level} | Conf: {conf:.4f} | Status: {status}")