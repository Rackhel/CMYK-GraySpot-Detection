"""
scripts/example_config_logger_usage.py

설정 및 로거 사용 예제 / Configuration and Logger Usage Example

이 스크립트는 새로운 ConfigManager와 Logger 시스템을 사용하는 방법을 보여줍니다.
This script demonstrates how to use the new ConfigManager and Logger systems.

실행 / Run:
    cd CMYK_MAIN
    python src/scripts/example_config_logger_usage.py
"""

import sys
from pathlib import Path

# CMYK_MAIN 루트와 src/ 를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # CMYK_MAIN/
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from src.config import get_config
from src.utils import (
    get_logger,
    log_epoch_summary,
    log_training_config,
    setup_logging,
)


def main():
    """
    메인 함수 / Main function.
    """

    # ─────────────────────────────────────────────────────────────
    # Step 1: 로깅 설정 / Setup Logging
    # ─────────────────────────────────────────────────────────────
    print("Step 1: Setting up logging...")

    setup_logging(
        log_dir=ROOT_DIR / "outputs" / "logs",
        level="INFO",  # DEBUG, INFO, WARNING, ERROR
        format_style="detailed",  # simple, detailed, json
        console=True,  # 콘솔 출력 / Console output
        file=True,  # 파일 출력 / File output
    )

    # 로거 인스턴스 획득 / Get logger instance
    logger = get_logger(__name__)
    logger.info("=" * 70)
    logger.info("Configuration & Logger Usage Example / 설정 및 로거 사용 예제")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # Step 2: 설정 로드 / Load Configuration
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 2: Loading configuration...")

    try:
        config = get_config()
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return

    # ─────────────────────────────────────────────────────────────
    # Step 3: 설정 검증 / Validate Configuration
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 3: Validating configuration...")

    if config.validate():
        logger.info("✓ Configuration is valid")
    else:
        logger.error("✗ Configuration has validation errors")
        return

    # ─────────────────────────────────────────────────────────────
    # Step 4: 필요한 디렉토리 생성 / Create Necessary Directories
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 4: Creating necessary directories...")

    try:
        config.create_necessary_directories()
        logger.info("✓ Directories created")
    except Exception as e:
        logger.error(f"✗ Failed to create directories: {e}")
        return

    # ─────────────────────────────────────────────────────────────
    # Step 5: 설정값 조회 / Query Configuration Values
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 5: Querying configuration values...")

    # 닷 표기법으로 접근 / Access using dot notation
    channels = config.get("data.channels")
    num_levels = config.get("data.num_levels")
    image_size = config.get("data.image_size")

    logger.info(f"Channels: {channels}")
    logger.info(f"Num Levels: {num_levels}")
    logger.info(f"Image Size: {image_size}x{image_size}")

    # 모델 설정 / Model configuration
    backbone = config.get("model.backbone")
    logger.info(f"Model Backbone: {backbone}")

    # 훈련 설정 / Training configuration
    phase2_epochs = config.get("phase2.epochs")
    phase2_lr = config.get("phase2.learning_rate")
    phase2_batch = config.get("phase2.batch_size")

    logger.info(
        f"Phase2 - Epochs: {phase2_epochs}, LR: {phase2_lr}, Batch: {phase2_batch}"
    )

    # 시스템 설정 / System configuration
    device = config.get("system.device_name")
    logger.info(f"Device: {device}")

    # ─────────────────────────────────────────────────────────────
    # Step 6: 경로 조회 / Query Paths
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 6: Querying paths...")

    # get_path()는 Path 객체를 반환 / get_path() returns Path objects
    try:
        labeled_dir = config.get_path("storage.labeled_dir")
        models_dir = config.get_path("storage.models_dir")
        logs_dir = config.get_path("storage.logs_dir")

        logger.info(f"Labeled Dir: {labeled_dir}")
        logger.info(f"Models Dir: {models_dir}")
        logger.info(f"Logs Dir: {logs_dir}")

        # 경로 확인 / Check paths
        logger.info(f"Labeled Dir exists: {labeled_dir.exists()}")
        logger.info(f"Models Dir exists: {models_dir.exists()}")
        logger.info(f"Logs Dir exists: {logs_dir.exists()}")
    except Exception as e:
        logger.error(f"Error accessing paths: {e}")

    # ─────────────────────────────────────────────────────────────
    # Step 7: 기본값이 있는 설정 조회 / Query with Defaults
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 7: Querying with default values...")

    # 존재하지 않는 설정에 기본값 사용
    custom_value = config.get("custom.setting", default="default_value")
    logger.info(f"Custom setting (with default): {custom_value}")

    # ─────────────────────────────────────────────────────────────
    # Step 8: 설정 저장 / Save Configuration Snapshot
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 8: Saving configuration snapshot...")

    try:
        config.save_config(output_dir=ROOT_DIR / "outputs" / "logs")
        logger.info("✓ Configuration snapshot saved")
    except Exception as e:
        logger.error(f"✗ Failed to save config snapshot: {e}")

    # ─────────────────────────────────────────────────────────────
    # Step 9: 훈련 설정 로깅 / Log Training Configuration
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 9: Logging training configuration...")

    log_training_config(config.config, logger=logger)

    # ─────────────────────────────────────────────────────────────
    # Step 10: 시뮬레이션된 에폭 로깅 / Simulated Epoch Logging
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 10: Simulating epoch logging...")

    # 가상의 훈련 루프 시뮬레이션
    for epoch in range(1, 4):
        train_loss = 0.5 - (epoch * 0.05)
        val_loss = 0.45 - (epoch * 0.04)
        metrics = {
            "accuracy": 0.85 + (epoch * 0.02),
            "precision": 0.84 + (epoch * 0.02),
            "recall": 0.83 + (epoch * 0.02),
            "f1": 0.83 + (epoch * 0.02),
        }

        log_epoch_summary(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics,
            logger=logger,
        )

    # ─────────────────────────────────────────────────────────────
    # Step 11: 다양한 로그 레벨 예제 / Demonstrate Different Log Levels
    # ─────────────────────────────────────────────────────────────
    logger.info("\nStep 11: Demonstrating different log levels...")

    logger.debug("This is a DEBUG message - shows detailed information")
    logger.info("This is an INFO message - shows general information")
    logger.warning("This is a WARNING message - shows potential issues")
    logger.error("This is an ERROR message - shows what went wrong")

    # ─────────────────────────────────────────────────────────────
    # Complete / 완료
    # ─────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("✓ Example completed successfully!")
    logger.info("=" * 70)
    logger.info("\nNext steps / 다음 단계:")
    logger.info("1. Check outputs/logs/ for log files / 로그 파일 확인")
    logger.info("2. Review config snapshots / 설정 스냅샷 검토")
    logger.info(
        "3. Integrate this setup into your training scripts / 훈련 스크립트에 통합"
    )


class ExampleTrainer:
    """
    로거 믹스인을 사용하는 예제 클래스 / Example class using logger mixin.

    LoggerMixin을 상속받으면 자동으로 self.logger를 사용할 수 있습니다.
    """

    from src.utils import LoggerMixin

    class Trainer(LoggerMixin):
        def __init__(self, config):
            self.config = config
            self.logger.info("Trainer initialized with config")

        def train_epoch(self, epoch):
            self.logger.info(f"Training epoch {epoch}...")
            # ... training logic ...
            self.logger.debug(f"Epoch {epoch} completed")


if __name__ == "__main__":
    main()
