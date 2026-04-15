"""
config/README.md

설정 관리 가이드 / Configuration Management Guide

프로젝트의 설정 시스템 사용 방법을 설명합니다.
Explains how to use the project's configuration system.
"""

# Configuration Management Guide / 설정 관리 가이드

## 개요 / Overview

이 프로젝트는 두 가지 핵심 구성 요소로 이루어져 있습니다:
This project consists of two core components:

1. **ConfigManager** - YAML 설정 파일 로드 및 관리 / Load and manage YAML config files
2. **Logger** - 중앙 집중식 로깅 시스템 / Centralized logging system

---

## ConfigManager 사용법 / How to use ConfigManager

### 기본 사용 / Basic Usage

```python
from src.config import get_config

# 설정 로드 / Load configuration
config = get_config()

# 설정값 조회 (닷 표기법) / Get config value using dot notation
learning_rate = config.get("phase2.learning_rate")
model_backbone = config.get("model.backbone")
num_channels = config.get("data.channels")

# 경로 조회 (절대경로로 변환됨) / Get path as Path object (auto-converted to absolute)
models_dir = config.get_path("storage.models_dir")
labeled_dir = config.get_path("storage.labeled_dir")

# 디렉토리 자동 생성 / Auto-create necessary directories
config.create_necessary_directories()
```

### 고급 사용 / Advanced Usage

```python
from src.config import get_config
from pathlib import Path

# 커스텀 경로로 설정 로드 / Load config from custom path
custom_config_path = Path("configs/custom.yaml")
config = get_config(config_path=custom_config_path)

# 설정 검증 / Validate configuration
if config.validate():
    print("Configuration is valid!")
else:
    print("Configuration has errors!")

# 설정값 조회 (기본값 지정 가능) / Get value with default
epochs = config.get("phase2.epochs", default=30)
optimizer = config.get("train.optimizer", default="adamw")

# 딕셔너리처럼 접근 / Dictionary-like access
phase2_config = config["phase2"]
model_config = config["model"]

# 현재 설정 저장 (기록용) / Save current config for record-keeping
config.save_config(output_dir=Path("outputs/logs"))
```

### 디바이스 설정 / Device Configuration

```python
config = get_config()

# 자동 감지 결과 확인 / Check auto-detected device
device = config.get("system.device_name")  # e.g., "cuda:0" or "cpu"
device_count = config.get("system.device_count")

print(f"Using device: {device}")
print(f"Total GPUs available: {device_count}")
```

### 경로 설정 / Path Configuration

모든 경로는 `config.yaml`에서 상대경로로 지정되지만, 자동으로 절대경로로 변환됩니다.
All paths in `config.yaml` are relative but automatically converted to absolute paths.

```yaml
# config.yaml에서 / In config.yaml
storage:
  data_root: "data_set"           # 상대경로 / Relative
  labeled_dir: "data_set/labeled" # 상대경로 / Relative
  models_dir: "data_set/models"   # 상대경로 / Relative
```

```python
# Python 코드에서 / In Python code
config = get_config()
labeled_dir = config.get_path("storage.labeled_dir")
# 반환값: /absolute/path/to/CMYK_MAIN/data_set/labeled
```

---

## Logger 사용법 / How to use Logger

### 기본 설정 / Basic Setup

```python
from src.utils import setup_logging, get_logger
from pathlib import Path

# 프로그램 시작 시 한 번 호출 / Call once at program start
setup_logging(
    log_dir=Path("outputs/logs"),
    level="INFO",
    format_style="detailed",
    console=True,
    file=True
)

# 로거 인스턴스 획득 / Get logger instance
logger = get_logger(__name__)

# 사용 / Use
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### 모듈에서 사용 / Using in modules

```python
# src/training/trainer.py
from src.utils import get_logger

logger = get_logger(__name__)

class CMYKTrainer:
    def train(self):
        logger.info("Training started")
        logger.debug(f"Config: {self.config}")
        logger.warning("Low validation accuracy")
        logger.error(f"Training failed: {error}")
```

### LoggerMixin 사용 / Using LoggerMixin

클래스에 자동으로 logger를 추가하려면:
To automatically add logger to your class:

```python
from src.utils import LoggerMixin

class MyTrainer(LoggerMixin):
    def train(self):
        self.logger.info("Training started")
        self.logger.debug(f"Epoch: {epoch}")
```

### 로그 레벨 설정 / Setting Log Levels

```python
from src.utils import setup_logging

# DEBUG: 상세 정보 / Detailed information
setup_logging(level="DEBUG")

# INFO: 일반 정보 (기본값) / General information (default)
setup_logging(level="INFO")

# WARNING: 경고만 / Warnings only
setup_logging(level="WARNING")

# ERROR: 에러만 / Errors only
setup_logging(level="ERROR")
```

### 포맷 스타일 선택 / Choosing Format Styles

```python
from src.utils import setup_logging

# 상세 포맷 (컬러) / Detailed format with colors
setup_logging(format_style="detailed")
# 출력: [2024-04-09 14:30:45] INFO src.training.trainer - Training started

# 간단한 포맷 / Simple format
setup_logging(format_style="simple")
# 출력: [INFO] Training started

# JSON 포맷 / JSON format (for production)
setup_logging(format_style="json")
# 출력: {"level": "INFO", "message": "Training started", ...}
```

### 편의 함수 / Convenience Functions

```python
from src.utils import log_training_config, log_epoch_summary, get_logger

config = {...}
logger = get_logger()

# 훈련 설정 로깅 / Log training configuration
log_training_config(config, logger=logger)

# 에폭 요약 로깅 / Log epoch summary
log_epoch_summary(
    epoch=1,
    train_loss=0.45,
    val_loss=0.38,
    metrics={"accuracy": 0.92, "f1": 0.89},
    logger=logger
)
```

---

## config.yaml 설정 설명 / config.yaml Settings Explanation

### system / 시스템 설정

```yaml
system:
  project_name: "grayspot"
  device: "auto"        # 자동 감지 (cuda/cpu)
  mixed_precision: false
```

### data / 데이터 설정

```yaml
data:
  channels: ["Y", "M", "C", "K"]
  num_levels: 6
  image_size: 128
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
```

### storage / 저장소 설정

```yaml
storage:
  data_root: "data_set"
  labeled_dir: "data_set/labeled"
  models_dir: "data_set/models"
  reports_dir: "data_set/reports"
  outputs_dir: "outputs"
  logs_dir: "outputs/logs"
```

### model / 모델 설정

```yaml
model:
  backbone: "efficientnet_b0"
  weights: null
  frozen_backbone: false
```

### phase0 / 자기지도 학습 (선택사항)

```yaml
phase0:
  enabled: true
  epochs: 10
  learning_rate: 1.0e-3
  temperature: 0.1
  projection_dim: 128
```

### phase2 / 지도 학습

```yaml
phase2:
  epochs: 30
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  dropout: 0.3
  oversample: true
  early_stopping:
    enabled: true
    patience: 5
```

### train / 훈련 공통 설정

```yaml
train:
  seed: 42
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clip: 1.0
```

### logging / 로깅 설정

```yaml
logging:
  level: "INFO"
  format: "detailed"
  console_output: true
  file_output: true
  log_interval: 50
```

---

## 전체 예제 / Complete Example

```python
from pathlib import Path
from src.config import get_config
from src.utils import setup_logging, get_logger, LoggerMixin

# 1. 로깅 설정 / Setup logging
setup_logging(
    log_dir=Path("outputs/logs"),
    level="INFO",
    format_style="detailed"
)

logger = get_logger(__name__)

# 2. 설정 로드 / Load configuration
config = get_config()
config.validate()
config.create_necessary_directories()

# 3. 설정 정보 출력 / Print config info
logger.info(f"Config loaded: {config}")
logger.info(f"Device: {config.get('system.device_name')}")
logger.info(f"Model: {config.get('model.backbone')}")

# 4. 훈련 클래스 / Training class
class Trainer(LoggerMixin):
    def __init__(self, cfg):
        self.config = cfg
        self.logger.info("Trainer initialized")
    
    def train(self):
        epochs = self.config.get("phase2.epochs")
        lr = self.config.get("phase2.learning_rate")
        
        self.logger.info(f"Starting training for {epochs} epochs with LR={lr}")
        
        for epoch in range(epochs):
            self.logger.debug(f"Epoch {epoch}/{epochs}")
            # ... training logic ...
            self.logger.info(f"Epoch {epoch} completed")

# 5. 사용 / Usage
if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
```

---

## 문제 해결 / Troubleshooting

### "Could not find project root directory" 에러
ProjectPath 감지가 실패했습니다. `CMYK_MAIN` 폴더에 `src/`와 `Dockerfile`이 있는지 확인하세요.

### 경로가 상대경로로 반환됨
`get_path()`를 사용하세요. `get()`은 문자열을 반환하고, `get_path()`는 Path 객체를 반환합니다.

### 로그가 파일에 저장되지 않음
`setup_logging(file=True, log_dir=Path("outputs/logs"))`를 확인하세요.

---

## 더 많은 정보 / More Information

- `src/config/config.yaml` - 전체 설정 파일 / Full configuration file
- `src/config/config_manager.py` - ConfigManager 구현 / ConfigManager implementation
- `src/utils/logger.py` - Logger 구현 / Logger implementation
