# CMYK Printer Grayspot Detection Pipeline

**Grayspot Defect Classification System** — Deep learning-based automated detection and classification of printer defects using CMYK channel analysis.

**Grayspot 결함 분류 시스템** — CMYK 채널 분석을 사용한 프린터 결함의 딥러닝 기반 자동 감지 및 분류.

---

## Table of Contents / 목차

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Inference](#inference)
8. [Docker Usage](#docker-usage)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum / 최소 사양

- **Python**: 3.10 or later (3.11.5 recommended)
- **Memory**: 8 GB RAM
- **Storage**: 20 GB (with dataset)
- **Disk I/O**: SSD recommended

### Recommended / 권장 사양

- **Python**: 3.11.5
- **GPU**: NVIDIA CUDA 11.8+ (4GB VRAM minimum)
- **Memory**: 16 GB RAM or more
- **Storage**: 50+ GB SSD

### Supported Platforms / 지원 플랫폼

- ✅ **Windows** (CPU / CUDA)
- ✅ **macOS** (CPU / MPS — Apple Silicon)
- ✅ **Linux** (CPU / CUDA)
- ✅ **Docker** (all platforms)

---

## Installation

### Option 1: Local Installation (Recommended for Development)

#### 1.1 Clone Repository / 저장소 복제

```bash
git clone <your-repo-url>
cd CMYK_MAIN
```

#### 1.2 Create Virtual Environment / 가상 환경 생성

```bash
# Using Python venv / Python venv 사용
python -m venv venv

# Activate virtual environment / 가상 환경 활성화
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

#### 1.3 Install Dependencies / 의존성 설치

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies (CPU)
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### PyTorch Installation by Platform / 플랫폼별 PyTorch 설치

```bash
# macOS — Apple Silicon (MPS)
pip install torch torchvision

# macOS — Intel (CPU only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Windows / Linux — GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Windows / Linux — CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Option 2: Docker Installation (Recommended for Production)

#### 2.1 Build Docker Image / Docker 이미지 빌드

```bash
# CPU version (default) / CPU 버전 (기본값)
docker build -t grayspot:latest .

# GPU version (CUDA 11.8) / GPU 버전 (CUDA 11.8)
docker build --build-arg TORCH_VERSION=cu118 -t grayspot:gpu .
```

#### 2.2 Run Docker Container / Docker 컨테이너 실행

```bash
# CPU
docker run --rm -it \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  grayspot:latest

# GPU (requires nvidia-docker)
docker run --rm -it --gpus all \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  grayspot:gpu
```

---

## Quick Start

### 1. Prepare Data / 데이터 준비

```bash
# Ensure data directory structure exists
mkdir -p data_set/labeled/{C,K,M,Y}
mkdir -p data_set/raw
mkdir -p outputs/logs
mkdir -p outputs/reports
```

### 2. Run Baseline Training / 기준선 훈련 실행

```bash
# All channels
python src/scripts/train.py

# Specific channel
python src/scripts/train.py --channel Y

# With browser auto-open
python src/scripts/train.py --open-browser
```

### 3. Generate Reports / 리포트 생성

Reports are generated automatically after training. To regenerate:

```bash
python src/scripts/train.py --report-only
```

Open `outputs/reports/baseline.html` in your browser.

---

## Project Structure

```
CMYK_MAIN/
├── src/
│   ├── config/              # Configuration management
│   │   ├── config.yaml      # Main configuration file
│   │   ├── config_manager.py
│   │   └── pyproject.toml
│   ├── data/                # Data loading & preprocessing
│   │   ├── dataset.py
│   │   ├── augmentation.py
│   │   └── preprocessing.py
│   ├── models/              # Model architectures
│   │   ├── backbone.py
│   │   ├── classifier.py
│   │   ├── grayspot_model.py
│   │   └── projection_head.py
│   ├── training/            # Training pipeline
│   │   ├── trainer.py
│   │   └── ...
│   ├── evaluation/          # Evaluation metrics
│   │   ├── metrics.py
│   │   ├── confusion.py
│   │   └── evaluator.py
│   ├── inference/           # Inference module
│   │   └── predictor.py
│   ├── reporting/           # Report generation
│   │   └── html_report.py
│   ├── scripts/             # Executable scripts
│   │   ├── train.py
│   │   ├── run_baseline.py
│   │   ├── run_optuna.py
│   │   └── generate_baseline_report.py
│   ├── notebooks/           # Jupyter notebooks
│   │   ├── 01_preprocessing.ipynb
│   │   ├── 02_model_test.ipynb
│   │   ├── 03_training.ipynb
│   │   ├── 04_evaluation.ipynb
│   │   ├── 05_contrastive.ipynb
│   │   └── 06_embedding_viz.ipynb
│   ├── tests/               # Unit tests
│   │   ├── test_training_phase0.py
│   │   ├── test_training_phase2.py
│   │   ├── test_evaluation.py
│   │   └── test_optuna.py
│   ├── utils/               # Utilities
│   │   ├── logger.py
│   │   └── __init__.py
│   └── tuning/              # Hyperparameter tuning
│       └── optuna_tuner.py
├── data_set/                # Data directory (git-ignored)
│   ├── labeled/
│   ├── raw/
│   ├── baseline/
│   ├── models/
│   ├── reports/
│   └── embedding_viz/
├── outputs/                 # Outputs directory
│   ├── logs/
│   ├── reports/
│   │   └── baseline.html
│   ├── baseline/
│   └── optuna/
├── config/                  # Docker config mount point
├── Dockerfile               # Multi-stage Docker image
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE

```

---

## Configuration

### Main Configuration File / 주요 설정 파일

Location: `src/config/config.yaml`

Key sections / 주요 섹션:

```yaml
system:
  device: "auto"              # auto | cuda | mps | cpu
  mixed_precision: false      # Enable AMP training

data:
  channels: ["Y", "M", "C", "K"]
  num_levels: 6               # Defect levels 0-5
  image_size: 128             # Input image size

phase2:
  epochs: 30
  batch_size: 16
  learning_rate: 1.0e-4
  early_stopping:
    enabled: true
    patience: 5

storage:
  data_root: "data_set"
  models_dir: "data_set/models"
  reports_dir: "data_set/reports"
  outputs_dir: "outputs"
```

See `src/config/config.yaml` for complete options / 전체 옵션은 `src/config/config.yaml` 참고.

---

## Training

### Baseline Training (Phase 2 only) / 기준선 훈련 (Phase 2만)

```bash
# Train all channels
python src/scripts/train.py

# Train specific channel
python src/scripts/train.py --channel Y

# With configuration override
python src/scripts/train.py --channel all --epochs 50
```

### Outputs / 출력

- **Models**: `data_set/baseline/best_{channel}.pt`
- **History**: `data_set/baseline/phase2_history_{channel}.csv`
- **Summary**: `data_set/baseline/baseline_summary.json`
- **Report**: `outputs/reports/baseline.html`

---

## Inference

### Using Predictor Class / Predictor 클래스 사용

```python
from src.inference.predictor import GrayspotPredictor
import numpy as np

# Initialize predictor
predictor = GrayspotPredictor()

# Load model for channel Y
predictor.load_model(channel="Y", model_path="data_set/baseline/best_Y.pt")

# Load and preprocess image
from src.data.preprocessing import preprocess_image
image = preprocess_image("path/to/image.jpg")
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Inference
results = predictor.predict(image, channel="Y")
predictions = results["predictions"]
confidences = results["confidences"]
```

### Batch Inference / 배치 추론

```python
# Multi-channel batch prediction
images_dict = {
    "Y": y_images,
    "M": m_images,
    "C": c_images,
    "K": k_images,
}
results = predictor.predict_batch(images_dict)
```

---

## Docker Usage

### Build / 빌드

```bash
# CPU (default)
docker build -t grayspot:latest .

# GPU (CUDA 11.8)
docker build --build-arg TORCH_VERSION=cu118 -t grayspot:gpu .

# Specific Python version
docker build --build-arg PYTHON_VERSION=3.11.5 -t grayspot:py311 .
```

### Run / 실행

```bash
# Interactive shell
docker run --rm -it \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  grayspot:latest bash

# Run training
docker run --rm \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  grayspot:latest \
  python src/scripts/train.py --channel Y

# Run with GPU
docker run --rm --gpus all \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  grayspot:gpu \
  python src/scripts/train.py
```

---

## Troubleshooting

### Issue: CUDA out of memory

```bash
# Reduce batch size in config.yaml
batch_size: 8  # Instead of 16

# Or run with CPU
python src/scripts/train.py --device cpu
```

### Issue: Module not found errors

```bash
# Ensure you're in the project root and have activated venv
cd CMYK_MAIN
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Image preprocessing errors

```bash
# Check image format and ensure proper loading
from src.data.preprocessing import preprocess_image
import cv2

image = cv2.imread("path/to/image.jpg", cv2.IMREAD_COLOR)
if image is None:
    print("Image loading failed!")
else:
    processed = preprocess_image(image)
```

---

## Version History

### v1.0.0 (April 2026)
- ✅ Baseline training pipeline complete
- ✅ Multi-channel CMYK support
- ✅ HTML report generation
- ✅ Docker support (CPU & GPU)
- ✅ Inference module with batch support
- ✅ Project code optimization and cleanup

### v0.7.0 (Previous)
- Initial Docker setup
- Basic training scripts

---

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "Add your feature"`
3. Push to branch: `git push origin feature/your-feature`
4. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation in `doc/`
- Refer to notebook examples in `src/notebooks/`

---

**Last Updated**: April 30, 2026  
**Python Version**: 3.11.5  
**PyTorch Version**: 2.1.0+

# Run evaluation / 평가 실행
docker run --rm \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:latest \
  python -m src.scripts.generate_baseline_report
```

#### 2.4.4 Run Options / 실행 옵션

```bash
# Keep container after execution (don't use --rm) / 실행 후 컨테이너 유지
docker run -d \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  --name cmyk-training \
  cmyk-project:latest

# View logs of running container / 실행 중인 컨테이너의 로그 보기
docker logs -f cmyk-training

# Stop container / 컨테이너 중지
docker stop cmyk-training

# Limit memory and CPU usage / 메모리 및 CPU 사용량 제한
docker run --rm \
  --memory=4g --cpus=2 \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:latest
```

#### 2.4.5 Docker Compose (Optional) / Docker Compose (선택 사항)

Create a `docker-compose.yml` file for easier management:
쉬운 관리를 위해 `docker-compose.yml` 파일을 작성하세요:

```yaml
version: '3.8'

services:
  cmyk-cpu:
    image: cmyk-project:latest
    build:
      context: .
      args:
        TORCH_VERSION: cpu
    volumes:
      - ./data_set:/app/data_set
      - ./outputs:/app/outputs
    working_dir: /app

  cmyk-gpu:
    image: cmyk-project:gpu
    build:
      context: .
      args:
        TORCH_VERSION: cu118
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data_set:/app/data_set
      - ./outputs:/app/outputs
    working_dir: /app
```

Then use:
```bash
# Build and run CPU version / CPU 버전 빌드 및 실행
docker-compose up cmyk-cpu

# Build and run GPU version / GPU 버전 빌드 및 실행
docker-compose up cmyk-gpu
```

---

## 3. Notes / 참고사항

### 3.1 Development Environment / 개발 환경

- OS: macOS (primary) / Windows (secondary) / OS: macOS (주 개발 환경) / Windows (보조 개발 환경)
- IDE: VS Code
- Extensions / 확장 프로그램: Container Tools, Prettier - Code Formatter, Prettier ESLint, Pylance, Python Debugger, Python, Jupyter, Live Preview

> Because the primary laptop is macOS, path installation or folder locations may differ from Windows.

> 주 개발 환경이 macOS이므로 경로 설치 또는 폴더 위치가 Windows와 다를 수 있습니다.
>
> The annotation will be kept as a Korean / English bilingual format for all team members.

> 주석은 모든 팀원을 위해 한국어 / 영어 병기 형식으로 작성됩니다.
>
> If there are awkward parts in the comments, a translator was used — please understand.

> 주석에 어색한 부분이 있다면 번역기를 사용한 것이니 양해 부탁드립니다.