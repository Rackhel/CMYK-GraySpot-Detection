# CMYK Printer Project
# CMYK 프린터 프로젝트

---
## 0. Python version

Python version의 경우 GPU 사용을 위해 3.11.5 버전을 사용합니다.
For Python version, use version 3.11.5 for GPU use.

## 1. PyTorch Installation / PyTorch 설치

PyTorch is installed automatically when building the Docker image.
PyTorch는 Docker 이미지 빌드 시 자동으로 설치됩니다.

If you are running locally without Docker, install PyTorch manually based on your environment.
Docker 없이 로컬에서 실행하는 경우 아래 안내에 따라 직접 설치하세요.

### 1.1 Local Installation / 로컬 설치

```bash
# macOS — Apple Silicon (MPS) / macOS — Apple Silicon (MPS 사용)
pip install torch torchvision

# macOS — Intel (CPU only) / macOS — Intel (CPU만 사용)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Windows / Linux — GPU (CUDA 11.8)
# Windows / Linux — GPU 사용 (CUDA 11.8)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# Windows / Linux — CPU only / Windows / Linux — CPU만 사용
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 2. Setup Instructions / 설치 방법

### 2.1 Clone Repository / 저장소 복제

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2.2 Local Dependency Installation / 로컬 패키지 설치

```bash
# Install remaining dependencies (local only, not needed for Docker)
# 나머지 패키지 설치 (로컬 전용, Docker 사용 시 불필요)
pip install -r requirements.txt
```

### 2.3 Docker Build / 도커 빌드

The Docker image uses an optimized layer strategy for efficient caching and minimal image size.
Docker 이미지는 효율적인 캐싱과 최소 이미지 크기를 위해 최적화된 계층 전략을 사용합니다.

#### 2.3.1 Build for CPU / CPU용 빌드

```bash
# Standard build (CPU, default TORCH_VERSION)
# 표준 빌드 (CPU, 기본 TORCH_VERSION)
docker build -t cmyk-project:latest .

# Alternative: specify version explicitly / 대안: 버전을 명시적으로 지정
docker build --build-arg TORCH_VERSION=cpu -t cmyk-project:cpu .
```

#### 2.3.2 Build for GPU / GPU용 빌드

```bash
# Build with CUDA 11.8 support (Windows / Linux)
# CUDA 11.8 지원 빌드 (Windows / Linux)
docker build --build-arg TORCH_VERSION=cu118 -t cmyk-project:gpu .

# Build with latest CUDA support / 최신 CUDA 지원 빌드
docker build --build-arg TORCH_VERSION=cu121 -t cmyk-project:gpu-latest .
```

#### 2.3.3 Build Options / 빌드 옵션

```bash
# Build with progress display / 진행 상황을 표시하며 빌드
docker build --progress=plain -t cmyk-project:latest .

# Build without cache (force full rebuild) / 캐시 없이 빌드 (전체 재빌드)
docker build --no-cache -t cmyk-project:latest .

# Build with label / 라벨과 함께 빌드
docker build -t cmyk-project:v0.1.0 .
```

---

### 2.4 Docker Run / 도커 실행

The container includes pre-configured volumes for data_set and outputs directories.
컨테이너는 data_set 및 outputs 디렉토리에 대해 미리 구성된 볼륨을 포함합니다.

#### 2.4.1 Preparation / 준비

```bash
# Create data directories on host machine / 호스트 머신에 데이터 디렉토리 생성
mkdir -p data_set outputs
```

#### 2.4.2 Run Default Command (Baseline Training) / 기본 명령어 실행 (기준선 훈련)

```bash
# CPU: Run baseline training
# CPU: 기준선 훈련 실행
docker run --rm \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:latest

# GPU: Run baseline training with GPU support
# GPU: GPU 지원으로 기준선 훈련 실행
docker run --rm --gpus all \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:gpu
```

#### 2.4.3 Run Custom Commands / 사용자 정의 명령어 실행

```bash
# Run specific channel training (e.g., Channel C)
# 특정 채널 훈련 실행 (예: 채널 C)
docker run --rm \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:latest \
  python -m src.scripts.run_baseline --channel C

# Run all channels / 모든 채널 실행
docker run --rm \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:latest \
  python -m src.scripts.run_baseline --channel all

# Interactive shell / 대화형 셸
docker run --rm -it \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  cmyk-project:latest \
  /bin/bash

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