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

```bash
# CPU (macOS / Windows / Linux without GPU)
# CPU (macOS / GPU 없는 Windows / Linux)
docker build -t cmyk-project .

# GPU (Windows / Linux + CUDA 11.8)
# GPU (Windows / Linux + CUDA 11.8)
docker build --build-arg TORCH_VERSION=cu118 -t cmyk-project-gpu .
```

### 2.4 Docker Run / 도커 실행

```bash
# Create a data_set folder if needed (for storing the data_set)
# 데이터셋 저장을 위한 data_set 폴더 생성
mkdir -p data_set

# CPU (remove --rm if you want to keep container)
# CPU (컨테이너를 유지하려면 --rm 제거)
docker run --rm -v ${PWD}/data_set:/app/data_set cmyk-project

# GPU
docker run --rm --gpus all -v ${PWD}/data_set:/app/data_set cmyk-project-gpu
```

---

## 3. Notes / 참고사항

### 3.1 Development Environment / 개발 환경

- OS: macOS (primary) / Windows (secondary) / OS: macOS (주 개발 환경) / Windows (보조 개발 환경)
- IDE: VS Code
- Extensions / 확장 프로그램: Container Tools, Prettier - Code Formatter, Prettier ESLint, Pylance, Python Debugger, Python, Jupyter

> Because the primary laptop is macOS, path installation or folder locations may differ from Windows.
> 주 개발 환경이 macOS이므로 경로 설치 또는 폴더 위치가 Windows와 다를 수 있습니다.
>
> The annotation will be kept as a Korean / English bilingual format for all team members.
> 주석은 모든 팀원을 위해 한국어 / 영어 병기 형식으로 작성됩니다.
>
> If there are awkward parts in the comments, a translator was used — please understand.
> 주석에 어색한 부분이 있다면 번역기를 사용한 것이니 양해 부탁드립니다.