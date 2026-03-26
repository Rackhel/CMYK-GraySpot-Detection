# CMYK Printer Project

## PyTorch Installation / PyTorch 설치

PyTorch is installed automatically when building the Docker image.
PyTorch는 Docker 이미지 빌드 시 자동으로 설치됩니다.

If you are running locally without Docker, install PyTorch manually based on your environment.
Docker 없이 로컬에서 실행하는 경우 아래 안내에 따라 직접 설치하세요.

```bash
# macOS (CPU only — CUDA not supported on macOS)
# macOS (CPU만 사용 — macOS는 CUDA 미지원)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Windows / Linux — GPU (CUDA 11.8)
# Windows / Linux — GPU 사용 (CUDA 11.8)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# Windows / Linux — CPU only
# Windows / Linux — CPU만 사용
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Currently the project runs on CPU only. GPU support will be added in Stage 3.
> 현재 프로젝트는 CPU 전용으로 동작합니다. GPU 지원은 Stage 3에서 추가됩니다.

---

## Setup Instructions

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install remaining dependencies (local only, not needed for Docker)
# 나머지 패키지 설치 (로컬 전용, Docker 사용 시 불필요)
pip install -r requirements.txt
```

### Docker Build / 도커 빌드

```bash
# CPU (macOS / GPU 없는 Windows / Linux)
docker build -t cmyk-project .

# GPU (Windows / Linux + CUDA 11.8)
docker build --build-arg TORCH_VERSION=cu118 -t cmyk-project-gpu .
```

### Docker Run / 도커 실행

```bash
# Create a data folder if needed (for storing the dataset)
mkdir -p data

# CPU (remove --rm if you want to keep container)
docker run --rm -v ${PWD}/data:/app/data cmyk-project

# GPU
docker run --rm --gpus all -v ${PWD}/data:/app/data cmyk-project-gpu
```

---

# Jin
# IDE uses VS Code, and extensions include Container Tools, Pretier - Code formatter, Pretier ESLint, Pylance, Python Debugger, Python, Jupyter