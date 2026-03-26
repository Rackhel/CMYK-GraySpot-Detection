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
# Because My laptop is MacOS, the path installation or folder location may be different.
# Since the PC at home is Windows, I will try to develop it in parallel.
# I haven't been able to learn because I haven't secured the data set yet.
# The annotation will be kept as a Korean/British flag for everyone.
# If there are a lot of awkward parts in the comment, I used a lot of translators, so please understand.
# IDE uses VS Code, and extensions include Container Tools, Pretier - Code formatter, Pretier ESLint, Pylance, Python Debugger, Python.

---

## 실행 순서 / Execution Order

### 1. 데이터 다운로드 + 폴더 초기화 / Data Download + Folder Initialization
```bash
python src/scripts/download_dataset.py
python src/scripts/setup_storage.py
```

### 2. 라벨링 전 검증 / Pre-labeling Validation
```bash
python src/tests/test_before_labeling.py
```

### 3. 라벨링 / Labeling
```
data/labeled/{channel}/level_{0~5}/ 폴더에 이미지를 직접 분류하여 넣는다.
Manually classify and place images into data/labeled/{channel}/level_{0~5}/ folders.
```

### 4. 라벨링 후 검증 / Post-labeling Validation
```bash
python src/tests/test_after_labeling.py
```

### 5. 학습 검증 / Training Validation
```bash
# 라벨이 이미 있을 때 Phase 2 직행 / Direct to Phase 2 when labels already exist
python src/tests/test_training.py --skip-phase0

# Swing Architecture 전체 실행 / Full Swing Architecture
python src/tests/test_training.py
```

### 6. 평가 검증 / Evaluation Validation
```bash
# HTML 리포트 포함 / Include HTML report
python src/tests/test_evaluation.py --report

# Swing Cycle 번호 지정 / Specify Swing Cycle number
python src/tests/test_evaluation.py --report --cycle 1
```

### 7. 추론 검증 / Inference Validation
```bash
# 샘플 이미지 자동 선택 / Auto-select sample image
python src/tests/test_inference.py

# 특정 이미지 지정 / Specify image
python src/tests/test_inference.py --image data/images/scan_001.png
```

> 각 단계에서 `[PASS]` 가 전부 출력되면 다음 단계로 진행한다.
> Proceed to the next step only when all items show `[PASS]`.