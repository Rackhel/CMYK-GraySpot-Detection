# Multi-stage Dockerfile for CMYK Printer Grayspot Detection Pipeline
# 멀티 스테이지 Dockerfile - CMYK 프린터 Grayspot 탐지 파이프라인

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — Install dependencies
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11.5-slim as builder

# Build arguments / 빌드 인자
# Default is cpu, pass cu118 for GPU / 기본값은 cpu, GPU 사용 시 cu118 전달
ARG TORCH_VERSION=cpu
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VERSION} && \
    pip install --no-cache-dir -r requirements.txt


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — Minimal production image
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11.5-slim

# Add metadata labels / 메타데이터 라벨 추가
LABEL maintainer="CMYK Printer Project Team"
LABEL description="CMYK Printer Grayspot Defect Classification Pipeline"
LABEL version="1.0.0"

# Set working directory / 작업 디렉토리 설정
WORKDIR /app

# Install runtime-only system dependencies (minimal footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopencv-core4.5 \
    libopencv-imgproc4.5 \
    libopencv-imgcodecs4.5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and configuration
COPY src ./src
COPY config ./config

# Create necessary directories for data and outputs
RUN mkdir -p /app/data_set /app/outputs /app/outputs/logs /app/outputs/reports

# Set environment variables for Python optimization and reproducibility
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    TORCH_HOME="/app/.torch" \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4

# Define volumes for data and outputs / 데이터 및 출력용 볼륨 정의
VOLUME ["/app/data_set", "/app/outputs"]

# Expose port for potential web interface
EXPOSE 8000

# Default command: show project info
CMD ["python", "-c", "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"]

# Default command / 기본 명령어
# Users can override this when running the container / 컨테이너 실행 시 오버라이드 가능
# Example: docker run -v ${PWD}/data_set:/app/data_set cmyk-project python -m src.scripts.run_baseline
CMD ["python", "-m", "src.scripts.run_baseline"]