# Multi-stage Dockerfile for CMYK Printer Project
# 멀티 스테이지 Dockerfile - CMYK 프린터 프로젝트

FROM python:3.11.5-slim

# Build arguments / 빌드 인자
# Default is cpu, pass cu118 for GPU / 기본값은 cpu, GPU 사용 시 cu118 전달
ARG TORCH_VERSION=cpu
ARG DEBIAN_FRONTEND=noninteractive

# Set working directory / 작업 디렉토리 설정
WORKDIR /app

# Add metadata labels / 메타데이터 라벨 추가
LABEL maintainer="CMYK Printer Project Team"
LABEL description="CMYK Printer Defect Classification Pipeline"
LABEL version="0.7.0"

# Install system dependencies completely in one layer / 시스템 의존성을 한 계층에서 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching / 계층 캐싱 최적화를 위해 requirements 먼저 복사
COPY requirements.txt .

# Install PyTorch and Python dependencies in one layer / PyTorch 및 Python 의존성을 한 계층에서 설치
# This layer will be cached unless requirements.txt changes
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VERSION} && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code / 소스 코드 복사
COPY src ./src

# Copy config directory / 설정 디렉토리 복사
COPY config ./config

# Create directories for mounting volumes / 볼륨 마운트용 디렉토리 생성
RUN mkdir -p /app/data_set /app/outputs

# Set environment variables for Python / Python 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/app/src:$PATH"

# Define volumes for data and outputs / 데이터 및 출력용 볼륨 정의
VOLUME ["/app/data_set", "/app/outputs"]

# Default command / 기본 명령어
# Users can override this when running the container / 컨테이너 실행 시 오버라이드 가능
# Example: docker run -v ${PWD}/data_set:/app/data_set cmyk-project python -m src.scripts.run_baseline
CMD ["python", "-m", "src.scripts.run_baseline"]