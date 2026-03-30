FROM python:3.11.5-slim

WORKDIR /app

# 기본값 cpu, GPU 사용 시 cu118 전달 / Default is cpu, pass cu118 for GPU
ARG TORCH_VERSION=cpu

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VERSION}
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN apt-get update && apt-get install -y unzip

CMD ["python", "src/scripts/download_dataset.py"]