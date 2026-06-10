# syntax=docker/dockerfile:1
FROM python:3.11-slim as base

ARG TORCH_VERSION=cpu
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libx11-6 \
        libxrender1 \
        libxkbcommon0 \
        libxcb1 \
        libx11-xcb1 \
        libxext6 \
        libxi6 \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/${TORCH_VERSION} \
    && python -m pip install --no-cache-dir PyQt6

COPY src ./src
COPY gui ./gui

ENV PYTHONUNBUFFERED=1

CMD ["bash"]