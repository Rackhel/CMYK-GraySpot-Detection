FROM python:3.10-slim

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY gui ./gui

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# Default: Starts a standard terminal session instead of a specific script
CMD ["/bin/bash"]
