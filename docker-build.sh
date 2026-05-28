#!/bin/bash
# Build and run Grayspot in Docker

echo "🐳 Building Docker Image..."
docker build --build-arg TORCH_VERSION=cpu -t grayspot:latest .
docker build --build-arg TORCH_VERSION=cu118 -t grayspot:gpu .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully!"
echo ""
echo "🚀 Run training:" 
echo "   docker run --rm -it -v \\${PWD}/data_set:/app/data_set -v \\${PWD}/outputs:/app/outputs grayspot:latest python src/scripts/train.py"
echo ""
echo "🚀 Run GPU training:"
echo "   docker run --rm --gpus all -v \\${PWD}/data_set:/app/data_set -v \\${PWD}/outputs:/app/outputs grayspot:gpu python src/scripts/train.py"
echo ""
echo "🚀 Open an interactive shell:" 
echo "   docker run --rm -it -v \\${PWD}/data_set:/app/data_set -v \\${PWD}/outputs:/app/outputs grayspot:latest bash"
echo ""
echo "🚀 Run GUI (Linux/X11 only):"
echo "   docker run --rm -it -e DISPLAY=\\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v \\${PWD}/data_set:/app/data_set -v \\${PWD}/outputs:/app/outputs grayspot:latest python -m gui.main"
echo ""
echo "Or use docker-compose (optional)."
