#!/bin/bash
# Build and run Grayspot in Docker

echo "🐳 Building Docker Image..."
docker build -t grayspot:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully!"
echo ""
echo "🚀 Run GUI (Streamlit):"
echo "   docker run --rm -it -p 8501:8501 -v \${PWD}/data_set:/app/data_set -v \${PWD}/outputs:/app/outputs grayspot:latest streamlit run gui/app.py"
echo ""
echo "🚀 Run Training:"
echo "   docker run --rm -it -v \${PWD}/data_set:/app/data_set -v \${PWD}/outputs:/app/outputs grayspot:latest python src/scripts/train.py"
echo ""
echo "Or use docker-compose (optional)."
