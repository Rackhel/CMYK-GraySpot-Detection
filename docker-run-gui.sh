#!/bin/bash
# Run Grayspot GUI in Docker with Streamlit

CONTAINER_NAME="grayspot-gui"
PORT=8501

echo "🚀 Starting Grayspot GUI..."
echo "   Container: $CONTAINER_NAME"
echo "   Port: http://localhost:$PORT"
echo ""

docker run --rm \
  --name "$CONTAINER_NAME" \
  -p $PORT:8501 \
  -v "${PWD}/data_set:/app/data_set" \
  -v "${PWD}/outputs:/app/outputs" \
  -v "${PWD}/src/config:/app/src/config" \
  grayspot:latest \
  streamlit run gui/app.py --server.port=$PORT --server.address=0.0.0.0

echo ""
echo "🛑 Container stopped."
