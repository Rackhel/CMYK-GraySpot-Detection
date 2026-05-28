#!/bin/bash
# Run the CMYK PyQt6 GUI in Docker with host display forwarding.
# This helper is intended for Linux/X11-style environments.

CONTAINER_NAME="grayspot-gui"

echo "Starting CMYK PyQt6 GUI..."
echo "Container: $CONTAINER_NAME"

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${PWD}/data_set:/app/data_set" \
  -v "${PWD}/outputs:/app/outputs" \
  -v "${PWD}/src/config:/app/src/config" \
  grayspot:latest \
  python -m gui.main

echo "Container stopped."
