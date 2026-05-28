#!/bin/bash
# Run the CMYK PyQt6 GUI in Docker with host display forwarding.
# This helper is intended for Linux/X11 environments only.

CONTAINER_NAME="grayspot-gui"

echo "Starting CMYK PyQt6 GUI..."
echo "Container: $CONTAINER_NAME"

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${PWD}/data_set:/app/data_set" \
  -v "${PWD}/outputs:/app/outputs" \
  -v "${PWD}/src:/app/src" \
  -v "${PWD}/gui:/app/gui" \
  grayspot:latest \
  python -m gui.main

echo "Container stopped."
