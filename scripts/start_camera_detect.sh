#!/bin/bash
# Launch camera bridge with ML detection enabled
# Usage: conda activate gz_garden && ./scripts/start_camera_detect.sh [model_path]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="${1:-${PROJECT_DIR}/ml/models/mil_vehicle_aerial_v3_full/weights/best.onnx}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model not found at $MODEL_PATH"
    echo "Running camera bridge WITHOUT detection."
    echo "Train a model first: cd ml && python scripts/train.py"
    echo ""
    exec python3 "${PROJECT_DIR}/backend/camera_bridge.py"
else
    echo "Starting camera bridge with detection: $MODEL_PATH"
    exec python3 "${PROJECT_DIR}/backend/camera_bridge.py" --detect "$MODEL_PATH"
fi
