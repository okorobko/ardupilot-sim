#!/usr/bin/env bash
# Start the Gazebo camera bridge (gz transport → Flask SocketIO)
# Must run in the gz_garden conda environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Starting Camera Bridge ==="
echo "  Gazebo topic: /camera"
echo "  Backend: http://localhost:5001"
echo ""

exec python3 "$PROJECT_DIR/backend/camera_bridge.py"
