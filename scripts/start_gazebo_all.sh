#!/usr/bin/env bash
# Launch full Gazebo simulation stack
# Order: Gazebo → SITL → ROS2 Bridge → YOLO → Backend
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARDU_GZ_DIR="$HOME/Documents/Projects/Drone/ardupilot_gazebo"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup SIGINT SIGTERM EXIT

# Set Gazebo paths
export GZ_SIM_SYSTEM_PLUGIN_PATH="${ARDU_GZ_DIR}/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="${ARDU_GZ_DIR}/models:${ARDU_GZ_DIR}/worlds:${PROJECT_DIR}/gazebo/models:${PROJECT_DIR}/gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"

# Activate venv for backend
VENV="${PROJECT_DIR}/venv"

echo "=== ArduPilot + Gazebo Simulation Stack ==="
echo ""

# 1. Start Gazebo
echo "[1/4] Starting Gazebo..."
gz sim "${PROJECT_DIR}/gazebo/worlds/drone_surveillance.sdf" -v 2 &
PIDS+=($!)
sleep 10

# 2. Start SITL (gazebo-iris)
echo "[2/4] Starting ArduPilot SITL (gazebo-iris)..."
bash "$SCRIPT_DIR/start_gazebo_sitl.sh" &
PIDS+=($!)
sleep 15

# 3. Start ROS2 bridge
echo "[3/4] Starting ROS2 camera bridge..."
bash "$SCRIPT_DIR/start_gz_bridge.sh" &
PIDS+=($!)
sleep 3

# 4. Start Flask backend
echo "[4/4] Starting web dashboard..."
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi
cd "$PROJECT_DIR/backend"
python3 app.py &
PIDS+=($!)

sleep 2

echo ""
echo "=========================================="
echo "  Gazebo:    Running (check GUI)"
echo "  SITL:      ArduCopter (gazebo-iris)"
echo "  Camera:    /camera → /camera/image_raw"
echo "  Dashboard: http://localhost:5001"
echo ""
echo "  Press Ctrl+C to stop all"
echo "=========================================="

if command -v open &>/dev/null; then
    open "http://localhost:5001"
fi

wait
