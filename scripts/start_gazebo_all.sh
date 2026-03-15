#!/usr/bin/env bash
# Launch full Gazebo simulation stack
# Order: Gazebo server → unpause → SITL → Backend → GUI
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

VENV="${PROJECT_DIR}/venv"

echo "=== ArduPilot + Gazebo Simulation Stack ==="
echo ""

# 1. Start Gazebo server (macOS requires -s for server-only)
echo "[1/5] Starting Gazebo server..."
gz sim -s "${PROJECT_DIR}/gazebo/worlds/drone_surveillance.sdf" -v 2 &
PIDS+=($!)
sleep 10

# 2. Unpause simulation
echo "[2/5] Unpausing simulation..."
gz service -s /world/drone_surveillance/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 5000 --req 'pause: false'
sleep 2

# 3. Start SITL with JSON model (matches ArduPilotPlugin protocol)
echo "[3/5] Starting ArduPilot SITL (JSON model)..."
bash "$SCRIPT_DIR/start_gazebo_sitl.sh" &
PIDS+=($!)
sleep 15

# 4. Start Flask backend
echo "[4/5] Starting web dashboard..."
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi
cd "$PROJECT_DIR/backend"
python3 app.py &
PIDS+=($!)
sleep 2

# 5. Launch Gazebo GUI (macOS requires separate process)
echo "[5/5] Launching Gazebo GUI..."
gz sim -g &
PIDS+=($!)

echo ""
echo "=========================================="
echo "  Gazebo:    Running (server + GUI)"
echo "  SITL:      ArduCopter (JSON model)"
echo "  Camera:    /camera (Gazebo transport)"
echo "  Dashboard: http://localhost:5001"
echo ""
echo "  Press Ctrl+C to stop all"
echo "=========================================="

if command -v open &>/dev/null; then
    open "http://localhost:5001"
fi

wait
