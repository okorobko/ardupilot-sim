#!/usr/bin/env bash
# Full military simulation stack with live vehicle detection
#
# Launches: Gazebo (military world) → SITL → Backend → Camera bridge (v3 detection) → GUI
#
# Prerequisites:
#   conda activate gz_garden
#   source venv/bin/activate  (or let this script use the venv)
#
# Usage:
#   conda activate gz_garden
#   ./scripts/start_military_detect.sh
#
# Then open: http://localhost:5001
# Fly over the military vehicles — detections appear as coloured boxes on the downward camera.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARDU_GZ_DIR="$HOME/Documents/Projects/Drone/ardupilot_gazebo"

MODEL_PATH="${PROJECT_DIR}/ml/models/mil_vehicle_aerial_v3_full/weights/best.onnx"
WORLD="${PROJECT_DIR}/gazebo/worlds/military_training.sdf"
WORLD_NAME="military_training"
VENV="${PROJECT_DIR}/venv"
ML_VENV="${PROJECT_DIR}/ml/venv_ml"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "Done."
}
trap cleanup SIGINT SIGTERM EXIT

# ── Gazebo environment ────────────────────────────────────────────────────────
export GZ_SIM_SYSTEM_PLUGIN_PATH="${ARDU_GZ_DIR}/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="${ARDU_GZ_DIR}/models:${ARDU_GZ_DIR}/worlds:${PROJECT_DIR}/gazebo/models:${PROJECT_DIR}/gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"

echo ""
echo "============================================================"
echo "  Military Vehicle Detection — Live Simulation"
echo "  World:  military_training.sdf"
echo "  Model:  $(basename $(dirname $MODEL_PATH))/best.onnx  (v3, mAP50=0.773)"
echo "  Dashboard: http://localhost:5001"
echo "============================================================"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Detection model not found: $MODEL_PATH"
    echo "        Run: ml/venv_ml/bin/python3 ml/scripts/export_all.py --weights ml/models/mil_vehicle_aerial_v3_full/weights/best.pt"
    exit 1
fi

if ! command -v gz &>/dev/null; then
    echo "[ERROR] gz not found. Activate the gz_garden conda env first:"
    echo "        conda activate gz_garden"
    exit 1
fi

# ── 1. Gazebo server ──────────────────────────────────────────────────────────
echo "[1/5] Starting Gazebo server (military world)..."
gz sim -s "$WORLD" -v 2 &
PIDS+=($!)
echo "      PID $! — waiting 12s for physics to initialise..."
sleep 12

# ── 2. Unpause ────────────────────────────────────────────────────────────────
echo "[2/5] Unpausing simulation..."
gz service -s "/world/${WORLD_NAME}/control" \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 6000 --req 'pause: false' && echo "      Unpaused." || echo "      [WARN] Unpause failed — try manually in Gazebo GUI."
sleep 1

# ── 3. ArduPilot SITL ─────────────────────────────────────────────────────────
echo "[3/5] Starting ArduPilot SITL (gazebo-iris JSON)..."
bash "$SCRIPT_DIR/start_gazebo_sitl.sh" &
PIDS+=($!)
echo "      PID $! — waiting 18s for GPS lock..."
sleep 18

# ── 4. Web backend ────────────────────────────────────────────────────────────
echo "[4/5] Starting web backend (http://localhost:5001)..."
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi
cd "$PROJECT_DIR/backend"
python3 app.py &
PIDS+=($!)
cd "$PROJECT_DIR"
sleep 3

# ── 5. Camera bridge + detection ─────────────────────────────────────────────
echo "[5/5] Starting camera bridge with v3 detection..."
# Use ml/venv_ml python for onnxruntime (CoreML acceleration)
if [ -f "${ML_VENV}/bin/python3" ]; then
    PYTHON="${ML_VENV}/bin/python3"
else
    PYTHON="python3"
fi

PYTHONPATH="${PROJECT_DIR}/backend" \
    "$PYTHON" "${PROJECT_DIR}/backend/camera_bridge.py" \
    --detect "$MODEL_PATH" &
PIDS+=($!)
sleep 2

# ── Gazebo GUI ────────────────────────────────────────────────────────────────
echo ""
echo "Starting Gazebo GUI (separate process — macOS requirement)..."
gz sim -g &
PIDS+=($!)

# ── Ready ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL SYSTEMS UP"
echo ""
echo "  Dashboard:   http://localhost:5001"
echo "  Controls:    WASD = fly  |  R/F = up/down  |  Q/E = yaw"
echo "  Detection:   coloured boxes on downward camera (top-right)"
echo ""
echo "  Vehicle colours:"
echo "    RED        = tank"
echo "    ORANGE     = apc_ifv"
echo "    YELLOW     = military_truck"
echo "    LIGHT-RED  = sp_artillery"
echo "    MAGENTA    = mlrs"
echo "    CYAN       = helicopter"
echo "    TEAL       = uav"
echo ""
echo "  Tip: take off to 40-60m, fly north/east over the vehicle"
echo "       formations for best detection coverage."
echo ""
echo "  Press Ctrl+C to stop everything."
echo "============================================================"
echo ""

if command -v open &>/dev/null; then
    open "http://localhost:5001"
fi

wait
