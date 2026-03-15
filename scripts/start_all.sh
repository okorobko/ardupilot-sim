#!/usr/bin/env bash
# Launch ArduPilot SITL + Flask backend, open browser
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Track child PIDs for cleanup
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

# Activate venv if it exists
VENV="${PROJECT_DIR}/venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

echo "=== ArduPilot Drone Simulator ==="
echo ""

# Start SITL in background
echo "[1/3] Starting ArduPilot SITL..."
bash "$SCRIPT_DIR/start_sitl.sh" &
PIDS+=($!)

# Wait for SITL to be ready (heartbeat on MAVLink port)
echo "[2/3] Waiting for SITL heartbeat..."
for i in $(seq 1 60); do
    if python3 -c "
from pymavlink import mavutil
import sys
try:
    conn = mavutil.mavlink_connection('udp:127.0.0.1:14550', source_system=255)
    msg = conn.wait_heartbeat(timeout=2)
    if msg:
        print('  Heartbeat received!')
        sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null; then
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  Timeout waiting for heartbeat. Check SITL output."
        exit 1
    fi
    sleep 1
done

# Start Flask backend
echo "[3/3] Starting web dashboard..."
cd "$PROJECT_DIR/backend"
python3 app.py &
PIDS+=($!)

sleep 2

# Open browser
echo ""
echo "=========================================="
echo "  Dashboard: http://localhost:5001"
echo "  Press Ctrl+C to stop all"
echo "=========================================="
echo ""

if command -v open &>/dev/null; then
    open "http://localhost:5001"
fi

# Wait for any child to exit
wait
