#!/usr/bin/env bash
# Launch ArduPilot SITL with Gazebo backend (gazebo-iris frame)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/config/drone.yaml"

# Parse home location from config — use grep/sed to avoid yaml dep in gz_garden env
_adir=$(grep 'ardupilot_dir' "${CONFIG_FILE}" | sed 's/.*: *//' | tr -d '"' | xargs)
ARDUPILOT_DIR="${_adir/#\~/$HOME}"
_lat=$(grep 'home_lat' "${CONFIG_FILE}" | sed 's/.*: *//')
_lon=$(grep 'home_lon' "${CONFIG_FILE}" | sed 's/.*: *//')
_alt=$(grep 'home_alt' "${CONFIG_FILE}" | sed 's/.*: *//')
_hdg=$(grep 'home_heading' "${CONFIG_FILE}" | sed 's/.*: *//')
HOME_LOC="${_lat},${_lon},${_alt},${_hdg}"

SIM_VEHICLE="${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"

if [ ! -f "$SIM_VEHICLE" ]; then
    echo "Error: sim_vehicle.py not found. Run scripts/install_ardupilot.sh first."
    exit 1
fi

echo "=== Starting ArduPilot SITL (Gazebo mode) ==="
echo "  Frame:  gazebo-iris"
echo "  Home:   $HOME_LOC"
echo ""
echo "  NOTE: Start Gazebo FIRST (scripts/start_gazebo.sh)"
echo ""

exec "$SIM_VEHICLE" \
    -v ArduCopter \
    --frame gazebo-iris \
    --model JSON \
    -l "$HOME_LOC" \
    --no-rebuild \
    --no-mavproxy
