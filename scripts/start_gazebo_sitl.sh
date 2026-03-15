#!/usr/bin/env bash
# Launch ArduPilot SITL with Gazebo backend (gazebo-iris frame)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/config/drone.yaml"

# Parse home location from config
eval "$(python3 -c "
import yaml, os
with open('${CONFIG_FILE}') as f:
    c = yaml.safe_load(f)
s = c['simulation']
adir = os.path.expanduser(s['ardupilot_dir'])
loc = f\"{s['home_lat']},{s['home_lon']},{s['home_alt']},{s['home_heading']}\"
print(f'ARDUPILOT_DIR=\"{adir}\"')
print(f'HOME_LOC=\"{loc}\"')
")"

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
    --model gazebo-iris \
    -l "$HOME_LOC" \
    --no-rebuild \
    --no-mavproxy
