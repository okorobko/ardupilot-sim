#!/usr/bin/env bash
# Launch ArduPilot SITL from drone.yaml configuration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_DIR}/config/drone.yaml"

# Parse config using Python
eval "$(python3 -c "
import yaml, os
with open('${CONFIG_FILE}') as f:
    c = yaml.safe_load(f)
s = c['simulation']
v = c['vehicle']
adir = os.path.expanduser(s['ardupilot_dir'])
vmap = {'copter':'ArduCopter','plane':'ArduPlane','rover':'Rover','sub':'ArduSub'}
vehicle = vmap.get(v['type'], 'ArduCopter')
frame = v.get('frame', 'quad')
loc = f\"{s['home_lat']},{s['home_lon']},{s['home_alt']},{s['home_heading']}\"
port = s.get('mavlink_port', 14550)
speed = s.get('speed_up', 1)
print(f'ARDUPILOT_DIR=\"{adir}\"')
print(f'VEHICLE=\"{vehicle}\"')
print(f'FRAME=\"{frame}\"')
print(f'HOME_LOC=\"{loc}\"')
print(f'MAV_PORT=\"{port}\"')
print(f'SPEED_UP=\"{speed}\"')
")"

SIM_VEHICLE="${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"

if [ ! -f "$SIM_VEHICLE" ]; then
    echo "Error: sim_vehicle.py not found at $SIM_VEHICLE"
    echo "Run scripts/install_ardupilot.sh first."
    exit 1
fi

echo "=== Starting ArduPilot SITL ==="
echo "  Vehicle: $VEHICLE"
echo "  Frame:   $FRAME"
echo "  Home:    $HOME_LOC"
echo "  Port:    $MAV_PORT"
echo "  Speed:   ${SPEED_UP}x"
echo ""

exec "$SIM_VEHICLE" \
    -v "$VEHICLE" \
    --frame "$FRAME" \
    -l "$HOME_LOC" \
    --out "udp:127.0.0.1:${MAV_PORT}" \
    --speedup "$SPEED_UP" \
    --no-rebuild
