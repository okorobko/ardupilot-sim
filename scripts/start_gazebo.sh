#!/usr/bin/env bash
# Launch Gazebo with the drone surveillance world
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARDU_GZ_DIR="$HOME/Documents/Projects/Drone/ardupilot_gazebo"

# Set Gazebo resource paths
export GZ_SIM_SYSTEM_PLUGIN_PATH="${ARDU_GZ_DIR}/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="${ARDU_GZ_DIR}/models:${ARDU_GZ_DIR}/worlds:${PROJECT_DIR}/gazebo/models:${PROJECT_DIR}/gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"

WORLD="${PROJECT_DIR}/gazebo/worlds/drone_surveillance.sdf"

echo "=== Starting Gazebo Harmonic ==="
echo "  World: $WORLD"
echo "  Plugin path: $GZ_SIM_SYSTEM_PLUGIN_PATH"
echo "  Resource path: $GZ_SIM_RESOURCE_PATH"
echo ""

exec gz sim "$WORLD" -v 4
