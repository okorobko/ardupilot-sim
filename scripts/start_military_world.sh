#!/bin/bash
# Launch the military training Gazebo world
# Usage: conda activate gz_garden && ./scripts/start_military_world.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Set Gazebo resource paths
export GZ_SIM_SYSTEM_PLUGIN_PATH="${HOME}/Documents/Projects/Drone/ardupilot_gazebo/build"
export GZ_SIM_RESOURCE_PATH="${HOME}/Documents/Projects/Drone/ardupilot_gazebo/models:${HOME}/Documents/Projects/Drone/ardupilot_gazebo/worlds:${PROJECT_DIR}/gazebo/models:${PROJECT_DIR}/gazebo/worlds"

echo "=================================================="
echo "  Military Training World (Gazebo)"
echo "  World: military_training.sdf"
echo "  Vehicles: tanks, APCs, artillery, MLRS, trucks, helicopters"
echo "=================================================="
echo ""
echo "Resource path: $GZ_SIM_RESOURCE_PATH"
echo ""

# Run Gazebo server only (GUI must be separate on macOS)
exec gz sim -s "${PROJECT_DIR}/gazebo/worlds/military_training.sdf"
