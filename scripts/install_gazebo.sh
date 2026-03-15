#!/usr/bin/env bash
# Install Gazebo Harmonic + ROS2 bridge + ardupilot_gazebo plugin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARDU_GZ_DIR="$HOME/Documents/Projects/Drone/ardupilot_gazebo"

echo "=== Gazebo Harmonic Installation ==="

# Step 1: Install Gazebo + ROS2 bridge via conda
echo "[1/3] Installing Gazebo Harmonic + ros-humble-ros-gz..."
conda activate ros2_humble
conda install -c conda-forge -c robostack-staging ros-humble-ros-gz -y

echo "  Verifying gz-sim..."
gz sim --version

# Step 2: Clone ardupilot_gazebo if not present
if [ -d "$ARDU_GZ_DIR" ]; then
    echo "[2/3] ardupilot_gazebo already cloned at $ARDU_GZ_DIR"
else
    echo "[2/3] Cloning ardupilot_gazebo..."
    git clone --depth 1 https://github.com/ArduPilot/ardupilot_gazebo.git "$ARDU_GZ_DIR"
fi

# Step 3: Build ardupilot_gazebo plugin
echo "[3/3] Building ardupilot_gazebo plugin..."
cd "$ARDU_GZ_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
make -j$(sysctl -n hw.ncpu)

echo ""
echo "=== Installation complete ==="
echo ""
echo "Add to your shell profile:"
echo "  export GZ_SIM_SYSTEM_PLUGIN_PATH=$ARDU_GZ_DIR/build"
echo "  export GZ_SIM_RESOURCE_PATH=$ARDU_GZ_DIR/models:$ARDU_GZ_DIR/worlds:$PROJECT_DIR/gazebo/models:$PROJECT_DIR/gazebo/worlds"
