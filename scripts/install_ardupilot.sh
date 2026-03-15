#!/usr/bin/env bash
# Install ArduPilot SITL on macOS (Apple Silicon native)
set -euo pipefail

ARDUPILOT_DIR="${HOME}/ardupilot"

echo "=== ArduPilot SITL Installation for macOS ==="

# Step 1: Install Homebrew dependencies
echo "[1/5] Installing Homebrew dependencies..."
brew install --quiet gcc-arm-none-eabi ccache gawk cmake python@3.11 2>/dev/null || true

# Step 2: Clone ArduPilot
if [ -d "${ARDUPILOT_DIR}" ]; then
    echo "[2/5] ArduPilot already cloned at ${ARDUPILOT_DIR}, pulling latest..."
    cd "${ARDUPILOT_DIR}"
    git pull
    git submodule update --init --recursive
else
    echo "[2/5] Cloning ArduPilot..."
    git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git "${ARDUPILOT_DIR}"
fi

cd "${ARDUPILOT_DIR}"

# Step 3: Install prerequisites
echo "[3/5] Installing ArduPilot prerequisites..."
Tools/environment_install/install-prereqs-mac.sh -y

# Reload profile to pick up PATH changes
export PATH="${ARDUPILOT_DIR}/Tools/autotest:${PATH}"

# Step 4: Build SITL for copter
echo "[4/5] Building SITL (copter)..."
./waf configure --board sitl
./waf copter

# Step 5: Verify
echo "[5/5] Verifying installation..."
if [ -f "build/sitl/bin/arducopter" ]; then
    echo "✓ ArduPilot SITL built successfully!"
    echo "  Binary: ${ARDUPILOT_DIR}/build/sitl/bin/arducopter"
    echo "  sim_vehicle.py: ${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"
else
    echo "✗ Build failed — arducopter binary not found."
    exit 1
fi

echo ""
echo "=== Installation complete ==="
echo "You can test with: ${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py -v ArduCopter -w"
