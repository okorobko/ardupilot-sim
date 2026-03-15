# ArduPilot Drone Simulator - Project Context

## Overview

ArduPilot-based quadcopter simulator with a web dashboard featuring 3D visualization, interactive map, and flight controls. Designed for macOS Apple Silicon with native SITL (no Docker). Supports two modes: standalone SITL and Gazebo Garden integration with dual cameras and a vehicle-populated world.

**Platform:** macOS Apple Silicon (M1/M2/M3/M4)
**Drone type:** Configurable via drone.yaml (default: quadcopter)
**Autopilot:** ArduPilot (not PX4)

## Architecture

### Standalone Mode (SITL only)

```
┌─────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────┐
│  ArduPilot SITL     │    │  Python Backend           │    │  Web Frontend          │
│  (sim_vehicle.py)   │    │  (Flask + SocketIO)       │    │  (Single HTML file)    │
│                     │    │                           │    │                        │
│  TCP:5760 (SERIAL0) ├───►│  MAVLinkBridge thread     ├───►│  Three.js r128 3D      │
│                     │    │  pymavlink                │ WS │  Leaflet map (OSM)     │
│  ArduCopter SITL    │    │  Port 5001                │    │  Console + telemetry   │
└─────────────────────┘    └──────────────────────────┘    └────────────────────────┘
```

### Gazebo Mode

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Gazebo Garden   │    │  ArduPilot SITL   │    │  Python Backend  │    │  Web Frontend    │
│  (gz-sim7)       │    │  (JSON model)     │    │  (Flask+SocketIO)│    │  (Three.js+map)  │
│                  │    │                   │    │                  │    │                  │
│  Physics world,  │JSON│  Receives sensors,│TCP │  MAVLink bridge, │ WS │  3D spectator +  │
│  15 vehicles,    ├───►│  sends servos via  ├───►│  camera relay,   ├───►│  vehicle boxes,  │
│  2 cameras,      │UDP │  ArduPilotPlugin  │5760│  keyboard fly    │    │  dual cam views, │
│  ArduPilotPlugin │    │  --model JSON     │    │  handler         │    │  keyboard ctrl   │
└────────┬─────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
         │
         │ gz topic (/camera, /chase_cam)
         ▼
┌──────────────────┐
│  Camera Bridge   │
│  camera_bridge.py│
│                  │
│  gz topic → RGB  │
│  → JPEG → base64 │
│  → SocketIO      │
└──────────────────┘
```

**Key choice:** pymavlink (not MAVSDK) -- ArduPilot native, no gRPC, no separate server process.
**Key choice:** TCP:5760 (SITL SERIAL0) -- not UDP:14550 (which requires MAVProxy --out).
**Key choice:** State-based command verification -- not COMMAND_ACK (avoids recv_match thread conflicts).
**Key choice:** JSON model protocol for Gazebo -- not old binary gazebo-iris model format.
**Key choice:** gz-sim7 via conda (gz_garden env) -- native Gazebo packages unavailable on macOS.

## Directory Structure

```
ardupilot-sim/
├── backend/
│   ├── app.py                    # Flask+SocketIO server, routes, command handler
│   ├── mavlink_bridge.py         # MAVLink reader thread, telemetry, commands, demo_roundtrip
│   ├── camera_bridge.py          # Gazebo camera → JPEG → SocketIO bridge (dual cam)
│   ├── config_loader.py          # YAML config with defaults, vehicle type mapping
│   └── requirements.txt          # flask, flask-socketio, pymavlink, pyyaml
├── config/
│   └── drone.yaml                # Vehicle/simulation/visualization config
├── frontend/
│   └── templates/
│       └── index.html            # Single-file dashboard (Three.js + Leaflet + controls)
├── gazebo/
│   ├── models/
│   │   └── iris_with_camera/
│   │       ├── model.config      # Gazebo model metadata
│   │       └── model.sdf         # Iris + downward cam (640x480) + chase cam (800x600)
│   └── worlds/
│       └── drone_surveillance.sdf # World: roads, 15 vehicles, GPS origin (Kyiv)
├── scripts/
│   ├── install_ardupilot.sh      # ArduPilot SITL macOS installation
│   ├── install_gazebo.sh         # Gazebo Garden + ardupilot_gazebo conda install
│   ├── start_sitl.sh             # SITL launcher (parses drone.yaml)
│   ├── start_all.sh              # Combined SITL + backend launcher (standalone mode)
│   ├── start_gazebo.sh           # Gazebo world launcher (gz sim)
│   ├── start_gazebo_sitl.sh      # SITL in Gazebo mode (--model JSON --frame gazebo-iris)
│   ├── start_gazebo_all.sh       # Full Gazebo stack: server → unpause → SITL → backend → GUI
│   ├── start_camera_bridge.sh    # Camera bridge launcher (requires gz_garden conda)
│   ├── start_gz_bridge.sh        # ROS2 gz-bridge for camera topics (alternative)
│   ├── test_all.py               # 51-test comprehensive suite
│   ├── test_ui.py                # 44-test UI structure verification
│   ├── test_map_coords.py        # Selenium browser coordinate test
│   └── test_roundtrip.py         # Standalone roundtrip demo test
├── venv/                         # Python virtual environment (standalone mode)
├── PLAN.md                       # Original implementation plan
├── CLAUDE.md                     # This file
└── README.md                     # Full project documentation
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Flight Simulator | ArduPilot SITL (native macOS) | Software-in-the-loop simulation |
| Physics Simulation | Gazebo Garden (gz-sim7, conda) | 3D world, cameras, vehicle physics |
| Gazebo Plugin | ardupilot_gazebo (ArduPilotPlugin) | SITL <-> Gazebo bridge (JSON protocol) |
| Control API | pymavlink | MAVLink message read/write |
| Backend | Flask + Flask-SocketIO | Web server, WebSocket bridge |
| Camera Bridge | camera_bridge.py (cv2 + gz CLI) | Gazebo camera topics -> JPEG -> SocketIO |
| 3D Rendering | Three.js r128 | Drone model, ground, trail, vehicle boxes |
| Map | Leaflet 1.9.4 + OSM tiles | GPS position tracking |
| WebSocket | Socket.IO 4.5.4 | Real-time telemetry + camera frames to browser |
| Config | PyYAML | drone.yaml loading |

## Common Commands

### Standalone Mode (SITL only)

```bash
# All-in-one:
./scripts/start_all.sh

# Manual (2 terminals):
./scripts/start_sitl.sh          # Terminal 1: SITL
source venv/bin/activate         # Terminal 2: Backend
cd backend && python3 app.py
```

### Gazebo Mode

```bash
# All-in-one:
conda activate gz_garden
./scripts/start_gazebo_all.sh

# Manual (4 terminals):
conda activate gz_garden
./scripts/start_gazebo.sh        # Terminal 1: Gazebo server

# Terminal 2: Unpause + SITL
gz service -s /world/drone_surveillance/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 5000 --req 'pause: false'
./scripts/start_gazebo_sitl.sh

source venv/bin/activate         # Terminal 3: Backend
cd backend && python3 app.py

conda activate gz_garden         # Terminal 4: Camera bridge (optional)
./scripts/start_camera_bridge.sh

# Gazebo GUI (optional, must be separate process on macOS):
conda activate gz_garden
gz sim -g
```

### Install ArduPilot (first time)

```bash
./scripts/install_ardupilot.sh
```

### Install Gazebo (first time)

```bash
conda create -n gz_garden python=3.11
conda activate gz_garden
./scripts/install_gazebo.sh
pip install opencv-python python-socketio[client]
```

### Run Tests

```bash
# Full suite (51 tests, ~5 min with demo)
python3 scripts/test_all.py

# Quick (skip roundtrip demo)
python3 scripts/test_all.py --quick

# UI tests (44 tests)
python3 scripts/test_ui.py

# Browser coordinate test
python3 scripts/test_map_coords.py

# Standalone roundtrip demo
python3 scripts/test_roundtrip.py
```

### Setup venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install python-socketio[client]   # for tests
```

## Gazebo Environment Setup

The Gazebo integration requires the `gz_garden` conda environment:

```bash
conda create -n gz_garden python=3.11
conda activate gz_garden
conda install -c conda-forge -c robostack-staging ros-humble-ros-gz
pip install opencv-python python-socketio[client]
```

Required environment variables (set in shell profile or scripts set them automatically):

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/Documents/Projects/Drone/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$HOME/Documents/Projects/Drone/ardupilot_gazebo/models:$HOME/Documents/Projects/Drone/ardupilot_gazebo/worlds:$PROJECT_DIR/gazebo/models:$PROJECT_DIR/gazebo/worlds
```

## Gazebo World Details

### drone_surveillance.sdf

- **SDF version**: 1.9 (Gazebo Garden compatible)
- **Physics**: 1ms step (`max_step_size=0.001`), real-time factor 1.0
- **GPS origin**: 50.450001, 30.523333 (Kyiv) -- matches drone.yaml
- **System plugins**: Physics, Sensors (ogre2), UserCommands, SceneBroadcaster, IMU, NavSat
- **Roads**: E-W at y=20, N-S at x=30, 8m wide with yellow center lines
- **15 vehicles** (static box models):
  - E-W road: red sedan, white sedan, blue sedan, black SUV, silver SUV, green car, yellow car
  - N-S road: red sedan, white SUV, blue car, orange truck
  - Parking lot: red car, white car, dark SUV, blue truck

### iris_with_camera Model

- Includes `iris_with_ardupilot` from ardupilot_gazebo (provides ArduPilotPlugin)
- **Downward camera**: 640x480 RGB, 30Hz, FOV 60deg, topic `/camera`
- **Chase camera**: 800x600 RGB, 15Hz, FOV 74deg, topic `/chase_cam`, mounted 3m behind + 2m above

### SITL Configuration for Gazebo

The SITL must use `--frame gazebo-iris --model JSON`:
- `gazebo-iris` tells ArduPilot to use Gazebo-compatible physics parameters
- `--model JSON` selects the JSON communication protocol (not old binary format)
- `--no-mavproxy` is used since the backend connects directly to TCP:5760

## Camera Bridge

`backend/camera_bridge.py` captures frames from Gazebo transport and sends them to the Flask backend:

1. Runs `gz topic -e -t /camera -n 1` to capture one raw frame
2. Parses width/height/data from the text output
3. Decodes RGB bytes, converts to BGR via OpenCV
4. Encodes as JPEG (quality 55), base64-encodes
5. Sends via SocketIO `camera_frame` event

Two streams run in parallel:
- `/camera` -> `camera_frame` at 5fps (downward view, main)
- `/chase_cam` -> `chase_frame` at 3fps (third-person view, overlay)

The Flask backend relays these events to all browser clients.

## Keyboard Flight Controls

Available when drone is armed and in GUIDED mode:

| Key | Action | Value |
|-----|--------|-------|
| W / Up Arrow | Fly forward | +3 m/s body-frame X |
| S / Down Arrow | Fly backward | -3 m/s body-frame X |
| A / Left Arrow | Strafe left | -3 m/s body-frame Y |
| D / Right Arrow | Strafe right | +3 m/s body-frame Y |
| R | Ascend | -3 m/s NED Z (up) |
| F | Descend | +3 m/s NED Z (down) |
| Q | Yaw left | -0.5 rad/s |
| E | Yaw right | +0.5 rad/s |

- Commands sent at 10Hz while keys held
- Body-to-NED conversion done in `mavlink_bridge.py` using current heading
- Keys highlight blue in the UI when pressed
- Releasing all keys sends zero-velocity to stop

## Surveillance Route Demo

The `demo_roundtrip` flies a surveillance route over the Gazebo vehicles:

1. Takeoff to 25m (good altitude for camera view)
2. WP1: E-W road west end (-20m east, +20m north) -- parked cars below
3. WP2: E-W road east end (+40m east, +20m north) -- along the road
4. WP3: N-S road south (+30m east, -30m north) -- more vehicles
5. WP4: Parking lot (-40m east, -22m north) -- clustered cars
6. Land at parking lot, wait 25 seconds
7. Takeoff to 35m, fly back to home
8. Land

## MAVLink Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5760/tcp | MAVLink | SITL SERIAL0 -- backend connects here |
| 14550/udp | MAVLink | MAVProxy forwarding (--out in start_sitl.sh) |

The backend connects to TCP:5760 (configured in `drone.yaml` as `mavlink_port`). The UDP:14550 output from start_sitl.sh is available for QGroundControl or other tools.

## Development Guidelines

1. **Single HTML file**: The frontend is one file (`frontend/templates/index.html`). No build step, no NPM. CDN-hosted libraries only.
2. **Three.js r128**: Do not upgrade past r128. Version r160+ removed `examples/js/` scripts. Upgrading requires switching to ES modules and a bundler.
3. **pymavlink threading**: The MAVLinkBridge reader thread owns `conn.recv_match`. Never call `recv_match` from command handlers -- use state-based polling instead.
4. **Config-driven**: Vehicle type, frame, home position, and visualization parameters all come from `drone.yaml`. Scripts parse this file for SITL launch parameters.
5. **Test before committing**: Run `python3 scripts/test_all.py --quick` for fast validation. Full suite includes a 3-5 minute roundtrip demo.
6. **Auto-arm on takeoff**: The takeoff command in app.py automatically sets GUIDED mode and arms if needed.
7. **Demo safety**: `demo_roundtrip` wraps everything in try/except and sets LAND mode on any error.
8. **Gazebo world must match Three.js scene**: Vehicle positions in `drone_surveillance.sdf` must stay in sync with `buildSceneVehicles()` in index.html. Three.js Z is negated from Gazebo Y.
9. **macOS Gazebo**: Always run server (`gz sim -s`) and GUI (`gz sim -g`) as separate processes. Combined mode causes OGRE-Next crashes on macOS.
10. **JSON model protocol**: When adding Gazebo support, always use `--model JSON` with `--frame gazebo-iris`. The old binary protocol is deprecated.

## ArduCopter Mode Numbers

Key modes used by the system:

| Number | Mode | Usage |
|--------|------|-------|
| 0 | STABILIZE | Manual flight |
| 2 | ALT_HOLD | Altitude hold |
| 4 | GUIDED | GPS waypoint commands, takeoff, keyboard flight |
| 5 | LOITER | Position hold |
| 6 | RTL | Return to launch |
| 9 | LAND | Auto landing |

Mode mapping is in `mavlink_bridge.py` (`COPTER_MODES` dict).

## Troubleshooting

### "MAVLink bridge connecting..." hangs

SITL is not running or not on TCP:5760. Start SITL first:
```bash
./scripts/start_sitl.sh
```

### Arm fails / "Arm timeout"

- SITL needs GPS lock first (wait for "EKF2 IMU0 is using GPS" in SITL output)
- Must be in GUIDED mode to arm via MAVLink
- Check that no pre-arm checks are failing in SITL console

### Map tiles not loading

The map uses OpenStreetMap tiles. Requires internet connection. If tiles appear gray, try resizing the browser window -- this triggers `map.invalidateSize()`.

### 3D scene blank / WebGL error

The 3D panel uses WebGL via Three.js. If it fails, the console will show "3D init error" but map and telemetry continue working. Check browser WebGL support.

### Tests fail with "python-socketio not installed"

```bash
source venv/bin/activate
pip install python-socketio[client]
```

### SITL build fails on Apple Silicon

- Do not use Docker (Rosetta 2 Unix socket issues)
- Ensure Homebrew deps are installed: `brew install gcc-arm-none-eabi ccache gawk cmake`
- Run ArduPilot's prerequisites: `cd ~/ardupilot && Tools/environment_install/install-prereqs-mac.sh -y`

### Gazebo: "No plugin found" / ArduPilotPlugin missing

Ensure environment variables are set:
```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/Documents/Projects/Drone/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$HOME/Documents/Projects/Drone/ardupilot_gazebo/models:$HOME/Documents/Projects/Drone/ardupilot_gazebo/worlds
```
Or use the start scripts which set these automatically.

### Gazebo: Server crashes on macOS

Do NOT run `gz sim <world>` (combined mode). Use separate server and GUI:
```bash
gz sim -s <world>   # Terminal 1: server only
gz sim -g           # Terminal 2: GUI only
```

### Gazebo: Simulation paused / drone does not move

Gazebo starts paused by default. Unpause via:
```bash
gz service -s /world/drone_surveillance/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 5000 --req 'pause: false'
```

### Camera bridge: no frames

- Ensure Gazebo is running and simulation is unpaused
- Ensure you are in the `gz_garden` conda environment
- Check that camera topics are publishing: `gz topic -l | grep cam`
- Check that the backend is running on port 5001

### Keyboard controls not working

- Drone must be armed and in GUIDED mode
- Click on the dashboard page first (keyboard events need page focus)
- Check the console for velocity commands being sent

## Apple Silicon Compatibility

### Works natively:
- ArduPilot SITL (`./waf configure --board sitl`)
- pymavlink (pure Python)
- All Homebrew dependencies
- Flask + Flask-SocketIO
- Gazebo Garden via conda (gz_garden environment)

### Does NOT work:
- Docker-based SITL (Rosetta 2 corrupts Unix domain sockets)
- Any Docker-based MAVLink tools on ARM64 macOS
- Native Gazebo system packages (use conda instead)
