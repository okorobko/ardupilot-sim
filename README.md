# ArduPilot Drone Simulator

A real-time drone simulator with 3D visualization, interactive map, and flight controls. Connects to ArduPilot SITL via MAVLink and presents a web-based dashboard with Three.js 3D rendering, Leaflet map tracking, console logging, and command controls. Optionally integrates with Gazebo Garden for physics simulation, dual cameras, and a vehicle-populated surveillance world.

## Architecture Overview

### Mode 1: Standalone (SITL only)

```
+-----------------------------+       +----------------------------------+       +----------------------------------+
|   ArduPilot SITL            |       |   Python Backend                 |       |   Web Frontend                   |
|   (sim_vehicle.py)          |       |   (Flask + SocketIO + pymavlink) |       |   (Three.js 3D + Leaflet map)    |
|                             |       |                                  |       |                                  |
|   Simulates flight physics, | TCP   |   Reads MAVLink messages,        | WS    |   Renders 3D drone model,        |
|   EKF, GPS, sensors         |------>|   emits telemetry at 10Hz,       |------>|   updates map marker/trail,      |
|                             | :5760 |   handles commands, runs demo    |       |   shows console logs,            |
|   SERIAL0 on TCP:5760       |       |   Serves on http://0.0.0.0:5001 |       |   provides flight controls        |
+-----------------------------+       +----------------------------------+       +----------------------------------+
```

### Mode 2: Gazebo Integration

```
+---------------------------+       +---------------------------+       +----------------------------+       +---------------------------+
|   Gazebo Garden           |       |   ArduPilot SITL          |       |   Python Backend           |       |   Web Frontend            |
|   (gz-sim7, conda)        |       |   (JSON model frame)      |       |   (Flask + SocketIO)       |       |   (Three.js + Leaflet)    |
|                           |       |                           |       |                            |       |                           |
|   3D physics world,       | JSON  |   Receives sensor data,   | TCP   |   MAVLink bridge,          | WS    |   3D spectator with       |
|   vehicles, cameras,      |------>|   sends servo outputs     |------>|   camera bridge relay,     |------>|   vehicle boxes,          |
|   ArduPilotPlugin         | UDP   |   via ArduPilotPlugin     | :5760 |   keyboard fly handler     |       |   dual camera overlays,   |
|                           |       |   --model JSON            |       |                            |       |   keyboard controls       |
+---------------------------+       +---------------------------+       +----------------------------+       +---------------------------+
                |
                | gz topic (/camera, /chase_cam)
                v
        +---------------------------+
        |   Camera Bridge           |
        |   (camera_bridge.py)      |
        |                           |
        |   gz topic -> JPEG ->     |
        |   base64 -> SocketIO      |
        +---------------------------+
```

### Data Flow

```
ArduPilot SITL
    |
    | MAVLink TCP:5760 (SERIAL0)
    |   HEARTBEAT, GLOBAL_POSITION_INT, ATTITUDE,
    |   SYS_STATUS, VFR_HUD, GPS_RAW_INT
    v
MAVLinkBridge (background thread)
    |
    | Parses messages, updates internal state
    | Emits "telemetry" at 10Hz via SocketIO
    | Emits "log" (DRONE tag) at 1Hz with MAVLink rates
    |
    v
Flask + SocketIO (port 5001)
    |
    | WebSocket: telemetry, log, config, command_result, demo_status,
    |            camera_frame, chase_frame, fly
    | REST: /api/status, /api/config
    |
    v
Browser (index.html)
    |
    +-- Three.js 3D scene (drone model, ground, trail, sky, vehicle boxes)
    +-- Leaflet map (OSM tiles, drone marker, trail, home marker)
    +-- Console log ([DRONE] [CONTROL] [DEMO] [SYS] tags)
    +-- Telemetry grid (alt, speed, heading, climb, GPS, battery)
    +-- Attitude indicator (canvas 2D)
    +-- Coordinate overlay on map
    +-- Flight controls (arm, disarm, takeoff, land, RTL, mode select, demo)
    +-- Keyboard controls (WASD/arrows, Q/E yaw, R/F altitude)
    +-- Camera overlays (downward cam + chase cam, click to expand)
```

## Module Descriptions

### backend/config_loader.py

Loads drone configuration from `config/drone.yaml` with deep-merge defaults. Every config key has a fallback so the YAML file can be sparse or missing entirely.

**Key design decisions:**
- Deep merge (`_deep_merge`) allows partial overrides -- you can set just `vehicle.type` without repeating all visualization defaults.
- `VEHICLE_MAP` translates user-friendly names (`copter`, `plane`, `rover`, `sub`) to ArduPilot binary names (`ArduCopter`, `ArduPlane`, etc.).
- `FRAME_ARM_COUNT` maps frame type to motor count for the 3D model (`quad`=4, `hexa`=6, `octa`=8, `tri`=3, `y6`=6).
- Derived fields (`ardupilot_vehicle`, `arm_count`) are added after merge so the frontend does not need to duplicate the mapping logic.
- `validate_config` checks that the ArduPilot directory exists and contains `sim_vehicle.py`, and validates lat/lon ranges. Returns a list of errors rather than raising, so the backend can start with warnings.

### backend/mavlink_bridge.py

The core telemetry reader. Runs a dedicated background thread that connects to ArduPilot SITL over TCP and continuously reads MAVLink messages.

**Key design decisions:**
- **pymavlink instead of MAVSDK**: pymavlink is ArduPilot's native library. No gRPC server needed, no separate process, no protobuf compilation. Direct access to all MAVLink messages and ArduPilot-specific mode numbers.
- **TCP:5760 connection** (SITL SERIAL0) instead of UDP:14550. The `--out` flag in `start_sitl.sh` creates a MAVProxy forwarding port, but the backend connects directly to SITL's primary serial port. TCP provides reliable delivery and simpler connection semantics.
- **State-based command verification** instead of COMMAND_ACK polling. After sending an arm command, the bridge polls `self.armed` (updated by the reader thread from HEARTBEAT messages) instead of calling `conn.recv_match(type='COMMAND_ACK')`. This avoids thread safety issues: `recv_match` in the command handler would race with `recv_match` in the reader loop, potentially consuming each other's messages.
- **10Hz telemetry emission**: consolidated telemetry dict emitted via SocketIO, keeping the frontend update smooth without flooding the WebSocket.
- **1Hz drone log**: emits a human-readable status line plus MAVLink message rate counts, allowing the console to show live SITL data rates.
- **ArduCopter mode mapping**: hard-coded dict of mode number to name (STABILIZE=0, GUIDED=4, LAND=9, etc.) since pymavlink does not provide this for ArduCopter custom modes.
- **`send_velocity` method**: body-frame to NED conversion using current heading. Supports forward/back, left/right, up/down, and yaw rate. Used by keyboard flight controls.
- **`demo_roundtrip` method**: a surveillance route demo (arm, takeoff to 25m, fly 4 waypoints over roads and vehicles, land, wait 25s, re-arm, takeoff to 35m, fly back, land). Uses `wait_altitude`, `wait_position`, and `wait_disarmed` helpers with configurable timeouts and progress callbacks. On any exception, attempts to set LAND mode as a safety fallback.

### backend/camera_bridge.py

Bridges Gazebo camera topics to the web dashboard via SocketIO. Runs in the `gz_garden` conda environment.

**Key design decisions:**
- **Direct `gz topic` capture**: uses `gz topic -e -t <topic> -n 1` subprocess calls to grab single frames, avoiding the need for a compiled gz-transport Python binding.
- **Dual camera streams**: downward camera (`/camera`, 640x480, 5fps) for surveillance view, chase camera (`/chase_cam`, 800x600, 3fps) for third-person view.
- **JPEG compression + base64**: frames are decoded from Gazebo's raw RGB, converted to JPEG (quality 55), base64-encoded, and sent via SocketIO `camera_frame` / `chase_frame` events.
- **Threading**: chase cam runs in a daemon thread, downward cam in the main thread.

### backend/app.py

Flask+SocketIO web server. Serves the dashboard, provides REST API, and routes SocketIO commands to the MAVLink bridge.

**Key design decisions:**
- **Auto-arm on takeoff**: the `takeoff` command handler automatically sets GUIDED mode and arms the vehicle if not already armed, reducing the number of clicks needed to fly.
- **Template folder** points to `frontend/templates/` (one level up from backend), keeping backend and frontend code separated.
- **Demo thread management**: only one demo flight can run at a time. The `_demo_thread` global tracks the active thread, and new requests are rejected if one is already in progress.
- **`allow_unsafe_werkzeug=True`**: required for Flask-SocketIO with the development Werkzeug server. Not for production use.
- **Config and telemetry on connect**: when a browser connects, it immediately receives the drone config and current telemetry snapshot so the UI can initialize without waiting for the next telemetry cycle.
- **Camera frame relay**: `camera_frame` and `chase_frame` SocketIO events are received from the camera bridge and broadcast to all browser clients.
- **Keyboard fly handler**: `fly` SocketIO event receives velocity commands (vx, vy, vz, yaw_rate) from the browser keyboard controls and forwards them to `bridge.send_velocity()`.
- **Log events**: command handlers emit `log` events with `CONTROL` and `DRONE` tags so the console shows both the command sent and the result received.

### frontend/templates/index.html

A single-file web dashboard containing all HTML, CSS, and JavaScript. No build step, no bundler, no NPM.

**Layout (CSS grid + flexbox):**
```
+---------------------------+------------+
|                           |  Commands  |
|   Three.js 3D Scene       |  (mode,    |
|   (panel-3d)              |  arm/dis,  |
|   + camera overlays       |  takeoff,  |
|     (down cam, chase cam) |  demo,     |
|                           |  keyboard) |
+------+----------+---------+------------+
| Console        | Telem + | Map          |
| [DRONE] msgs   | Attitude| (Leaflet)    |
| [CONTROL] cmds | grid    |              |
| [DEMO] status  |         |              |
+------+----------+---------+--------------+
```

**Key design decisions:**
- **Three.js r128** (CDN script tag). Uses `examples/js/controls/OrbitControls.js` via `<script>` tag from unpkg CDN without a module bundler. Versions r160+ removed `examples/js/` in favor of ES modules.
- **try-catch around 3D init**: `try { init3D(); } catch(e) { ... }` prevents WebGL failures (e.g., headless environments, missing GPU) from breaking the map and console. The map initializes independently via `setTimeout(initMap, 500)`.
- **Procedural drone model**: built dynamically from config (`arm_count`, `body_color`, `prop_color`). Arms are distributed evenly around a circle. Propellers spin when armed (alternating direction). A red cone on the nose indicates heading.
- **Vehicle boxes in 3D scene**: `buildSceneVehicles()` creates colored box meshes matching Gazebo world vehicle positions -- 13 cars/SUVs on E-W and N-S roads plus a parking lot. This provides a visual reference in the Three.js spectator view even without Gazebo rendering.
- **Dual camera overlays**: downward camera and chase camera are displayed as click-to-expand overlays in the 3D panel. Show "NO SIGNAL" placeholders until the camera bridge connects.
- **Keyboard flight controls**: W/S (forward/back), A/D (left/right), R/F (up/down), Q/E (yaw left/right), plus arrow keys as WASD alternatives. Sends velocity commands at 10Hz while keys are held. Speed: 3 m/s, yaw rate: 0.5 rad/s. Requires GUIDED mode and armed state.
- **GPS to local meters**: `dx = (lon - home_lon) * 111320 * cos(home_lat * PI/180)`, `dz = -(lat - home_lat) * 110540`. The negative Z is because Three.js Z axis points toward the camera while north is "forward".
- **Leaflet map**: OpenStreetMap light tiles (not dark CartoDB). Drone marker is an SVG arrow rotated by heading. Flight trail as polyline. Home marker as red circle. Map follows drone by default; dragging disables follow, double-click re-enables.
- **Map container sizing**: `initMap` checks if the container has dimensions before initializing Leaflet (CSS grid may not have laid out yet). Retries after 300ms if too small. Aggressive `invalidateSize()` calls at 100/300/600/1000/2000/4000ms after init to ensure tiles render.
- **Console**: tagged log lines with color-coded tags ([DRONE]=blue, [CONTROL]=yellow, [DEMO]=purple, [SYS]=gray). Auto-scrolls, capped at 500 lines. Telemetry rate logs use a muted gray to avoid visual noise.
- **Attitude indicator**: canvas 2D drawing with sky/ground halves, horizon line shifted by pitch, rotated by roll. Crosshair overlay.
- **Coordinate overlay**: absolute-positioned div on top of the map showing `lat, lon  alt=Xm`, updated every telemetry frame regardless of map readiness.

### config/drone.yaml

Central configuration file read by both the backend (config_loader.py) and the SITL launch scripts.

```yaml
vehicle:
  type: copter        # copter | plane | rover | sub
  frame: quad         # quad | hexa | octa | tri | y6

simulation:
  ardupilot_dir: ~/ardupilot
  home_lat: 50.450001    # Kyiv, Ukraine
  home_lon: 30.523333
  home_alt: 180          # meters AMSL
  home_heading: 0
  speed_up: 1            # SITL speed multiplier
  mavlink_port: 5760     # TCP port for SITL SERIAL0

visualization:
  ground_size: 500       # 3D ground plane size in meters
  camera_follow: true
  camera_distance: 20
  show_trail: true
  trail_length: 200
  drone_model:
    body_color: "#333333"
    prop_color: "#22aa44"
```

### Gazebo World & Models

#### gazebo/worlds/drone_surveillance.sdf

Gazebo Garden world (SDF 1.9) with:
- **Physics**: 1ms step for ArduPilot compatibility
- **System plugins**: Physics, Sensors (OGRE2), UserCommands, SceneBroadcaster, IMU, NavSat
- **GPS origin**: Kyiv (50.450001, 30.523333) matching drone.yaml
- **Ground plane**: 500x500m green surface
- **Roads**: E-W road at y=20 and N-S road at x=30 (8m wide, dark asphalt with yellow center lines)
- **15 vehicles**: procedural box models with car proportions
  - Sedans (4.5x1.8x1.4m): red, white, blue, green, yellow
  - SUVs (4.8x2.0x1.7m): black, silver, white, dark blue
  - Trucks (6.0x2.2x2.5m): orange cab + cargo box
  - Vehicles placed on roads and in a parking lot cluster
- **Drone**: includes `iris_with_camera` model at origin

#### gazebo/models/iris_with_camera/model.sdf

Custom drone model that includes the standard `iris_with_ardupilot` (from ardupilot_gazebo) and adds:
- **Downward camera**: 640x480 RGB, 30Hz, 60-degree FOV, publishes to `/camera` gz topic. Fixed to base_link pointing straight down (1.5708 rad pitch).
- **Chase camera**: 800x600 RGB, 15Hz, 74-degree FOV, publishes to `/chase_cam` gz topic. Mounted 3m behind and 2m above the drone, angled forward-down (0.4 rad pitch).

### Scripts

#### scripts/install_ardupilot.sh

Automated ArduPilot SITL installation for macOS. Steps:
1. Install Homebrew dependencies (gcc-arm-none-eabi, ccache, gawk, cmake, python@3.11)
2. Clone ArduPilot with submodules to `~/ardupilot` (or pull if already cloned)
3. Run ArduPilot's macOS prerequisites installer
4. Build SITL with `./waf configure --board sitl && ./waf copter`
5. Verify the `arducopter` binary exists

#### scripts/install_gazebo.sh

Gazebo Garden installation via conda:
1. Installs `ros-humble-ros-gz` from conda-forge/robostack-staging (includes gz-sim7)
2. Clones `ardupilot_gazebo` plugin repository
3. Builds the plugin with cmake (links against conda Gazebo libraries)
4. Prints required environment variable exports (`GZ_SIM_SYSTEM_PLUGIN_PATH`, `GZ_SIM_RESOURCE_PATH`)

#### scripts/start_sitl.sh

Launches ArduPilot SITL by parsing `config/drone.yaml` via an inline Python snippet. Extracts vehicle type, frame, home location, MAVLink port, and speed multiplier. Runs `sim_vehicle.py` with `--no-rebuild` flag to skip recompilation. The `--out` flag forwards MAVLink to `udp:127.0.0.1:{port}`.

#### scripts/start_gazebo.sh

Launches Gazebo with the drone surveillance world. Sets `GZ_SIM_SYSTEM_PLUGIN_PATH` and `GZ_SIM_RESOURCE_PATH` to include ardupilot_gazebo and project models/worlds. Runs `gz sim` with verbose logging.

#### scripts/start_gazebo_sitl.sh

Launches ArduPilot SITL in Gazebo mode:
- Frame: `gazebo-iris` (uses ArduPilotPlugin for physics)
- Model: `JSON` (new JSON-based protocol, not the old binary `gazebo-iris` model)
- Flags: `--no-rebuild --no-mavproxy`
- Parses home location from `drone.yaml`

#### scripts/start_gazebo_all.sh

Combined launcher for the full Gazebo simulation stack (5 steps):
1. Start Gazebo server (`gz sim -s`, server-only mode required on macOS)
2. Unpause simulation via `gz service` (Gazebo starts paused by default)
3. Start SITL with JSON model (`start_gazebo_sitl.sh`)
4. Start Flask backend (`app.py`)
5. Launch Gazebo GUI (`gz sim -g`, separate process required on macOS)

Includes process cleanup on SIGINT/SIGTERM and opens the dashboard in the browser.

#### scripts/start_camera_bridge.sh

Wrapper to launch `backend/camera_bridge.py`. Must be run in the `gz_garden` conda environment where `gz` CLI and OpenCV are available.

#### scripts/start_gz_bridge.sh

Launches the ROS2 gz-bridge for camera topics (`/camera` -> `/camera/image_raw`). Requires ROS2 Humble with `ros_gz_bridge` package. This is an alternative to the Python camera bridge for ROS2-based pipelines.

#### scripts/start_all.sh

Combined launcher (standalone mode) that:
1. Activates the Python venv if present
2. Starts SITL in background
3. Polls for a MAVLink heartbeat (up to 60 seconds) using a pymavlink one-liner
4. Starts the Flask backend
5. Opens `http://localhost:5001` in the default browser
6. Traps SIGINT/SIGTERM to cleanly kill all child processes

#### scripts/test_all.py

Comprehensive 51-test suite organized in 8 sections:
1. **Backend Connectivity** (4 tests): API reachability, SITL connection, GPS fix, satellite count
2. **Config Endpoint** (8 tests): vehicle type/frame, home lat/lon ranges, visualization config, arm count
3. **Telemetry Streaming** (12 tests): SocketIO config event on connect, telemetry event rate, all telemetry fields present (position, attitude, heading, battery, GPS, armed, flight_mode, connected, lat/lon/alt, roll/pitch/yaw)
4. **ARM/DISARM** (2 tests): arm in GUIDED mode, disarm
5. **Takeoff & Land** (4 tests): auto-arm takeoff to 15m, altitude check, armed check, mode check, land and disarm
6. **Mode Changes** (4 tests): cycle through STABILIZE, LOITER, ALT_HOLD, GUIDED
7. **Map Data** (7 tests): non-zero lat/lon, near home position, altitude updates during flight, multiple altitude values, heading, position updates during descent
8. **Roundtrip Demo** (10 tests): demo completion, success, takeoff/flight/landing/wait/return phases, drone back near home, disarmed after demo

Supports `--quick` flag to skip the roundtrip demo. Uses `python-socketio[client]` for WebSocket testing and `urllib.request` for REST API testing.

#### scripts/test_ui.py

44-test UI structure verification suite organized in 8 sections:
1. **Required HTML Elements** (22 tests): checks all expected element IDs exist in the served HTML
2. **Map Tile Provider** (2 tests): OSM tiles present, dark CartoDB tiles absent
3. **Map Theme** (1 test): Leaflet container background is not dark
4. **Telemetry Layout** (4 tests): CSS grid on `.bottom-telem`, grid columns, attitude cell in column 3 spanning rows
5. **Map Coordinate Overlay** (3 tests): `map-coords` div exists with correct class, JS update code present
6. **Bottom Row Layout** (4 tests): console, telem, map present in correct order
7. **Map Position Updates** (5 tests): SocketIO telemetry positions received, valid lat/lon floats
8. **Coordinate Overlay JS** (3 tests): lat/lon `toFixed(6)`, alt `toFixed(1)` in update code

#### scripts/test_map_coords.py

Deep-dive browser-level test for map coordinate updates. Tests at multiple levels:
1. Backend status check (connected, non-zero lat)
2. HTML structure verification (map-coords element present)
3. JS `updateMap` function analysis (references map-coords, sets textContent, checks data.position)
4. Telemetry handler verification (updateMap called from telemetry)
5. SocketIO telemetry flow (config events, telemetry events, position data types)
6. JS logic simulation (format expected coordinate string)
7. Selenium browser test (headless Chrome, waits for map-coords to update from "Waiting for GPS...", checks for JS errors, socket connection status, telemetry data presence). Auto-installs Selenium if not present.

#### scripts/test_roundtrip.py

Standalone roundtrip demo test. Connects via SocketIO, triggers `demo_roundtrip`, streams progress messages to stdout with timestamps, and reports final position/mode/armed state. Simpler than test_all.py's demo test -- useful for manual verification.

## Key Technical Decisions

### pymavlink instead of MAVSDK

MAVSDK uses a gRPC server (`mavsdk_server`) as a separate process, adding complexity and potential connection issues. pymavlink talks MAVLink directly and is maintained by the ArduPilot project. It provides raw message access, which is needed for ArduPilot-specific features like custom mode numbers and `SET_POSITION_TARGET_GLOBAL_INT`.

### State-Based Command Verification

After sending a command (e.g., arm), the bridge checks `self.armed` in a polling loop rather than using `conn.recv_match(type='COMMAND_ACK')`. The reason: the reader thread is continuously calling `recv_match(blocking=True)` to process telemetry. If the command handler also calls `recv_match`, it creates a race condition where either thread could consume the other's expected message. State-based verification is simpler and thread-safe.

### TCP:5760 Instead of UDP:14550

ArduPilot SITL exposes SERIAL0 on TCP port 5760 by default. The `--out udp:127.0.0.1:14550` flag creates a MAVProxy forwarding port, which requires MAVProxy to be installed and running. Connecting directly to TCP:5760 is more reliable (TCP guarantees delivery), eliminates the MAVProxy dependency for the backend connection, and is how SITL natively communicates.

### JSON Model Protocol (Gazebo)

When running with Gazebo, SITL uses `--model JSON` instead of the old binary `gazebo-iris` model format. The JSON protocol is what `ardupilot_gazebo`'s ArduPilotPlugin speaks -- it sends sensor data as JSON over UDP and receives servo commands. The frame is still `gazebo-iris` (which tells ArduPilot to expect Gazebo-style physics), but `--model JSON` selects the communication protocol.

### Gazebo Garden (gz-sim7) via Conda

Gazebo Harmonic/Garden is installed via conda (`gz_garden` environment) rather than system packages. This provides a self-contained installation on macOS where native Gazebo packages are not available. The conda environment includes `gz-sim7`, rendering engines, and all dependencies.

### Three.js r128

The frontend uses Three.js r128 loaded via CDN script tags. This version supports `examples/js/controls/OrbitControls.js` via a simple `<script>` tag without a module bundler. Versions r160+ removed the `examples/js/` directory in favor of ES modules.

### try-catch Around 3D Init

```javascript
try { init3D(); } catch(e) { clog('SYS', '3D init error: '+e.message, 'log-err'); }
```

If WebGL initialization fails (headless browser, missing GPU, or a Three.js error), this prevents the entire page from breaking. The map, console, and telemetry panels continue to work independently.

### CSS Grid Layout with Flexbox Bottom Panel

The main layout uses CSS grid with two columns and two rows. The bottom panel uses flexbox to arrange console, telemetry, and map side by side. The telemetry panel itself uses a nested CSS grid (2 data columns + 1 attitude column spanning 3 rows) to keep all values visible without scrolling.

### 1Hz Drone Log Emission

Every second, the bridge emits two log lines: a human-readable status line (position, altitude, heading, speed, mode, armed state) and a MAVLink message rate summary. This shows the frontend operator exactly what data is flowing from SITL, making it easy to diagnose connection or data issues.

## Directory Structure

```
ardupilot-sim/
├── backend/
│   ├── app.py                    # Flask+SocketIO server, routes, command handling
│   ├── mavlink_bridge.py         # pymavlink reader thread, telemetry, commands, demo
│   ├── camera_bridge.py          # Gazebo camera → JPEG → SocketIO bridge
│   ├── config_loader.py          # YAML config loading with defaults and validation
│   └── requirements.txt          # flask, flask-socketio, pymavlink, pyyaml
├── config/
│   └── drone.yaml                # Vehicle, simulation, and visualization config
├── frontend/
│   └── templates/
│       └── index.html            # Single-file dashboard (Three.js + Leaflet + controls)
├── gazebo/
│   ├── models/
│   │   └── iris_with_camera/
│   │       ├── model.config      # Gazebo model metadata
│   │       └── model.sdf         # Iris + downward camera + chase camera
│   └── worlds/
│       └── drone_surveillance.sdf # World with roads, 15 vehicles, GPS origin
├── scripts/
│   ├── install_ardupilot.sh      # ArduPilot SITL installation for macOS
│   ├── install_gazebo.sh         # Gazebo Garden + ardupilot_gazebo installation
│   ├── start_sitl.sh             # SITL launcher (parses drone.yaml)
│   ├── start_all.sh              # Combined SITL + backend launcher (standalone)
│   ├── start_gazebo.sh           # Gazebo world launcher
│   ├── start_gazebo_sitl.sh      # SITL in Gazebo mode (JSON model)
│   ├── start_gazebo_all.sh       # Full Gazebo stack launcher (5 steps)
│   ├── start_camera_bridge.sh    # Camera bridge launcher
│   ├── start_gz_bridge.sh        # ROS2 gz-bridge for camera topics
│   ├── test_all.py               # 51-test comprehensive test suite
│   ├── test_ui.py                # 44-test UI structure verification
│   ├── test_map_coords.py        # Browser-level Selenium coordinate test
│   └── test_roundtrip.py         # Standalone roundtrip demo test
├── venv/                         # Python virtual environment
├── PLAN.md                       # Original implementation plan
├── CLAUDE.md                     # AI assistant project context
└── README.md                     # This file
```

## Setup

### Prerequisites

- macOS (Apple Silicon or Intel)
- Homebrew
- Python 3.11+
- Git
- For Gazebo mode: conda (Miniforge/Mambaforge recommended)

### 1. Install ArduPilot SITL

```bash
cd ardupilot-sim
./scripts/install_ardupilot.sh
```

This clones ArduPilot to `~/ardupilot`, installs dependencies, and builds the SITL binary. Takes 5-10 minutes on first run.

### 2. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

For running tests that use SocketIO client:

```bash
pip install python-socketio[client]
```

For browser-level tests:

```bash
pip install selenium
```

### 3. (Optional) Install Gazebo Garden

```bash
# Create and activate conda environment
conda create -n gz_garden python=3.11
conda activate gz_garden

# Install Gazebo + ardupilot_gazebo
./scripts/install_gazebo.sh

# Add to your shell profile:
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/Documents/Projects/Drone/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$HOME/Documents/Projects/Drone/ardupilot_gazebo/models:$HOME/Documents/Projects/Drone/ardupilot_gazebo/worlds:$(pwd)/gazebo/models:$(pwd)/gazebo/worlds
```

For the camera bridge, also install OpenCV and python-socketio in the conda env:

```bash
conda activate gz_garden
pip install opencv-python python-socketio[client]
```

## Running

### Option A: Standalone (SITL only)

#### All-in-one

```bash
./scripts/start_all.sh
```

Starts SITL, waits for heartbeat, starts backend, opens browser.

#### Manual (two terminals)

Terminal 1 -- SITL:
```bash
./scripts/start_sitl.sh
```

Terminal 2 -- Backend:
```bash
source venv/bin/activate
cd backend
python3 app.py
```

Then open http://localhost:5001 in a browser.

### Option B: Gazebo Mode

#### All-in-one

```bash
conda activate gz_garden
./scripts/start_gazebo_all.sh
```

This starts (in order): Gazebo server, unpauses simulation, SITL with JSON model, Flask backend, Gazebo GUI. Opens the dashboard in the browser.

#### Manual (four terminals)

Terminal 1 -- Gazebo server (must start first):
```bash
conda activate gz_garden
./scripts/start_gazebo.sh
```

Terminal 2 -- Unpause + SITL:
```bash
# Unpause (after Gazebo loads the world)
gz service -s /world/drone_surveillance/control \
    --reqtype gz.msgs.WorldControl \
    --reptype gz.msgs.Boolean \
    --timeout 5000 --req 'pause: false'

# Start SITL
./scripts/start_gazebo_sitl.sh
```

Terminal 3 -- Backend:
```bash
source venv/bin/activate
cd backend
python3 app.py
```

Terminal 4 -- Camera bridge (optional, for live camera feeds):
```bash
conda activate gz_garden
./scripts/start_camera_bridge.sh
```

Then open http://localhost:5001 in a browser.

#### Gazebo GUI (optional, separate window)

On macOS, the Gazebo GUI must run as a separate process from the server:
```bash
conda activate gz_garden
gz sim -g
```

## Keyboard Flight Controls

The dashboard supports real-time keyboard flight control when the drone is armed and in GUIDED mode.

| Key | Action |
|-----|--------|
| W / Up Arrow | Fly forward |
| S / Down Arrow | Fly backward |
| A / Left Arrow | Strafe left |
| D / Right Arrow | Strafe right |
| R | Ascend |
| F | Descend |
| Q | Yaw left |
| E | Yaw right |

- Speed: 3 m/s
- Yaw rate: 0.5 rad/s
- Commands sent at 10Hz while keys are held
- Keys highlight in the command panel when active
- Releasing all keys sends a zero-velocity stop command

## Testing

All tests require SITL running and backend on localhost:5001.

```bash
# Full test suite (51 tests, includes roundtrip demo ~3-5 min)
python3 scripts/test_all.py

# Quick tests (skip roundtrip demo)
python3 scripts/test_all.py --quick

# UI structure tests (44 tests)
python3 scripts/test_ui.py

# Browser coordinate test (uses Selenium)
python3 scripts/test_map_coords.py

# Standalone roundtrip demo
python3 scripts/test_roundtrip.py
```

## macOS Gazebo Limitations

- **Server and GUI must be separate processes**: on macOS, `gz sim` cannot run server and GUI in one process due to OGRE-Next rendering issues. Use `gz sim -s` for server and `gz sim -g` for GUI in separate terminals. The `start_gazebo_all.sh` script handles this automatically.
- **OGRE-Next rendering**: macOS uses OGRE-Next (ogre2) for Gazebo rendering. Some visual features (shadows, advanced materials) may behave differently than on Linux. The world SDF specifies `<render_engine>ogre2</render_engine>` for the Sensors plugin.
- **Conda environment required**: native Gazebo packages are not available for macOS. The conda `gz_garden` environment provides gz-sim7 and all dependencies.
- **ArduPilotPlugin build**: the plugin must be built against the conda Gazebo libraries (`-DCMAKE_PREFIX_PATH=$CONDA_PREFIX`).

## Apple Silicon Notes

- ArduPilot SITL compiles natively on Apple Silicon with `./waf configure --board sitl`. No Rosetta translation needed.
- pymavlink is pure Python -- no native compilation issues.
- Docker is NOT recommended. The same Rosetta 2 Unix domain socket issues that affect PX4 SITL also affect ArduPilot in Docker containers.
- Homebrew dependencies install natively for ARM64.
- The `install_ardupilot.sh` script runs ArduPilot's official macOS prerequisites installer, which handles Apple Silicon-specific toolchain setup.
