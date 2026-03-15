# ArduPilot Drone Simulator with 3D Visualization & Map

## Context

The user has a working PX4-based tailsitter project at `small-tailsitter/` with Flask+SocketIO web dashboard, ROS2 pipeline, and MAVSDK integration. Now they want a **separate ArduPilot-based simulator** for a standard quadcopter, with configurable drone type, 3D visual environment, and map view. ArduPilot is not yet installed on the system.

## Architecture

```
ArduPilot SITL (sim_vehicle.py)
    │ MAVLink UDP :14550
    ▼
Python Backend (Flask + SocketIO + pymavlink)
    │ WebSocket
    ▼
Web Frontend (Three.js 3D scene + Leaflet map)
```

Key choice: **pymavlink** instead of MAVSDK — it's the standard ArduPilot library, simpler (no gRPC), and has ArduPilot-specific mode mappings. No ROS2 layer needed.

## Directory Structure

```
/Users/olkorobko/Documents/projects/drone/ardupilot-sim/
├── config/
│   └── drone.yaml                # Configurable drone type, frame, home position
├── backend/
│   ├── app.py                    # Flask + SocketIO server
│   ├── mavlink_bridge.py         # pymavlink reader thread → WebSocket emitter
│   ├── config_loader.py          # Loads & validates drone.yaml
│   └── requirements.txt          # pymavlink, flask, flask-socketio, pyyaml
├── frontend/
│   └── templates/
│       └── index.html            # Single-file: Three.js 3D + Leaflet map + controls
├── scripts/
│   ├── install_ardupilot.sh      # Clone & build ArduPilot for macOS ARM
│   ├── start_sitl.sh             # Launch sim_vehicle.py from config
│   └── start_all.sh              # Launch SITL + backend, open browser
└── README.md
```

## Implementation Steps

### Step 1: ArduPilot Installation Script (`scripts/install_ardupilot.sh`)
- Install brew deps: `gcc-arm-none-eabi`, `ccache`, `gawk`, `cmake`
- Clone ArduPilot to `~/ardupilot`: `git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git`
- Run `Tools/environment_install/install-prereqs-mac.sh`
- Build SITL: `./waf configure --board sitl && ./waf copter`
- Verify: `Tools/autotest/sim_vehicle.py -v ArduCopter -w` outputs heartbeat

### Step 2: Drone Configuration File (`config/drone.yaml`)
```yaml
vehicle:
  type: copter        # copter | plane | rover | sub
  frame: quad         # quad | hexa | octa | tri | y6

simulation:
  ardupilot_dir: ~/ardupilot
  home_lat: 50.450001
  home_lon: 30.523333
  home_alt: 180
  home_heading: 0
  speed_up: 1
  mavlink_port: 14550

visualization:
  ground_size: 500
  camera_follow: true
  camera_distance: 20
  show_trail: true
  trail_length: 200
  drone_model:
    body_color: "#333333"
    prop_color: "#22aa44"
```

### Step 3: Config Loader (`backend/config_loader.py`)
- Load YAML, apply defaults for missing keys
- Map types: `copter` → `ArduCopter`, `plane` → `ArduPlane`, etc.
- Validate `ardupilot_dir` exists and has `sim_vehicle.py`

### Step 4: MAVLink Bridge (`backend/mavlink_bridge.py`)
- `MAVLinkBridge` class with background thread
- Connect via `mavutil.mavlink_connection(f'udp:127.0.0.1:{port}')`
- Parse messages: `GLOBAL_POSITION_INT`, `ATTITUDE`, `HEARTBEAT`, `SYS_STATUS`, `VFR_HUD`, `GPS_RAW_INT`
- Emit telemetry to SocketIO at ~10Hz
- Send commands: arm/disarm (`MAV_CMD_COMPONENT_ARM_DISARM`), takeoff, set_mode (GUIDED/LAND/RTL/LOITER)
- Request message streams via `MAV_CMD_SET_MESSAGE_INTERVAL`

### Step 5: Flask Backend (`backend/app.py`)
- Follow pattern from `small-tailsitter/scripts/web_dashboard.py`
- Flask + SocketIO on port 5001
- Start MAVLinkBridge in background thread
- REST endpoints: `/api/status`, `/api/config`
- SocketIO events: `command` (arm, disarm, takeoff, land, set_mode)
- Serve `frontend/templates/index.html`
- Load drone config and pass to frontend via template or API

### Step 6: Web Frontend (`frontend/templates/index.html`)
Single HTML file with split-pane layout:

**Left panel — Three.js 3D scene:**
- Ground plane with grid texture (configurable size)
- Sky: `HemisphereLight` + `DirectionalLight`
- Procedural quadcopter model (body `BoxGeometry`, 4 arms `CylinderGeometry`, 4 spinning prop discs)
  - Dynamically build N arms based on `vehicle.frame` (quad=4, hexa=6, octa=8)
- Apply attitude (roll/pitch/yaw) to drone group
- GPS→local meters conversion: `dx = (lon-home_lon) * 111320 * cos(home_lat)`, `dy = (lat-home_lat) * 110540`
- Flight trail via `THREE.Line` with `BufferGeometry`
- `OrbitControls` with follow mode

**Right panel — Leaflet map (top) + controls (bottom):**
- Dark CartoDB tiles, drone marker rotated by heading
- Flight trail polyline, home marker
- Controls: mode selector, arm/disarm, takeoff (altitude input), land, RTL
- Telemetry: altitude, speed, battery, GPS sats, heading, flight mode
- Canvas attitude indicator (reuse pattern from existing dashboard)

**CDN libs:** Three.js r160, Leaflet 1.9.4, Socket.IO 4.5.4

### Step 7: SITL Launch Script (`scripts/start_sitl.sh`)
- Parse `config/drone.yaml` via Python one-liner
- Launch: `sim_vehicle.py -v ArduCopter --frame quad -l lat,lon,alt,hdg --out udp:127.0.0.1:14550`

### Step 8: Combined Startup (`scripts/start_all.sh`)
- Start SITL in background
- Wait for heartbeat (poll with timeout)
- Start Flask backend
- Open `http://localhost:5001` in browser
- Trap SIGINT to kill all processes

### Step 9: Python venv + requirements
```
ardupilot-sim/venv/
requirements.txt: flask, flask-socketio, pymavlink, pyyaml
```

## Key Reference Files (patterns to reuse)
- `small-tailsitter/scripts/web_dashboard.py` — Flask+SocketIO structure, telemetry threading, command handling
- `small-tailsitter/scripts/templates/dashboard.html` — Dark UI theme, Leaflet map, attitude indicator, SocketIO client
- `small-tailsitter/scripts/start_px4_sitl.sh` — SITL launch script pattern

## Verification
1. **SITL running**: `sim_vehicle.py` outputs "EKF2 IMU0 is using GPS" within ~30s
2. **Backend connected**: Console shows "Heartbeat received"
3. **Map view**: Drone marker at Kyiv (50.45, 30.52)
4. **3D view**: Quadcopter model visible on ground plane
5. **Arm + takeoff**: Click Arm → Takeoff(10m) → drone rises in both views, altitude reads ~10m
6. **Attitude**: 3D model tilts with drone orientation
7. **Config test**: Change `frame: hexa` in `drone.yaml`, restart → 6-arm model appears

## Apple Silicon Notes
- ArduPilot SITL compiles natively with `--board sitl` (no Rosetta)
- pymavlink is pure Python — no compilation issues
- Docker NOT recommended (same Rosetta socket issues as PX4)

## Progress

- [x] Step 0: Create directory structure
- [x] Step 1: ArduPilot installation script
- [x] Step 2: Drone configuration file
- [x] Step 3: Config loader
- [x] Step 4: MAVLink bridge
- [x] Step 5: Flask backend
- [x] Step 6: Web frontend
- [x] Step 7: SITL launch script
- [x] Step 8: Combined startup script
- [x] Step 9: Python venv + requirements
