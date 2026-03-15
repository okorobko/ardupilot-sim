# ArduPilot Drone Simulator - Project Context

## Overview

ArduPilot-based quadcopter simulator with a web dashboard featuring 3D visualization, interactive map, and flight controls. Designed for macOS Apple Silicon with native SITL (no Docker).

**Platform:** macOS Apple Silicon (M1/M2/M3/M4)
**Drone type:** Configurable via drone.yaml (default: quadcopter)
**Autopilot:** ArduPilot (not PX4)

## Architecture

```
┌─────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────┐
│  ArduPilot SITL     │    │  Python Backend           │    │  Web Frontend          │
│  (sim_vehicle.py)   │    │  (Flask + SocketIO)       │    │  (Single HTML file)    │
│                     │    │                           │    │                        │
│  TCP:5760 (SERIAL0) ├───►│  MAVLinkBridge thread     ├───►│  Three.js r149 3D      │
│                     │    │  pymavlink                │ WS │  Leaflet map (OSM)     │
│  ArduCopter SITL    │    │  Port 5001                │    │  Console + telemetry   │
└─────────────────────┘    └──────────────────────────┘    └────────────────────────┘
```

**Key choice:** pymavlink (not MAVSDK) -- ArduPilot native, no gRPC, no separate server process.
**Key choice:** TCP:5760 (SITL SERIAL0) -- not UDP:14550 (which requires MAVProxy --out).
**Key choice:** State-based command verification -- not COMMAND_ACK (avoids recv_match thread conflicts).

## Directory Structure

```
ardupilot-sim/
├── backend/
│   ├── app.py                    # Flask+SocketIO server, routes, command handler
│   ├── mavlink_bridge.py         # MAVLink reader thread, telemetry, commands, demo_roundtrip
│   ├── config_loader.py          # YAML config with defaults, vehicle type mapping
│   └── requirements.txt          # flask, flask-socketio, pymavlink, pyyaml
├── config/
│   └── drone.yaml                # Vehicle/simulation/visualization config
├── frontend/
│   └── templates/
│       └── index.html            # Single-file dashboard (Three.js + Leaflet + controls)
├── scripts/
│   ├── install_ardupilot.sh      # ArduPilot SITL macOS installation
│   ├── start_sitl.sh             # SITL launcher (parses drone.yaml)
│   ├── start_all.sh              # Combined SITL + backend launcher
│   ├── test_all.py               # 51-test comprehensive suite
│   ├── test_ui.py                # 44-test UI structure verification
│   ├── test_map_coords.py        # Selenium browser coordinate test
│   └── test_roundtrip.py         # Standalone roundtrip demo test
├── venv/                         # Python virtual environment
├── PLAN.md                       # Original implementation plan
└── CLAUDE.md                     # This file
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Flight Simulator | ArduPilot SITL (native macOS) | Software-in-the-loop simulation |
| Control API | pymavlink | MAVLink message read/write |
| Backend | Flask + Flask-SocketIO | Web server, WebSocket bridge |
| 3D Rendering | Three.js r149 | Drone model, ground, trail |
| Map | Leaflet 1.9.4 + OSM tiles | GPS position tracking |
| WebSocket | Socket.IO 4.5.4 | Real-time telemetry to browser |
| Config | PyYAML | drone.yaml loading |

## Common Commands

### Start Everything

```bash
./scripts/start_all.sh
```

### Start Individually

```bash
# Terminal 1: SITL
./scripts/start_sitl.sh

# Terminal 2: Backend
source venv/bin/activate
cd backend && python3 app.py
```

### Install ArduPilot (first time)

```bash
./scripts/install_ardupilot.sh
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

## MAVLink Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5760/tcp | MAVLink | SITL SERIAL0 -- backend connects here |
| 14550/udp | MAVLink | MAVProxy forwarding (--out in start_sitl.sh) |

The backend connects to TCP:5760 (configured in `drone.yaml` as `mavlink_port`). The UDP:14550 output from start_sitl.sh is available for QGroundControl or other tools.

## Development Guidelines

1. **Single HTML file**: The frontend is one file (`frontend/templates/index.html`). No build step, no NPM. CDN-hosted libraries only.
2. **Three.js r149**: Do not upgrade past r149. Version r160+ removed `examples/js/` scripts. Upgrading requires switching to ES modules and a bundler.
3. **pymavlink threading**: The MAVLinkBridge reader thread owns `conn.recv_match`. Never call `recv_match` from command handlers -- use state-based polling instead.
4. **Config-driven**: Vehicle type, frame, home position, and visualization parameters all come from `drone.yaml`. Scripts parse this file for SITL launch parameters.
5. **Test before committing**: Run `python3 scripts/test_all.py --quick` for fast validation. Full suite includes a 3-5 minute roundtrip demo.
6. **Auto-arm on takeoff**: The takeoff command in app.py automatically sets GUIDED mode and arms if needed.
7. **Demo safety**: `demo_roundtrip` wraps everything in try/except and sets LAND mode on any error.

## ArduCopter Mode Numbers

Key modes used by the system:

| Number | Mode | Usage |
|--------|------|-------|
| 0 | STABILIZE | Manual flight |
| 2 | ALT_HOLD | Altitude hold |
| 4 | GUIDED | GPS waypoint commands, takeoff |
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

## Apple Silicon Compatibility

### Works natively:
- ArduPilot SITL (`./waf configure --board sitl`)
- pymavlink (pure Python)
- All Homebrew dependencies
- Flask + Flask-SocketIO

### Does NOT work:
- Docker-based SITL (Rosetta 2 corrupts Unix domain sockets)
- Any Docker-based MAVLink tools on ARM64 macOS
