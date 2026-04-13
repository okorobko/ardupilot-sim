# Phase 5 Report — Flight Guidance (Mini-Map + Survey)

**Date:** 2026-04-13
**Status:** COMPLETE

---

## Changes made

### Mini-map vehicle dots

- Plotted all 19 military vehicles from `military_training.sdf` on the Leaflet map
- Each dot is color-coded by class (same colors as detection boxes):
  - RED = tank (4 vehicles)
  - ORANGE = apc_ifv (4 vehicles)
  - LIGHT-RED = sp_artillery (3 vehicles)
  - MAGENTA = mlrs (2 vehicles)
  - YELLOW = military_truck (4 vehicles)
  - CYAN = helicopter (2 vehicles)
- Tooltips show vehicle name on hover
- Positions converted from Gazebo local (meters) to GPS using home origin (50.450001, 30.523333)

### SURVEY auto-fly button

Added "SURVEY" button next to "ROUNDTRIP" in the Demo section. Flies an automated pattern over all 6 formations:

| Waypoint | Formation | Gazebo coords (x, y) |
|----------|-----------|---------------------|
| WP1 | Tank column | (-55, 0) |
| WP2 | Artillery battery | (60, 45) |
| WP3 | MLRS section | (-67, 52) |
| WP4 | Supply convoy | (0, -30) |
| WP5 | Helicopter LZ | (70, -42) |
| WP6 | Staging area | (20, 20) |

Flight parameters:
- Cruise altitude: 45m (optimal for detection FOV)
- 5-second hover at each formation for detection
- Returns to home and lands after completing all waypoints
- Progress shown in demo status bar

### Backend

- `app.py`: Added `demo_survey` SocketIO handler
- `mavlink_bridge.py`: Added `demo_survey()` method with 6 waypoints, GUIDED mode, auto-arm

---

## Files modified

| File | Change |
|------|--------|
| `frontend/templates/index.html` | SURVEY button, plotMilitaryVehicles(), vehicle data array |
| `backend/app.py` | demo_survey SocketIO handler |
| `backend/mavlink_bridge.py` | demo_survey() method with 6-formation waypoint route |

---

## Updated plan summary

| Phase | Status | Key result |
|-------|--------|------------|
| 1 | Done | Bbox fix (normalized→pixel) |
| 2 | Done | Chase-cam detection + enlarged overlay |
| 3 | Done | Live detection panel + SERIAL0 keepalive |
| 4 | Done | SimpleTracker: stable track IDs, no flicker |
| 5 | **Done** | **Mini-map dots + SURVEY auto-fly** |
| 6 | Skip | Not needed (7ms latency) |

All phases complete.
