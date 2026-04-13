# Phase 3 Report — Detection Info Panel + SERIAL0 Fix

**Date:** 2026-04-13
**Status:** COMPLETE

---

## Changes made

### Detection info panel (right sidebar)

Added live "Detections" section in the commands panel showing:
- **Total count** of all detections since page load
- **Rolling rate** (detections/second over 5s window)
- **Inference latency** (ms)
- **Per-class breakdown** with colored dots, class name, peak confidence, count
- **Freshness fade**: rows dim if class not seen in last 2 seconds
- Both down-cam and chase-cam detections feed into the same panel

### SERIAL0 keepalive fix

**Root cause found:** ArduPilot SITL stops sending heartbeats on ALL serial ports when SERIAL0 (TCP:5760) disconnects. The startup script briefly connected to trigger SERIAL1 init, then disconnected — killing heartbeats.

**Fix:** Keep a persistent socket connection to 5760 alive for the entire session. Added `python3 -c "import socket,time; s=socket.socket(); s.connect(('127.0.0.1',5760)); time.sleep(99999)"` as a background process in `start_military_detect.sh`.

### Heartbeat timeout increase

Increased `wait_heartbeat` timeout from 30s to 60s to accommodate Gazebo JSON initialization time.

---

## Test results

Flight test at 15.8m altitude:

| Class | Count | Notes |
|-------|-------|-------|
| uav | 13 | Most frequent — drone sees its own UAV model? |
| sp_artillery | 5 | Distinctive shape, high mAP |
| tank | 1 | Lower confidence at this altitude |
| **Total** | **19** | Over ~10 seconds of hover |

Detection rate: ~1.9 detections/second
Inference latency: 6-12 ms (CoreML)

---

## Files modified

| File | Change |
|------|--------|
| `frontend/templates/index.html` | Detection info panel CSS + HTML + JS (updateDetPanel) |
| `backend/mavlink_bridge.py` | Increased heartbeat timeout to 60s |
| `config/drone.yaml` | Set mavlink_port to 5762 (SERIAL1) |
| `scripts/start_military_detect.sh` | SERIAL0 keepalive connection + SERIAL1 trigger |

---

## Next step

→ **Phase 4**: Temporal smoothing (SimpleTracker) to kill bbox flicker.
