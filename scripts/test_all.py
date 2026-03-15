#!/usr/bin/env python3
"""Comprehensive test suite for ArduPilot Drone Simulator.

Tests:
  1. Backend connectivity & API
  2. Config endpoint
  3. Telemetry streaming via SocketIO
  4. ARM / DISARM commands
  5. Takeoff (auto-arm) and landing
  6. Mode changes
  7. Map data (position updates with real GPS coordinates)
  8. Roundtrip demo flight (100m out, land, wait, fly back)

Usage:
    python3 scripts/test_all.py            # run all tests
    python3 scripts/test_all.py --quick    # skip demo (faster)

Requires: SITL running + backend on localhost:5001
"""

import json
import math
import sys
import time
import urllib.request

BASE = "http://localhost:5001"
PASS = 0
FAIL = 0


def status():
    return json.loads(urllib.request.urlopen(f"{BASE}/api/status", timeout=5).read())


def config():
    return json.loads(urllib.request.urlopen(f"{BASE}/api/config", timeout=5).read())


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  -- {detail}")


def wait_for(predicate, timeout=60, interval=0.5):
    """Wait until predicate(status()) returns True."""
    start = time.time()
    while time.time() - start < timeout:
        s = status()
        if predicate(s):
            return s
        time.sleep(interval)
    return status()


# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  ArduPilot Simulator — Test Suite")
print("=" * 60)

# Import socketio
try:
    import socketio
except ImportError:
    print("ERROR: python-socketio not installed. pip install python-socketio[client]")
    sys.exit(1)


# ── Test 1: Backend connectivity ──────────────────────────
print("\n[1/8] Backend Connectivity")
try:
    s = status()
    check("API /api/status reachable", True)
    check("SITL connected", s["connected"], f"connected={s['connected']}")
    check("GPS fix", s["gps"]["fix_type"] >= 3, f"fix_type={s['gps']['fix_type']}")
    check("Satellites > 0", s["gps"]["satellites"] > 0, f"sats={s['gps']['satellites']}")
except Exception as e:
    check("API reachable", False, str(e))
    print("\nCannot continue without backend. Exiting.")
    sys.exit(1)


# ── Test 2: Config endpoint ──────────────────────────────
print("\n[2/8] Config Endpoint")
try:
    c = config()
    check("Config has vehicle", "vehicle" in c)
    check("Vehicle type is copter", c["vehicle"]["type"] == "copter", c["vehicle"]["type"])
    check("Vehicle frame is quad", c["vehicle"]["frame"] == "quad", c["vehicle"]["frame"])
    check("Config has simulation", "simulation" in c)
    check("Home lat valid", 49 < c["simulation"]["home_lat"] < 52, c["simulation"]["home_lat"])
    check("Home lon valid", 29 < c["simulation"]["home_lon"] < 32, c["simulation"]["home_lon"])
    check("Config has visualization", "visualization" in c)
    check("Arm count = 4", c["vehicle"].get("arm_count") == 4, c["vehicle"].get("arm_count"))
except Exception as e:
    check("Config endpoint", False, str(e))


# ── Test 3: Telemetry streaming ──────────────────────────
print("\n[3/8] Telemetry Streaming (SocketIO)")
sio = socketio.Client()
telem_data = []
config_data = []

@sio.on("config")
def on_cfg(data):
    config_data.append(data)

@sio.on("telemetry")
def on_telem(data):
    telem_data.append(data)

sio.connect(BASE)
time.sleep(3)

check("Config event received on connect", len(config_data) > 0)
check("Telemetry events received", len(telem_data) > 5, f"count={len(telem_data)}")

if telem_data:
    t = telem_data[-1]
    check("Telemetry has position", "position" in t)
    check("Telemetry has attitude", "attitude" in t)
    check("Telemetry has heading", "heading" in t)
    check("Telemetry has battery", "battery" in t)
    check("Telemetry has gps", "gps" in t)
    check("Telemetry has armed", "armed" in t)
    check("Telemetry has flight_mode", "flight_mode" in t)
    check("Telemetry has connected", "connected" in t)
    check("Position has lat/lon/alt", all(k in t["position"] for k in ["lat", "lon", "alt"]))
    check("Attitude has roll/pitch/yaw", all(k in t["attitude"] for k in ["roll", "pitch", "yaw"]))

sio.disconnect()


# ── Test 4: ARM / DISARM ─────────────────────────────────
print("\n[4/8] ARM / DISARM")

sio = socketio.Client()
cmd_results = []

@sio.on("command_result")
def on_cmd(data):
    cmd_results.append(data)

@sio.on("telemetry")
def on_t(data):
    pass

sio.connect(BASE)
time.sleep(1)

# Ensure disarmed first
s = status()
if s["armed"]:
    sio.emit("command", {"action": "land"})
    wait_for(lambda s: not s["armed"], timeout=60)
    time.sleep(2)

# Set GUIDED mode for arming
sio.emit("command", {"action": "set_mode", "mode": "GUIDED"})
time.sleep(1)

# ARM
cmd_results.clear()
sio.emit("command", {"action": "arm"})
time.sleep(3)
s = status()
check("ARM command accepted", s["armed"], f"armed={s['armed']}")

# DISARM
sio.emit("command", {"action": "disarm"})
time.sleep(3)
s = status()
check("DISARM command accepted", not s["armed"], f"armed={s['armed']}")

sio.disconnect()


# ── Test 5: Takeoff & Land ───────────────────────────────
print("\n[5/8] Takeoff (auto-arm) & Land")

sio = socketio.Client()

@sio.on("command_result")
def on_cmd5(data):
    pass

@sio.on("telemetry")
def on_t5(data):
    pass

sio.connect(BASE)
time.sleep(1)

# Takeoff to 15m (should auto-arm + set GUIDED)
sio.emit("command", {"action": "takeoff", "altitude": 15})

# Wait for altitude
s = wait_for(lambda s: s["position"]["alt"] > 12, timeout=45)
check("Takeoff reached 12m+", s["position"]["alt"] > 12, f"alt={s['position']['alt']:.1f}")
check("Armed during flight", s["armed"])
check("Mode is GUIDED", s["flight_mode"] == "GUIDED", s["flight_mode"])

# Record position for map test
takeoff_lat = s["position"]["lat"]
takeoff_lon = s["position"]["lon"]

# Land
sio.emit("command", {"action": "land"})
s = wait_for(lambda s: not s["armed"], timeout=90)
check("Landed and disarmed", not s["armed"], f"armed={s['armed']}, alt={s['position']['alt']:.1f}")

sio.disconnect()
time.sleep(3)


# ── Test 6: Mode changes ────────────────────────────────
print("\n[6/8] Mode Changes")

sio = socketio.Client()
mode_log = []

@sio.on("telemetry")
def on_t6(data):
    if data.get("flight_mode") and (not mode_log or mode_log[-1] != data["flight_mode"]):
        mode_log.append(data["flight_mode"])

@sio.on("command_result")
def on_cmd6(data):
    pass

sio.connect(BASE)
time.sleep(1)

test_modes = ["STABILIZE", "LOITER", "ALT_HOLD", "GUIDED"]
for mode in test_modes:
    sio.emit("command", {"action": "set_mode", "mode": mode})
    time.sleep(1.5)
    s = status()
    check(f"Mode set to {mode}", s["flight_mode"] == mode, f"actual={s['flight_mode']}")

sio.disconnect()


# ── Test 7: Map data (GPS position tracking) ────────────
print("\n[7/8] Map Data (GPS Position Tracking)")

sio = socketio.Client()
positions = []

@sio.on("telemetry")
def on_t7(data):
    if data.get("position"):
        positions.append(data["position"])

@sio.on("command_result")
def on_cmd7(data):
    pass

sio.connect(BASE)
time.sleep(2)

c = config()
home_lat = c["simulation"]["home_lat"]
home_lon = c["simulation"]["home_lon"]

# Get current position
s = status()
check("Position lat non-zero", s["position"]["lat"] != 0, f"lat={s['position']['lat']}")
check("Position lon non-zero", s["position"]["lon"] != 0, f"lon={s['position']['lon']}")
check("Position near home lat", abs(s["position"]["lat"] - home_lat) < 0.01,
      f"lat={s['position']['lat']}, home={home_lat}")
check("Position near home lon", abs(s["position"]["lon"] - home_lon) < 0.01,
      f"lon={s['position']['lon']}, home={home_lon}")

# Takeoff and fly to check position changes
positions.clear()
sio.emit("command", {"action": "takeoff", "altitude": 20})
# Wait longer — takeoff needs GUIDED + arm + actual climb
s = wait_for(lambda s: s["position"]["alt"] > 15, timeout=60)
check("Altitude updates for map", s["position"]["alt"] > 15, f"alt={s['position']['alt']:.1f}")

# Collect more telemetry while in air
time.sleep(3)
unique_alts = set(round(p["alt"], 0) for p in positions)
check("Multiple altitude values streamed", len(unique_alts) > 3, f"unique_alts={len(unique_alts)}")

# Check heading is a number
check("Heading is numeric", isinstance(s.get("heading", None), (int, float)), f"heading={s.get('heading')}")

# Land
positions.clear()
sio.emit("command", {"action": "land"})
wait_for(lambda s: s["position"]["alt"] < 5, timeout=60)
time.sleep(3)

# Check position updated during descent
if positions:
    descent_alts = [p["alt"] for p in positions]
    check("Position updated during descent", max(descent_alts) > min(descent_alts) + 2,
          f"alt range: {min(descent_alts):.1f}-{max(descent_alts):.1f}")
else:
    check("Position updated during descent", False, "no positions received")

wait_for(lambda s: not s["armed"], timeout=60)

sio.disconnect()
time.sleep(5)


# ── Test 8: Roundtrip Demo ──────────────────────────────
skip_demo = "--quick" in sys.argv
if skip_demo:
    print("\n[8/8] Roundtrip Demo — SKIPPED (--quick)")
else:
    print("\n[8/8] Roundtrip Demo (100m out & back)")

    sio = socketio.Client()
    demo_msgs = []
    demo_done = {"value": False, "success": False}

    @sio.on("demo_status")
    def on_demo(data):
        demo_msgs.append(data["message"])
        if data.get("done"):
            demo_done["value"] = True
            demo_done["success"] = "complete" in data["message"].lower() or "Complete" in data["message"]

    @sio.on("command_result")
    def on_cmd8(data):
        pass

    @sio.on("telemetry")
    def on_t8(data):
        pass

    sio.connect(BASE)
    time.sleep(1)

    # Record home position
    s = status()
    home_lat_actual = s["position"]["lat"]
    home_lon_actual = s["position"]["lon"]

    # Start demo
    sio.emit("demo_roundtrip")

    # Wait for completion (up to 5 min)
    start = time.time()
    while not demo_done["value"] and time.time() - start < 300:
        time.sleep(1)

    check("Demo completed", demo_done["value"], f"msgs={len(demo_msgs)}")
    check("Demo success", demo_done["success"], demo_msgs[-1] if demo_msgs else "no messages")

    # Check demo included expected phases
    all_msgs = " ".join(demo_msgs).lower()
    check("Demo had takeoff phase", "taking off" in all_msgs or "climbing" in all_msgs, "no takeoff msg")
    check("Demo had flight phase", "flying" in all_msgs or "en route" in all_msgs, "no flight msg")
    check("Demo had landing phase", "landing" in all_msgs or "landed" in all_msgs, "no landing msg")
    check("Demo had wait phase", "waiting" in all_msgs, "no wait msg")
    check("Demo had return phase", "return" in all_msgs or "back" in all_msgs or "home" in all_msgs, "no return msg")

    # Check drone is back near home
    time.sleep(2)
    s = status()
    dist_from_home = math.sqrt(
        ((s["position"]["lat"] - home_lat_actual) * 110540) ** 2 +
        ((s["position"]["lon"] - home_lon_actual) * 111320 * math.cos(math.radians(home_lat_actual))) ** 2
    )
    check("Drone back near home (<10m)", dist_from_home < 10, f"dist={dist_from_home:.1f}m")
    check("Drone disarmed after demo", not s["armed"], f"armed={s['armed']}")

    sio.disconnect()


# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
