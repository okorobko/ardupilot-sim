"""Flask + SocketIO backend for ArduPilot drone simulator."""

import os
import sys
import threading

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from config_loader import load_config, validate_config
from mavlink_bridge import MAVLinkBridge

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "frontend", "templates"),
)
app.config["SECRET_KEY"] = "ardupilot-sim-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# Load configuration
config = load_config()
errors = validate_config(config)
if errors:
    print("Configuration warnings:")
    for e in errors:
        print(f"  - {e}")

# Create MAVLink bridge
bridge = MAVLinkBridge(socketio, port=config["simulation"]["mavlink_port"])

# Demo flight state
_demo_thread = None


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "connected": bridge.connected,
        "armed": bridge.armed,
        "flight_mode": bridge.flight_mode,
        "position": bridge.position,
        "attitude": bridge.attitude,
        "heading": bridge.heading,
        "groundspeed": bridge.groundspeed,
        "battery": bridge.battery,
        "gps": bridge.gps,
    })


@app.route("/api/config")
def api_config():
    return jsonify(config)


# ── SocketIO events ──────────────────────────────────────────

@socketio.on("connect")
def handle_connect(*args, **kwargs):
    print("Browser client connected")
    emit("config", config)
    emit("telemetry", {
        "position": bridge.position,
        "attitude": bridge.attitude,
        "heading": bridge.heading,
        "groundspeed": 0,
        "airspeed": 0,
        "climb_rate": 0,
        "battery": bridge.battery,
        "gps": bridge.gps,
        "armed": bridge.armed,
        "flight_mode": bridge.flight_mode,
        "connected": bridge.connected,
    })


@socketio.on("command")
def handle_command(data):
    cmd = data.get("action", "")
    print(f"Command: {cmd}")

    if cmd == "arm":
        socketio.emit("log", {"message": "MAV_CMD_COMPONENT_ARM_DISARM (arm=1)", "tag": "CONTROL"})
        result = bridge.arm()
        socketio.emit("log", {"message": f"ACK arm: {result['message']}", "tag": "DRONE"})
    elif cmd == "disarm":
        socketio.emit("log", {"message": "MAV_CMD_COMPONENT_ARM_DISARM (arm=0)", "tag": "CONTROL"})
        result = bridge.disarm()
        socketio.emit("log", {"message": f"ACK disarm: {result['message']}", "tag": "DRONE"})
    elif cmd == "takeoff":
        import time
        altitude = data.get("altitude", 10)
        # Set GUIDED mode
        socketio.emit("log", {"message": "SET_MODE GUIDED", "tag": "CONTROL"})
        bridge.set_mode("GUIDED")
        time.sleep(1)
        # Auto-arm if not armed
        if not bridge.armed:
            socketio.emit("log", {"message": "MAV_CMD_COMPONENT_ARM_DISARM (auto-arm)", "tag": "CONTROL"})
            arm_result = bridge.arm()
            socketio.emit("log", {"message": f"ACK arm: {arm_result['message']}", "tag": "DRONE"})
            if not arm_result["success"]:
                result = arm_result
                emit("command_result", result)
                return
            time.sleep(1)
        # Takeoff
        socketio.emit("log", {"message": f"MAV_CMD_NAV_TAKEOFF alt={altitude}m", "tag": "CONTROL"})
        result = bridge.takeoff(altitude)
        socketio.emit("log", {"message": f"ACK takeoff: {result['message']}", "tag": "DRONE"})
    elif cmd == "land":
        socketio.emit("log", {"message": "SET_MODE LAND", "tag": "CONTROL"})
        result = bridge.set_mode("LAND")
    elif cmd == "rtl":
        socketio.emit("log", {"message": "SET_MODE RTL", "tag": "CONTROL"})
        result = bridge.set_mode("RTL")
    elif cmd == "set_mode":
        mode = data.get("mode", "LOITER")
        socketio.emit("log", {"message": f"SET_MODE {mode}", "tag": "CONTROL"})
        result = bridge.set_mode(mode)
    else:
        result = {"success": False, "message": f"Unknown command: {cmd}"}

    emit("command_result", result)


@socketio.on("fly")
def handle_fly(data):
    """Handle keyboard flight velocity commands."""
    vx = data.get("vx", 0)  # forward/back
    vy = data.get("vy", 0)  # left/right
    vz = data.get("vz", 0)  # up/down (NED: negative = up)
    yr = data.get("yaw_rate", 0)
    bridge.send_velocity(vx, vy, vz, yr)


@socketio.on("camera_frame")
def handle_camera_frame(data):
    """Forward camera frame from bridge to all browser clients."""
    socketio.emit("camera_frame", data)


@socketio.on("demo_roundtrip")
def handle_demo_roundtrip(data=None):
    global _demo_thread
    if _demo_thread and _demo_thread.is_alive():
        emit("command_result", {"success": False, "message": "Demo already in progress"})
        return

    emit("command_result", {"success": True, "message": "Roundtrip demo started!"})

    def _run():
        def status_cb(msg):
            socketio.emit("demo_status", {"message": msg})

        result = bridge.demo_roundtrip(status_callback=status_cb)
        socketio.emit("demo_status", {"message": result["message"], "done": True})

    _demo_thread = threading.Thread(target=_run, daemon=True)
    _demo_thread.start()


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ArduPilot Drone Simulator — Web Dashboard")
    print("=" * 60)

    # Start MAVLink bridge
    print("\n[1/2] Starting MAVLink bridge...")
    bridge.start()

    # Start web server
    print("[2/2] Starting web server...")
    print("=" * 60)
    print(f"  Dashboard: http://localhost:5001")
    print(f"  MAVLink:   udp:127.0.0.1:{config['simulation']['mavlink_port']}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")

    socketio.run(app, host="0.0.0.0", port=5001, debug=False,
                 allow_unsafe_werkzeug=True)
