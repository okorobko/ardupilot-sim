"""MAVLink bridge: reads telemetry from ArduPilot SITL, emits via SocketIO."""

import math
import time
import threading
from pymavlink import mavutil


class MAVLinkBridge:
    """Connects to ArduPilot SITL via MAVLink UDP and streams telemetry."""

    def __init__(self, socketio, port=14550):
        self.socketio = socketio
        self.port = port
        self.conn = None
        self.running = False
        self._thread = None

        # Telemetry state
        self.connected = False
        self.armed = False
        self.flight_mode = "UNKNOWN"
        self.position = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.attitude = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.heading = 0
        self.groundspeed = 0.0
        self.airspeed = 0.0
        self.climb_rate = 0.0
        self.battery = {"voltage": 0.0, "remaining": 100}
        self.gps = {"fix_type": 0, "satellites": 0}

        # ArduCopter mode mapping
        self.COPTER_MODES = {
            0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO",
            4: "GUIDED", 5: "LOITER", 6: "RTL", 7: "CIRCLE",
            9: "LAND", 11: "DRIFT", 13: "SPORT", 14: "FLIP",
            15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE",
            18: "THROW", 19: "AVOID_ADSB", 20: "GUIDED_NOGPS",
            21: "SMART_RTL", 22: "FLOWHOLD", 23: "FOLLOW",
            24: "ZIGZAG", 25: "SYSTEMID", 26: "AUTOROTATE",
            27: "AUTO_RTL",
        }

    def start(self):
        """Start the bridge in a background thread."""
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the bridge."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        """Main loop: connect and read messages."""
        print(f"  MAVLink bridge connecting to tcp:127.0.0.1:{self.port}...")
        self.conn = mavutil.mavlink_connection(
            f"tcp:127.0.0.1:{self.port}",
            source_system=255,
            source_component=0,
        )

        # Wait for heartbeat
        print("  Waiting for heartbeat...")
        self.conn.wait_heartbeat(timeout=60)
        self.connected = True
        print(f"  Heartbeat received (system {self.conn.target_system}, "
              f"component {self.conn.target_component})")

        # Request message streams at desired rates
        self._request_streams()

        last_emit = 0
        last_log = 0
        msg_counts = {}
        while self.running:
            msg = self.conn.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue

            msg_type = msg.get_type()
            msg_counts[msg_type] = msg_counts.get(msg_type, 0) + 1

            if msg_type == "HEARTBEAT":
                self._handle_heartbeat(msg)
            elif msg_type == "GLOBAL_POSITION_INT":
                self._handle_position(msg)
            elif msg_type == "ATTITUDE":
                self._handle_attitude(msg)
            elif msg_type == "SYS_STATUS":
                self._handle_sys_status(msg)
            elif msg_type == "VFR_HUD":
                self._handle_vfr_hud(msg)
            elif msg_type == "GPS_RAW_INT":
                self._handle_gps_raw(msg)

            now = time.time()

            # Emit consolidated telemetry at ~10Hz
            if now - last_emit >= 0.1:
                last_emit = now
                self._emit_telemetry()

            # Emit drone log at ~1Hz showing live SITL data
            if now - last_log >= 1.0:
                last_log = now
                self._emit_drone_log(msg_counts)
                msg_counts = {}

    def _request_streams(self):
        """Request specific message streams at desired rates."""
        # Request all data streams at 10Hz
        self.conn.mav.request_data_stream_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,  # Hz
            1,   # start
        )

    def _handle_heartbeat(self, msg):
        if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
            return
        self.armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        mode_num = msg.custom_mode
        self.flight_mode = self.COPTER_MODES.get(mode_num, f"MODE_{mode_num}")

    def _handle_position(self, msg):
        self.position = {
            "lat": msg.lat / 1e7,
            "lon": msg.lon / 1e7,
            "alt": msg.relative_alt / 1000.0,
        }
        self.heading = msg.hdg / 100.0

    def _handle_attitude(self, msg):
        self.attitude = {
            "roll": math.degrees(msg.roll),
            "pitch": math.degrees(msg.pitch),
            "yaw": math.degrees(msg.yaw),
        }

    def _handle_sys_status(self, msg):
        self.battery = {
            "voltage": msg.voltage_battery / 1000.0,
            "remaining": msg.battery_remaining if msg.battery_remaining >= 0 else 100,
        }

    def _handle_vfr_hud(self, msg):
        self.groundspeed = msg.groundspeed
        self.airspeed = msg.airspeed
        self.climb_rate = msg.climb
        self.heading = msg.heading

    def _handle_gps_raw(self, msg):
        self.gps = {
            "fix_type": msg.fix_type,
            "satellites": msg.satellites_visible,
        }

    def _emit_telemetry(self):
        """Emit current telemetry state via SocketIO."""
        self.socketio.emit("telemetry", {
            "position": self.position,
            "attitude": self.attitude,
            "heading": self.heading,
            "groundspeed": round(self.groundspeed, 1),
            "airspeed": round(self.airspeed, 1),
            "climb_rate": round(self.climb_rate, 1),
            "battery": self.battery,
            "gps": self.gps,
            "armed": self.armed,
            "flight_mode": self.flight_mode,
            "connected": self.connected,
        })

    def _emit_drone_log(self, msg_counts):
        """Emit a drone log line showing live SITL state at 1Hz."""
        pos = self.position
        att = self.attitude
        # Show MAVLink message rates
        rates = ", ".join(f"{k}:{v}" for k, v in sorted(msg_counts.items())
                         if k not in ("BAD_DATA",))
        self.socketio.emit("log", {
            "tag": "DRONE",
            "message": (
                f"pos=({pos['lat']:.6f},{pos['lon']:.6f}) "
                f"alt={pos['alt']:.1f}m "
                f"hdg={self.heading:.0f} "
                f"spd={self.groundspeed:.1f}m/s "
                f"mode={self.flight_mode} "
                f"{'ARMED' if self.armed else 'DISARMED'}"
            ),
        })
        self.socketio.emit("log", {
            "tag": "DRONE",
            "message": f"MAVLink rx/s: {rates}",
            "cls": "log-telem",
        })

    # ── Commands ──────────────────────────────────────────────

    def arm(self):
        """Arm the vehicle. Waits for armed state instead of ACK."""
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        # Wait for state change instead of ACK (avoids recv_match conflicts)
        for _ in range(30):
            time.sleep(0.2)
            if self.armed:
                return {"success": True, "message": "Armed"}
        return {"success": False, "message": "Arm timeout"}

    def disarm(self):
        """Disarm the vehicle. Waits for disarmed state."""
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        for _ in range(30):
            time.sleep(0.2)
            if not self.armed:
                return {"success": True, "message": "Disarmed"}
        return {"success": False, "message": "Disarm timeout"}

    def takeoff(self, altitude=10):
        """Takeoff to specified altitude. Waits for positive climb."""
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude,
        )
        # Wait for drone to start climbing (alt > 1m)
        for _ in range(40):
            time.sleep(0.25)
            if self.position["alt"] > 1.0:
                return {"success": True, "message": f"Taking off to {altitude}m"}
        return {"success": False, "message": "Takeoff timeout"}

    def set_mode(self, mode_name):
        """Set flight mode by name."""
        # Reverse lookup mode number
        mode_num = None
        for num, name in self.COPTER_MODES.items():
            if name == mode_name.upper():
                mode_num = num
                break

        if mode_num is None:
            return {"success": False, "message": f"Unknown mode: {mode_name}"}

        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_num,
        )
        return {"success": True, "message": f"Mode set to {mode_name}"}

    def send_velocity(self, vx, vy, vz, yaw_rate=0):
        """Send velocity command in body frame, converted to NED.

        vx: forward (m/s), vy: right (m/s), vz: down (m/s), yaw_rate: rad/s
        Converts body→NED manually and holds yaw via position target.
        """
        # Body to NED conversion using current heading
        hdg_rad = math.radians(self.heading)
        vn = vx * math.cos(hdg_rad) - vy * math.sin(hdg_rad)
        ve = vx * math.sin(hdg_rad) + vy * math.cos(hdg_rad)
        vd = vz  # NED down

        if yaw_rate != 0:
            # Yaw rate mode
            type_mask = 0b0000011111000111  # vx,vy,vz + yaw_rate
            yaw = 0
        else:
            # Hold current heading
            type_mask = 0b0000101111000111  # vx,vy,vz + yaw
            yaw = hdg_rad
            yaw_rate = 0

        self.conn.mav.set_position_target_local_ned_send(
            0,
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,
            vn, ve, vd,
            0, 0, 0,
            yaw, yaw_rate,
        )

    def goto_position(self, lat, lon, alt):
        """Send drone to a GPS position at given altitude (GUIDED mode)."""
        self.conn.mav.set_position_target_global_int_send(
            0,  # time_boot_ms
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # type_mask: use only lat/lon/alt
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,  # vx, vy, vz
            0, 0, 0,  # afx, afy, afz
            0, 0,      # yaw, yaw_rate
        )

    def wait_altitude(self, target_alt, tolerance=2.0, timeout=60, callback=None):
        """Wait until drone reaches target altitude."""
        start = time.time()
        while time.time() - start < timeout:
            if abs(self.position["alt"] - target_alt) < tolerance:
                return True
            if callback:
                callback(self.position["alt"])
            time.sleep(0.5)
        return False

    def wait_position(self, target_lat, target_lon, tolerance=3.0, timeout=120, callback=None):
        """Wait until drone reaches target lat/lon (within tolerance meters)."""
        start = time.time()
        while time.time() - start < timeout:
            dlat = (self.position["lat"] - target_lat) * 110540
            dlon = (self.position["lon"] - target_lon) * 111320 * math.cos(math.radians(target_lat))
            dist = math.sqrt(dlat**2 + dlon**2)
            if dist < tolerance:
                return True
            if callback:
                callback(dist, self.position["alt"])
            time.sleep(0.5)
        return False

    def wait_disarmed(self, timeout=60):
        """Wait until drone disarms (landed)."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.armed:
                return True
            time.sleep(0.5)
        return False

    def demo_roundtrip(self, status_callback=None):
        """Demo flight: 100m out at 50m, land, wait 25s, fly back at 75m.

        Runs in a background thread. status_callback(msg) is called with progress.
        """
        def log(msg):
            print(f"  [Demo] {msg}")
            if status_callback:
                status_callback(msg)

        home_lat = self.position["lat"]
        home_lon = self.position["lon"]

        # Target: 100m north
        target_lat = home_lat + (100.0 / 110540.0)
        target_lon = home_lon

        try:
            # ── LEG 1: Fly 100m at 50m altitude ──
            log("Setting GUIDED mode...")
            self.set_mode("GUIDED")
            time.sleep(1)

            log("Arming...")
            result = self.arm()
            if not result["success"]:
                log(f"Arm failed: {result['message']}")
                return {"success": False, "message": result["message"]}
            time.sleep(1)

            log("Taking off to 50m...")
            self.takeoff(50)
            self.wait_altitude(50, tolerance=3, timeout=60,
                               callback=lambda alt: log(f"Climbing... {alt:.1f}m"))

            log(f"Flying 100m north to ({target_lat:.6f}, {target_lon:.6f})...")
            self.goto_position(target_lat, target_lon, 50)
            self.wait_position(target_lat, target_lon, tolerance=3, timeout=60,
                               callback=lambda d, a: log(f"En route... {d:.0f}m remaining, alt={a:.1f}m"))

            log("Reached waypoint. Landing...")
            self.set_mode("LAND")
            self.wait_disarmed(timeout=120)
            log("Landed. Waiting 25 seconds...")
            time.sleep(25)

            # ── LEG 2: Fly back at 75m altitude ──
            log("Setting GUIDED mode for return...")
            self.set_mode("GUIDED")
            time.sleep(1)

            log("Arming for return...")
            result = self.arm()
            if not result["success"]:
                log(f"Arm failed: {result['message']}")
                return {"success": False, "message": result["message"]}
            time.sleep(1)

            log("Taking off to 75m...")
            self.takeoff(75)
            self.wait_altitude(75, tolerance=3, timeout=90,
                               callback=lambda alt: log(f"Climbing... {alt:.1f}m"))

            log(f"Flying back to home ({home_lat:.6f}, {home_lon:.6f})...")
            self.goto_position(home_lat, home_lon, 75)
            self.wait_position(home_lat, home_lon, tolerance=3, timeout=60,
                               callback=lambda d, a: log(f"Returning... {d:.0f}m remaining, alt={a:.1f}m"))

            log("Back at home. Landing...")
            self.set_mode("LAND")
            self.wait_disarmed(timeout=120)
            log("Demo complete! Landed at home.")

            return {"success": True, "message": "Roundtrip demo complete!"}

        except Exception as e:
            log(f"Error: {e}")
            try:
                self.set_mode("LAND")
            except Exception:
                pass
            return {"success": False, "message": str(e)}

