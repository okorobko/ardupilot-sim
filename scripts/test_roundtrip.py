#!/usr/bin/env python3
"""Test: Fly 100m north at 50m, land, wait 25s, fly back at 75m.

Usage: python3 scripts/test_roundtrip.py
Requires: SITL running + backend on localhost:5001
"""

import json
import time
import urllib.request


BASE = "http://localhost:5001"


def get_status():
    return json.loads(urllib.request.urlopen(f"{BASE}/api/status").read())


def main():
    print("=" * 60)
    print("  Roundtrip Demo Test")
    print("  Leg 1: 100m north at 50m altitude")
    print("  Land + wait 25s")
    print("  Leg 2: fly back home at 75m altitude")
    print("=" * 60)

    # Check connectivity
    s = get_status()
    if not s["connected"]:
        print("ERROR: Backend not connected to SITL")
        return

    print(f"Connected. Home: ({s['position']['lat']:.6f}, {s['position']['lon']:.6f})")

    # Trigger demo via SocketIO
    import socketio

    sio = socketio.Client()
    demo_done = {"value": False}

    @sio.on("demo_status")
    def on_status(data):
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {data['message']}")
        if data.get("done"):
            demo_done["value"] = True

    @sio.on("command_result")
    def on_result(data):
        if not data["success"] and "already" in data.get("message", "").lower():
            print(f"  Result: {data['message']}")

    sio.connect(BASE)
    print("\nStarting demo...\n")
    sio.emit("demo_roundtrip")

    # Wait for completion
    while not demo_done["value"]:
        time.sleep(1)

    # Final status
    time.sleep(2)
    s = get_status()
    print(f"\nFinal: alt={s['position']['alt']:.1f}m "
          f"mode={s['flight_mode']} armed={s['armed']}")
    print("\nTest complete!")

    sio.disconnect()


if __name__ == "__main__":
    main()
