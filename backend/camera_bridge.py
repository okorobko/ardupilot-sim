#!/usr/bin/env python3
"""Camera bridge: streams Gazebo cameras to Flask backend via SocketIO.

Captures frames from two Gazebo camera topics:
  - /chase_cam (800x600) → main 3D panel (third-person view)
  - /camera (640x480) → overlay (downward view for ML detection)

Usage (requires gz_garden conda env):
    conda activate gz_garden
    python3 backend/camera_bridge.py
"""

import base64
import re
import subprocess
import sys
import time
import threading

import cv2
import numpy as np
import socketio

BACKEND_URL = "http://localhost:5001"
JPEG_QUALITY = 55


def capture_frame(topic, width=640, height=480):
    """Capture one frame from a Gazebo camera topic."""
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-t", topic, "-n", "1"],
            capture_output=True, timeout=5,
        )
        text = result.stdout.decode("utf-8", errors="replace")

        w_match = re.search(r"^width:\s*(\d+)", text, re.M)
        h_match = re.search(r"^height:\s*(\d+)", text, re.M)
        if not w_match or not h_match:
            return None

        w = int(w_match.group(1))
        h = int(h_match.group(1))

        data_match = re.search(r'^data:\s*"(.*)"', text, re.M | re.S)
        if not data_match:
            return None

        raw_bytes = data_match.group(1).encode("utf-8").decode("unicode_escape").encode("latin-1")
        expected = w * h * 3
        if len(raw_bytes) < expected:
            return None

        img = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape((h, w, 3))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, jpeg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None


def stream_camera(sio, topic, event_name, fps, label):
    """Stream a camera topic to the backend."""
    interval = 1.0 / fps
    count = 0
    print(f"  [{label}] Streaming {topic} → {event_name} at {fps}fps", flush=True)

    while True:
        start = time.time()
        b64 = capture_frame(topic)
        if b64:
            sio.emit(event_name, {"data": b64})
            count += 1
            if count % (fps * 10) == 0:
                print(f"  [{label}] {count} frames sent", flush=True)
        elapsed = time.time() - start
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    print("=" * 50, flush=True)
    print("  Gazebo Camera Bridge (dual)", flush=True)
    print("  Down cam:  /camera → camera_frame (5fps, main)", flush=True)
    print("  Chase cam: /chase_cam → chase_frame (3fps, overlay)", flush=True)
    print(f"  Backend: {BACKEND_URL}", flush=True)
    print("=" * 50, flush=True)

    sio = socketio.Client()
    print("Connecting to backend...", flush=True)
    sio.connect(BACKEND_URL)
    print("Connected!", flush=True)

    # Stream chase cam in a thread (overlay, lower fps)
    t_chase = threading.Thread(
        target=stream_camera,
        args=(sio, "/chase_cam", "chase_frame", 3, "CHASE"),
        daemon=True,
    )
    t_chase.start()

    # Stream downward cam in main thread (main view, higher fps)
    try:
        stream_camera(sio, "/camera", "camera_frame", 5, "DOWN")
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        sio.disconnect()


if __name__ == "__main__":
    main()
