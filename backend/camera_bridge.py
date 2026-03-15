#!/usr/bin/env python3
"""Camera bridge: captures frames from Gazebo camera and sends to Flask backend.

Captures single frames from Gazebo `/camera` topic using `gz topic -e -n 1`,
converts raw RGB data to JPEG, and sends via SocketIO to the web dashboard.

Usage (requires gz_garden conda env):
    conda activate gz_garden
    python3 backend/camera_bridge.py
"""

import base64
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import socketio

BACKEND_URL = "http://localhost:5001"
GZ_TOPIC = "/camera"
FPS = 5  # Target frames per second
JPEG_QUALITY = 55


def capture_frame():
    """Capture one frame from Gazebo camera topic."""
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-t", GZ_TOPIC, "-n", "1"],
            capture_output=True, timeout=5,
        )
        text = result.stdout.decode("utf-8", errors="replace")

        # Parse dimensions
        w_match = re.search(r"^width:\s*(\d+)", text, re.M)
        h_match = re.search(r"^height:\s*(\d+)", text, re.M)
        if not w_match or not h_match:
            return None

        width = int(w_match.group(1))
        height = int(h_match.group(1))

        # Extract the data field — escaped binary between quotes
        data_match = re.search(r'^data:\s*"(.*)"', text, re.M | re.S)
        if not data_match:
            return None

        raw_escaped = data_match.group(1)

        # Decode octal escapes (\220 etc) and other escapes
        raw_bytes = raw_escaped.encode("utf-8").decode("unicode_escape").encode("latin-1")

        expected = width * height * 3
        if len(raw_bytes) < expected:
            return None

        # Convert to image
        img = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape((height, width, 3))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Encode JPEG
        _, jpeg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  Frame error: {e}")
        return None


def main():
    print("=" * 50, flush=True)
    print("  Gazebo Camera Bridge", flush=True)
    print(f"  Topic: {GZ_TOPIC}", flush=True)
    print(f"  Backend: {BACKEND_URL}", flush=True)
    print(f"  Target FPS: {FPS}", flush=True)
    print("=" * 50, flush=True)

    sio = socketio.Client()
    print("Connecting to backend...", flush=True)
    sio.connect(BACKEND_URL)
    print("Connected!", flush=True)

    frame_count = 0
    interval = 1.0 / FPS

    try:
        while True:
            start = time.time()

            b64 = capture_frame()
            if b64:
                sio.emit("camera_frame", {"data": b64})
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"  Sent {frame_count} frames", flush=True)
            else:
                if frame_count == 0:
                    print("  Waiting for camera frames...", flush=True)

            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\nStopping after {frame_count} frames")
    finally:
        sio.disconnect()


if __name__ == "__main__":
    main()
