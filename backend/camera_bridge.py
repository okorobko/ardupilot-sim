#!/usr/bin/env python3
"""Camera bridge: streams Gazebo cameras to Flask backend via SocketIO.

Captures frames from two Gazebo camera topics:
  - /chase_cam (800x600) → main 3D panel (third-person view)
  - /camera (640x480) → overlay (downward view for ML detection)

Optionally runs YOLOv8n military vehicle detection on downward camera frames.

Usage (requires gz_garden conda env):
    conda activate gz_garden
    python3 backend/camera_bridge.py [--detect path/to/model.onnx]
"""

import argparse
import base64
import json
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

# Global detector instance (None if detection disabled)
_detector = None


def capture_frame_raw(topic, width=640, height=480):
    """Capture one raw BGR frame from a Gazebo camera topic.

    Returns:
        numpy array (H, W, 3) BGR or None on failure.
    """
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
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def frame_to_b64(img_bgr):
    """Encode BGR frame as base64 JPEG."""
    _, jpeg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


def capture_frame(topic, width=640, height=480):
    """Capture one frame from a Gazebo camera topic as base64 JPEG."""
    img = capture_frame_raw(topic, width, height)
    if img is None:
        return None
    return frame_to_b64(img)


def stream_camera(sio, topic, event_name, fps, label, run_detection=False,
                   detection_event="detection_results", tracker=None):
    """Stream a camera topic to the backend.

    Args:
        sio: SocketIO client.
        topic: Gazebo camera topic.
        event_name: SocketIO event name for frame data.
        fps: Target frames per second.
        label: Label for logging.
        run_detection: If True, run ML detection on frames and emit results.
        detection_event: SocketIO event name for detection results.
    """
    interval = 1.0 / fps
    count = 0
    detect_count = 0
    print(f"  [{label}] Streaming {topic} → {event_name} at {fps}fps"
          f"{' + detection' if run_detection else ''}", flush=True)

    while True:
        start = time.time()

        if run_detection and _detector is not None:
            # Capture raw frame for detection
            img_bgr = capture_frame_raw(topic)
            if img_bgr is not None:
                # Run detection
                raw_detections = _detector.detect(img_bgr)
                latency = _detector.latency_ms

                # Apply tracker for temporal smoothing (if available)
                if tracker is not None:
                    detections = tracker.update(raw_detections)
                else:
                    detections = raw_detections

                # Encode frame (without drawn boxes — frontend draws them)
                b64 = frame_to_b64(img_bgr)
                sio.emit(event_name, {"data": b64})

                # Emit tracked detection results (always emit so frontend can clear)
                sio.emit(detection_event, {
                    "detections": detections,
                    "latency_ms": round(latency, 1),
                    "frame_id": count,
                })
                if detections:
                    detect_count += 1

                count += 1
                if count % (fps * 10) == 0:
                    print(f"  [{label}] {count} frames, {detect_count} w/detections, "
                          f"latency={latency:.1f}ms", flush=True)
        else:
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
    global _detector

    parser = argparse.ArgumentParser(description="Gazebo Camera Bridge")
    parser.add_argument("--detect", type=str, default=None,
                        help="Path to ONNX model for vehicle detection")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--backend", type=str, default=BACKEND_URL,
                        help=f"Backend URL (default: {BACKEND_URL})")
    args = parser.parse_args()

    # Initialize detector and trackers if model provided
    down_tracker = None
    chase_tracker = None
    if args.detect:
        try:
            from detector import VehicleDetector, SimpleTracker
            _detector = VehicleDetector(
                args.detect,
                conf_threshold=args.conf,
            )
            down_tracker = SimpleTracker(iou_threshold=0.3, min_hits=2, max_age=8)
            chase_tracker = SimpleTracker(iou_threshold=0.3, min_hits=2, max_age=8)
            print(f"  Detector loaded: {args.detect}", flush=True)
            print(f"  Confidence threshold: {args.conf}", flush=True)
        except Exception as e:
            print(f"  WARNING: Failed to load detector: {e}", flush=True)
            print(f"  Continuing without detection...", flush=True)

    detect_enabled = _detector is not None

    print("=" * 50, flush=True)
    print("  Gazebo Camera Bridge (dual)", flush=True)
    print("  Down cam:  /camera → camera_frame (5fps, main)", flush=True)
    print("  Chase cam: /chase_cam → chase_frame (3fps, overlay)", flush=True)
    if detect_enabled:
        print("  Detection: ENABLED (military vehicles)", flush=True)
    print(f"  Backend: {args.backend}", flush=True)
    print("=" * 50, flush=True)

    sio = socketio.Client()
    print("Connecting to backend...", flush=True)
    sio.connect(args.backend)
    print("Connected!", flush=True)

    # Stream chase cam in a thread (with detection + tracking)
    t_chase = threading.Thread(
        target=stream_camera,
        args=(sio, "/chase_cam", "chase_frame", 3, "CHASE"),
        kwargs={"run_detection": detect_enabled,
                "detection_event": "detection_results_chase",
                "tracker": chase_tracker},
        daemon=True,
    )
    t_chase.start()

    # Stream downward cam in main thread (main view, higher fps + detection + tracking)
    try:
        stream_camera(sio, "/camera", "camera_frame", 5, "DOWN",
                       run_detection=detect_enabled, tracker=down_tracker)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        sio.disconnect()


if __name__ == "__main__":
    main()
