#!/usr/bin/env python3
"""Generate synthetic training data from Gazebo simulator with military vehicles.

Captures frames from a Gazebo world by spawning a virtual camera at various
positions and altitudes, auto-labels objects using known 3D poses projected
through a pinhole camera model, and applies domain randomization.

Supports two modes:
  - Live: captures frames from a running Gazebo instance via `gz topic`
  - Offline (--offline): augments existing images using a vehicle_poses.json

Output is in YOLO format: image files + corresponding .txt label files.

Usage:
  # Live mode (Gazebo must be running):
  python generate_synthetic.py --num-images 1000

  # Offline augmentation mode:
  python generate_synthetic.py --offline --source-dir ./captured_frames

  # Custom altitudes and world:
  python generate_synthetic.py --altitudes "10,20,40,60" --world my_world
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Continuing without progress bars.\n")

    class tqdm:
        """Minimal fallback when tqdm is not available."""

        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable
            self._total = kwargs.get("total", None)
            desc = kwargs.get("desc", "")
            if desc:
                print(f"  {desc}...")

        def __iter__(self):
            return iter(self._iterable) if self._iterable else iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_postfix_str(self, s):
            pass

        @staticmethod
        def write(msg):
            print(msg)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VehiclePose:
    """A vehicle's pose and bounding box in the world."""
    name: str
    class_name: str  # YOLO class: sedan, suv, truck, apc, tank, etc.
    class_id: int
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    length: float = 4.5  # along local X
    width: float = 1.8   # along local Y
    height: float = 1.4   # along local Z


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    width: int = 640
    height: int = 480
    fov_horizontal: float = math.radians(60)  # radians

    @property
    def fx(self) -> float:
        return self.width / (2.0 * math.tan(self.fov_horizontal / 2.0))

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Vehicle class mapping
# ---------------------------------------------------------------------------

# YOLO class IDs for the military / vehicle detection task.
CLASS_MAP = {
    "sedan": 0,
    "suv": 1,
    "truck": 2,
    "apc": 3,
    "tank": 4,
    "helicopter": 5,
    "car": 0,  # alias
}

# Vehicle dimension presets (length, width, height).
VEHICLE_DIMENSIONS = {
    "sedan": (4.5, 1.8, 1.4),
    "car": (4.3, 1.7, 1.3),
    "suv": (4.8, 2.0, 1.7),
    "truck": (6.0, 2.2, 2.5),
    "apc": (6.5, 2.8, 2.3),
    "tank": (7.0, 3.6, 2.4),
}


# ---------------------------------------------------------------------------
# SDF world parser
# ---------------------------------------------------------------------------

def parse_vehicles_from_sdf(sdf_path: Path) -> list[VehiclePose]:
    """Extract vehicle models and their poses from a Gazebo SDF world file.

    Looks for <model> elements whose names match known vehicle patterns
    (car_, suv_, truck_, park_, apc_, tank_) and extracts their <pose> and
    bounding box <size> from the first <collision> geometry.

    Args:
        sdf_path: Path to the .sdf world file.

    Returns:
        List of VehiclePose objects for each detected vehicle model.
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    vehicles = []

    # Patterns that identify vehicle models (prefix -> class)
    vehicle_prefixes = {
        "car_": "sedan",
        "suv_": "suv",
        "truck_": "truck",
        "park_red": "sedan",
        "park_white": "car",
        "park_suv": "suv",
        "park_truck": "truck",
        "apc_": "apc",
        "tank_": "tank",
        "heli_": "helicopter",
    }

    # Search all <model> elements at any depth
    for model in root.iter("model"):
        model_name = model.get("name", "")

        # Determine vehicle class from name prefix
        class_name = None
        for prefix, cls in vehicle_prefixes.items():
            if model_name.startswith(prefix) or model_name == prefix.rstrip("_"):
                class_name = cls
                break

        if class_name is None:
            continue

        # Parse pose: "x y z roll pitch yaw"
        pose_elem = model.find("pose")
        if pose_elem is not None and pose_elem.text:
            parts = pose_elem.text.strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            roll = float(parts[3]) if len(parts) > 3 else 0.0
            pitch = float(parts[4]) if len(parts) > 4 else 0.0
            yaw = float(parts[5]) if len(parts) > 5 else 0.0
        else:
            x, y, z, roll, pitch, yaw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Try to get bounding box from collision geometry
        length, width, height = VEHICLE_DIMENSIONS.get(
            class_name, (4.5, 1.8, 1.4)
        )
        for collision in model.iter("collision"):
            size_elem = collision.find(".//box/size")
            if size_elem is not None and size_elem.text:
                sp = size_elem.text.strip().split()
                length, width, height = float(sp[0]), float(sp[1]), float(sp[2])
                break

        class_id = CLASS_MAP.get(class_name, 0)
        vehicles.append(VehiclePose(
            name=model_name,
            class_name=class_name,
            class_id=class_id,
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            length=length, width=width, height=height,
        ))

    return vehicles


# ---------------------------------------------------------------------------
# 3D-to-2D projection (pinhole camera model)
# ---------------------------------------------------------------------------

def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler angles (ZYX convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return R_z @ R_y @ R_x


def build_extrinsic(cam_x: float, cam_y: float, cam_z: float,
                    cam_roll: float, cam_pitch: float, cam_yaw: float) -> np.ndarray:
    """Build the 4x4 world-to-camera transform (extrinsic matrix).

    The camera in Gazebo looks down its +X axis. We convert to the standard
    CV convention where the camera looks down +Z, with +X right and +Y down.

    Args:
        cam_x, cam_y, cam_z: Camera position in world frame.
        cam_roll, cam_pitch, cam_yaw: Camera orientation in world frame (rad).

    Returns:
        4x4 extrinsic matrix that transforms world points to camera frame.
    """
    # Rotation from world to Gazebo camera body
    R_world_to_body = rotation_matrix(cam_roll, cam_pitch, cam_yaw).T

    # Gazebo camera convention: +X forward, +Y left, +Z up
    # OpenCV convention: +Z forward, +X right, +Y down
    # Transform: x_cv = y_gz_neg, y_cv = z_gz_neg, z_cv = x_gz
    gz_to_cv = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0],
    ], dtype=np.float64)

    R = gz_to_cv @ R_world_to_body
    t = R @ np.array([-cam_x, -cam_y, -cam_z])

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic


def project_3d_bbox_to_2d(
    vehicle: VehiclePose,
    cam_x: float, cam_y: float, cam_z: float,
    cam_roll: float, cam_pitch: float, cam_yaw: float,
    intrinsics: CameraIntrinsics,
) -> Optional[tuple[float, float, float, float]]:
    """Project a vehicle's 3D oriented bounding box to a 2D YOLO bbox.

    Computes the 8 corners of the 3D bounding box in world coordinates,
    projects them through the pinhole camera model, and returns the
    enclosing 2D axis-aligned bounding box in YOLO normalized format.

    Args:
        vehicle: Vehicle pose and dimensions.
        cam_x, cam_y, cam_z: Camera world position.
        cam_roll, cam_pitch, cam_yaw: Camera world orientation (rad).
        intrinsics: Camera intrinsic parameters.

    Returns:
        Tuple (cx, cy, w, h) in YOLO normalized coords [0, 1], or None if
        the vehicle is behind the camera or fully outside the frame.
    """
    # 8 corners of the local bounding box (centered at vehicle origin)
    hl, hw, hh = vehicle.length / 2, vehicle.width / 2, vehicle.height / 2
    local_corners = np.array([
        [+hl, +hw, +hh],
        [+hl, +hw, -hh],
        [+hl, -hw, +hh],
        [+hl, -hw, -hh],
        [-hl, +hw, +hh],
        [-hl, +hw, -hh],
        [-hl, -hw, +hh],
        [-hl, -hw, -hh],
    ])

    # Rotate corners by vehicle yaw and translate to world position
    R_veh = rotation_matrix(vehicle.roll, vehicle.pitch, vehicle.yaw)
    world_corners = (R_veh @ local_corners.T).T + np.array([vehicle.x, vehicle.y, vehicle.z])

    # Build camera extrinsic
    extrinsic = build_extrinsic(cam_x, cam_y, cam_z, cam_roll, cam_pitch, cam_yaw)
    K = intrinsics.K

    # Project each corner
    pixels = []
    for corner in world_corners:
        p_hom = np.append(corner, 1.0)
        p_cam = extrinsic @ p_hom
        # Behind camera check
        if p_cam[2] <= 0:
            return None
        p_img = K @ p_cam[:3]
        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]
        pixels.append((u, v))

    pixels = np.array(pixels)
    u_min, v_min = pixels.min(axis=0)
    u_max, v_max = pixels.max(axis=0)

    # Clip to image bounds
    u_min = max(0.0, u_min)
    v_min = max(0.0, v_min)
    u_max = min(float(intrinsics.width), u_max)
    v_max = min(float(intrinsics.height), v_max)

    # Discard if too small or fully outside
    if u_max - u_min < 2.0 or v_max - v_min < 2.0:
        return None

    # Convert to YOLO normalized format: center_x, center_y, width, height
    cx = ((u_min + u_max) / 2.0) / intrinsics.width
    cy = ((v_min + v_max) / 2.0) / intrinsics.height
    bw = (u_max - u_min) / intrinsics.width
    bh = (v_max - v_min) / intrinsics.height

    return (cx, cy, bw, bh)


# ---------------------------------------------------------------------------
# Gazebo frame capture (same pattern as camera_bridge.py)
# ---------------------------------------------------------------------------

def capture_gz_frame(topic: str, width: int = 640, height: int = 480,
                     timeout: float = 5.0) -> Optional[np.ndarray]:
    """Capture a single frame from a Gazebo camera topic via gz CLI.

    Uses `gz topic -e -t <topic> -n 1` and parses the text output to
    extract raw RGB image data, following the same pattern as
    backend/camera_bridge.py.

    Args:
        topic: Gazebo transport topic name (e.g., /synthetic_cam).
        width: Expected image width.
        height: Expected image height.
        timeout: Subprocess timeout in seconds.

    Returns:
        BGR numpy array (H, W, 3) or None on failure.
    """
    try:
        result = subprocess.run(
            ["gz", "topic", "-e", "-t", topic, "-n", "1"],
            capture_output=True, timeout=timeout,
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

        raw = (
            data_match.group(1)
            .encode("utf-8")
            .decode("unicode_escape")
            .encode("latin-1")
        )
        expected = w * h * 3
        if len(raw) < expected:
            return None

        img = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((h, w, 3))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gazebo service helpers
# ---------------------------------------------------------------------------

def set_gz_camera_pose(world: str, cam_model: str,
                       x: float, y: float, z: float,
                       roll: float, pitch: float, yaw: float) -> bool:
    """Move the synthetic camera model to a new pose via gz service.

    Uses the /world/<world>/set_pose service to teleport the camera model.

    Args:
        world: Gazebo world name.
        cam_model: Name of the camera model entity.
        x, y, z: Target world position.
        roll, pitch, yaw: Target orientation in radians.

    Returns:
        True if the service call succeeded.
    """
    # Convert Euler to quaternion for gz.msgs.Pose
    qx, qy, qz, qw = _euler_to_quaternion(roll, pitch, yaw)

    req = (
        f'name: "{cam_model}", '
        f'position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}'
    )
    try:
        subprocess.run(
            [
                "gz", "service",
                "-s", f"/world/{world}/set_pose",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000",
                "--req", req,
            ],
            capture_output=True, timeout=5,
        )
        return True
    except Exception:
        return False


def set_gz_sun_angle(world: str, direction: tuple[float, float, float]) -> bool:
    """Change the sun light direction for domain randomization.

    Args:
        world: Gazebo world name.
        direction: (x, y, z) direction vector for the directional light.

    Returns:
        True if the service call succeeded.
    """
    dx, dy, dz = direction
    req = (
        f'name: "sun", '
        f'type: 1, '
        f'direction: {{x: {dx}, y: {dy}, z: {dz}}}'
    )
    try:
        subprocess.run(
            [
                "gz", "service",
                "-s", f"/world/{world}/light_config",
                "--reqtype", "gz.msgs.Light",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000",
                "--req", req,
            ],
            capture_output=True, timeout=5,
        )
        return True
    except Exception:
        return False


def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (rad) to quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return (qx, qy, qz, qw)


# ---------------------------------------------------------------------------
# Domain randomization (post-capture augmentation)
# ---------------------------------------------------------------------------

def apply_augmentation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random brightness, contrast, and noise augmentations.

    Args:
        img: Input BGR image (uint8).
        rng: Numpy random generator for reproducibility.

    Returns:
        Augmented BGR image (uint8).
    """
    result = img.astype(np.float32)

    # Random brightness shift [-30, +30]
    brightness = rng.uniform(-30, 30)
    result += brightness

    # Random contrast [0.7, 1.3]
    contrast = rng.uniform(0.7, 1.3)
    mean = result.mean()
    result = (result - mean) * contrast + mean

    # Random Gaussian noise
    if rng.random() < 0.5:
        sigma = rng.uniform(3, 15)
        noise = rng.normal(0, sigma, result.shape).astype(np.float32)
        result += noise

    # Random color jitter: slight hue/saturation shift
    if rng.random() < 0.3:
        hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.8, 1.2), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # Random Gaussian blur
    if rng.random() < 0.2:
        ksize = rng.choice([3, 5])
        result = cv2.GaussianBlur(
            np.clip(result, 0, 255).astype(np.uint8),
            (ksize, ksize), 0,
        ).astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def random_sun_direction(rng: np.random.Generator) -> tuple[float, float, float]:
    """Generate a random sun direction vector for domain randomization.

    The sun elevation angle is kept between 20 and 70 degrees to avoid
    unrealistic lighting (straight overhead or near-horizon).
    """
    azimuth = rng.uniform(0, 2 * math.pi)
    elevation = rng.uniform(math.radians(20), math.radians(70))

    dx = math.cos(azimuth) * math.cos(elevation)
    dy = math.sin(azimuth) * math.cos(elevation)
    dz = -math.sin(elevation)  # points downward

    return (dx, dy, dz)


# ---------------------------------------------------------------------------
# YOLO output helpers
# ---------------------------------------------------------------------------

def write_yolo_label(label_path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    """Write a YOLO format label file.

    Each line: <class_id> <cx> <cy> <width> <height>
    All coordinates are normalized to [0, 1].
    """
    with label_path.open("w") as f:
        for class_id, cx, cy, w, h in labels:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def write_classes_file(output_dir: Path, class_names: list[str]) -> None:
    """Write a classes.txt file for the dataset."""
    classes_path = output_dir / "classes.txt"
    with classes_path.open("w") as f:
        for name in class_names:
            f.write(f"{name}\n")


# ---------------------------------------------------------------------------
# Camera position grid generation
# ---------------------------------------------------------------------------

def generate_camera_positions(
    vehicles: list[VehiclePose],
    altitudes: list[float],
    grid_step: float = 10.0,
    margin: float = 20.0,
) -> list[tuple[float, float, float]]:
    """Generate a grid of camera positions covering the vehicle area.

    Creates positions at each altitude level over a grid that spans the
    bounding area of all vehicles plus a margin.

    Args:
        vehicles: Known vehicle poses (to determine coverage area).
        altitudes: List of camera altitudes in meters.
        grid_step: Spacing between grid points in meters.
        margin: Extra margin around vehicle bounding area in meters.

    Returns:
        List of (x, y, z) camera positions.
    """
    if not vehicles:
        return []

    xs = [v.x for v in vehicles]
    ys = [v.y for v in vehicles]

    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    positions = []
    x = x_min
    while x <= x_max:
        y = y_min
        while y <= y_max:
            for alt in altitudes:
                positions.append((x, y, alt))
            y += grid_step
        x += grid_step

    return positions


# ---------------------------------------------------------------------------
# Live capture pipeline
# ---------------------------------------------------------------------------

def run_live_capture(
    args: argparse.Namespace,
    vehicles: list[VehiclePose],
    intrinsics: CameraIntrinsics,
    output_dir: Path,
) -> dict[str, int]:
    """Run the live Gazebo capture pipeline.

    Iterates through camera positions, moves the camera, randomizes the
    sun angle, captures a frame, projects labels, applies augmentation,
    and saves the result in YOLO format.

    Args:
        args: Parsed command-line arguments.
        vehicles: Known vehicle poses from the world file.
        intrinsics: Camera intrinsic parameters.
        output_dir: Directory for output images and labels.

    Returns:
        Dict mapping class names to number of instances generated.
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    altitudes = [float(a) for a in args.altitudes.split(",")]
    positions = generate_camera_positions(vehicles, altitudes)

    rng = np.random.default_rng(seed=42)
    class_counts: dict[str, int] = {}
    image_idx = 0
    max_images = args.num_images

    cam_model = "synthetic_camera"
    world = args.world

    # If we have more positions than needed images, subsample
    if len(positions) > max_images:
        indices = rng.choice(len(positions), size=max_images, replace=False)
        positions = [positions[i] for i in sorted(indices)]
    else:
        # Cycle through positions if we need more images
        full_cycles = max_images // len(positions)
        remainder = max_images % len(positions)
        expanded = positions * full_cycles + positions[:remainder]
        positions = expanded

    print(f"\nGenerating {len(positions)} images from {len(vehicles)} vehicles")
    print(f"Altitudes: {altitudes}")
    print(f"Camera topic: {args.camera_topic}")
    print(f"Output: {output_dir}\n")

    pbar = tqdm(total=len(positions), desc="Capturing frames")
    failed_captures = 0

    for pos_idx, (px, py, pz) in enumerate(positions):
        if image_idx >= max_images:
            break

        # Domain randomization: small random offsets to position and tilt
        cam_x = px + rng.uniform(-2.0, 2.0)
        cam_y = py + rng.uniform(-2.0, 2.0)
        cam_z = pz + rng.uniform(-1.0, 1.0)

        # Camera points downward: pitch ~-90 deg with small random perturbation
        cam_roll = rng.uniform(-0.1, 0.1)
        cam_pitch = math.radians(-90) + rng.uniform(-0.15, 0.15)
        cam_yaw = rng.uniform(-math.pi, math.pi)

        # Randomize sun angle
        sun_dir = random_sun_direction(rng)
        set_gz_sun_angle(world, sun_dir)

        # Move camera
        set_gz_camera_pose(world, cam_model, cam_x, cam_y, cam_z,
                           cam_roll, cam_pitch, cam_yaw)

        # Brief settle time for the renderer
        time.sleep(0.05)

        # Capture frame
        frame = capture_gz_frame(args.camera_topic, intrinsics.width, intrinsics.height)
        if frame is None:
            failed_captures += 1
            pbar.update(1)
            if failed_captures > 20 and image_idx == 0:
                print("\nERROR: Cannot capture any frames. Is Gazebo running with "
                      f"a camera on topic '{args.camera_topic}'?")
                print("Hint: You may need to spawn a camera model first, or use "
                      "--offline mode.")
                sys.exit(1)
            continue

        # Apply post-capture augmentation
        frame = apply_augmentation(frame, rng)

        # Project vehicle labels
        labels = []
        for v in vehicles:
            bbox = project_3d_bbox_to_2d(
                v, cam_x, cam_y, cam_z, cam_roll, cam_pitch, cam_yaw, intrinsics
            )
            if bbox is not None:
                cx, cy, bw, bh = bbox
                labels.append((v.class_id, cx, cy, bw, bh))
                class_counts[v.class_name] = class_counts.get(v.class_name, 0) + 1

        # Save image and label
        img_name = f"synthetic_{image_idx:06d}.jpg"
        lbl_name = f"synthetic_{image_idx:06d}.txt"

        cv2.imwrite(str(images_dir / img_name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_yolo_label(labels_dir / lbl_name, labels)

        image_idx += 1
        pbar.set_postfix_str(f"saved={image_idx}, labels={len(labels)}")
        pbar.update(1)

    pbar.close()

    if failed_captures > 0:
        print(f"\nWarning: {failed_captures} frame captures failed "
              f"(out of {len(positions)} attempts)")

    return class_counts


# ---------------------------------------------------------------------------
# Offline augmentation pipeline
# ---------------------------------------------------------------------------

def run_offline_augmentation(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, int]:
    """Run offline augmentation on existing images.

    Reads images from a source directory, applies augmentations, and writes
    augmented copies. If a vehicle_poses.json is present, generates labels.

    Args:
        args: Parsed command-line arguments.
        output_dir: Directory for output images and labels.

    Returns:
        Dict mapping class names to number of instances generated.
    """
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    src_images = sorted([
        p for p in source_dir.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not src_images:
        print(f"ERROR: No images found in {source_dir}")
        sys.exit(1)

    # Load vehicle poses if available
    poses_file = source_dir / "vehicle_poses.json"
    vehicle_poses_data = None
    if poses_file.exists():
        with poses_file.open() as f:
            vehicle_poses_data = json.load(f)
        print(f"Loaded vehicle poses from {poses_file}")

    rng = np.random.default_rng(seed=42)
    class_counts: dict[str, int] = {}
    image_idx = 0
    max_images = args.num_images

    # Determine number of augmentations per source image
    augs_per_image = max(1, max_images // len(src_images))
    remainder = max_images - augs_per_image * len(src_images)

    print(f"\nOffline augmentation: {len(src_images)} source images")
    print(f"Generating ~{augs_per_image} augmentations per image")
    print(f"Target: {max_images} total images")
    print(f"Output: {output_dir}\n")

    pbar = tqdm(total=min(max_images, len(src_images) * augs_per_image + max(0, remainder)),
                desc="Augmenting")

    for src_path in src_images:
        if image_idx >= max_images:
            break

        img = cv2.imread(str(src_path))
        if img is None:
            continue

        n_augs = augs_per_image + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        for aug_i in range(n_augs):
            if image_idx >= max_images:
                break

            augmented = apply_augmentation(img.copy(), rng)

            img_name = f"synthetic_{image_idx:06d}.jpg"
            lbl_name = f"synthetic_{image_idx:06d}.txt"

            cv2.imwrite(str(images_dir / img_name), augmented,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Write labels from vehicle_poses.json if available
            labels = []
            if vehicle_poses_data:
                src_key = src_path.stem
                if src_key in vehicle_poses_data:
                    for obj in vehicle_poses_data[src_key]:
                        cid = CLASS_MAP.get(obj.get("class", "sedan"), 0)
                        cx = obj["cx"]
                        cy = obj["cy"]
                        bw = obj["w"]
                        bh = obj["h"]
                        labels.append((cid, cx, cy, bw, bh))
                        cname = obj.get("class", "sedan")
                        class_counts[cname] = class_counts.get(cname, 0) + 1

            write_yolo_label(labels_dir / lbl_name, labels)

            image_idx += 1
            pbar.update(1)

    pbar.close()
    return class_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(class_counts: dict[str, int], total_images: int, output_dir: Path) -> None:
    """Print a summary of the generated dataset."""
    print("\n" + "=" * 60)
    print("  Synthetic Data Generation Summary")
    print("=" * 60)
    print(f"  Output directory : {output_dir}")
    print(f"  Total images     : {total_images}")
    print(f"  Total labels     : {sum(class_counts.values())}")
    print()
    if class_counts:
        print("  Instances per class:")
        for cls_name in sorted(class_counts, key=class_counts.get, reverse=True):
            print(f"    {cls_name:<20s} {class_counts[cls_name]:>6d}")
    else:
        print("  No labels generated (no vehicles visible or offline without poses).")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from Gazebo simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --num-images 5000\n"
            "  %(prog)s --altitudes '10,20,40' --world military_training\n"
            "  %(prog)s --offline --source-dir ./captured_frames\n"
        ),
    )
    parser.add_argument(
        "--world",
        default="military_training",
        help="Gazebo world name (default: military_training)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "synthetic",
        help="Output directory for generated data (default: ../data/synthetic)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5000,
        help="Number of images to generate (default: 5000)",
    )
    parser.add_argument(
        "--altitudes",
        default="15,25,35,50,75",
        help="Comma-separated camera altitudes in meters (default: 15,25,35,50,75)",
    )
    parser.add_argument(
        "--camera-topic",
        default="/synthetic_cam",
        help="Gazebo camera topic for frame capture (default: /synthetic_cam)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Camera image width in pixels (default: 640)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Camera image height in pixels (default: 480)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera horizontal field of view in degrees (default: 60)",
    )
    parser.add_argument(
        "--sdf-world",
        type=Path,
        default=None,
        help="Path to SDF world file to parse vehicle poses from. "
             "Auto-detected from project structure if not specified.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: augment existing images instead of live Gazebo capture.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Source directory for offline mode (contains images and optional vehicle_poses.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    intrinsics = CameraIntrinsics(
        width=args.image_width,
        height=args.image_height,
        fov_horizontal=math.radians(args.fov),
    )

    # Resolve SDF world file for vehicle poses
    sdf_path = args.sdf_world
    if sdf_path is None:
        # Auto-detect from project structure
        project_root = Path(__file__).resolve().parent.parent.parent
        candidates = [
            project_root / "gazebo" / "worlds" / "drone_surveillance.sdf",
            project_root / "gazebo" / "worlds" / f"{args.world}.sdf",
        ]
        for candidate in candidates:
            if candidate.exists():
                sdf_path = candidate
                break

    # Parse vehicles from SDF
    vehicles: list[VehiclePose] = []
    if sdf_path and sdf_path.exists():
        print(f"Parsing vehicle poses from: {sdf_path}")
        vehicles = parse_vehicles_from_sdf(sdf_path)
        print(f"Found {len(vehicles)} vehicles:")
        for v in vehicles:
            print(f"  {v.name}: {v.class_name} at ({v.x:.1f}, {v.y:.1f}, {v.z:.1f})")
    elif not args.offline:
        print("WARNING: No SDF world file found. Labels will be empty.")
        print("  Specify --sdf-world or place an SDF file in gazebo/worlds/")

    # Write class names file
    output_dir.mkdir(parents=True, exist_ok=True)
    used_classes = sorted(set(
        name for name, cid in CLASS_MAP.items()
        if name not in ("car",)  # skip aliases
    ), key=lambda n: CLASS_MAP[n])
    write_classes_file(output_dir, used_classes)

    # Run pipeline
    if args.offline:
        if args.source_dir is None:
            print("ERROR: --offline requires --source-dir")
            sys.exit(1)
        class_counts = run_offline_augmentation(args, output_dir)
    else:
        class_counts = run_live_capture(args, vehicles, intrinsics, output_dir)

    # Count actual output images
    images_dir = output_dir / "images"
    total_images = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0

    print_summary(class_counts, total_images, output_dir)


if __name__ == "__main__":
    main()
