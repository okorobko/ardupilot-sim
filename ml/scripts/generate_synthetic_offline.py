#!/usr/bin/env python3
"""Generate synthetic training data WITHOUT a running Gazebo instance.

Uses the 3D vehicle poses from the military_training.sdf world file and
projects them onto random satellite/aerial background images using a
pinhole camera model. Renders colored vehicle silhouettes at the projected
positions with domain randomization.

This is the offline alternative to generate_synthetic.py (live mode) for
when Gazebo cannot render frames (e.g., headless macOS).

Focuses on the 3 missing classes: sp_artillery(3), mlrs(4), helicopter(5).

Usage:
    python generate_synthetic_offline.py --num-images 1200
    python generate_synthetic_offline.py --num-images 500 --classes sp_artillery,helicopter
"""

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse projection math from generate_synthetic.py
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic import (
    CameraIntrinsics,
    VehiclePose,
    parse_vehicles_from_sdf,
    rotation_matrix,
    build_extrinsic,
)

random.seed(42)
np.random.seed(42)

PROJECT = Path(__file__).resolve().parent.parent.parent
SDF_PATH = PROJECT / "gazebo" / "worlds" / "military_training.sdf"
BG_DIR = PROJECT / "ml" / "data" / "raw" / "mvrsd" / "images" / "train"
OUTPUT_DIR = PROJECT / "ml" / "data" / "synthetic"

# 7-class military schema (must match dataset.yaml)
CLASS_NAMES = ["tank", "apc_ifv", "military_truck", "sp_artillery",
               "mlrs", "helicopter", "uav"]

# Vehicle top-down colors (olive/camo tones, BGR format)
VEHICLE_COLORS = {
    "tank":           [(45, 65, 55), (50, 70, 60), (40, 55, 45)],
    "apc":            [(50, 70, 55), (45, 65, 50), (55, 75, 60)],
    "military_truck": [(50, 75, 60), (55, 80, 65), (45, 60, 50)],
    "sp_artillery":   [(40, 60, 48), (35, 55, 42), (45, 65, 52)],
    "mlrs":           [(50, 70, 55), (45, 65, 50), (55, 75, 60)],
    "helicopter":     [(55, 75, 60), (50, 70, 55), (60, 80, 65)],
}


def project_vehicle_polygon(
    vehicle: VehiclePose,
    cam_x: float, cam_y: float, cam_z: float,
    cam_roll: float, cam_pitch: float, cam_yaw: float,
    intrinsics: CameraIntrinsics,
) -> tuple[list[tuple[int, int]], tuple[float, float, float, float]] | tuple[None, None]:
    """Project a vehicle's top face as a polygon + YOLO bbox."""
    # Top face corners (z = height/2, the visible face from above)
    hl, hw = vehicle.length / 2, vehicle.width / 2
    hz = vehicle.height  # top surface at full height (vehicles sit on ground)
    top_corners = np.array([
        [+hl, +hw, hz],
        [+hl, -hw, hz],
        [-hl, -hw, hz],
        [-hl, +hw, hz],
    ])

    # Rotate by vehicle yaw and translate
    R_veh = rotation_matrix(vehicle.roll, vehicle.pitch, vehicle.yaw)
    world_corners = (R_veh @ top_corners.T).T + np.array([vehicle.x, vehicle.y, vehicle.z])

    # Build camera extrinsic
    extrinsic = build_extrinsic(cam_x, cam_y, cam_z, cam_roll, cam_pitch, cam_yaw)
    K = intrinsics.K

    # Project each corner
    pixels = []
    for corner in world_corners:
        p_hom = np.append(corner, 1.0)
        p_cam = extrinsic @ p_hom
        if p_cam[2] <= 0:
            return None, None
        p_img = K @ p_cam[:3]
        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]
        pixels.append((int(round(u)), int(round(v))))

    # Also project full 3D bbox for YOLO label
    hl2, hw2, hh2 = vehicle.length / 2, vehicle.width / 2, vehicle.height / 2
    all_corners = np.array([
        [+hl2, +hw2, +hh2], [+hl2, +hw2, -hh2],
        [+hl2, -hw2, +hh2], [+hl2, -hw2, -hh2],
        [-hl2, +hw2, +hh2], [-hl2, +hw2, -hh2],
        [-hl2, -hw2, +hh2], [-hl2, -hw2, -hh2],
    ])
    world_all = (R_veh @ all_corners.T).T + np.array([vehicle.x, vehicle.y, vehicle.z])

    all_pixels = []
    for corner in world_all:
        p_hom = np.append(corner, 1.0)
        p_cam = extrinsic @ p_hom
        if p_cam[2] <= 0:
            return None, None
        p_img = K @ p_cam[:3]
        all_pixels.append((p_img[0] / p_img[2], p_img[1] / p_img[2]))

    all_pixels = np.array(all_pixels)
    u_min = max(0.0, all_pixels[:, 0].min())
    v_min = max(0.0, all_pixels[:, 1].min())
    u_max = min(float(intrinsics.width), all_pixels[:, 0].max())
    v_max = min(float(intrinsics.height), all_pixels[:, 1].max())

    if u_max - u_min < 3.0 or v_max - v_min < 3.0:
        return None, None

    cx = ((u_min + u_max) / 2.0) / intrinsics.width
    cy = ((v_min + v_max) / 2.0) / intrinsics.height
    bw = (u_max - u_min) / intrinsics.width
    bh = (v_max - v_min) / intrinsics.height

    return pixels, (cx, cy, bw, bh)


def render_vehicle(img: np.ndarray, polygon: list[tuple[int, int]],
                   class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Render a vehicle silhouette on the image."""
    colors = VEHICLE_COLORS.get(class_name, [(50, 70, 55)])
    base_color = list(colors[rng.integers(len(colors))])

    # Slight color randomization
    color = tuple(max(0, min(255, c + int(rng.integers(-15, 15)))) for c in base_color)

    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Fill polygon
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)

    # Slight transparency for realism
    alpha = rng.uniform(0.7, 0.95)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add edge line (shadow/outline)
    edge_color = tuple(max(0, c - 30) for c in color)
    cv2.polylines(img, [pts], True, edge_color, 1)

    return img


def generate_camera_around_vehicles(
    vehicles: list[VehiclePose],
    n_positions: int,
    altitudes: list[float],
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float, float, float]]:
    """Generate camera positions centered near individual vehicles.

    Instead of uniformly sampling the bounding area (which misses
    clustered vehicles at high altitudes), we generate positions around
    each vehicle with altitude-dependent offsets.
    """
    positions = []
    per_vehicle = max(1, n_positions // len(vehicles))
    remainder = n_positions - per_vehicle * len(vehicles)

    for vi, v in enumerate(vehicles):
        n = per_vehicle + (1 if vi < remainder else 0)
        for _ in range(n):
            alt = altitudes[rng.integers(len(altitudes))]

            # Offset proportional to altitude — higher up = wider spread
            # At 15m, offset ~5m; at 75m, offset ~25m
            max_offset = alt * 0.35
            cx = v.x + rng.uniform(-max_offset, max_offset)
            cy = v.y + rng.uniform(-max_offset, max_offset)
            cz = alt

            # Camera pointing downward: pitch=+90deg in Gazebo convention
            # (rotates camera +X axis to point toward ground)
            cam_roll = rng.uniform(-0.08, 0.08)
            cam_pitch = math.radians(90) + rng.uniform(-0.15, 0.15)
            cam_yaw = rng.uniform(-math.pi, math.pi)

            positions.append((cx, cy, cz, cam_roll, cam_pitch, cam_yaw))

    rng.shuffle(positions)
    return positions


def apply_domain_randomization(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations to synthetic image."""
    result = img.astype(np.float32)

    # Brightness
    result += rng.uniform(-25, 25)

    # Contrast
    contrast = rng.uniform(0.75, 1.25)
    mean = result.mean()
    result = (result - mean) * contrast + mean

    # Gaussian noise
    if rng.random() < 0.4:
        sigma = rng.uniform(5, 20)
        result += rng.normal(0, sigma, result.shape).astype(np.float32)

    # Color jitter
    if rng.random() < 0.3:
        hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data offline using 3D projection onto backgrounds")
    parser.add_argument("--num-images", type=int, default=1200,
                        help="Total images to generate (default: 1200)")
    parser.add_argument("--altitudes", default="15,25,35,50,75",
                        help="Comma-separated altitudes in meters")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class names to focus on (default: all)")
    parser.add_argument("--bg-dir", type=Path, default=BG_DIR,
                        help="Background images directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--sdf", type=Path, default=SDF_PATH,
                        help="SDF world file path")
    args = parser.parse_args()

    intrinsics = CameraIntrinsics(width=640, height=480,
                                  fov_horizontal=math.radians(60))
    altitudes = [float(a) for a in args.altitudes.split(",")]

    # Parse vehicles from SDF
    print(f"Parsing vehicles from {args.sdf}")
    all_vehicles = parse_vehicles_from_sdf(args.sdf)
    print(f"Found {len(all_vehicles)} vehicles total")

    # Filter to requested classes if specified
    if args.classes:
        target_names = set(args.classes.split(","))
        vehicles = [v for v in all_vehicles if v.class_name in target_names]
        print(f"Filtered to {len(vehicles)} vehicles of classes: {target_names}")
    else:
        vehicles = all_vehicles

    if not vehicles:
        print("ERROR: No vehicles found matching criteria")
        sys.exit(1)

    for v in vehicles:
        print(f"  {v.name}: {v.class_name} (class {v.class_id}) "
              f"at ({v.x:.1f}, {v.y:.1f}, {v.z:.1f})")

    # Load background images
    bg_images = sorted([p for p in args.bg_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not bg_images:
        print(f"ERROR: No background images in {args.bg_dir}")
        sys.exit(1)
    print(f"\nLoaded {len(bg_images)} background images from {args.bg_dir}")

    # Setup output
    images_dir = args.output_dir / "images"
    labels_dir = args.output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    # Generate camera positions
    positions = generate_camera_around_vehicles(
        vehicles, args.num_images, altitudes, rng)

    print(f"\nGenerating {len(positions)} synthetic images...")
    class_counts = {}
    saved = 0
    empty_frames = 0

    for idx, (cx, cy, cz, cr, cp, cyw) in enumerate(positions):
        # Pick random background
        bg_path = bg_images[rng.integers(len(bg_images))]
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        bg = cv2.resize(bg, (intrinsics.width, intrinsics.height))

        # Project and render vehicles
        labels = []
        for v in vehicles:
            polygon, bbox = project_vehicle_polygon(
                v, cx, cy, cz, cr, cp, cyw, intrinsics)
            if polygon is None:
                continue

            # Check polygon is within frame
            in_frame = all(0 <= x < intrinsics.width and 0 <= y < intrinsics.height
                           for x, y in polygon)
            if not in_frame:
                # At least partially visible
                any_visible = any(0 <= x < intrinsics.width and 0 <= y < intrinsics.height
                                  for x, y in polygon)
                if not any_visible:
                    continue

            bg = render_vehicle(bg, polygon, v.class_name, rng)
            yolo_cx, yolo_cy, yolo_w, yolo_h = bbox
            labels.append((v.class_id, yolo_cx, yolo_cy, yolo_w, yolo_h))
            class_counts[v.class_name] = class_counts.get(v.class_name, 0) + 1

        if not labels:
            empty_frames += 1
            continue

        # Apply domain randomization
        bg = apply_domain_randomization(bg, rng)

        # Save
        img_name = f"synth_{saved:06d}.jpg"
        lbl_name = f"synth_{saved:06d}.txt"
        cv2.imwrite(str(images_dir / img_name), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])

        with open(labels_dir / lbl_name, "w") as f:
            for cls_id, c_x, c_y, w, h in labels:
                f.write(f"{cls_id} {c_x:.6f} {c_y:.6f} {w:.6f} {h:.6f}\n")

        saved += 1
        if (saved) % 200 == 0:
            print(f"  {saved}/{args.num_images} images saved...")

    print(f"\nDone: {saved} images with labels, {empty_frames} empty frames skipped")
    print(f"\nPer-class instance counts:")
    for name in sorted(class_counts, key=lambda n: -class_counts[n]):
        print(f"  {name:<20} {class_counts[name]:>6}")


if __name__ == "__main__":
    main()
