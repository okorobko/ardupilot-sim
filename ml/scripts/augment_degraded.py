#!/usr/bin/env python3
"""Offline degraded image augmentation for drone imagery robustness.

Generates 2-3 degraded variants per training image to simulate real-world
drone camera conditions: thermal/IR, motion blur, compression artifacts,
fog/smoke, low light, sensor noise, FPV distortion, and watermark overlays.

Input:  YOLO dataset directory with images/ and labels/ subdirs
Output: Augmented images + copied labels in the same directory structure

Usage:
    python augment_degraded.py --input ml/data/aerial/train --variants 2
    python augment_degraded.py --input ml/data/aerial/train --variants 3 --preview 10
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)


# ── Augmentation functions ────────────────────────────────────

def thermal_ir(img):
    """Simulate thermal/IR camera: grayscale + CLAHE + random colormap."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(2.0, 4.0),
                            tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Randomly choose: inverted grayscale, or a thermal colormap
    mode = random.choice(["invert", "inferno", "jet", "hot", "bone"])
    if mode == "invert":
        result = cv2.cvtColor(255 - enhanced, cv2.COLOR_GRAY2BGR)
    elif mode == "inferno":
        result = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    elif mode == "jet":
        result = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    elif mode == "hot":
        result = cv2.applyColorMap(enhanced, cv2.COLORMAP_HOT)
    else:
        result = cv2.applyColorMap(enhanced, cv2.COLORMAP_BONE)

    return result


def jpeg_compression(img):
    """Simulate FPV video stream compression artifacts (quality 15-40)."""
    quality = random.randint(15, 40)
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def motion_blur(img):
    """Simulate drone movement blur (10-30px kernel)."""
    size = random.randint(10, 30)
    angle = random.uniform(0, 360)
    # Build directional motion blur kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad), np.sin(rad)
    for i in range(size):
        t = i - center
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def downscale_upscale(img):
    """Simulate low-resolution distant objects (downscale 25-50% then upscale)."""
    h, w = img.shape[:2]
    scale = random.uniform(0.25, 0.50)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def low_brightness(img):
    """Simulate night/twilight conditions (reduce brightness 40-75%)."""
    factor = random.uniform(0.30, 0.65)
    result = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    return result


def fog_overlay(img):
    """Simulate fog/smoke/haze overlay."""
    h, w = img.shape[:2]
    # Create smooth fog pattern using multiple Gaussian blobs
    fog = np.zeros((h, w), dtype=np.float32)
    n_blobs = random.randint(3, 8)
    for _ in range(n_blobs):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(min(h, w) // 4, min(h, w))
        Y, X = np.ogrid[:h, :w]
        dist = ((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        blob = np.exp(-dist / (2 * (radius ** 2)))
        fog += blob

    fog = (fog / fog.max()) * random.uniform(0.3, 0.7)
    fog_color = np.full_like(img, random.randint(180, 240), dtype=np.uint8)
    fog_3c = np.stack([fog] * 3, axis=-1).astype(np.float32)

    result = (img.astype(np.float32) * (1 - fog_3c) +
              fog_color.astype(np.float32) * fog_3c)
    return result.clip(0, 255).astype(np.uint8)


def poisson_noise(img):
    """Simulate low-light sensor noise."""
    # Scale controls noise intensity (lower = more noise)
    scale = random.uniform(5.0, 20.0)
    noisy = np.random.poisson(img.astype(np.float32) / scale) * scale
    return noisy.clip(0, 255).astype(np.uint8)


def barrel_distortion(img):
    """Simulate FPV fisheye lens distortion."""
    h, w = img.shape[:2]
    k1 = random.uniform(-0.4, -0.1)  # negative = barrel
    k2 = random.uniform(-0.1, 0.0)
    camera_matrix = np.array([[w, 0, w / 2],
                               [0, w, h / 2],
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)
    result = cv2.undistort(img, camera_matrix, dist_coeffs)
    return result


def text_watermark(img):
    """Simulate Telegram/social media watermarks and unit insignia text."""
    result = img.copy()
    h, w = result.shape[:2]

    texts = [
        "TELEGRAM", "t.me/channel", "@unit_name", "Z", "V",
        "DRONE CAM", "FPV", "БПЛА", "REC", "CAM-01",
        "RESTRICTED", "MIL-OPS", "2024", "LIVE",
    ]

    n_texts = random.randint(1, 3)
    for _ in range(n_texts):
        text = random.choice(texts)
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX,
                              cv2.FONT_HERSHEY_DUPLEX,
                              cv2.FONT_HERSHEY_PLAIN])
        scale = random.uniform(0.4, 1.2)
        thickness = random.randint(1, 2)
        color = random.choice([(255, 255, 255), (0, 255, 0),
                               (0, 0, 255), (255, 255, 0)])

        x = random.randint(5, max(6, w - 100))
        y = random.randint(20, h - 10)

        # Semi-transparent overlay
        overlay = result.copy()
        cv2.putText(overlay, text, (x, y), font, scale, color, thickness)
        alpha = random.uniform(0.3, 0.8)
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    return result


def gaussian_noise(img):
    """Add Gaussian noise to simulate general sensor noise."""
    sigma = random.uniform(10, 35)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    result = (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    return result


# ── Augmentation profiles ─────────────────────────────────────
# Each profile simulates a realistic degradation scenario.
# Profiles are weighted by how common these conditions are in real footage.

PROFILES = [
    # (name, function_pipeline, weight)
    ("thermal",       [thermal_ir],                              3),
    ("thermal_noisy", [thermal_ir, poisson_noise],               2),
    ("jpeg_artifact", [jpeg_compression],                        2),
    ("motion_blur",   [motion_blur],                             2),
    ("lowres",        [downscale_upscale],                       2),
    ("lowres_jpeg",   [downscale_upscale, jpeg_compression],     1),
    ("night",         [low_brightness, gaussian_noise],          2),
    ("night_blur",    [low_brightness, motion_blur],             1),
    ("foggy",         [fog_overlay],                             2),
    ("foggy_lowres",  [fog_overlay, downscale_upscale],          1),
    ("fpv_distort",   [barrel_distortion, jpeg_compression],     1),
    ("watermark",     [text_watermark],                          1),
    ("wm_jpeg",       [text_watermark, jpeg_compression],        1),
    ("noisy",         [gaussian_noise],                          1),
    ("harsh",         [low_brightness, poisson_noise, jpeg_compression], 1),
]


def pick_profiles(n):
    """Pick n distinct augmentation profiles weighted by frequency."""
    names, pipelines, weights = zip(*PROFILES)
    total = sum(weights)
    probs = [w / total for w in weights]

    chosen = []
    remaining_idx = list(range(len(PROFILES)))
    remaining_probs = list(probs)

    for _ in range(min(n, len(PROFILES))):
        total_p = sum(remaining_probs)
        norm = [p / total_p for p in remaining_probs]
        idx = np.random.choice(remaining_idx, p=norm)
        pos = remaining_idx.index(idx)
        chosen.append((names[idx], pipelines[idx]))
        remaining_idx.pop(pos)
        remaining_probs.pop(pos)

    return chosen


def apply_pipeline(img, pipeline):
    """Apply a sequence of augmentation functions."""
    result = img.copy()
    for fn in pipeline:
        result = fn(result)
    return result


# ── Main ──────────────────────────────────────────────────────

def augment_dataset(input_dir, variants=2, preview=0):
    """Generate degraded variants for all images in a YOLO dataset split."""
    input_dir = Path(input_dir)
    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"

    if not img_dir.exists():
        print(f"Error: {img_dir} not found")
        sys.exit(1)

    images = sorted([f for f in img_dir.iterdir()
                     if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"Found {len(images)} images in {img_dir}")
    print(f"Generating {variants} degraded variants per image...")

    # Preview mode: save sample augmentations and exit
    if preview > 0:
        preview_dir = input_dir / "augment_preview"
        preview_dir.mkdir(exist_ok=True)
        sample = random.sample(images, min(preview, len(images)))
        for img_path in sample:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            cv2.imwrite(str(preview_dir / f"{img_path.stem}_original.jpg"), img)
            for name, pipeline, _w in PROFILES:
                result = apply_pipeline(img, pipeline)
                cv2.imwrite(str(preview_dir / f"{img_path.stem}_{name}.jpg"),
                            result)
        print(f"Preview saved to {preview_dir} ({len(sample)} images × "
              f"{len(PROFILES)} profiles)")
        return

    # Full augmentation
    created = 0
    profile_counts = {}

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: cannot read {img_path.name}, skipping")
            continue

        # Pick distinct profiles for this image
        profiles = pick_profiles(variants)

        for profile_name, pipeline in profiles:
            # Generate augmented image
            augmented = apply_pipeline(img, pipeline)

            # Save with augmentation suffix
            aug_name = f"{img_path.stem}_aug_{profile_name}{img_path.suffix}"
            cv2.imwrite(str(img_dir / aug_name), augmented,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Copy label file (bounding boxes unchanged — augmentations are
            # pixel-level only, no geometric transforms)
            lbl_src = lbl_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                lbl_dst = lbl_dir / (Path(aug_name).stem + ".txt")
                lbl_dst.write_text(lbl_src.read_text())

            profile_counts[profile_name] = profile_counts.get(profile_name, 0) + 1
            created += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(images)} images "
                  f"({created} augmented so far)")

    print(f"\nDone: {created} augmented images created from {len(images)} originals")
    print(f"Total dataset size: {len(images) + created} images")
    print(f"\nProfile distribution:")
    for name, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} {count:>5}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded image variants for drone detection training")
    parser.add_argument("--input", required=True,
                        help="Path to YOLO dataset split (must have images/ and labels/)")
    parser.add_argument("--variants", type=int, default=2,
                        help="Number of degraded variants per image (default: 2)")
    parser.add_argument("--preview", type=int, default=0,
                        help="Generate preview of N sample images with all profiles, "
                             "then exit (for visual inspection)")
    args = parser.parse_args()
    augment_dataset(args.input, args.variants, args.preview)


if __name__ == "__main__":
    main()
