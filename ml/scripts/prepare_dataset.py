#!/usr/bin/env python3
"""
prepare_dataset.py

Takes raw downloaded datasets (xView, DOTA, Roboflow) and synthetic data,
filters to relevant vehicle/aircraft classes, remaps labels to 7 target
classes, converts all annotations to YOLO format, merges sources, and
splits into train/val/test.

Target classes:
    0 - tank
    1 - apc_ifv
    2 - military_truck
    3 - sp_artillery
    4 - mlrs
    5 - helicopter
    6 - uav
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

TARGET_CLASSES = {
    0: "tank",
    1: "apc_ifv",
    2: "military_truck",
    3: "sp_artillery",
    4: "mlrs",
    5: "helicopter",
    6: "uav",
}

# xView integer class ID -> target class ID
XVIEW_CLASS_MAP = {
    73: 0,   # tank
    74: 1,   # APC / Fighting vehicle
    71: 2,   # truck
    72: 2,   # utility truck
    76: 2,   # truck w/ box
    77: 2,   # truck w/ liquid
    75: 3,   # self-propelled artillery
    15: 5,   # helicopter
}

# DOTA text label -> target class ID
DOTA_CLASS_MAP = {
    "large-vehicle": 2,   # military_truck (approximate)
    "helicopter": 5,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def obb_to_aabb(coords):
    """Convert oriented bounding box (x1,y1,...,x4,y4) to axis-aligned bbox.

    Parameters
    ----------
    coords : list[float]
        Eight values: x1, y1, x2, y2, x3, y3, x4, y4.

    Returns
    -------
    tuple (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert pixel bbox to YOLO normalised format.

    Returns (x_center, y_center, width, height) all in [0, 1].
    """
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return x_center, y_center, w, h


def _image_dimensions(img_path):
    """Return (width, height) for an image without heavy dependencies.

    Tries PIL first, falls back to reading PNG/JPEG header bytes.
    """
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            return im.size  # (w, h)
    except Exception:
        pass

    # Minimal header-based fallback for PNG
    with open(img_path, "rb") as f:
        header = f.read(32)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            import struct

            w, h = struct.unpack(">II", header[16:24])
            return w, h

    raise RuntimeError(f"Cannot determine dimensions for {img_path}")


# ---------------------------------------------------------------------------
# Source processors
# ---------------------------------------------------------------------------


def process_xview(raw_dir: Path):
    """Process xView dataset.

    Expected layout under *raw_dir/xview/*:
        images/          — .tif or .png images
        xView_train.geojson (or similar .geojson annotation file)

    Returns list of (image_path, list[(class_id, x_c, y_c, w, h)]).
    """
    xview_dir = raw_dir / "xview"
    if not xview_dir.exists():
        print("[xView] Directory not found, skipping.")
        return []

    # Locate geojson
    geojsons = list(xview_dir.glob("*.geojson"))
    if not geojsons:
        print("[xView] No .geojson annotation file found, skipping.")
        return []

    images_dir = xview_dir / "images"
    if not images_dir.exists():
        # Try alternate layouts
        images_dir = xview_dir / "train_images"
    if not images_dir.exists():
        images_dir = xview_dir  # images might be alongside geojson

    # Build a map: image_filename -> list of (class_id, bbox_pixels)
    image_annotations: dict[str, list] = defaultdict(list)

    for gj_path in geojsons:
        print(f"[xView] Parsing {gj_path.name} ...")
        with open(gj_path) as f:
            data = json.load(f)

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            type_id = props.get("type_id")
            if type_id is None:
                continue
            type_id = int(type_id)
            if type_id not in XVIEW_CLASS_MAP:
                continue

            target_cls = XVIEW_CLASS_MAP[type_id]
            image_name = props.get("image_id", "")
            if not image_name:
                continue

            # bbox in geojson is [xmin, ymin, xmax, ymax] in pixel coords
            bounds = props.get("bounds_imcoords", "")
            if not bounds:
                continue
            parts = [float(x) for x in bounds.split(",")]
            if len(parts) != 4:
                continue
            xmin, ymin, xmax, ymax = parts
            image_annotations[image_name].append((target_cls, xmin, ymin, xmax, ymax))

    results = []
    for img_name, annots in image_annotations.items():
        img_path = images_dir / img_name
        if not img_path.exists():
            continue
        try:
            img_w, img_h = _image_dimensions(img_path)
        except RuntimeError:
            continue
        yolo_annots = []
        for cls_id, xmin, ymin, xmax, ymax in annots:
            xc, yc, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            if w > 0 and h > 0:
                yolo_annots.append((cls_id, xc, yc, w, h))
        if yolo_annots:
            results.append((img_path, yolo_annots))

    print(f"[xView] Processed {len(results)} images with relevant annotations.")
    return results


def process_dota(raw_dir: Path):
    """Process DOTA dataset.

    Expected layout under *raw_dir/dota/*:
        images/   — .png images
        labels/   — .txt label files (one per image, OBB format)

    DOTA label format per line:
        x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    """
    dota_dir = raw_dir / "dota"
    if not dota_dir.exists():
        print("[DOTA] Directory not found, skipping.")
        return []

    images_dir = dota_dir / "images"
    labels_dir = dota_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        # Try alternate: labelTxt
        labels_dir = dota_dir / "labelTxt"
    if not images_dir.exists():
        print("[DOTA] images/ subdirectory not found, skipping.")
        return []
    if not labels_dir.exists():
        print("[DOTA] labels/ subdirectory not found, skipping.")
        return []

    results = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        # Find matching image
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        try:
            img_w, img_h = _image_dimensions(img_path)
        except RuntimeError:
            continue

        yolo_annots = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # DOTA files sometimes start with header lines (imagesource, gsd)
                if len(parts) < 9:
                    continue
                try:
                    coords = [float(parts[i]) for i in range(8)]
                except ValueError:
                    continue
                category = parts[8]
                if category not in DOTA_CLASS_MAP:
                    continue
                target_cls = DOTA_CLASS_MAP[category]
                xmin, ymin, xmax, ymax = obb_to_aabb(coords)
                xc, yc, w, h = bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                if w > 0 and h > 0:
                    yolo_annots.append((target_cls, xc, yc, w, h))

        if yolo_annots:
            results.append((img_path, yolo_annots))

    print(f"[DOTA] Processed {len(results)} images with relevant annotations.")
    return results


def process_roboflow(raw_dir: Path):
    """Process Roboflow dataset (already YOLO format).

    Expected layout under *raw_dir/roboflow/*:
        images/  — image files
        labels/  — .txt YOLO label files
        data.yaml or classes.txt — class name list

    If Roboflow was exported with its own class IDs we remap them based on
    a name-based lookup.
    """
    rf_dir = raw_dir / "roboflow"
    if not rf_dir.exists():
        print("[Roboflow] Directory not found, skipping.")
        return []

    images_dir = rf_dir / "images"
    labels_dir = rf_dir / "labels"

    # Support flat layout too (train/images, train/labels, …)
    if not images_dir.exists():
        for sub in ("train", "valid", "test"):
            sub_img = rf_dir / sub / "images"
            sub_lbl = rf_dir / sub / "labels"
            if sub_img.exists() and sub_lbl.exists():
                images_dir = sub_img
                labels_dir = sub_lbl
                break

    if not images_dir.exists() or not labels_dir.exists():
        print("[Roboflow] images/ and labels/ not found, skipping.")
        return []

    # Build Roboflow class -> target class map from class names
    rf_class_map: dict[int, int] = {}
    name_to_target = {
        "tank": 0,
        "apc": 1, "apc_ifv": 1, "ifv": 1, "fighting_vehicle": 1,
        "military_truck": 2, "truck": 2,
        "sp_artillery": 3, "self_propelled_artillery": 3, "artillery": 3,
        "mlrs": 4, "rocket_launcher": 4,
        "helicopter": 5, "heli": 5,
        "uav": 6, "drone": 6,
    }

    # Try to read class list from data.yaml or classes.txt
    class_names: list[str] = []
    data_yaml = rf_dir / "data.yaml"
    classes_txt = rf_dir / "classes.txt"
    if data_yaml.exists():
        try:
            import yaml  # type: ignore

            with open(data_yaml) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            if isinstance(names, dict):
                class_names = [names[k] for k in sorted(names.keys())]
            else:
                class_names = list(names)
        except Exception:
            pass
    elif classes_txt.exists():
        with open(classes_txt) as f:
            class_names = [l.strip() for l in f if l.strip()]

    if class_names:
        for idx, name in enumerate(class_names):
            normalised = name.lower().replace(" ", "_").replace("-", "_")
            if normalised in name_to_target:
                rf_class_map[idx] = name_to_target[normalised]
    else:
        # Fallback: assume Roboflow IDs match target IDs (0-6)
        print("[Roboflow] No class list found; assuming IDs match target IDs.")
        rf_class_map = {i: i for i in range(7)}

    results = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        yolo_annots = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    orig_cls = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                if orig_cls not in rf_class_map:
                    continue
                target_cls = rf_class_map[orig_cls]
                yolo_annots.append((target_cls, xc, yc, w, h))

        if yolo_annots:
            results.append((img_path, yolo_annots))

    print(f"[Roboflow] Processed {len(results)} images with relevant annotations.")
    return results


def process_synthetic(synthetic_dir: Path):
    """Process synthetic data generated by the simulator.

    Expected layout:
        images/  — .png / .jpg images
        labels/  — .txt YOLO label files (already in target class IDs)
    """
    if not synthetic_dir or not synthetic_dir.exists():
        print("[Synthetic] Directory not found, skipping.")
        return []

    images_dir = synthetic_dir / "images"
    labels_dir = synthetic_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        print("[Synthetic] images/ or labels/ not found, skipping.")
        return []

    results = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        yolo_annots = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                if cls_id not in TARGET_CLASSES:
                    continue
                yolo_annots.append((cls_id, xc, yc, w, h))

        if yolo_annots:
            results.append((img_path, yolo_annots))

    print(f"[Synthetic] Processed {len(results)} images with relevant annotations.")
    return results


# ---------------------------------------------------------------------------
# Split & write
# ---------------------------------------------------------------------------


def write_yolo_label(label_path: Path, annotations):
    """Write a list of (class_id, xc, yc, w, h) to a YOLO .txt file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for cls_id, xc, yc, w, h in annotations:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def copy_sample(img_path: Path, annotations, dest_images: Path, dest_labels: Path, prefix: str, idx: int):
    """Copy an image and write its label file into the destination dirs."""
    suffix = img_path.suffix
    new_name = f"{prefix}_{idx:06d}"
    dest_img = dest_images / (new_name + suffix)
    dest_lbl = dest_labels / (new_name + ".txt")

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, dest_img)
    write_yolo_label(dest_lbl, annotations)


def compute_stats(samples, label: str):
    """Print per-class statistics for a list of samples."""
    class_counts: dict[int, int] = defaultdict(int)
    for _, annots in samples:
        for cls_id, *_ in annots:
            class_counts[cls_id] += 1

    total = sum(class_counts.values())
    print(f"\n  [{label}] {len(samples)} images, {total} annotations")
    for cls_id in sorted(TARGET_CLASSES.keys()):
        count = class_counts.get(cls_id, 0)
        print(f"    {cls_id} ({TARGET_CLASSES[cls_id]:>16s}): {count}")
    return dict(class_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and merge datasets into unified YOLO format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root directory containing raw dataset folders (xview/, dota/, roboflow/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the unified dataset.",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=None,
        help="Directory with synthetic images/labels (default: ../data/synthetic/ relative to this script).",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train / val / test split ratios (default: 0.70 0.15 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Resolve synthetic dir default
    if args.synthetic_dir is None:
        default_synthetic = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"
        args.synthetic_dir = default_synthetic

    train_ratio, val_ratio, test_ratio = args.split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. Process each source
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Processing raw datasets ...")
    print("=" * 60)

    xview_samples = process_xview(args.raw_dir)
    dota_samples = process_dota(args.raw_dir)
    roboflow_samples = process_roboflow(args.raw_dir)
    synthetic_samples = process_synthetic(args.synthetic_dir)

    real_samples = xview_samples + dota_samples + roboflow_samples

    # -----------------------------------------------------------------------
    # 2. Per-source statistics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Per-source statistics")
    print("=" * 60)

    stats = {}
    if xview_samples:
        stats["xview"] = compute_stats(xview_samples, "xView")
    if dota_samples:
        stats["dota"] = compute_stats(dota_samples, "DOTA")
    if roboflow_samples:
        stats["roboflow"] = compute_stats(roboflow_samples, "Roboflow")
    if synthetic_samples:
        stats["synthetic"] = compute_stats(synthetic_samples, "Synthetic")

    total_real = len(real_samples)
    total_synthetic = len(synthetic_samples)
    print(f"\nTotal real-world samples : {total_real}")
    print(f"Total synthetic samples  : {total_synthetic}")

    if total_real == 0 and total_synthetic == 0:
        print("\nNo data found. Nothing to do.")
        return

    # -----------------------------------------------------------------------
    # 3. Split real-world data
    # -----------------------------------------------------------------------
    random.shuffle(real_samples)

    n_test = int(round(total_real * test_ratio))
    n_val = int(round(total_real * val_ratio))
    n_train = total_real - n_val - n_test

    test_samples = real_samples[:n_test]
    val_samples_real = real_samples[n_test:n_test + n_val]
    train_samples_real = real_samples[n_test + n_val:]

    # Synthetic data goes into train and val only (NOT test)
    if synthetic_samples:
        random.shuffle(synthetic_samples)
        syn_val_count = int(round(len(synthetic_samples) * (val_ratio / (train_ratio + val_ratio))))
        syn_train = synthetic_samples[syn_val_count:]
        syn_val = synthetic_samples[:syn_val_count]
    else:
        syn_train = []
        syn_val = []

    train_samples = train_samples_real + syn_train
    val_samples = val_samples_real + syn_val

    # Shuffle train and val after merging
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"\nSplit sizes:")
    print(f"  train : {len(train_samples)} ({len(train_samples_real)} real + {len(syn_train)} synthetic)")
    print(f"  val   : {len(val_samples)} ({len(val_samples_real)} real + {len(syn_val)} synthetic)")
    print(f"  test  : {len(test_samples)} (real-world only)")

    # -----------------------------------------------------------------------
    # 4. Write output
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Writing unified dataset to {args.output_dir} ...")
    print("=" * 60)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_class_stats = {}

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_img_dir = output_dir / split_name / "images"
        split_lbl_dir = output_dir / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        class_counts: dict[int, int] = defaultdict(int)
        for idx, (img_path, annots) in enumerate(samples):
            copy_sample(img_path, annots, split_img_dir, split_lbl_dir, split_name, idx)
            for cls_id, *_ in annots:
                class_counts[cls_id] += 1

        split_class_stats[split_name] = {
            "num_images": len(samples),
            "annotations_per_class": {
                TARGET_CLASSES[k]: v for k, v in sorted(class_counts.items())
            },
        }
        print(f"  {split_name}: {len(samples)} images written.")

    # -----------------------------------------------------------------------
    # 5. Write data.yaml (for YOLO training tools)
    # -----------------------------------------------------------------------
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {list(TARGET_CLASSES.values())}\n")

    print(f"\n  data.yaml written to {data_yaml_path}")

    # -----------------------------------------------------------------------
    # 6. Write split_info.json
    # -----------------------------------------------------------------------
    split_info = {
        "seed": args.seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "sources": {
            "xview": len(xview_samples),
            "dota": len(dota_samples),
            "roboflow": len(roboflow_samples),
            "synthetic": len(synthetic_samples),
        },
        "splits": split_class_stats,
        "target_classes": TARGET_CLASSES,
        "note": "Test split contains only real-world data. Synthetic data is in train/val only.",
    }

    split_info_path = output_dir / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"  split_info.json written to {split_info_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
