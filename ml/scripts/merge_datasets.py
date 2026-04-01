#!/usr/bin/env python3
"""Merge all downloaded datasets into unified 7-class YOLO format.

Maps source classes → target classes:
  0: tank       ← t-72, t-64, t-80, tank
  1: apc_ifv    ← bmp-1/2/3, bmd-2, btr-70/80, mt-lb, APC
  2: military_truck ← Vehicle (generic), undefined vehicle
  3: sp_artillery   ← (synthetic only for now)
  4: mlrs           ← bm-21
  5: helicopter     ← (from other sources)
  6: uav            ← drone detection dataset

Output: ml/data/processed/{train,val,test}/{images,labels}/
"""

import os
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

random.seed(42)

PROJECT = Path(__file__).resolve().parent.parent.parent
RAW = PROJECT / "ml" / "data" / "raw"
OUT = PROJECT / "ml" / "data" / "processed"

TARGET_CLASSES = ["tank", "apc_ifv", "military_truck", "sp_artillery",
                  "mlrs", "helicopter", "uav"]

# ── Class mappings per dataset ─────────────────────────────────

MAPPINGS = {
    "roboflow_russian_tanks": {
        # src class index: target class index
        # data.yaml: bmp-1(0), bmp-2(1), bmp-3(2), mt-lb(3), t-72(4)
        0: 1,  # bmp-1 → apc_ifv
        1: 1,  # bmp-2 → apc_ifv
        2: 1,  # bmp-3 → apc_ifv
        3: 1,  # mt-lb → apc_ifv
        4: 0,  # t-72 → tank
    },
    "roboflow_mil_recognition": {
        # air-fighter(0), armoured personnel carrier(1), bomber(2),
        # soldier(3), tank(4), undefined vehicle(5)
        1: 1,  # APC → apc_ifv
        4: 0,  # tank → tank
        5: 2,  # undefined vehicle → military_truck
        # skip: air-fighter(0), bomber(2), soldier(3)
    },
    "roboflow_russian_annotated": {
        # bm-21(0), bmd-2(1), bmp-1(2), bmp-2(3), btr-70(4),
        # btr-80(5), mt-lb(6), t-64(7), t-72(8), t-80(9)
        0: 4,  # bm-21 → mlrs
        1: 1,  # bmd-2 → apc_ifv
        2: 1,  # bmp-1 → apc_ifv
        3: 1,  # bmp-2 → apc_ifv
        4: 1,  # btr-70 → apc_ifv
        5: 1,  # btr-80 → apc_ifv
        6: 1,  # mt-lb → apc_ifv
        7: 0,  # t-64 → tank
        8: 0,  # t-72 → tank
        9: 0,  # t-80 → tank
    },
    "roboflow_mil_detection": {
        # Person(0), Trench(1), Vehicle(2)
        2: 2,  # Vehicle → military_truck
        # skip: Person(0), Trench(1)
    },
}


def load_roboflow_dataset(name, mapping):
    """Load a Roboflow YOLOv8 dataset and remap labels."""
    base = RAW / name
    samples = []  # list of (image_path, [(target_cls, cx, cy, w, h), ...])

    for split in ["train", "valid", "test"]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = []
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    src_cls = int(parts[0])
                    if src_cls in mapping:
                        target_cls = mapping[src_cls]
                        boxes.append((target_cls, *[float(x) for x in parts[1:5]]))

            if boxes:  # only keep images with at least one mapped annotation
                samples.append((img_path, boxes))

    return samples


def load_drone_dataset():
    """Load HF drone detection dataset (already class 6)."""
    base = RAW / "hf_drone_detection"
    samples = []

    img_dir = base / "images"
    lbl_dir = base / "labels"
    if not img_dir.exists():
        return samples

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                boxes.append((int(parts[0]), *[float(x) for x in parts[1:5]]))
        if boxes:
            samples.append((img_path, boxes))

    return samples


def main():
    print("=" * 60)
    print("  Merging datasets → 7-class YOLO format")
    print("=" * 60)

    all_samples = []
    source_counts = {}

    # Load Roboflow datasets
    for name, mapping in MAPPINGS.items():
        samples = load_roboflow_dataset(name, mapping)
        source_counts[name] = len(samples)
        all_samples.extend(samples)
        print(f"  {name}: {len(samples)} images")

    # Load drone dataset
    drone_samples = load_drone_dataset()
    source_counts["hf_drone_detection"] = len(drone_samples)
    all_samples.extend(drone_samples)
    print(f"  hf_drone_detection: {len(drone_samples)} images")

    print(f"\n  Total: {len(all_samples)} images")

    # Count per-class annotations
    class_counts = defaultdict(int)
    for _, boxes in all_samples:
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1

    print(f"\n  Per-class annotation counts:")
    for i, name in enumerate(TARGET_CLASSES):
        print(f"    {i}: {name:<20} {class_counts.get(i, 0):>5}")

    # Shuffle and split: 70% train, 15% val, 15% test
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    # Create output directories
    for split in ["train", "val", "test"]:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files
    total_written = 0
    for split, samples in splits.items():
        print(f"\n  Writing {split}: {len(samples)} images...")
        for idx, (img_path, boxes) in enumerate(samples):
            ext = img_path.suffix
            fname = f"{split}_{idx:05d}{ext}"

            # Copy image
            shutil.copy2(img_path, OUT / split / "images" / fname)

            # Write label
            lines = []
            for cls_id, cx, cy, w, h in boxes:
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (OUT / split / "labels" / fname.replace(ext, ".txt")).write_text(
                "\n".join(lines) + "\n"
            )
            total_written += 1

    # Write dataset.yaml pointing to processed data
    dataset_yaml = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 7,
        "names": TARGET_CLASSES,
    }
    yaml_path = OUT / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\n{'=' * 60}")
    print(f"  DONE: {total_written} images written")
    print(f"    train: {len(splits['train'])}")
    print(f"    val:   {len(splits['val'])}")
    print(f"    test:  {len(splits['test'])}")
    print(f"  Dataset config: {yaml_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
