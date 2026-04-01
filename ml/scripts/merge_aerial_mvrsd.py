#!/usr/bin/env python3
"""Merge MV-RSD satellite dataset into the aerial training set.

MV-RSD classes → our 7-class mapping:
  AFV (Armored Fighting Vehicle) → split: larger AFVs → tank (0), smaller → apc_ifv (1)
  SMV (Small Military Vehicle)   → apc_ifv (1)
  LMV (Large Military Vehicle)   → military_truck (2)
  MCV (Military Construction Vehicle) → military_truck (2)
  CV  (Civilian Vehicle)         → SKIP (not target)

Output: adds images+labels to ml/data/aerial/{train,val}/
"""

import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

random.seed(42)

PROJECT = Path(__file__).resolve().parent.parent.parent
MVRSD = PROJECT / "ml" / "data" / "raw" / "mvrsd"
AERIAL = PROJECT / "ml" / "data" / "aerial"

CLASS_MAP = {
    "AFV": None,   # Special handling: split by bbox size
    "SMV": 1,      # apc_ifv
    "LMV": 2,      # military_truck
    "MCV": 2,      # military_truck
    "CV": None,    # Skip civilian
}

# AFV bbox area threshold: larger = tank, smaller = apc
AFV_AREA_THRESHOLD = 8000  # pixels (roughly 90x90)


def convert_xml_to_yolo(xml_path, img_w=640, img_h=640):
    """Convert XML annotation to YOLO format lines."""
    tree = ET.parse(xml_path)
    lines = []
    has_military = False

    for obj in tree.findall(".//object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Skip civilian vehicles
        if name == "CV":
            continue

        # Map class
        if name == "AFV":
            area = (xmax - xmin) * (ymax - ymin)
            cls_id = 0 if area >= AFV_AREA_THRESHOLD else 1  # tank vs apc
        elif name in CLASS_MAP and CLASS_MAP[name] is not None:
            cls_id = CLASS_MAP[name]
        else:
            continue

        has_military = True

        # Convert to YOLO format (normalized cx, cy, w, h)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0.001, min(1, w))
        h = max(0.001, min(1, h))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines, has_military


def main():
    print("=" * 60)
    print("  Merging MV-RSD → Aerial Dataset")
    print("=" * 60)

    # Process train and val splits
    added = defaultdict(int)
    cls_counts = defaultdict(int)
    total_annotations = 0

    for split in ["train", "val"]:
        xml_dir = MVRSD / "labels" / split / "xml"
        img_dir = MVRSD / "images" / split

        if not xml_dir.exists():
            print(f"  Skipping {split}: {xml_dir} not found")
            continue

        xml_files = sorted(xml_dir.glob("*.xml"))
        print(f"\n  Processing {split}: {len(xml_files)} XML files...")

        out_img = AERIAL / split / "images"
        out_lbl = AERIAL / split / "labels"

        for xml_path in xml_files:
            lines, has_military = convert_xml_to_yolo(xml_path)

            if not has_military or not lines:
                continue

            # Find matching image
            img_name = xml_path.stem + ".jpg"
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            # Copy with mvrsd_ prefix to avoid name collisions
            out_name = f"mvrsd_{img_name}"
            shutil.copy2(img_path, out_img / out_name)

            lbl_name = f"mvrsd_{xml_path.stem}.txt"
            (out_lbl / lbl_name).write_text("\n".join(lines) + "\n")

            added[split] += 1
            total_annotations += len(lines)
            for line in lines:
                cls_id = int(line.split()[0])
                cls_counts[cls_id] += 1

    names = ["tank", "apc_ifv", "military_truck", "sp_artillery",
             "mlrs", "helicopter", "uav"]

    print(f"\n{'=' * 60}")
    print(f"  DONE: Added {sum(added.values())} MV-RSD images to aerial set")
    for split, count in added.items():
        print(f"    {split}: +{count}")
    print(f"  Total new annotations: {total_annotations}")
    print(f"\n  Per-class (MV-RSD only):")
    for i, n in enumerate(names):
        c = cls_counts.get(i, 0)
        print(f"    {i}: {n:<20} {c:>5}")

    # Count total aerial dataset now
    print(f"\n  Total aerial dataset after merge:")
    for split in ["train", "val", "test"]:
        imgs = list((AERIAL / split / "images").glob("*.*"))
        print(f"    {split}: {len(imgs)} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
