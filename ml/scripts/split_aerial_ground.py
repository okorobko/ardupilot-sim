#!/usr/bin/env python3
"""Split the merged dataset into aerial and ground subsets.

Aerial: objects occupy < 8% of image area (overhead drone view, high altitude)
Ground: objects occupy > 8% of image area (close-up, ground-level, low altitude)

Output:
  ml/data/aerial/{train,val,test}/{images,labels}/
  ml/data/ground/{train,val,test}/{images,labels}/
"""

import shutil
import yaml
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT / "ml" / "data" / "processed"
AERIAL_DIR = PROJECT / "ml" / "data" / "aerial"
GROUND_DIR = PROJECT / "ml" / "data" / "ground"
THRESHOLD = 0.08  # 8% of image area

CLASS_NAMES = ["tank", "apc_ifv", "military_truck", "sp_artillery",
               "mlrs", "helicopter", "uav"]


def main():
    print("=" * 60)
    print("  Splitting dataset: Aerial vs Ground")
    print(f"  Threshold: max object area < {THRESHOLD*100:.0f}% = aerial")
    print("=" * 60)

    for out_dir in [AERIAL_DIR, GROUND_DIR]:
        for split in ["train", "val", "test"]:
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    aerial_count = defaultdict(int)
    ground_count = defaultdict(int)
    aerial_cls = defaultdict(int)
    ground_cls = defaultdict(int)

    for split in ["train", "val", "test"]:
        img_dir = PROCESSED / split / "images"
        lbl_dir = PROCESSED / split / "labels"

        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            # Find matching image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / (lbl_path.stem + ext)
                if p.exists():
                    img_path = p
                    break
            if not img_path:
                continue

            # Analyze object sizes
            max_area = 0
            classes = []
            for line in lbl_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    bw, bh = float(parts[3]), float(parts[4])
                    max_area = max(max_area, bw * bh)
                    classes.append(int(parts[0]))

            if not classes:
                continue

            # Classify
            if max_area < THRESHOLD:
                dest = AERIAL_DIR
                aerial_count[split] += 1
                for c in classes:
                    aerial_cls[c] += 1
            else:
                dest = GROUND_DIR
                ground_count[split] += 1
                for c in classes:
                    ground_cls[c] += 1

            # Copy files
            shutil.copy2(img_path, dest / split / "images" / img_path.name)
            shutil.copy2(lbl_path, dest / split / "labels" / lbl_path.name)

    # Write dataset.yaml for each
    for name, out_dir, cls_counts in [
        ("aerial", AERIAL_DIR, aerial_cls),
        ("ground", GROUND_DIR, ground_cls),
    ]:
        cfg = {
            "path": str(out_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 7,
            "names": CLASS_NAMES,
        }
        with open(out_dir / "dataset.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # Print summary
    for label, counts, cls_c in [
        ("AERIAL (>5m altitude)", aerial_count, aerial_cls),
        ("GROUND (<5m altitude)", ground_count, ground_cls),
    ]:
        total = sum(counts.values())
        print(f"\n  {label}: {total} images")
        for split in ["train", "val", "test"]:
            print(f"    {split}: {counts[split]}")
        print(f"  Per-class annotations:")
        for i, n in enumerate(CLASS_NAMES):
            c = cls_c.get(i, 0)
            bar = "#" * min(c // 20, 40)
            print(f"    {i}: {n:<20} {c:>5}  {bar}")

    print(f"\n  Output:")
    print(f"    {AERIAL_DIR}")
    print(f"    {GROUND_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
