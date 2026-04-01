#!/usr/bin/env python3
"""Download open-source aerial military vehicle datasets.

Downloads from sources that don't require registration:
  1. HuggingFace: Military-Aircraft-Detection (4.4K images, helicopters + aircraft)
  2. HuggingFace: Drone Detection Dataset (54K images, UAV class)
  3. VEDAI: Vehicle Detection in Aerial Imagery (overhead vehicle photos)

For datasets requiring registration (downloaded separately):
  - xView: https://challenge.xviewdataset.org (tanks, APCs, trucks, helicopters)
  - Roboflow: Use API key with download_datasets.py
  - Kaggle Military Aircraft: kaggle datasets download -d a2015003713/militaryaircraftdetectiondataset

Usage:
    pip install datasets Pillow tqdm
    python download_open_datasets.py [--output-dir ../data/raw] [--datasets all]
"""

import argparse
import csv
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *a, **kw):
            self._it = iterable
            desc = kw.get("desc", "")
            if desc:
                print(f"  {desc}...")
        def __iter__(self):
            return iter(self._it) if self._it else iter([])
        def update(self, n=1): pass
        def close(self): pass


def download_hf_military_aircraft(output_dir: Path, max_images: int = 2000):
    """Download Military-Aircraft-Detection from HuggingFace.

    Contains ~4.4K images of military aircraft including helicopters.
    We filter to helicopter classes for our 'helicopter' target class,
    and keep a sample of other aircraft for negative mining.
    """
    print("\n" + "=" * 60)
    print("  [1/3] HuggingFace: Military-Aircraft-Detection")
    print("  Source: Illia56/Military-Aircraft-Detection (Apache 2.0)")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' package not installed.")
        print("  Run: pip install datasets")
        return 0

    dest = output_dir / "hf_military_aircraft"
    dest_images = dest / "images"
    dest_labels = dest / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    print("  Downloading dataset from HuggingFace...")
    try:
        ds = load_dataset("Illia56/Military-Aircraft-Detection", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0

    # Helicopter-related classes (maps to our class 5: helicopter)
    HELI_CLASSES = {
        "AH-64", "AH64", "Apache", "CH-47", "CH47", "Mi-24", "Mi24",
        "Mi-28", "Mi28", "Ka-52", "Ka52", "UH-60", "UH60", "BlackHawk",
        "Helicopter", "V-22", "V22",
    }
    # UAV classes (maps to our class 6: uav)
    UAV_CLASSES = {"MQ-9", "MQ9", "RQ-4", "RQ4", "TB2", "TB-2", "Drone", "UAV"}

    saved = 0
    heli_count = 0
    uav_count = 0
    other_count = 0

    print(f"  Processing {len(ds)} images (max {max_images})...")

    for i, sample in enumerate(tqdm(ds, desc="HF Military Aircraft", total=len(ds))):
        if saved >= max_images:
            break

        try:
            image = sample.get("image")
            objects = sample.get("objects", {})
            if image is None or not objects:
                continue

            bboxes = objects.get("bbox", [])
            categories = objects.get("category", [])
            if not bboxes:
                continue

            # Get image dimensions
            w, h = image.size

            # Process annotations
            yolo_lines = []
            has_target = False

            for bbox, cat in zip(bboxes, categories):
                # Determine class mapping
                cat_str = str(cat) if not isinstance(cat, str) else cat

                # Check if it's a helicopter
                is_heli = any(hc.lower() in cat_str.lower() for hc in HELI_CLASSES)
                is_uav = any(uc.lower() in cat_str.lower() for uc in UAV_CLASSES)

                if is_heli:
                    target_class = 5  # helicopter
                    has_target = True
                    heli_count += 1
                elif is_uav:
                    target_class = 6  # uav
                    has_target = True
                    uav_count += 1
                else:
                    continue  # Skip non-target classes

                # Convert bbox to YOLO format
                # HF datasets bbox format: [x_min, y_min, width, height] (COCO)
                if len(bbox) == 4:
                    x_min, y_min, bw, bh = bbox
                    cx = (x_min + bw / 2) / w
                    cy = (y_min + bh / 2) / h
                    nw = bw / w
                    nh = bh / h

                    if 0 < nw < 1 and 0 < nh < 1:
                        yolo_lines.append(
                            f"{target_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                        )

            if not yolo_lines:
                # Keep some images without targets as background/negatives
                if other_count < max_images // 10:
                    other_count += 1
                else:
                    continue

            # Save image
            fname = f"hf_aircraft_{i:05d}.jpg"
            image.save(str(dest_images / fname), "JPEG", quality=90)

            # Save labels
            label_path = dest_labels / fname.replace(".jpg", ".txt")
            label_path.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")

            saved += 1

        except Exception as e:
            continue

    print(f"  Saved: {saved} images ({heli_count} helicopter, {uav_count} UAV annotations)")
    _write_classes_txt(dest)
    return saved


def download_hf_drone_detection(output_dir: Path, max_images: int = 3000):
    """Download drone/UAV detection dataset from HuggingFace.

    Contains 54K images but we sample a subset for UAV class training.
    Maps to our class 6: uav.
    """
    print("\n" + "=" * 60)
    print("  [2/3] HuggingFace: Drone Detection Dataset")
    print("  Source: pathikg/drone-detection-dataset (MIT)")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' package not installed.")
        return 0

    dest = output_dir / "hf_drone_detection"
    dest_images = dest / "images"
    dest_labels = dest / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    print("  Downloading dataset (streaming mode for efficiency)...")
    try:
        ds = load_dataset("pathikg/drone-detection-dataset", split="train",
                          streaming=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0

    saved = 0
    annotations_count = 0
    TARGET_CLASS = 6  # uav

    print(f"  Processing images (max {max_images})...")

    for i, sample in enumerate(tqdm(ds, desc="HF Drone Detection",
                                     total=max_images)):
        if saved >= max_images:
            break

        try:
            image = sample.get("image")
            objects = sample.get("objects", {})
            if image is None:
                continue

            w, h = image.size
            bboxes = objects.get("bbox", [])

            yolo_lines = []
            for bbox in bboxes:
                if len(bbox) == 4:
                    x_min, y_min, bw, bh = bbox
                    cx = (x_min + bw / 2) / w
                    cy = (y_min + bh / 2) / h
                    nw = bw / w
                    nh = bh / h
                    if 0 < nw < 1 and 0 < nh < 1:
                        yolo_lines.append(
                            f"{TARGET_CLASS} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                        )
                        annotations_count += 1

            if not yolo_lines:
                continue

            fname = f"hf_drone_{i:05d}.jpg"
            image.save(str(dest_images / fname), "JPEG", quality=90)
            (dest_labels / fname.replace(".jpg", ".txt")).write_text(
                "\n".join(yolo_lines) + "\n"
            )
            saved += 1

        except Exception as e:
            continue

    print(f"  Saved: {saved} images ({annotations_count} UAV annotations)")
    _write_classes_txt(dest)
    return saved


def download_vedai(output_dir: Path):
    """Download VEDAI (Vehicle Detection in Aerial Imagery).

    Small overhead vehicle images at 512x512 and 1024x1024.
    Classes include cars, trucks, tractors, camping cars, boats, vans,
    pick-ups, planes, others.
    Maps to: military_truck (class 2) for trucks/large vehicles.
    """
    print("\n" + "=" * 60)
    print("  [3/3] VEDAI: Vehicle Detection in Aerial Imagery")
    print("  Source: https://downloads.greyc.fr/vedai/")
    print("=" * 60)

    import urllib.request

    dest = output_dir / "vedai"
    dest.mkdir(parents=True, exist_ok=True)

    base_url = "https://downloads.greyc.fr/vedai/"
    files = [
        ("Annotations512.tar", "Annotations (512x512)"),
        ("Vehicules512.tar.001", "Images 512x512 part 1"),
        ("Vehicules512.tar.002", "Images 512x512 part 2"),
    ]

    downloaded = 0
    for fname, desc in files:
        fpath = dest / fname
        if fpath.exists():
            print(f"  Already exists: {fname}")
            downloaded += 1
            continue

        url = base_url + fname
        print(f"  Downloading {desc}: {fname}...")
        try:
            urllib.request.urlretrieve(url, str(fpath))
            downloaded += 1
            print(f"    Done ({fpath.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Extract if tar files exist
    if downloaded > 0:
        import tarfile
        for fname, _ in files:
            fpath = dest / fname
            if fpath.exists() and fpath.suffix == ".tar":
                print(f"  Extracting {fname}...")
                try:
                    with tarfile.open(str(fpath)) as tf:
                        tf.extractall(path=str(dest))
                except Exception as e:
                    print(f"    Extraction note: {e}")

    print(f"  Downloaded {downloaded}/{len(files)} files")
    print(f"  Note: VEDAI needs manual label conversion (run prepare_dataset.py)")
    return downloaded


def _write_classes_txt(dest: Path):
    """Write classes.txt for the dataset."""
    classes = [
        "tank", "apc_ifv", "military_truck", "sp_artillery",
        "mlrs", "helicopter", "uav",
    ]
    (dest / "classes.txt").write_text("\n".join(classes) + "\n")


def print_summary(counts: dict, output_dir: Path):
    """Print download summary."""
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    total = sum(counts.values())
    for source, count in counts.items():
        status = f"{count} images/files" if count > 0 else "FAILED"
        print(f"  {source:<40} {status}")
    print(f"  {'─' * 55}")
    print(f"  {'Total':<40} {total}")
    print(f"\n  Output directory: {output_dir}")

    print(f"\n  Class coverage from downloads:")
    print(f"    helicopter (5): HF Military-Aircraft-Detection")
    print(f"    uav (6):        HF Drone Detection Dataset")
    print(f"    vehicles (2):   VEDAI (trucks in aerial view)")
    print(f"")
    print(f"  Still needed (require registration):")
    print(f"    tank (0), apc_ifv (1), sp_artillery (3), mlrs (4):")
    print(f"      → xView: https://challenge.xviewdataset.org")
    print(f"      → Roboflow: military-vehicle-detection datasets")
    print(f"      → Gazebo synthetic generation")
    print(f"")
    print(f"  Next steps:")
    print(f"    1. python prepare_dataset.py  (merge + convert to YOLO)")
    print(f"    2. python generate_synthetic.py  (Gazebo synthetic data)")
    print(f"    3. python train.py  (train YOLOv8n)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download open-source aerial military vehicle datasets")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data" / "raw"),
                        help="Output directory")
    parser.add_argument("--datasets", type=str, default="all",
                        choices=["all", "aircraft", "drone", "vedai"],
                        help="Which datasets to download")
    parser.add_argument("--max-aircraft", type=int, default=2000,
                        help="Max images from aircraft dataset")
    parser.add_argument("--max-drone", type=int, default=3000,
                        help="Max images from drone dataset")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Open-Source Aerial Military Dataset Downloader")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    counts = {}

    if args.datasets in ("all", "aircraft"):
        counts["HF Military-Aircraft"] = download_hf_military_aircraft(
            output_dir, max_images=args.max_aircraft)

    if args.datasets in ("all", "drone"):
        counts["HF Drone Detection"] = download_hf_drone_detection(
            output_dir, max_images=args.max_drone)

    if args.datasets in ("all", "vedai"):
        counts["VEDAI"] = download_vedai(output_dir)

    print_summary(counts, output_dir)


if __name__ == "__main__":
    main()
