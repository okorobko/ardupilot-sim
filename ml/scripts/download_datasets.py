#!/usr/bin/env python3
"""
Download real-world aerial/satellite datasets for military vehicle detection training.

Supported datasets:
  - xView: Large-scale satellite imagery with bounding boxes (manual download required)
  - DOTA v2: Oriented object detection in aerial images (manual download required)
  - Roboflow: Community datasets accessible via API (API key required)

Usage:
  python download_datasets.py --datasets all
  python download_datasets.py --datasets xview dota
  python download_datasets.py --datasets roboflow --output-dir ./my_data
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Continuing without progress bars.\n")

    class tqdm:
        """Minimal fallback when tqdm is not available."""

        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable
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

        @staticmethod
        def write(msg):
            print(msg)


XVIEW_URL = "https://challenge.xviewdataset.org/download-links"
DOTA_URL = "https://captain-whu.github.io/DOTA/dataset.html"

XVIEW_EXPECTED_FILES = [
    "train_images.tgz",
    "train_labels.tgz",
    "val_images.tgz",
]

DOTA_EXPECTED_FILES = [
    "train.zip",
    "val.zip",
    "train_labels.zip",
    "val_labels.zip",
]


def setup_directories(output_dir: Path) -> dict[str, Path]:
    """Create the raw data subdirectories and return their paths."""
    dirs = {}
    for name in ("xview", "dota", "roboflow"):
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs


# ---------------------------------------------------------------------------
# xView
# ---------------------------------------------------------------------------

def download_xview(data_dir: Path) -> dict:
    """Check for xView dataset files; print manual download instructions if missing."""
    print("\n" + "=" * 60)
    print("xView Dataset")
    print("=" * 60)

    result = {"name": "xView", "found": [], "missing": []}

    for fname in XVIEW_EXPECTED_FILES:
        fpath = data_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  [OK]  {fname} ({size_mb:.1f} MB)")
            result["found"].append(fname)
        else:
            print(f"  [--]  {fname} not found")
            result["missing"].append(fname)

    if result["missing"]:
        print(f"\nxView requires manual download (free registration):")
        print(f"  1. Visit: {XVIEW_URL}")
        print(f"  2. Register / sign in")
        print(f"  3. Download the following files:")
        for f in result["missing"]:
            print(f"       - {f}")
        print(f"  4. Place them in: {data_dir.resolve()}")
    else:
        print("\nAll xView files present.")

    # If archives are present but not extracted, offer to extract
    for fname in result["found"]:
        fpath = data_dir / fname
        extracted_marker = data_dir / f".{fname}.extracted"
        if fpath.suffix == ".tgz" and not extracted_marker.exists():
            answer = _prompt_yn(f"  Extract {fname}?")
            if answer:
                import tarfile

                print(f"  Extracting {fname}...")
                with tarfile.open(fpath, "r:gz") as tar:
                    members = tar.getmembers()
                    for member in tqdm(members, desc=f"Extracting {fname}"):
                        tar.extract(member, path=data_dir)
                extracted_marker.touch()
                print(f"  Done extracting {fname}.")

    return result


# ---------------------------------------------------------------------------
# DOTA v2
# ---------------------------------------------------------------------------

def download_dota(data_dir: Path) -> dict:
    """Check for DOTA v2 dataset files; print manual download instructions if missing."""
    print("\n" + "=" * 60)
    print("DOTA v2 Dataset")
    print("=" * 60)

    result = {"name": "DOTA v2", "found": [], "missing": []}

    for fname in DOTA_EXPECTED_FILES:
        fpath = data_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  [OK]  {fname} ({size_mb:.1f} MB)")
            result["found"].append(fname)
        else:
            print(f"  [--]  {fname} not found")
            result["missing"].append(fname)

    if result["missing"]:
        print(f"\nDOTA v2 requires manual download (registration needed):")
        print(f"  1. Visit: {DOTA_URL}")
        print(f"  2. Register and request access to DOTA v2.0")
        print(f"  3. Download the following splits:")
        for f in result["missing"]:
            print(f"       - {f}")
        print(f"  4. Place them in: {data_dir.resolve()}")
        print(f"\n  Tip: For military vehicle detection, the relevant DOTA classes include:")
        print(f"       large-vehicle, small-vehicle, helicopter, plane, ship")
    else:
        print("\nAll DOTA v2 files present.")

    # Extract zips if present
    for fname in result["found"]:
        fpath = data_dir / fname
        extracted_marker = data_dir / f".{fname}.extracted"
        if fpath.suffix == ".zip" and not extracted_marker.exists():
            answer = _prompt_yn(f"  Extract {fname}?")
            if answer:
                print(f"  Extracting {fname}...")
                with zipfile.ZipFile(fpath, "r") as zf:
                    members = zf.namelist()
                    for member in tqdm(members, desc=f"Extracting {fname}"):
                        zf.extract(member, path=data_dir)
                extracted_marker.touch()
                print(f"  Done extracting {fname}.")

    return result


# ---------------------------------------------------------------------------
# Roboflow
# ---------------------------------------------------------------------------

def download_roboflow(data_dir: Path) -> dict:
    """Download military vehicle aerial datasets from Roboflow."""
    print("\n" + "=" * 60)
    print("Roboflow Community Datasets")
    print("=" * 60)

    result = {"name": "Roboflow", "found": [], "missing": []}

    # Known community datasets for military / vehicle aerial detection.
    # These are public datasets; an API key is still required by the client.
    roboflow_datasets = [
        {
            "workspace": "ds",
            "project": "military-vehicles-aerial",
            "version": 1,
            "format": "yolov8",
        },
    ]

    try:
        from roboflow import Roboflow
    except ImportError:
        print("\n  roboflow Python package is not installed.")
        print("  Install it with:  pip install roboflow")
        print("  Then set your API key via ROBOFLOW_API_KEY env var or use:")
        print('    rf = Roboflow(api_key="YOUR_KEY")')
        result["missing"].append("roboflow package not installed")
        return result

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("\n  ROBOFLOW_API_KEY environment variable is not set.")
        print("  1. Sign up at https://app.roboflow.com")
        print("  2. Go to Settings -> API Keys")
        print("  3. Export:  export ROBOFLOW_API_KEY='your_key_here'")
        print("  4. Re-run this script with --datasets roboflow")
        result["missing"].append("ROBOFLOW_API_KEY not set")
        return result

    try:
        rf = Roboflow(api_key=api_key)

        for ds_info in tqdm(roboflow_datasets, desc="Roboflow datasets"):
            workspace = ds_info["workspace"]
            project_name = ds_info["project"]
            version = ds_info["version"]
            fmt = ds_info["format"]

            dest = data_dir / f"{workspace}_{project_name}_v{version}"
            if dest.exists() and any(dest.iterdir()):
                print(f"\n  [OK]  {project_name} v{version} already downloaded at {dest}")
                result["found"].append(project_name)
                continue

            print(f"\n  Downloading {workspace}/{project_name} v{version} ({fmt})...")
            try:
                project = rf.workspace(workspace).project(project_name)
                dataset = project.version(version).download(fmt, location=str(dest))
                print(f"  [OK]  Saved to {dest}")
                result["found"].append(project_name)
            except Exception as e:
                print(f"  [FAIL] Could not download {project_name}: {e}")
                print(f"         Try browsing https://universe.roboflow.com/ for alternative datasets.")
                result["missing"].append(project_name)

    except Exception as e:
        print(f"\n  Roboflow API error: {e}")
        result["missing"].append(f"API error: {e}")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt_yn(prompt: str) -> bool:
    """Ask a yes/no question; return True for yes. Non-interactive defaults to no."""
    if not sys.stdin.isatty():
        return False
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_summary(results: list[dict]) -> None:
    """Print a final summary table."""
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_found = 0
    total_missing = 0

    for r in results:
        found = len(r["found"])
        missing = len(r["missing"])
        total_found += found
        total_missing += missing

        status = "READY" if missing == 0 else "INCOMPLETE"
        print(f"  {r['name']:.<30} {status}  ({found} found, {missing} missing)")

    print("-" * 60)
    if total_missing == 0:
        print("  All datasets are available.")
    else:
        print(f"  {total_missing} item(s) still need attention. See details above.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download aerial/satellite datasets for military vehicle detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --datasets all\n"
            "  %(prog)s --datasets xview dota\n"
            "  %(prog)s --datasets roboflow --output-dir ./my_data/raw\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Root directory for downloaded data (default: ../data/raw)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["xview", "dota", "roboflow", "all"],
        default=["all"],
        help="Which datasets to download (default: all)",
    )
    args = parser.parse_args()

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"xview", "dota", "roboflow"}

    output_dir: Path = args.output_dir.resolve()
    print(f"Output directory: {output_dir}")

    dirs = setup_directories(output_dir)
    results = []

    if "xview" in selected:
        results.append(download_xview(dirs["xview"]))

    if "dota" in selected:
        results.append(download_dota(dirs["dota"]))

    if "roboflow" in selected:
        results.append(download_roboflow(dirs["roboflow"]))

    print_summary(results)


if __name__ == "__main__":
    main()
