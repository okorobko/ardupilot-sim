#!/usr/bin/env python3
"""
Export a trained YOLOv8 model to ONNX, TensorRT FP16, and TFLite INT8 formats.
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def file_size_mb(path: Path) -> str:
    """Return human-readable file size in MB."""
    if path.exists():
        size = path.stat().st_size / (1024 * 1024)
        return f"{size:.2f} MB"
    return "N/A"


def export_onnx(model: YOLO, imgsz: int, output_dir: Path) -> None:
    """Export to ONNX format."""
    print("\n" + "-" * 50)
    print("Exporting to ONNX ...")
    try:
        onnx_path = model.export(format="onnx", imgsz=imgsz)
        onnx_path = Path(onnx_path)
        if output_dir != onnx_path.parent:
            dest = output_dir / onnx_path.name
            onnx_path.rename(dest)
            onnx_path = dest
        print(f"  [OK]  ONNX saved: {onnx_path}  ({file_size_mb(onnx_path)})")
    except Exception as exc:
        print(f"  [FAIL] ONNX export failed: {exc}")


def export_tensorrt(model: YOLO, imgsz: int, output_dir: Path) -> None:
    """Export to TensorRT FP16 engine (requires NVIDIA GPU + TensorRT)."""
    print("\n" + "-" * 50)
    print("Exporting to TensorRT FP16 ...")
    print("  NOTE: Requires a machine with an NVIDIA GPU and TensorRT installed.")
    try:
        engine_path = model.export(format="engine", imgsz=imgsz, half=True)
        engine_path = Path(engine_path)
        if output_dir != engine_path.parent:
            dest = output_dir / engine_path.name
            engine_path.rename(dest)
            engine_path = dest
        print(f"  [OK]  TensorRT saved: {engine_path}  ({file_size_mb(engine_path)})")
    except Exception as exc:
        print(f"  [FAIL] TensorRT export failed: {exc}")
        print("         Make sure you are on a machine with an NVIDIA GPU and TensorRT.")


def export_tflite_int8(model: YOLO, imgsz: int, output_dir: Path, calib_dir: Path) -> None:
    """Export to TFLite INT8 with calibration images."""
    print("\n" + "-" * 50)
    print("Exporting to TFLite INT8 ...")
    if not calib_dir.exists():
        print(f"  [WARN] Calibration directory not found: {calib_dir}")
        print("         INT8 quantisation requires representative images.")
        print("         Place calibration images in: {calib_dir}")
        return
    try:
        tflite_path = model.export(
            format="tflite",
            imgsz=imgsz,
            int8=True,
            data=str(calib_dir),
        )
        tflite_path = Path(tflite_path)
        if output_dir != tflite_path.parent:
            dest = output_dir / tflite_path.name
            tflite_path.rename(dest)
            tflite_path = dest
        print(f"  [OK]  TFLite INT8 saved: {tflite_path}  ({file_size_mb(tflite_path)})")
    except Exception as exc:
        print(f"  [FAIL] TFLite INT8 export failed: {exc}")


def print_hailo_instructions() -> None:
    """Print instructions for Hailo HEF export."""
    print("\n" + "-" * 50)
    print("Hailo HEF Export (manual)")
    print("-" * 50)
    print(
        "  Hailo HEF export requires the Hailo SDK (Dataflow Compiler).\n"
        "  Steps:\n"
        "    1. Install Hailo SDK: https://hailo.ai/developer-zone/\n"
        "    2. Convert ONNX model to Hailo Archive (.har):\n"
        "         hailo parser onnx model.onnx\n"
        "    3. Optimise with calibration data:\n"
        "         hailo optimize model.har --calib-path ./calibration/\n"
        "    4. Compile to HEF:\n"
        "         hailo compiler model.har\n"
        "    The resulting .hef file can be deployed on Hailo-8/8L accelerators."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX, TensorRT FP16, and TFLite INT8"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained best.pt weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store exported models (default: same as weights dir)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        print(f"[ERROR] Weights file not found: {args.weights}")
        sys.exit(1)

    output_dir = args.output_dir or args.weights.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_dir = Path(__file__).resolve().parent.parent / "data" / "calibration"

    print(f"Weights:    {args.weights}")
    print(f"Image size: {args.imgsz}")
    print(f"Output dir: {output_dir}")

    model = YOLO(str(args.weights))

    export_onnx(model, args.imgsz, output_dir)
    export_tensorrt(model, args.imgsz, output_dir)
    export_tflite_int8(model, args.imgsz, output_dir, calib_dir)
    print_hailo_instructions()

    print("\n" + "=" * 50)
    print("Export pipeline complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
