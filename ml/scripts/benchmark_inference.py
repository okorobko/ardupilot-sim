#!/usr/bin/env python3
"""
Benchmark inference latency for ONNX, TensorRT, and TFLite models.
Reports mean / median / p95 / p99 latency and FPS.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Runners for each backend
# ---------------------------------------------------------------------------

def benchmark_onnx(model_path: str, imgsz: int, warmup: int, iterations: int) -> list[float]:
    """Benchmark ONNX model using onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime is not installed. pip install onnxruntime")
        sys.exit(1)

    providers = ort.get_available_providers()
    print(f"  ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(model_path, providers=providers)

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    # Replace dynamic dims with concrete values
    shape = [
        d if isinstance(d, int) else 1 if i == 0 else imgsz
        for i, d in enumerate(input_shape)
    ]
    # Ensure standard BCHW: [1, 3, imgsz, imgsz]
    if len(shape) == 4:
        shape[0] = 1
        shape[2] = imgsz
        shape[3] = imgsz

    dummy = np.random.randn(*shape).astype(np.float32)

    print(f"  Input: {input_name}  shape={shape}")
    print(f"  Warmup ({warmup} iters) ...", end="", flush=True)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})
    print(" done")

    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append(time.perf_counter() - t0)

    return latencies


def benchmark_tensorrt(model_path: str, imgsz: int, warmup: int, iterations: int) -> list[float]:
    """Benchmark TensorRT engine (via ultralytics YOLO predict interface)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"  Warmup ({warmup} iters) ...", end="", flush=True)
    for _ in range(warmup):
        model.predict(dummy, verbose=False)
    print(" done")

    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        latencies.append(time.perf_counter() - t0)

    return latencies


def benchmark_tflite(model_path: str, imgsz: int, warmup: int, iterations: int) -> list[float]:
    """Benchmark TFLite model."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite  # type: ignore[import]
        except ImportError:
            print("[ERROR] Neither tflite_runtime nor tensorflow is installed.")
            sys.exit(1)

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    dummy = np.random.randn(*input_shape).astype(input_dtype)
    input_index = input_details[0]["index"]

    print(f"  Input shape={list(input_shape)}  dtype={input_dtype}")
    print(f"  Warmup ({warmup} iters) ...", end="", flush=True)
    for _ in range(warmup):
        interpreter.set_tensor(input_index, dummy)
        interpreter.invoke()
    print(" done")

    latencies: list[float] = []
    for _ in range(iterations):
        interpreter.set_tensor(input_index, dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append(time.perf_counter() - t0)

    return latencies


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(latencies: list[float], model_path: str) -> None:
    """Print a formatted table of latency statistics."""
    arr = np.array(latencies) * 1000  # convert to ms

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    fps = 1000.0 / mean if mean > 0 else 0.0

    width = 50
    print("\n" + "=" * width)
    print(f"  Benchmark Results: {Path(model_path).name}")
    print("=" * width)
    print(f"  {'Iterations:':<20} {len(latencies)}")
    print(f"  {'Mean latency:':<20} {mean:>8.2f} ms")
    print(f"  {'Median latency:':<20} {median:>8.2f} ms")
    print(f"  {'P95 latency:':<20} {p95:>8.2f} ms")
    print(f"  {'P99 latency:':<20} {p99:>8.2f} ms")
    print(f"  {'Throughput:':<20} {fps:>8.1f} FPS")
    print("=" * width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def detect_format(model_path: Path) -> str:
    """Auto-detect model format from file extension."""
    suffix = model_path.suffix.lower()
    fmt_map = {
        ".onnx": "onnx",
        ".engine": "tensorrt",
        ".trt": "tensorrt",
        ".tflite": "tflite",
    }
    fmt = fmt_map.get(suffix)
    if fmt is None:
        print(f"[ERROR] Unrecognised model extension: {suffix}")
        print(f"        Supported: {', '.join(fmt_map.keys())}")
        sys.exit(1)
    return fmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency for ONNX / TensorRT / TFLite models"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model file (.onnx, .engine/.trt, .tflite)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of timed iterations (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    fmt = detect_format(args.model)
    model_str = str(args.model)

    print(f"Model:      {args.model}")
    print(f"Format:     {fmt}")
    print(f"Image size: {args.imgsz}")
    print(f"Warmup:     {args.warmup}")
    print(f"Iterations: {args.iterations}")

    runners = {
        "onnx": benchmark_onnx,
        "tensorrt": benchmark_tensorrt,
        "tflite": benchmark_tflite,
    }

    latencies = runners[fmt](model_str, args.imgsz, args.warmup, args.iterations)
    print_results(latencies, model_str)


if __name__ == "__main__":
    main()
