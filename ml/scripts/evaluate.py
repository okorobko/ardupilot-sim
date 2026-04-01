#!/usr/bin/env python3
"""
Per-class mAP evaluation for a trained YOLOv8 model.
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 model on the test set"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml",
        help="Path to dataset YAML (default: ../configs/dataset.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON (default: <weights_dir>/eval_results.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        print(f"[ERROR] Weights not found: {args.weights}")
        sys.exit(1)
    if not args.dataset.exists():
        print(f"[ERROR] Dataset config not found: {args.dataset}")
        sys.exit(1)

    output_path = args.output or (args.weights.resolve().parent / "eval_results.json")

    print(f"Weights:   {args.weights}")
    print(f"Dataset:   {args.dataset}")
    print(f"Image sz:  {args.imgsz}")
    print(f"Conf thr:  {args.conf_thres}")
    print(f"IoU thr:   {args.iou_thres}")
    print()

    model = YOLO(str(args.weights))

    results = model.val(
        data=str(args.dataset),
        imgsz=args.imgsz,
        conf=args.conf_thres,
        iou=args.iou_thres,
        verbose=True,
    )

    # ---- Extract metrics ----
    class_names = results.names  # dict {idx: name}
    box = results.box  # ultralytics Metric object

    # Per-class arrays (numpy)
    ap50 = box.ap50            # mAP@0.5 per class
    ap = box.ap                # mAP@0.5:0.95 per class
    precision = box.p          # precision per class
    recall = box.r             # recall per class

    # ---- Print per-class table ----
    header = f"{'Class':<25} {'mAP@0.5':>10} {'mAP@.5:.95':>12} {'Precision':>10} {'Recall':>10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    per_class_results = {}
    for idx in sorted(class_names.keys()):
        name = class_names[idx]
        if idx < len(ap50):
            row = {
                "mAP50": float(ap50[idx]),
                "mAP50_95": float(ap[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
            }
            per_class_results[name] = row
            print(
                f"{name:<25} {row['mAP50']:>10.4f} {row['mAP50_95']:>12.4f} "
                f"{row['precision']:>10.4f} {row['recall']:>10.4f}"
            )

    print("-" * len(header))
    print(
        f"{'ALL (mean)':<25} {box.map50:>10.4f} {box.map:>12.4f} "
        f"{precision.mean():>10.4f} {recall.mean():>10.4f}"
    )
    print("=" * len(header))

    # ---- Confusion matrix summary ----
    print("\nConfusion Matrix Summary")
    print("-" * 40)
    try:
        cm = results.confusion_matrix
        matrix = cm.matrix  # numpy array [pred x true]
        for idx in sorted(class_names.keys()):
            if idx < matrix.shape[0]:
                tp = int(matrix[idx, idx])
                fp = int(matrix[idx, :].sum() - tp)
                fn = int(matrix[:, idx].sum() - tp)
                print(f"  {class_names[idx]:<20}  TP={tp:>5}  FP={fp:>5}  FN={fn:>5}")
    except Exception as exc:
        print(f"  Could not extract confusion matrix: {exc}")

    # ---- Save results JSON ----
    output_data = {
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "imgsz": args.imgsz,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "overall": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision_mean": float(precision.mean()),
            "recall_mean": float(recall.mean()),
        },
        "per_class": per_class_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
