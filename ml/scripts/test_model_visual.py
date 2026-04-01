#!/usr/bin/env python3
"""Visual model testing — run inference on test images and save annotated results.

Produces:
  - Annotated test images with bounding boxes (predictions vs ground truth)
  - Per-class detection statistics
  - Confidence distribution analysis
  - Side-by-side ground truth vs prediction comparisons
  - Summary mosaic of best/worst detections

Usage:
    python test_model_visual.py [--model path/to/best.onnx] [--num 50]
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))
from detector import VehicleDetector, CLASS_NAMES, CLASS_COLORS

PROJECT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT / "ml"
DEFAULT_MODEL = ML_DIR / "models" / "mil_vehicle_v1" / "weights" / "best.onnx"
TEST_DIR = ML_DIR / "data" / "processed" / "test"
OUTPUT_DIR = ML_DIR / "test_results"

# Colors for ground truth (green) vs prediction (class-colored)
GT_COLOR = (0, 255, 0)  # green for ground truth
MISS_COLOR = (0, 0, 255)  # red for missed GT


def load_gt_labels(label_path, img_w, img_h):
    """Load YOLO ground truth labels and convert to pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        cx, cy, w, h = [float(x) for x in parts[1:5]]
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append({"class_id": cls_id, "bbox": [x1, y1, x2, y2]})
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def match_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Match predictions to ground truth, return TP/FP/FN."""
    matched_gt = set()
    tp, fp = [], []

    # Sort predictions by confidence (highest first)
    preds_sorted = sorted(pred_boxes, key=lambda x: x["confidence"], reverse=True)

    for pred in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            gt_cls = gt_boxes[best_gt_idx]["class_id"]
            if gt_cls == pred["class_id"]:
                tp.append(pred)
                matched_gt.add(best_gt_idx)
            else:
                fp.append(pred)  # wrong class
        else:
            fp.append(pred)  # no matching GT

    fn = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
    return tp, fp, fn


def draw_annotated(img, gt_boxes, pred_boxes, tp, fp, fn):
    """Draw ground truth and predictions on image."""
    result = img.copy()
    h, w = result.shape[:2]

    # Draw FN (missed ground truth) in red dashed
    for gt in fn:
        x1, y1, x2, y2 = gt["bbox"]
        cv2.rectangle(result, (x1, y1), (x2, y2), MISS_COLOR, 2)
        label = f"MISS: {CLASS_NAMES[gt['class_id']]}"
        cv2.putText(result, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, MISS_COLOR, 1)

    # Draw TP (correct predictions) in green
    for pred in tp:
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        color = (0, 220, 0)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{pred['class_name']} {pred['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Draw FP (false positives) in orange
    for pred in fp:
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        color = (0, 165, 255)  # orange
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"FP: {pred['class_name']} {pred['confidence']:.0%}"
        cv2.putText(result, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return result


def draw_gt_only(img, gt_boxes):
    """Draw only ground truth boxes."""
    result = img.copy()
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt["bbox"]
        cv2.rectangle(result, (x1, y1), (x2, y2), GT_COLOR, 2)
        label = CLASS_NAMES[gt["class_id"]]
        cv2.putText(result, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)
    return result


def draw_pred_only(img, pred_boxes):
    """Draw only prediction boxes."""
    result = img.copy()
    for pred in pred_boxes:
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        cls_id = pred["class_id"]
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        label = f"{pred['class_name']} {pred['confidence']:.0%}"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return result


def create_side_by_side(img, gt_boxes, pred_boxes, filename):
    """Create GT vs Prediction side-by-side comparison."""
    gt_img = draw_gt_only(img, gt_boxes)
    pred_img = draw_pred_only(img, pred_boxes)

    h, w = img.shape[:2]
    # Add labels
    cv2.putText(gt_img, "GROUND TRUTH", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_COLOR, 2)
    cv2.putText(pred_img, "PREDICTIONS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    combined = np.hstack([gt_img, pred_img])
    # Add filename at bottom
    cv2.putText(combined, filename, (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return combined


def create_mosaic(images, cols=4, cell_size=(320, 240)):
    """Create a mosaic grid of images."""
    if not images:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)

    rows = (len(images) + cols - 1) // cols
    mosaic = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, cell_size)
        mosaic[r * cell_size[1]:(r + 1) * cell_size[1],
               c * cell_size[0]:(c + 1) * cell_size[0]] = resized

    return mosaic


def main():
    parser = argparse.ArgumentParser(description="Visual model testing")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--test-dir", type=str, default=str(TEST_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--num", type=int, default=100,
                        help="Max images to process")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotated").mkdir(exist_ok=True)
    (output / "side_by_side").mkdir(exist_ok=True)
    (output / "per_class").mkdir(exist_ok=True)

    print("=" * 60)
    print("  Military Vehicle Detection — Visual Test Suite")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Test dir: {args.test_dir}")
    print(f"  Output: {args.output}")
    print(f"  Conf threshold: {args.conf}")

    # Load detector
    detector = VehicleDetector(args.model, conf_threshold=args.conf)

    test_dir = Path(args.test_dir)
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    images = sorted(img_dir.glob("*.*"))[:args.num]
    print(f"  Images: {len(images)}")
    print()

    # Per-class stats
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_conf = defaultdict(list)
    latencies = []

    # Collect best/worst for mosaic
    best_detections = []  # (score, img)
    worst_misses = []  # (miss_count, img)
    per_class_examples = defaultdict(list)  # class_id -> [(img, ...)]

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        # Load ground truth
        gt_boxes = load_gt_labels(lbl_path, w, h)

        # Run detection
        preds = detector.detect(img)
        latencies.append(detector.latency_ms)

        # Match
        tp, fp, fn = match_detections(gt_boxes, preds)

        # Update stats
        for p in tp:
            class_tp[p["class_id"]] += 1
            class_conf[p["class_id"]].append(p["confidence"])
        for p in fp:
            class_fp[p["class_id"]] += 1
        for g in fn:
            class_fn[g["class_id"]] += 1

        # Save annotated image
        annotated = draw_annotated(img, gt_boxes, preds, tp, fp, fn)
        cv2.imwrite(str(output / "annotated" / img_path.name), annotated)

        # Save side-by-side (for images with GT)
        if gt_boxes:
            sbs = create_side_by_side(img, gt_boxes, preds, img_path.name)
            cv2.imwrite(str(output / "side_by_side" / img_path.name), sbs)

        # Collect for mosaics
        if tp:
            avg_conf = sum(p["confidence"] for p in tp) / len(tp)
            best_detections.append((avg_conf, annotated.copy(), img_path.name))
        if fn:
            worst_misses.append((len(fn), annotated.copy(), img_path.name))

        # Per-class examples
        for p in tp:
            if len(per_class_examples[p["class_id"]]) < 8:
                crop = annotated.copy()
                per_class_examples[p["class_id"]].append(crop)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(images)}...")

    # ── Create mosaics ──────────────────────────────────────

    # Best detections mosaic (top 16 by confidence)
    best_detections.sort(key=lambda x: x[0], reverse=True)
    best_imgs = [x[1] for x in best_detections[:16]]
    if best_imgs:
        mosaic = create_mosaic(best_imgs, cols=4)
        cv2.imwrite(str(output / "mosaic_best_detections.jpg"), mosaic)

    # Worst misses mosaic (top 16 by miss count)
    worst_misses.sort(key=lambda x: x[0], reverse=True)
    worst_imgs = [x[1] for x in worst_misses[:16]]
    if worst_imgs:
        mosaic = create_mosaic(worst_imgs, cols=4)
        cv2.imwrite(str(output / "mosaic_worst_misses.jpg"), mosaic)

    # Per-class example mosaics
    for cls_id, imgs in per_class_examples.items():
        if imgs:
            mosaic = create_mosaic(imgs, cols=4)
            cv2.imwrite(str(output / "per_class" / f"{CLASS_NAMES[cls_id]}_examples.jpg"),
                        mosaic)

    # ── Print results ────────────────────────────────────────

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Inference: {np.median(latencies):.1f}ms median, "
          f"{np.mean(latencies):.1f}ms mean, "
          f"{1000/np.median(latencies):.0f} FPS")

    print(f"\n  {'Class':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} "
          f"{'Recall':>8} {'Avg Conf':>9}")
    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} {'─'*10} {'─'*8} {'─'*9}")

    total_tp = total_fp = total_fn = 0
    results_data = {}

    for i, name in enumerate(CLASS_NAMES):
        tp_c = class_tp.get(i, 0)
        fp_c = class_fp.get(i, 0)
        fn_c = class_fn.get(i, 0)
        total_tp += tp_c
        total_fp += fp_c
        total_fn += fn_c

        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        avg_c = np.mean(class_conf[i]) if class_conf[i] else 0

        status = "" if (tp_c + fn_c) > 0 else "(no data)"
        print(f"  {name:<20} {tp_c:>5} {fp_c:>5} {fn_c:>5} {prec:>10.3f} "
              f"{rec:>8.3f} {avg_c:>9.3f} {status}")

        results_data[name] = {
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "avg_confidence": round(avg_c, 4),
        }

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec + 1e-6)

    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} {'─'*10} {'─'*8}")
    print(f"  {'TOTAL':<20} {total_tp:>5} {total_fp:>5} {total_fn:>5} "
          f"{overall_prec:>10.3f} {overall_rec:>8.3f}")
    print(f"\n  F1 Score: {f1:.3f}")

    # Save JSON results
    report = {
        "model": args.model,
        "test_images": len(images),
        "conf_threshold": args.conf,
        "median_latency_ms": round(np.median(latencies), 1),
        "fps": round(1000 / np.median(latencies), 1),
        "overall": {
            "precision": round(overall_prec, 4),
            "recall": round(overall_rec, 4),
            "f1": round(f1, 4),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        },
        "per_class": results_data,
    }
    with open(output / "test_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Output saved to: {output}")
    print(f"  - annotated/        ({len(images)} images with boxes)")
    print(f"  - side_by_side/     (GT vs predictions)")
    print(f"  - per_class/        (examples per class)")
    print(f"  - mosaic_best_detections.jpg")
    print(f"  - mosaic_worst_misses.jpg")
    print(f"  - test_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
