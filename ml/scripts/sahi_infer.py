#!/usr/bin/env python3
"""
SAHI (Slicing Aided Hyper Inference) tuning script.

Runs sliced inference over the val set and compares mAP50 vs standard
single-pass inference.  No external sahi package required — pure numpy/cv2.

Usage:
    python3 sahi_infer.py --model best.onnx --dataset dataset_aerial_v3.yaml
    python3 sahi_infer.py --model best.onnx --dataset dataset_aerial_v3.yaml \
        --slice-sizes 256 320 384 --overlaps 0.2 0.3 0.5 --max-images 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add backend to path so we can import VehicleDetector
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))
from detector import VehicleDetector, CLASS_NAMES


# ---------------------------------------------------------------------------
# Slicing helpers
# ---------------------------------------------------------------------------

def make_slices(img_w, img_h, slice_size, overlap):
    """Generate (x1, y1, x2, y2) slice windows covering the full image."""
    step = int(slice_size * (1 - overlap))
    slices = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)
            x1 = max(x2 - slice_size, 0)
            y1 = max(y2 - slice_size, 0)
            slices.append((x1, y1, x2, y2))
            if x2 == img_w:
                break
            x += step
        if y2 == img_h:
            break
        y += step
    return slices


def detect_sahi(detector, frame, slice_size, overlap):
    """Run sliced inference and return merged detections in original coords."""
    h, w = frame.shape[:2]
    all_boxes = []   # [x1, y1, x2, y2]
    all_scores = []
    all_class_ids = []

    for (sx1, sy1, sx2, sy2) in make_slices(w, h, slice_size, overlap):
        tile = frame[sy1:sy2, sx1:sx2]
        dets = detector.detect(tile)
        for d in dets:
            bx1, by1, bx2, by2 = d["bbox"]
            # Map tile-local coords back to full image
            all_boxes.append([bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1])
            all_scores.append(d["confidence"])
            all_class_ids.append(d["class_id"])

    if not all_boxes:
        return []

    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    class_ids = np.array(all_class_ids, dtype=np.int32)

    # Per-class NMS then merge
    keep_indices = _nms(boxes, scores, iou_threshold=detector.iou_threshold)
    detections = []
    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        detections.append({
            "class_id": int(class_ids[i]),
            "class_name": CLASS_NAMES[class_ids[i]],
            "confidence": float(scores[i]),
            "bbox": [round(float(x1), 1), round(float(y1), 1),
                     round(float(x2), 1), round(float(y2), 1)],
        })
    return detections


def _nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_val_pairs(dataset_yaml, max_images=None):
    """Return list of (image_path, label_path) for the val split."""
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["path"])
    img_dir = base / cfg["val"]
    lbl_dir = base / cfg["val"].replace("images", "labels")

    pairs = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))

    if max_images:
        pairs = pairs[:max_images]
    return pairs


def parse_label(lbl_path, img_w, img_h):
    """Parse YOLO label file → list of (class_id, x1, y1, x2, y2)."""
    gts = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            gts.append((cls, x1, y1, x2, y2))
    return gts


def iou_single(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / (areaA + areaB - inter + 1e-6)


def compute_ap(detections_all, gt_all, iou_thresh=0.5):
    """
    Compute per-class AP50 from detection/gt lists.

    detections_all: list of (score, class_id, x1, y1, x2, y2, img_idx)
    gt_all:         list of (class_id, x1, y1, x2, y2, img_idx)
    """
    num_classes = len(CLASS_NAMES)
    aps = []

    for cls in range(num_classes):
        dets = [(s, b, idx) for s, c, *b, idx in
                [(d[0], d[1], *d[2:5], d[6]) for d in detections_all] if c == cls]
        # unpack properly
        dets = [(d[0], d[1], d[2], d[3], d[4], d[5], d[6])
                for d in detections_all if d[1] == cls]
        gts = [(g[1], g[2], g[3], g[4], g[5]) for g in gt_all if g[0] == cls]

        # Group GTs by image
        gt_by_img = {}
        for g in gt_all:
            if g[0] != cls:
                continue
            img_idx = g[5]
            gt_by_img.setdefault(img_idx, []).append(list(g[1:5]))

        n_gt = sum(len(v) for v in gt_by_img.values())
        if n_gt == 0:
            continue

        # Sort dets by descending score
        dets_sorted = sorted(dets, key=lambda d: -d[0])
        matched = {img_idx: [False] * len(boxes)
                   for img_idx, boxes in gt_by_img.items()}

        tp = np.zeros(len(dets_sorted))
        fp = np.zeros(len(dets_sorted))

        for di, det in enumerate(dets_sorted):
            score, cls_id, x1, y1, x2, y2, img_idx = det
            best_iou = 0
            best_j = -1
            for j, gt_box in enumerate(gt_by_img.get(img_idx, [])):
                iou = iou_single((x1, y1, x2, y2), gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and not matched[img_idx][best_j]:
                tp[di] = 1
                matched[img_idx][best_j] = True
            else:
                fp[di] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / (n_gt + 1e-6)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)

        # AP via 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t].max() if (recall >= t).any() else 0.0
            ap += p / 11
        aps.append((CLASS_NAMES[cls], ap, n_gt))

    return aps


def evaluate(detector, pairs, detect_fn, label="standard"):
    """Run detection over pairs and return mAP50."""
    all_dets = []
    all_gts = []

    t0 = time.perf_counter()
    for img_idx, (img_path, lbl_path) in enumerate(pairs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        dets = detect_fn(frame)
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            all_dets.append((d["confidence"], d["class_id"],
                             x1, y1, x2, y2, img_idx))

        for cls, gx1, gy1, gx2, gy2 in parse_label(lbl_path, w, h):
            all_gts.append((cls, gx1, gy1, gx2, gy2, img_idx))

    elapsed = time.perf_counter() - t0
    aps = compute_ap(all_dets, all_gts)
    map50 = np.mean([ap for _, ap, _ in aps]) if aps else 0.0

    fps = len(pairs) / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Images: {len(pairs)}  |  Time: {elapsed:.1f}s  |  FPS: {fps:.1f}")
    print(f"  mAP50: {map50:.4f}")
    print(f"  {'Class':<18} {'AP50':>8} {'#GT':>6}")
    print(f"  {'-'*36}")
    for cls_name, ap, n_gt in sorted(aps, key=lambda x: -x[1]):
        print(f"  {cls_name:<18} {ap:>8.4f} {n_gt:>6}")
    print(f"{'='*60}")

    return map50, fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--slice-sizes", type=int, nargs="+", default=[320])
    p.add_argument("--overlaps", type=float, nargs="+", default=[0.3])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-images", type=int, default=None,
                   help="Limit val images for quick sweeps")
    p.add_argument("--save-json", type=Path, default=None,
                   help="Save results as JSON for later comparison")
    return p.parse_args()


def main():
    args = parse_args()

    detector = VehicleDetector(
        str(args.model),
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    pairs = load_val_pairs(str(args.dataset), max_images=args.max_images)
    print(f"Val images: {len(pairs)}")

    results = {}

    # Baseline: standard single-pass
    map50_base, fps_base = evaluate(
        detector, pairs,
        detect_fn=detector.detect,
        label="BASELINE (single-pass 640px)",
    )
    results["baseline"] = {"map50": map50_base, "fps": fps_base}

    # SAHI sweeps
    best_map = map50_base
    best_cfg = None
    for ss in args.slice_sizes:
        for ov in args.overlaps:
            n_tiles = len(make_slices(640, 640, ss, ov))
            label = f"SAHI slice={ss} overlap={ov:.0%} (~{n_tiles} tiles/frame)"
            map50, fps = evaluate(
                detector, pairs,
                detect_fn=lambda frame, s=ss, o=ov: detect_sahi(detector, frame, s, o),
                label=label,
            )
            key = f"sahi_{ss}_{int(ov*100)}"
            results[key] = {"map50": map50, "fps": fps,
                            "slice_size": ss, "overlap": ov, "n_tiles": n_tiles}
            if map50 > best_map:
                best_map = map50
                best_cfg = (ss, ov)

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  Baseline mAP50: {map50_base:.4f}  ({fps_base:.1f} FPS)")
    if best_cfg:
        ss, ov = best_cfg
        k = f"sahi_{ss}_{int(ov*100)}"
        gain = (best_map - map50_base) * 100
        print(f"  Best SAHI:      {best_map:.4f}  slice={ss} overlap={ov:.0%}  "
              f"(+{gain:.1f}pp)  {results[k]['fps']:.1f} FPS")
    else:
        print("  SAHI did not improve over baseline.")
    print("="*60)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {args.save_json}")


if __name__ == "__main__":
    main()
