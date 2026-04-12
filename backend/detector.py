#!/usr/bin/env python3
"""ONNX-based military vehicle detector for drone camera frames.

Runs YOLOv8n inference on camera frames and returns bounding box detections.
Designed to be called from camera_bridge.py on each captured frame.

Usage:
    detector = VehicleDetector("path/to/model.onnx")
    detections = detector.detect(frame_bgr)
    # detections = [{"class_id": 0, "class_name": "tank", "confidence": 0.87,
    #                "bbox": [x1, y1, x2, y2]}]
"""

import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# 7 military vehicle classes matching ml/configs/dataset.yaml
CLASS_NAMES = [
    "tank",
    "apc_ifv",
    "military_truck",
    "sp_artillery",
    "mlrs",
    "helicopter",
    "uav",
]

# Display colors per class (BGR)
CLASS_COLORS = [
    (0, 0, 255),      # tank — red
    (0, 140, 255),     # apc_ifv — orange
    (0, 200, 200),     # military_truck — yellow
    (80, 80, 255),     # sp_artillery — light red
    (255, 0, 128),     # mlrs — magenta
    (255, 200, 0),     # helicopter — cyan
    (200, 200, 0),     # uav — teal
]


class VehicleDetector:
    """YOLOv8n ONNX inference wrapper for military vehicle detection."""

    def __init__(self, model_path, conf_threshold=0.35, iou_threshold=0.45,
                 input_size=640):
        """Initialize detector with ONNX model.

        Args:
            model_path: Path to .onnx model file.
            conf_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            input_size: Model input size (square).
        """
        if ort is None:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.num_classes = len(CLASS_NAMES)

        # Create ONNX Runtime session
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[2] and input_shape[3]:
            self.input_size = input_shape[2]

        self._last_latency_ms = 0.0

    @property
    def latency_ms(self):
        """Last inference latency in milliseconds."""
        return self._last_latency_ms

    def preprocess(self, frame):
        """Preprocess BGR frame for YOLOv8 input.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            blob: (1, 3, input_size, input_size) float32 tensor.
            scale: (scale_x, scale_y) for mapping back to original coords.
            pad: (pad_x, pad_y) letterbox padding.
        """
        h, w = frame.shape[:2]
        size = self.input_size

        # Compute scale to fit within input_size (letterbox)
        scale = min(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2

        # Resize and pad
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)  # Add batch dimension

        return blob, (scale, scale), (pad_x, pad_y)

    def postprocess(self, output, scale, pad, orig_shape):
        """Post-process YOLOv8 output to detections.

        YOLOv8 output shape: (1, num_classes + 4, num_detections)
        Where first 4 values are cx, cy, w, h.

        Args:
            output: Raw model output array.
            scale: (scale_x, scale_y) from preprocessing.
            pad: (pad_x, pad_y) from preprocessing.
            orig_shape: (orig_h, orig_w) of input frame.

        Returns:
            List of detection dicts.
        """
        # Output shape: (1, 4 + num_classes, N) — transpose to (N, 4 + num_classes)
        predictions = output[0].transpose()

        # Filter by confidence
        class_scores = predictions[:, 4:]
        max_scores = class_scores.max(axis=1)
        mask = max_scores > self.conf_threshold
        predictions = predictions[mask]
        max_scores = max_scores[mask]
        class_ids = class_scores[mask].argmax(axis=1)

        if len(predictions) == 0:
            return []

        # Extract boxes (cx, cy, w, h) — model outputs normalized coords (0-1)
        cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        # Scale from normalized to input_size pixel space
        cx, cy, w, h = cx * self.input_size, cy * self.input_size, w * self.input_size, h * self.input_size
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Remove padding and scale back to original image coords
        pad_x, pad_y = pad
        scale_x, scale_y = scale
        x1 = (x1 - pad_x) / scale_x
        y1 = (y1 - pad_y) / scale_y
        x2 = (x2 - pad_x) / scale_x
        y2 = (y2 - pad_y) / scale_y

        # Clip to image bounds
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        indices = self._nms(boxes, max_scores, self.iou_threshold)

        detections = []
        for i in indices:
            detections.append({
                "class_id": int(class_ids[i]),
                "class_name": CLASS_NAMES[class_ids[i]],
                "confidence": float(max_scores[i]),
                "bbox": [
                    round(float(x1[i]), 1),
                    round(float(y1[i]), 1),
                    round(float(x2[i]), 1),
                    round(float(y2[i]), 1),
                ],
            })

        return detections

    def detect(self, frame):
        """Run detection on a BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            List of detection dicts with keys:
                class_id, class_name, confidence, bbox [x1, y1, x2, y2]
        """
        t0 = time.perf_counter()

        blob, scale, pad = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})
        detections = self.postprocess(
            outputs[0], scale, pad, frame.shape[:2]
        )

        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        return detections

    def detect_sahi(self, frame, slice_size=320, overlap=0.3):
        """Run sliced inference (SAHI) for improved small-object recall.

        Tiles the frame into overlapping slices, runs detect() on each, then
        merges all predictions back to original frame coordinates with NMS.

        Args:
            frame: BGR image (H, W, 3) uint8.
            slice_size: Side length of each square tile in pixels.
            overlap: Fractional overlap between adjacent tiles (0–1).

        Returns:
            Same format as detect().
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        step = int(slice_size * (1 - overlap))

        all_boxes, all_scores, all_class_ids = [], [], []

        y = 0
        while y < h:
            x = 0
            while x < w:
                x2 = min(x + slice_size, w)
                y2 = min(y + slice_size, h)
                x1 = max(x2 - slice_size, 0)
                y1 = max(y2 - slice_size, 0)

                tile = frame[y1:y2, x1:x2]
                for d in self.detect(tile):
                    bx1, by1, bx2, by2 = d["bbox"]
                    all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                    all_scores.append(d["confidence"])
                    all_class_ids.append(d["class_id"])

                if x2 == w:
                    break
                x += step
            if y2 == h:
                break
            y += step

        self._last_latency_ms = (time.perf_counter() - t0) * 1000

        if not all_boxes:
            return []

        boxes = np.array(all_boxes, dtype=np.float32)
        scores = np.array(all_scores, dtype=np.float32)
        class_ids = np.array(all_class_ids, dtype=np.int32)

        indices = self._nms(boxes, scores, self.iou_threshold)
        detections = []
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                "class_id": int(class_ids[i]),
                "class_name": CLASS_NAMES[class_ids[i]],
                "confidence": float(scores[i]),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2), 1), round(float(y2), 1)],
            })
        return detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame.

        Args:
            frame: BGR image (modified in-place).
            detections: List of detection dicts from detect().

        Returns:
            frame with drawn detections.
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_id = det["class_id"]
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            label = f"{det['class_name']} {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        """Non-maximum suppression."""
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

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
