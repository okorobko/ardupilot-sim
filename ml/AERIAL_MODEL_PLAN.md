# Aerial Military Vehicle Detection Model — Training Plan

## Problem Statement

The current unified model (v1) achieves 0.652 mAP50 overall but performs poorly on aerial/drone imagery:
- **Ground-level images**: objects fill 10-80% of frame — model works well (80% precision)
- **Aerial/drone images**: objects are 0.5-8% of frame — model struggles (low recall, many misses)

These are fundamentally different visual domains requiring separate models.

---

## Two-Model Architecture

```
Drone Camera Feed (640x480)
        │
        ├── altitude > 5m ──→ AERIAL MODEL (overhead, small objects)
        │                      YOLOv8s + SAHI slicing
        │
        └── altitude ≤ 5m ──→ GROUND MODEL (close-up, large objects)
                               YOLOv8n (current v1 model)
```

The autopilot provides altitude via MAVLink telemetry — `camera_bridge.py` already has access to this through the backend. The model switch is a simple altitude threshold.

---

## Current Aerial Dataset Analysis

| Class | Annotations | % of Total | Quality |
|-------|-------------|------------|---------|
| uav | 1,273 | 48% | Good — clear drone silhouettes |
| military_truck | 690 | 26% | Weak — generic "Vehicle" labels |
| apc_ifv | 367 | 14% | Moderate — small blurry blobs |
| tank | 311 | 12% | Moderate — needs more overhead data |
| mlrs | 4 | 0.2% | Critical gap |
| sp_artillery | 0 | 0% | Critical gap |
| helicopter | 0 | 0% | Critical gap |

**Key problems:**
1. Most "aerial" tank/APC images are actually low-quality drone footage from Ukraine — thermal, blurry, IR
2. Almost no true top-down satellite/overhead imagery of military vehicles
3. MLRS has only 4 aerial annotations (all ground-level BM-21 photos)
4. Object sizes are tiny (0.5-5% of frame) — standard YOLO misses them

---

## Recommended Model: YOLOv8s + SAHI

### Why YOLOv8s (Small) instead of YOLOv8n (Nano)

| Property | YOLOv8n (current) | YOLOv8s (recommended) |
|----------|-------------------|-----------------------|
| Parameters | 3.2M | **11.2M** |
| GFLOPs | 8.7 | **28.6** |
| ONNX size | 11.7 MB | **~23 MB** |
| COCO mAP50 | 37.3 | **44.9** |
| Small object mAP | Poor | **Significantly better** |

YOLOv8s has 3.5x more parameters, giving it much better feature extraction for small objects. The extra cost is acceptable because:
- Jetson Orin Nano runs YOLOv8s at ~20 FPS FP16 (still real-time)
- Aerial mode is used at altitude where update rate matters less
- SAHI slicing already reduces per-slice resolution

### Why SAHI (Slicing Aided Hyper Inference)

Standard YOLO at 640x640 input with a tank at 50m altitude:
- Tank appears as ~25x15 pixels — **too small** for reliable detection

**SAHI solution:** Slice the image into overlapping patches, run detection on each, merge results:

```
Original 640x480 image
┌─────────────────────────────┐
│  ┌──────────┬──────────┐    │
│  │ Slice 1  │ Slice 2  │    │   Each slice: 320x320
│  │ 320x320  │ 320x320  │    │   Overlap: 20%
│  ├──────────┼──────────┤    │   Effective resolution: 2x
│  │ Slice 3  │ Slice 4  │    │
│  │ 320x320  │ 320x320  │    │   Objects now fill 4x more
│  └──────────┴──────────┘    │   of each patch
└─────────────────────────────┘
```

SAHI typically improves small object mAP by **15-30%** with a 4x inference cost — still viable at 5 FPS aerial camera rate.

---

## Training Plan

### Phase 0: Dataset Augmentation (Critical)

The aerial dataset needs significant expansion before training.

#### A. MV-RSD Dataset (being downloaded)
- 3,000 satellite images at 640x640, 0.3m GSD
- 32,626 military vehicle annotations
- Google Earth overhead view — exactly what we need
- **Expected impact:** +1500 aerial images for tank/APC/truck

#### B. Gazebo Synthetic Data
- Run `generate_synthetic.py` with military_training.sdf world
- Camera at altitudes: 15, 25, 35, 50, 75, 100m
- Each altitude produces different object scales
- Domain randomization: sun angle, ground texture, noise
- **Target:** 2,000+ synthetic aerial images
- **Critical for:** sp_artillery (0 data), helicopter (0 data), mlrs (4 aerial)

#### C. xView Dataset (registration needed)
- WorldView-3 satellite imagery, 0.3m resolution
- Classes: tanks (73), fighting vehicles (74), trucks (71/72/76/77), helicopters (15)
- **Expected impact:** +2000 overhead military vehicle annotations

#### D. Tile Augmentation from Existing Data
- Take existing ground-level images of tanks/APCs
- Synthetically shrink + paste onto aerial background images
- Simulates what these vehicles look like from above
- Cheap way to generate aerial training data from ground photos

### Phase 1: Preprocessing

```
Target aerial dataset after augmentation:

| Class            | Current | + MV-RSD | + Synthetic | + xView | Total  |
|------------------|---------|----------|-------------|---------|--------|
| tank             |     311 |     +800 |        +400 |    +500 | ~2,000 |
| apc_ifv          |     367 |     +600 |        +400 |    +300 | ~1,700 |
| military_truck   |     690 |     +400 |        +300 |    +800 | ~2,200 |
| sp_artillery     |       0 |        0 |        +500 |      +0 |   ~500 |
| mlrs             |       4 |        0 |        +300 |      +0 |   ~300 |
| helicopter       |       0 |        0 |        +300 |    +200 |   ~500 |
| uav              |   1,273 |        0 |          +0 |      +0 | ~1,300 |
```

### Phase 2: Model Training

```yaml
# aerial_train_config.yaml
model:
  name: yolov8s            # Small, not Nano — better for small objects
  pretrained: yolov8s.pt   # COCO pretrained
  imgsz: 640

warmup:
  epochs: 15
  lr0: 0.001
  freeze: 10
  batch: 16

training:
  epochs: 200              # More epochs — harder task needs longer training
  lr0: 0.01
  lrf: 0.005               # Lower final LR — fine-grained convergence
  batch: 16
  optimizer: SGD
  momentum: 0.937
  weight_decay: 0.0005

augmentation:
  degrees: 360.0           # Full rotation — overhead has NO orientation
  flipud: 0.5
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.15              # Higher mixup — more regularization needed
  scale: 0.9               # Aggressive scale — simulate altitude changes
  translate: 0.2
  hsv_h: 0.02
  hsv_s: 0.8               # Higher saturation jitter — IR/thermal variation
  hsv_v: 0.5
  copy_paste: 0.3          # Paste small vehicles onto backgrounds
  erasing: 0.5             # Random erasing — robustness to occlusion
```

#### Key Differences from Ground Model

| Parameter | Ground Model | Aerial Model | Why |
|-----------|-------------|--------------|-----|
| Architecture | YOLOv8n | **YOLOv8s** | More capacity for small features |
| Epochs | 50 | **200** | Harder task, more diverse data |
| Scale aug | 0.5 | **0.9** | Simulate 10-100m altitude range |
| Degrees | 180 | **360** | Overhead = truly rotationless |
| Copy-paste | 0.0 | **0.3** | Paste small vehicles on backgrounds |
| Mixup | 0.1 | **0.15** | More regularization |
| LRF | 0.01 | **0.005** | Finer convergence for small objects |

### Phase 3: SAHI Integration

After training the base YOLOv8s model, integrate SAHI:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="aerial_best.pt",
    confidence_threshold=0.3,
)

result = get_sliced_prediction(
    image,
    model,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

### Phase 4: Export & Deploy

| Platform | Format | Expected FPS | Notes |
|----------|--------|-------------|-------|
| Apple Silicon | ONNX FP32 | 40-60 | Without SAHI |
| Apple Silicon | ONNX + SAHI | 10-15 | 4 slices per frame |
| Jetson Orin Nano | TensorRT FP16 | 15-25 | Without SAHI |
| Jetson Orin Nano | TRT + SAHI | 5-8 | Sufficient for surveillance |

---

## Alternative Models Considered

| Model | Params | Small Object mAP | Latency | Verdict |
|-------|--------|------------------|---------|---------|
| YOLOv8n | 3.2M | Poor | 10ms | Too weak for aerial |
| **YOLOv8s** | **11.2M** | **Good** | **20ms** | **Best balance** |
| YOLOv8m | 25.9M | Very good | 45ms | Too slow for edge |
| RT-DETR-l | 32M | Excellent | 60ms | Too heavy, good accuracy |
| YOLOv8s + SAHI | 11.2M | Excellent | 80ms | Best accuracy, OK speed |
| YOLO-World (zero-shot) | 20M+ | Moderate | 100ms+ | No training needed but unreliable |
| DINO v2 + detection head | 86M | Excellent | 200ms+ | Too heavy for edge |

**Recommendation: YOLOv8s + SAHI** — the sweet spot between detection quality for small objects and edge deployability.

---

## Expected Performance Targets

| Metric | Current (unified v1) | Aerial Model Target |
|--------|---------------------|---------------------|
| mAP50 (aerial test) | ~0.35 | **>= 0.55** |
| Tank recall (aerial) | ~0.20 | **>= 0.50** |
| APC recall (aerial) | ~0.25 | **>= 0.50** |
| UAV recall | 0.95 | **>= 0.90** |
| FPS (no SAHI) | 99 | **40+** |
| FPS (with SAHI) | N/A | **8-15** |
| Model size | 11.7 MB | **~23 MB** |

---

## Implementation Steps

```
Step 1: [NOW]    Split dataset → aerial / ground            ✅ DONE (2142 / 1897)
Step 2: [NEXT]   Add MV-RSD satellite dataset               (downloading)
Step 3:          Generate Gazebo synthetic aerial data       (script ready)
Step 4:          Register + download xView                   (manual)
Step 5:          Merge all aerial sources
Step 6:          Train YOLOv8s aerial model (200 epochs)
Step 7:          Integrate SAHI slicing
Step 8:          Export ONNX + benchmark
Step 9:          Add altitude-based model switching
Step 10:         Retrain ground model (v2) on ground subset
Step 11:         End-to-end test in Gazebo simulator
```

---

## File Structure (after implementation)

```
ml/
├── data/
│   ├── aerial/           # Aerial subset (>5m altitude)
│   │   ├── dataset.yaml
│   │   ├── train/val/test/
│   ├── ground/           # Ground subset (<5m altitude)
│   │   ├── dataset.yaml
│   │   ├── train/val/test/
│   └── processed/        # Original merged (kept for reference)
├── models/
│   ├── mil_vehicle_v1/           # Current unified model
│   ├── mil_vehicle_aerial_v1/    # NEW: aerial model (YOLOv8s)
│   └── mil_vehicle_ground_v1/    # NEW: ground model (YOLOv8n retrained)
├── configs/
│   ├── aerial_train_config.yaml  # NEW
│   └── ground_train_config.yaml  # NEW
```

## Detector Architecture (after implementation)

```python
class DualModeDetector:
    def __init__(self, aerial_model, ground_model, altitude_threshold=5.0):
        self.aerial = VehicleDetector(aerial_model)
        self.ground = VehicleDetector(ground_model)
        self.threshold = altitude_threshold

    def detect(self, frame, altitude):
        if altitude > self.threshold:
            return self.aerial.detect_with_sahi(frame)
        else:
            return self.ground.detect(frame)
```
