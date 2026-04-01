# Military Vehicle Detection Model — Training Report

**Date:** 2026-03-29
**Model:** YOLOv8n (Nano)
**Task:** 7-class military vehicle detection from aerial/drone imagery
**Final mAP@0.5:** 0.652 | **Inference:** 10ms / 99 FPS on Apple Silicon

---

## 1. Model Architecture

### YOLOv8n (Nano) — Selected for Edge Deployment

| Property | Value |
|----------|-------|
| Architecture | YOLOv8 Nano (Ultralytics v8.4.30) |
| Parameters | 3.2M |
| GFLOPs | ~8.7 |
| Input size | 640 x 640 |
| Backbone | CSPDarknet (modified) |
| Neck | PANet + C2f blocks |
| Head | Decoupled anchor-free |
| Base weights | COCO pretrained (`yolov8n.pt`) |
| Output | 7 classes, anchor-free detection |

**Why YOLOv8n:**
- Smallest YOLO variant with competitive accuracy
- 11.7 MB ONNX — fits all edge targets (Jetson, Hailo-8L, NXP i.MX8M)
- Already proven in the project's tailsitter ONNX pipeline
- Ultralytics provides one-command export to ONNX, TensorRT, TFLite

---

## 2. Dataset

### 2.1 Sources

| Source | Images | Type | License |
|--------|--------|------|---------|
| **Roboflow: Russian-military-annotated** | 993 | Real drone/aerial — BM-21, BMD-2, BMP-1/2, BTR-70/80, MT-LB, T-64/72/80 | CC BY 4.0 |
| **Roboflow: Military Vehicle Recognition** | 1,320 | Ukraine war recon drone — tanks, APCs, aircraft, soldiers | CC BY 4.0 |
| **Roboflow: RussianTankDroneImagesLowQuality** | 448 | Real FPV drone footage — BMP-1/2/3, MT-LB, T-72 | CC BY 4.0 |
| **Roboflow: Military Vehicle Detection** | 624 | Aerial — persons, trenches, vehicles | MIT |
| **HuggingFace: Drone Detection Dataset** | 1,500 | Ground-to-air UAV/drone detection | MIT |
| **Total** | **4,039** (after filtering to target classes) | | |

### 2.2 Class Mapping

Source dataset classes were remapped to 7 unified target classes:

| Target Class | ID | Source Mappings | Train Annot. | Val | Test |
|---|---|---|---|---|---|
| **tank** | 0 | T-72, T-64, T-80, "tank" | 814 | 196 | 191 |
| **apc_ifv** | 1 | BMP-1/2/3, BMD-2, BTR-70/80, MT-LB, APC | 907 | 185 | 168 |
| **military_truck** | 2 | "Vehicle", "undefined vehicle" | 481 | 136 | 126 |
| **sp_artillery** | 3 | *(no real data — synthetic only, not yet generated)* | 0 | 0 | 0 |
| **mlrs** | 4 | BM-21 Grad | 117 | 18 | 20 |
| **helicopter** | 5 | *(no data in current sources)* | 0 | 0 | 0 |
| **uav** | 6 | Drone detection dataset | 1,083 | 221 | 224 |
| **Total** | | | **3,402** | **756** | **729** |

### 2.3 Split

| Split | Images | Annotations | Purpose |
|-------|--------|-------------|---------|
| Train | 2,827 | 3,402 | Model training |
| Val | 605 | 756 | Epoch validation, early stopping |
| Test | 607 | 729 | Final evaluation |

**Split ratio:** 70% / 15% / 15% (shuffled, random seed 42)

### 2.4 Class Distribution Observations

- **Well-represented:** tank (814), apc_ifv (907), uav (1,083)
- **Moderate:** military_truck (481), mlrs (117)
- **Missing:** sp_artillery (0), helicopter (0) — zero training samples
- The model cannot detect sp_artillery or helicopter until data is added

---

## 3. Training Strategy

### Two-Phase Transfer Learning

The model was trained using a two-phase approach, starting from COCO-pretrained weights:

### Phase 1: Frozen Backbone Warmup

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Frozen layers | First 10 (entire backbone) |
| Optimizer | AdamW (auto-selected) |
| Learning rate | 0.001 |
| Batch size | 16 |
| Device | Apple MPS (Metal) |
| Duration | 5.3 hours |

**Purpose:** Allow the detection head to learn new class representations without disrupting pretrained backbone features.

**Phase 1 Results:**

| Epoch | box_loss | cls_loss | Precision | Recall | mAP50 | mAP50-95 |
|-------|----------|----------|-----------|--------|-------|----------|
| 1 | 1.697 | 4.051 | 0.451 | 0.244 | 0.237 | 0.090 |
| 5 | 1.472 | 2.140 | 0.552 | 0.311 | 0.324 | 0.152 |
| 7 | 1.414 | 1.915 | 0.810 | 0.388 | **0.518** | 0.223 |
| 10 | 1.338 | 1.723 | 0.660 | 0.440 | **0.543** | 0.244 |

### Phase 2: Full Fine-Tune

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Frozen layers | None (all unfrozen) |
| Optimizer | SGD |
| Learning rate | 0.01 → 0.0001 (cosine decay, lrf=0.01) |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup | 3 epochs (bias LR 0.1, momentum 0.8) |
| Batch size | 16 |
| Device | Apple MPS (Metal) |
| Duration | 10.4 hours |

### Augmentation (Aerial-Optimized)

| Augmentation | Value | Rationale |
|---|---|---|
| `degrees` | 180.0 | Full rotation — aerial views have no canonical orientation |
| `flipud` | 0.5 | Vertical flip common for overhead |
| `fliplr` | 0.5 | Horizontal flip |
| `mosaic` | 1.0 | Combine 4 images — more objects per batch |
| `mixup` | 0.1 | Blend images for regularization |
| `scale` | 0.5 | Simulate altitude variation |
| `hsv_h/s/v` | 0.015/0.7/0.4 | Color jitter for lighting variation |
| Close mosaic | Last 10 epochs | Standard practice for final convergence |

---

## 4. Training Results

### 4.1 Loss Curves (Phase 2)

*(See `ml/models/mil_vehicle_v1/results.png`)*

All three losses decreased consistently across 50 epochs:
- **Box loss:** 1.73 → 1.18 (32% reduction)
- **Class loss:** 2.46 → 1.23 (50% reduction)
- **DFL loss:** 1.86 → 1.50 (19% reduction)

Validation losses also decreased steadily, with no signs of overfitting:
- **Val box:** 2.13 → 1.50
- **Val cls:** 2.23 → 1.39
- **Val dfl:** 2.10 → 1.60

### 4.2 Metric Progression (Phase 2)

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1 | 0.575 | 0.359 | 0.379 | 0.130 |
| 10 | 0.236 | 0.363 | 0.287 | 0.101 |
| 20 | 0.697 | 0.399 | 0.471 | 0.232 |
| 30 | 0.466 | 0.408 | 0.428 | 0.225 |
| 35 | 0.551 | 0.562 | 0.583 | 0.306 |
| 40 | 0.699 | 0.462 | 0.546 | 0.308 |
| 45 | 0.559 | 0.587 | 0.627 | 0.356 |
| **50** | **0.624** | **0.614** | **0.652** | **0.396** |

Note: mAP50 dipped in epochs 1-10 of Phase 2 as expected — the backbone was unfrozen with a 10x higher LR, causing initial instability that resolved by epoch 15.

### 4.3 Per-Class Performance (Final)

*(From Precision-Recall curve — `ml/models/mil_vehicle_v1/BoxPR_curve.png`)*

| Class | mAP@0.5 | Performance |
|-------|---------|-------------|
| **uav** | **0.932** | Excellent — largest dataset, distinct appearance |
| **mlrs** | **0.778** | Very good — BM-21 is visually distinct (launcher rack) |
| **apc_ifv** | **0.667** | Good — well-represented, multiple sub-types |
| **tank** | **0.565** | Moderate — confused with APC sometimes |
| **military_truck** | **0.316** | Weak — generic "vehicle" label in source data |
| **sp_artillery** | N/A | No data — cannot detect |
| **helicopter** | N/A | No data — cannot detect |

### 4.4 Confusion Matrix Analysis

*(From `ml/models/mil_vehicle_v1/confusion_matrix_normalized.png`)*

Key observations:
- **UAV:** 90% correct classification — best performing class
- **MLRS:** 78% correct — strong despite only 117 training samples
- **APC/IFV:** 57% correct, 11% confused as tank — expected overlap (similar silhouettes)
- **Tank:** 45% correct, 9% confused as APC — needs more training data
- **Military truck:** 16% correct, 66% missed as background — weak "Vehicle" labels from source
- **Background false positives:** UAV has 26% background FP rate (detects drones that aren't there)

---

## 5. Exported Model

| Format | File | Size | Notes |
|--------|------|------|-------|
| PyTorch | `best.pt` | ~6.2 MB | Training checkpoint |
| **ONNX** | **`best.onnx`** | **11.7 MB** | Primary deployment format |

ONNX export configuration:
- Opset version: 12
- Simplified with onnxslim
- Static input shape: [1, 3, 640, 640]
- Output shape: [1, 11, 8400] (4 box + 7 class scores, 8400 anchors)

---

## 6. Inference Benchmarks

### Apple Silicon (Development Machine)

| Metric | Measured | Target | Status |
|--------|----------|--------|--------|
| **Median latency** | **10.0 ms** | < 40 ms | **PASS** (4x margin) |
| **FPS** | **99.4** | >= 25 | **PASS** (4x margin) |
| **Model size** | **11.7 MB** | < 15 MB | **PASS** |

Runtime: ONNX Runtime with CoreML Execution Provider
Preprocessing: letterbox 640x640, BGR→RGB, normalize [0,1]
Postprocessing: confidence threshold 0.35, NMS IoU 0.45

### Projected Edge Performance

| Platform | Precision | Expected FPS | Expected Latency |
|----------|-----------|--------------|------------------|
| **Jetson Orin Nano** | FP16 TensorRT | 30-50 | 20-33 ms |
| **RPi 5 + Hailo-8L** | INT8 HEF | 25-40 | 25-40 ms |
| **NXP i.MX8M Plus** | INT8 TFLite | 8-15 | 70-120 ms |

---

## 7. Test Suite Results

**56/56 tests PASS** across 7 sections:

| Section | Tests | Status |
|---------|-------|--------|
| Config (YAML, classes, gitignore) | 9/9 | PASS |
| Gazebo Models (6 vehicles, XML, dimensions) | 8/8 | PASS |
| World (19 vehicles, formations, GPS) | 7/7 | PASS |
| Detector (preprocessing, NMS, postprocess) | 12/12 | PASS |
| Pipeline (7 scripts, compilation, features) | 8/8 | PASS |
| Integration (camera_bridge, app.py, frontend) | 6/6 | PASS |
| KPI (model size, latency, FPS, classes, format) | 6/6 | PASS |

---

## 8. Known Limitations & Next Steps

### Limitations
1. **sp_artillery (class 3):** Zero training data — model cannot detect self-propelled artillery
2. **helicopter (class 5):** Zero training data — model cannot detect helicopters
3. **military_truck (class 2):** Low mAP (0.316) due to generic "Vehicle" labels in source data
4. **Tank-APC confusion:** 11% of APCs misclassified as tanks and vice versa
5. **Training data mostly ground-level:** Drone detection dataset is ground-to-air, not aerial overhead

### Recommended Next Steps

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| **HIGH** | Add MV-RSD dataset (3K satellite images, being downloaded) | +15-20% mAP for tank/APC |
| **HIGH** | Generate Gazebo synthetic data for sp_artillery & helicopter | Enable 2 missing classes |
| **MEDIUM** | Add xView dataset (register at xviewdataset.org) | +tanks, helicopters, trucks |
| **MEDIUM** | Increase epochs to 150 (current: 50) | +5-10% mAP convergence |
| **LOW** | Retrain military_truck with better-labeled data | Fix 0.316 mAP |
| **LOW** | Add ByteTrack for persistent vehicle IDs | Tracking, not detection |

### Estimated v2 Performance (with MV-RSD + synthetic + 150 epochs)

| Metric | Current v1 | Projected v2 |
|--------|-----------|-------------|
| mAP50 | 0.652 | 0.75-0.80 |
| Classes active | 5/7 | 7/7 |
| Training images | 4,039 | ~10,000 |

---

## 9. Reproducibility

```bash
# Full pipeline from scratch
cd ardupilot-sim

# 1. Download datasets
source ml/venv_ml/bin/activate
python ml/scripts/download_open_datasets.py --datasets drone
python ml/scripts/merge_datasets.py

# 2. Train (Phase 1 + Phase 2 + ONNX export)
python ml/scripts/train.py --phase both

# 3. Run test suite
python scripts/test_ml_pipeline.py

# 4. Run with detection
./scripts/start_camera_detect.sh ml/models/mil_vehicle_v1/weights/best.onnx
```

### Environment
- **Hardware:** Apple Silicon (M-series), MPS backend
- **Python:** 3.11.14
- **PyTorch:** 2.11.0
- **Ultralytics:** 8.4.30
- **ONNX Runtime:** 1.24.4
- **Training time:** ~15.7 hours total (Phase 1: 5.3h, Phase 2: 10.4h)
