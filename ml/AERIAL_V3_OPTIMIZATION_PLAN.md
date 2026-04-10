# Aerial Model v3 — Optimization for Low-Quality Drone Imagery

## Context

Aerial model v2 achieves mAP50=0.808 on validation (clean satellite imagery) but only F1=0.665 on real drone footage (thermal, IR, blurry, smoke). Tank recall is 46.7%, truck recall 14.1%. Three classes have zero data (sp_artillery, helicopter, mlrs). Model is 42.7MB — too large for RPi/NXP edge devices.

**Goal:** Improve real-world drone footage performance to F1>0.80, enable all 7 classes, reduce model size to <15MB for edge deployment.

---

## Phase 1: Degraded Image Augmentation Pipeline (HIGHEST IMPACT)

**Create `ml/scripts/augment_degraded.py`** — offline augmentation that generates 2-3 degraded variants per training image.

| Augmentation | Simulates | Implementation |
|---|---|---|
| Grayscale + CLAHE + invert | Thermal/IR camera | OpenCV: cvtColor, CLAHE, 255-img |
| Colormap (INFERNO/JET/HOT) | Thermal display modes | cv2.applyColorMap |
| JPEG compression (q=15-40) | FPV video stream | albumentations.ImageCompression |
| Barrel distortion | FPV fisheye lens | albumentations.OpticalDistortion |
| Motion blur (10-30px) | Drone movement | albumentations.MotionBlur |
| Downscale 25-50% + upscale | Low-res distant objects | albumentations.Downscale |
| Brightness 10-40% | Night/twilight | numpy multiply |
| Smoke/fog overlay | Battlefield conditions | albumentations.RandomFog + custom |
| Text/watermark overlay | Telegram/unit insignia | cv2.putText with alpha blend |
| Poisson noise | Low-light sensor noise | numpy random |

**Apply offline:** For each training image, generate 2 degraded variants → triples effective dataset (6K → 18K images).

**Files:** New `ml/scripts/augment_degraded.py`, modify `ml/scripts/merge_datasets.py`

---

## Phase 2: Missing Class Data (3 classes with 0 data)

### 2a: Fix merge_datasets.py to include helicopter data
The `hf_military_aircraft` dataset is downloaded but NOT merged — the merge script skips it. Add helicopter class mapping.

### 2b: Fix generate_synthetic.py CLASS_MAP
Currently missing sp_artillery, mlrs, helicopter prefixes in `parse_vehicles_from_sdf()`. Add:
```
"sp_art" → class 3, "mlrs"/"grad" → class 4, "helo" → class 5
```

### 2c: Generate Gazebo synthetic data
Run `generate_synthetic.py` on `military_training.sdf` (has 3x SP arty, 2x MLRS, 2x helicopters):
- 500 images for sp_artillery (altitudes 15-75m)
- 400 images for mlrs
- 300 images for helicopter
- Domain randomization: sun angle, camera jitter, ground texture
- Apply degraded augmentation pipeline (Phase 1) to synthetic images

### 2d: Oversample rare classes
Duplicate images containing mlrs 5x, sp_artillery 3x, helicopter 3x during merge to reduce class imbalance.

**Files:** `ml/scripts/merge_datasets.py`, `ml/scripts/generate_synthetic.py`

---

## Phase 3: SAHI Sliced Inference for Small Objects

**Add `detect_sahi()` to `backend/detector.py`:**
- Slice 640x480 input into 320x320 tiles with 25% overlap (~6 tiles)
- Run detection on each tile + full frame (7 inferences total)
- Map tile-local coordinates back to full-frame
- Global NMS across all detections

**Expected:** +15-25% recall for objects <32px. Latency: 25ms × 7 = ~175ms (5.7 FPS) — sufficient for 5fps camera.

**Add `--sahi` flag to `camera_bridge.py`** to enable sliced inference.

**Implement manually** (not using sahi library) — cleaner, no heavy dependency, works with existing ONNX pipeline.

**Files:** `backend/detector.py`, `backend/camera_bridge.py`

---

## Phase 4: Model Architecture — YOLOv8n-p2

**Switch from YOLOv8s (42.7MB) to YOLOv8n-p2 (~10MB):**
- YOLOv8n base: 3.2M params, 11.7MB ONNX
- P2 head addition: stride-4 feature map for tiny objects (+~1M params)
- Result: ~4.2M params, ~10MB ONNX — fits all edge targets
- P2 head specifically helps objects 8-16px which is our sweet spot at altitude

**Why not keep YOLOv8s:** 42.7MB doesn't fit RPi/NXP. With better data (Phase 1-2) and SAHI (Phase 3), YOLOv8n-p2 should close the gap.

**Why not super-resolution/CBAM/distillation:** Data quality is the bottleneck, not model capacity. These add latency without proportional accuracy gains on edge hardware.

**Files:** `ml/configs/train_config.yaml`, `ml/scripts/train.py`

---

## Phase 5: Training

**Dataset after Phase 1-2:** ~18,000-20,000 images (6K original + 12K augmented + 2K synthetic)

**Two-phase training:**
- Phase 1: 15 epochs, frozen backbone, LR 0.001
- Phase 2: 200 epochs, SGD, LR 0.01→0.0001, cosine decay

**Key config changes:**
```yaml
model: yolov8n-p2.yaml
epochs: 200
close_mosaic: 15
label_smoothing: 0.1
mosaic: 0.0  # disabled for MPS (dense annotations bug)
```

**Device:** MPS with mosaic=0.0 (workaround for known MPS bug). Estimated: ~20-30 hours.

**Files:** `ml/configs/train_config.yaml`, `ml/scripts/train.py`

---

## Phase 6: INT8 Quantization

**Create calibration set** at `ml/data/calibration/` — 300 diverse images (daylight, thermal, night, all classes).

**Export:**
- ONNX INT8: ~3-4MB (Apple Silicon, Jetson)
- TFLite INT8: ~3.5MB (NXP)
- Hailo HEF: ~3MB (RPi 5 + Hailo-8L)

**Expected accuracy loss:** <2% mAP with good calibration data.

**Files:** `ml/scripts/export_all.py`, `ml/data/calibration/`

---

## Phase 7: Hard Test Set & Evaluation

**Create `ml/data/test_hard/`** — 200 images of the hardest cases:
- Real thermal/IR drone frames
- Low-res, smoky, night conditions
- All 7 classes represented

**Add to `ml/scripts/evaluate.py`:**
- `--test-hard` flag
- Per-condition metrics (thermal/daylight/night/smoke)
- Confidence sweep (eval at 0.15, 0.25, 0.35)
- Small object AP (<32px)

**Files:** `ml/scripts/evaluate.py`, `ml/scripts/test_model_visual.py`

---

## Implementation Order

| Step | What | Days | Impact |
|------|------|------|--------|
| 1 | Build degraded augmentation pipeline | 2 | HIGHEST |
| 2 | Fix merge/synthetic scripts for missing classes | 1 | HIGH |
| 3 | Generate synthetic data (needs Gazebo running) | 2 | HIGH |
| 4 | Apply offline augmentation to full dataset | 1 | HIGH |
| 5 | Train YOLOv8n-p2 with augmented data (200 epochs) | 2-3 | HIGH |
| 6 | Implement SAHI in detector.py | 1 | HIGH |
| 7 | Build hard test set + evaluation framework | 1 | MEDIUM |
| 8 | INT8 quantization + edge export | 1 | MEDIUM |
| 9 | Integrate SAHI into camera_bridge.py | 0.5 | MEDIUM |
| 10 | End-to-end test in Gazebo simulator | 0.5 | MEDIUM |
| **Total** | | **~12-14 days** | |

---

## Expected Results

| Metric | Current v2 | Target v3 |
|--------|-----------|-----------|
| F1 on real drone footage | 0.665 | **0.80+** |
| Tank recall (hard test) | 46.7% | **70%+** |
| APC recall | 40.7% | **65%+** |
| Truck recall | 14.1% | **45%+** |
| sp_artillery recall | 0% | **50%+** |
| helicopter recall | 0% | **55%+** |
| mlrs recall | 0% | **40%+** |
| UAV recall | 95.5% | **95%+** |
| Active classes | 4/7 | **7/7** |
| ONNX model size | 42.7 MB | **~10 MB** |
| INT8 model size | N/A | **~3-4 MB** |

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `ml/scripts/augment_degraded.py` | NEW — degraded image augmentation pipeline |
| `ml/scripts/merge_datasets.py` | Add helicopter data loading, oversampling |
| `ml/scripts/generate_synthetic.py` | Fix CLASS_MAP for sp_artillery/mlrs/helicopter |
| `ml/configs/train_config.yaml` | YOLOv8n-p2, 200 epochs, label_smoothing |
| `ml/scripts/train.py` | Support p2 architecture |
| `backend/detector.py` | Add detect_sahi() method |
| `backend/camera_bridge.py` | Add --sahi flag |
| `ml/scripts/evaluate.py` | Add --test-hard, per-condition metrics |
| `ml/scripts/export_all.py` | INT8 calibration path |

## Verification

1. **Offline:** Run `test_model_visual.py` on hard test set — F1 > 0.80
2. **Per-class:** All 7 classes detected with recall > 40%
3. **Edge:** ONNX < 15MB, INT8 < 5MB, inference < 40ms on target hardware
4. **SAHI:** Compare mAP with/without SAHI on small objects
5. **Sim-in-loop:** Fly over military_training.sdf, detect all 19 vehicles in dashboard
