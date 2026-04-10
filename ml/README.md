# ML Pipeline — Military Vehicle Detection

Aerial military vehicle detection from drone camera footage.
Trained on 27K degraded + synthetic images; runs on-drone via ONNX + CoreML.

---

## Model Card — Aerial v3

| Property | Value |
|---|---|
| Architecture | YOLOv8n (standard, no P2 head) |
| Input size | 640×640 px |
| Parameters | 3.0M |
| Classes | 7 (see below) |
| Training images | 26,994 |
| Val mAP50 | 0.726 |
| **Test mAP50** | **0.773** |
| Test mAP50-95 | 0.513 |
| ONNX inference | 147 FPS (Apple M4, CoreML) |

### Classes

| ID | Name | Test mAP50 | Notes |
|----|------|-----------|-------|
| 0 | tank | 0.834 | T-72 and similar MBTs |
| 1 | apc_ifv | 0.623 | BMP-2, wheeled APCs — most confused class |
| 2 | military_truck | 0.691 | 6×6 cargo / logistic vehicles |
| 3 | sp_artillery | 0.947 | Self-propelled artillery (distinctive shape) |
| 4 | mlrs | — | MLRS / Grad (limited test samples) |
| 5 | helicopter | — | Mi-24 and similar (limited test samples) |
| 6 | uav | 0.912 | Drones / fixed-wing UAVs |

*mlrs and helicopter classes scored 0 test instances — not enough val coverage, not model failure.*

### Export Artifacts

| File | Size | Use case |
|------|------|----------|
| `models/mil_vehicle_aerial_v3_full/weights/best.pt` | 12 MB | PyTorch fine-tuning / re-export |
| `models/mil_vehicle_aerial_v3_full/weights/best.onnx` | 11.7 MB | Production (drone backend, CoreML) |
| `models/mil_vehicle_aerial_v3_full/weights/best_saved_model/best_dynamic_range_quant.tflite` | 3.2 MB | Edge deployment (RPi, NXP, Coral) |
| `models/mil_vehicle_aerial_v3_full/weights/best_saved_model/best_float16.tflite` | 5.9 MB | Mobile GPU |

*`.tflite` files are gitignored — regenerate from `best.pt` with `scripts/export_all.py`.*

---

## Dataset Pipeline

### Sources (27K training images total)

| Source | Images | Classes | Notes |
|--------|--------|---------|-------|
| MV-RSD (Google Earth) | 3,000 | tank, truck, APC | Satellite, clean |
| Roboflow military annotated | ~4,000 | tank, APC, truck | Mixed quality |
| Roboflow aerial tanks | 618 | tank | Overhead |
| Roboflow pure tank | 1,036 | tank | High quality |
| HuggingFace drone detection | 1,500 | UAV | Drone footage |
| Offline synthetic (v3) | 1,151 | sp_artillery, mlrs, helicopter | 3D projection |
| Degraded augmentation | ×2 per image | all | 15 degradation profiles |

### Degradation Profiles (`scripts/augment_degraded.py`)

15 profiles applied offline — 2 variants per image, tripling the effective dataset:

| Profile | Simulates |
|---------|-----------|
| thermal_ir | Thermal/IR camera output |
| jpeg_compression (q=15–40) | FPV video compression |
| motion_blur (10–30px) | Drone vibration / movement |
| downscale_upscale (25–50%) | Distant small objects |
| low_brightness (0.2–0.4×) | Night / twilight |
| fog_overlay | Smoke, battlefield obscurant |
| poisson_noise | Low-light sensor noise |
| barrel_distortion | FPV fisheye lens |
| text_watermark | Telegram channel overlays |
| gaussian_noise | General sensor noise |

### Synthetic Data (`scripts/generate_synthetic_offline.py`)

Pure offline 3D projection — no Gazebo rendering needed:
- Camera pinhole model at altitudes 15–75m, pitch +90° (nadir)
- 3D vehicle bounding boxes projected to 2D pixel coords
- Satellite background tiles from `data/aerial/backgrounds/`
- Generated 1,151 base images → 2,302 augmented variants
- Classes covered: sp_artillery (7.5×3.2m), mlrs (7.2×2.5m), helicopter (17.5×3.5m)

---

## Training

### Two-Phase Protocol

**Phase 1 — Frozen backbone warmup** (15 epochs)
```
lr0: 0.001  |  freeze: 10 layers  |  batch: 8
```

**Phase 2 — Full fine-tune** (200 epochs target, cosine decay)
```
lr0: 0.01  |  lrf: 0.01 (final LR = 0.0001)  |  optimizer: SGD
momentum: 0.937  |  weight_decay: 0.0005
```

### Key Config Decisions

| Setting | Value | Reason |
|---------|-------|--------|
| `base_weights` | yolov8n.pt | YOLOv8n-p2 dropped — MPS libc++abi abort (uncatchable) |
| `mosaic` | 0.0 | MPS TAL shape-mismatch crash with dense annotations + mosaic |
| `batch` | 8 | MPS memory stability |
| `label_smoothing` | 0.1 | Noisy synthetic labels |
| `degrees` | 180° | Aerial imagery has no canonical orientation |
| `device` | mps | Apple M4 Pro GPU |

### MPS Stability Notes

Training on Apple Silicon MPS requires several workarounds:

1. **TAL shape mismatch crash** — occurs with >25 labels/image or mosaic enabled.
   - Fix: cap labels at 15/image, `mosaic=0.0`
   - `ultralytics/utils/tal.py` patched to catch RuntimeError and fall back to CPU

2. **P2 head libc++abi abort** — `c10::AcceleratorError` is uncatchable in Python.
   - Fix: use standard YOLOv8n (no P2 head); recover small-object sensitivity via SAHI at inference

3. **Process killed by macOS** — energy saver kills long-running background processes.
   - Fix: `caffeinate -s python3 scripts/train.py ...`

### Running Training

```bash
source ml/venv_ml/bin/activate

# Full pipeline (warmup + fine-tune)
caffeinate -s python3 ml/scripts/train.py \
  --config ml/configs/train_config.yaml \
  --dataset ml/configs/dataset_aerial_v3.yaml \
  --name mil_vehicle_aerial_v3

# Resume from checkpoint
caffeinate -s python3 ml/scripts/train.py \
  --config ml/configs/train_config.yaml \
  --dataset ml/configs/dataset_aerial_v3.yaml \
  --name mil_vehicle_aerial_v3 \
  --phase full \
  --resume ml/models/mil_vehicle_aerial_v3_full/weights/last.pt
```

---

## Inference

### Standard (single-pass, 640px)

```python
from backend.detector import VehicleDetector

detector = VehicleDetector(
    "ml/models/mil_vehicle_aerial_v3_full/weights/best.onnx",
    conf_threshold=0.30,
    iou_threshold=0.45,
)
detections = detector.detect(frame_bgr)
# [{"class_id": 0, "class_name": "tank", "confidence": 0.87, "bbox": [x1,y1,x2,y2]}]
```

### SAHI Sliced (for higher-resolution inputs >1280px)

```python
detections = detector.detect_sahi(frame_bgr, slice_size=320, overlap=0.3)
```

**Note:** SAHI was benchmarked and does NOT improve recall on 640px input (baseline 147 FPS / mAP50=0.683 vs SAHI best 23 FPS / mAP50=0.504). SAHI is only beneficial when source images exceed 1280px. The method is implemented and ready for higher-resolution cameras.

---

## Evaluation

```bash
# Val set
python3 ml/scripts/evaluate.py \
  --weights ml/models/mil_vehicle_aerial_v3_full/weights/best.pt \
  --dataset ml/configs/dataset_aerial_v3.yaml

# Test set
python3 ml/scripts/evaluate.py \
  --weights ml/models/mil_vehicle_aerial_v3_full/weights/best.pt \
  --dataset ml/configs/dataset_aerial_v3.yaml \
  --split test
```

Results saved to `ml/models/mil_vehicle_aerial_v3_full/eval_test_results.json`.

---

## Export

```bash
# ONNX (primary — runs on drone backend via CoreML)
python3 ml/scripts/export_all.py \
  --weights ml/models/mil_vehicle_aerial_v3_full/weights/best.pt \
  --output-dir ml/models/mil_vehicle_aerial_v3_full/weights/

# TFLite INT8 (edge deployment)
python3 -c "
from ultralytics import YOLO
YOLO('ml/models/mil_vehicle_aerial_v3_full/weights/best.pt').export(
    format='tflite', imgsz=640, int8=True,
    data='ml/configs/dataset_aerial_v3.yaml'
)
"
```

---

## Directory Structure

```
ml/
├── configs/
│   ├── train_config.yaml        # Hyperparameters, augmentation, export settings
│   ├── dataset_aerial_v3.yaml   # Aerial v3 dataset paths + 7-class schema
│   └── dataset.yaml             # Legacy ground dataset config
├── data/                        # gitignored — downloaded + generated datasets
│   ├── aerial/
│   │   ├── train/images/        # 26,994 training images
│   │   ├── val/images/          # 1,812 validation images
│   │   └── test/images/         # 1,017 test images
│   └── synthetic/               # Offline-generated synthetic frames
├── models/                      # Training runs (large files via Git LFS)
│   ├── mil_vehicle_aerial_v3_warmup/   # Phase 1 weights
│   └── mil_vehicle_aerial_v3_full/     # Phase 2 weights + exports
├── scripts/
│   ├── train.py                 # Two-phase training entry point
│   ├── augment_degraded.py      # Offline degradation augmentation (15 profiles)
│   ├── generate_synthetic.py    # Gazebo-assisted synthetic generation
│   ├── generate_synthetic_offline.py  # Pure 3D projection (no Gazebo)
│   ├── merge_datasets.py        # Multi-source dataset merge + split
│   ├── export_all.py            # ONNX / TFLite / TRT export pipeline
│   ├── evaluate.py              # Per-class mAP evaluation
│   ├── sahi_infer.py            # SAHI parameter sweep + benchmark
│   ├── benchmark_inference.py   # Latency benchmarks
│   └── test_model_visual.py     # Visual debug tool
├── venv_ml/                     # gitignored — Python 3.11 venv
│   └── requirements.txt
└── README.md                    # This file
```

---

## Python Environment

```bash
python3.11 -m venv ml/venv_ml
source ml/venv_ml/bin/activate
pip install ultralytics==8.4.30 onnxruntime pyyaml opencv-python numpy
# For TFLite export:
pip install "tensorflow>=2.19,<=2.21" tf_keras onnx2tf onnx-graphsurgeon ai-edge-litert
```

**Note:** Use `ml/venv_ml/` (Python 3.11), not the main project `venv/` (Python 3.14). Ultralytics requires Python ≤ 3.11 for MPS training.
