#!/usr/bin/env python3
"""Comprehensive test suite for the Military Vehicle Detection ML pipeline.

Tests are organized into sections:
  1. Dataset & Config Integrity
  2. Gazebo Model Validation
  3. World File Validation
  4. Detector Unit Tests (preprocessing, NMS, postprocessing)
  5. Pipeline Script Validation
  6. Integration Tests (camera_bridge, app.py, frontend)
  7. KPI Acceptance Gates (model performance targets)

Run:
    python3 scripts/test_ml_pipeline.py           # all tests
    python3 scripts/test_ml_pipeline.py --quick    # skip KPI tests (no model needed)
    python3 scripts/test_ml_pipeline.py --kpi      # only KPI tests (needs trained model)

Output:
    Console summary + ml_pipeline_kpi_report.json
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ML_DIR = PROJECT_DIR / "ml"
GAZEBO_DIR = PROJECT_DIR / "gazebo"
BACKEND_DIR = PROJECT_DIR / "backend"
FRONTEND_DIR = PROJECT_DIR / "frontend"
CONFIG_DIR = PROJECT_DIR / "config"

sys.path.insert(0, str(BACKEND_DIR))

# ═══════════════════════════════════════════════════════════════════
# Test framework
# ═══════════════════════════════════════════════════════════════════

class TestResult:
    def __init__(self, name, section, passed, message="", value=None, target=None):
        self.name = name
        self.section = section
        self.passed = passed
        self.message = message
        self.value = value      # Measured value (for KPIs)
        self.target = target    # Target value (for KPIs)

results = []
_all_tests = []  # Registry of all test functions

def test(name, section="General"):
    """Decorator to register and run a test."""
    def decorator(func):
        def wrapper():
            try:
                ret = func()
                if ret is None:
                    results.append(TestResult(name, section, True))
                elif isinstance(ret, tuple):
                    passed, msg = ret[0], ret[1]
                    val = ret[2] if len(ret) > 2 else None
                    tgt = ret[3] if len(ret) > 3 else None
                    results.append(TestResult(name, section, passed, msg, val, tgt))
                else:
                    results.append(TestResult(name, section, bool(ret)))
            except Exception as e:
                results.append(TestResult(name, section, False, f"Exception: {e}"))
        wrapper._test_name = name
        wrapper._test_section = section
        _all_tests.append(wrapper)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════
# Section 1: Dataset & Config Integrity
# ═══════════════════════════════════════════════════════════════════

EXPECTED_CLASSES = ["tank", "apc_ifv", "military_truck", "sp_artillery",
                    "mlrs", "helicopter", "uav"]

@test("dataset.yaml exists", "Config")
def _():
    assert (ML_DIR / "configs" / "dataset.yaml").exists()

@test("dataset.yaml has 7 classes", "Config")
def _():
    import yaml
    with open(ML_DIR / "configs" / "dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["nc"] == 7, f"Expected 7 classes, got {cfg['nc']}"
    names = [cfg["names"][i] for i in range(7)]
    assert names == EXPECTED_CLASSES, f"Class mismatch: {names}"

@test("train_config.yaml exists and valid", "Config")
def _():
    import yaml
    path = ML_DIR / "configs" / "train_config.yaml"
    assert path.exists()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert "warmup" in cfg, "Missing warmup section"
    assert "training" in cfg, "Missing training section"
    assert "augmentation" in cfg, "Missing augmentation section"
    assert cfg["augmentation"]["degrees"] == 180.0, "Aerial rotation should be 180"
    assert cfg["augmentation"]["flipud"] == 0.5, "Vertical flip should be 0.5"

@test("train_config has correct phase params", "Config")
def _():
    import yaml
    with open(ML_DIR / "configs" / "train_config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["warmup"]["epochs"] == 10
    assert cfg["warmup"]["freeze"] == 10
    assert cfg["training"]["epochs"] == 150
    assert cfg["training"]["optimizer"] == "SGD"

@test("ml/requirements.txt has ultralytics", "Config")
def _():
    txt = (ML_DIR / "requirements.txt").read_text()
    assert "ultralytics" in txt
    assert "onnxruntime" in txt
    assert "albumentations" in txt

@test("dataset.yaml class names match detector.py", "Config")
def _():
    import yaml
    with open(ML_DIR / "configs" / "dataset.yaml") as f:
        cfg = yaml.safe_load(f)
    from detector import CLASS_NAMES
    yaml_names = [cfg["names"][i] for i in range(cfg["nc"])]
    assert yaml_names == CLASS_NAMES, f"Mismatch: {yaml_names} vs {CLASS_NAMES}"

@test("drone.yaml has detection section", "Config")
def _():
    import yaml
    with open(CONFIG_DIR / "drone.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "detection" in cfg, "Missing detection section"
    det = cfg["detection"]
    assert "enabled" in det
    assert "model_path" in det
    assert "conf_threshold" in det
    assert len(det["classes"]) == 7

@test("data directories exist", "Config")
def _():
    for subdir in ["raw", "processed", "synthetic", "calibration"]:
        path = ML_DIR / "data" / subdir
        assert path.exists(), f"Missing {path}"

@test(".gitignore covers ml/data and ml/models", "Config")
def _():
    txt = (PROJECT_DIR / ".gitignore").read_text()
    assert "ml/data/" in txt, "ml/data/ not in .gitignore"
    assert "ml/models/" in txt, "ml/models/ not in .gitignore"


# ═══════════════════════════════════════════════════════════════════
# Section 2: Gazebo Model Validation
# ═══════════════════════════════════════════════════════════════════

MILITARY_MODELS = {
    "tank_t72": {"hull": (7.0, 3.6, 1.2), "has_turret": True, "has_barrel": True},
    "apc_bmp2": {"hull": (6.7, 3.1, 1.1), "has_turret": True, "has_barrel": False},
    "military_truck": {"has_cab": True, "has_cargo": True},
    "sp_artillery": {"hull": (7.5, 3.4, 1.5), "has_turret": True, "has_barrel": True},
    "mlrs_grad": {"has_cab": True, "has_launcher": True},
    "helicopter_mi24": {"has_fuselage": True, "has_rotor": True, "has_tail": True},
}

@test("all 6 military vehicle models exist", "Gazebo Models")
def _():
    for name in MILITARY_MODELS:
        model_dir = GAZEBO_DIR / "models" / name
        assert model_dir.exists(), f"Missing model dir: {name}"
        assert (model_dir / "model.sdf").exists(), f"Missing model.sdf: {name}"
        assert (model_dir / "model.config").exists(), f"Missing model.config: {name}"

@test("all model SDFs are valid XML", "Gazebo Models")
def _():
    for name in MILITARY_MODELS:
        sdf_path = GAZEBO_DIR / "models" / name / "model.sdf"
        tree = ET.parse(sdf_path)
        root = tree.getroot()
        assert root.tag == "sdf", f"{name}: root tag is {root.tag}, expected 'sdf'"

@test("all models are static", "Gazebo Models")
def _():
    for name in MILITARY_MODELS:
        sdf_path = GAZEBO_DIR / "models" / name / "model.sdf"
        tree = ET.parse(sdf_path)
        static = tree.find(".//static")
        assert static is not None and static.text.strip() == "true", \
            f"{name}: model not static"

@test("tank_t72 dimensions correct", "Gazebo Models")
def _():
    sdf_path = GAZEBO_DIR / "models" / "tank_t72" / "model.sdf"
    tree = ET.parse(sdf_path)
    visuals = tree.findall(".//visual")
    visual_names = [v.get("name", "") for v in visuals]
    assert "hull" in visual_names, "Missing hull visual"
    assert "turret" in visual_names, "Missing turret visual"
    assert "gun_barrel" in visual_names, "Missing gun_barrel visual"
    # Check hull size
    for v in visuals:
        if v.get("name") == "hull":
            size = v.find(".//size").text.split()
            dims = [float(x) for x in size]
            assert abs(dims[0] - 7.0) < 0.1, f"Hull length: {dims[0]}"
            assert abs(dims[1] - 3.6) < 0.1, f"Hull width: {dims[1]}"

@test("all models have collision geometry", "Gazebo Models")
def _():
    for name in MILITARY_MODELS:
        sdf_path = GAZEBO_DIR / "models" / name / "model.sdf"
        tree = ET.parse(sdf_path)
        collision = tree.find(".//collision")
        assert collision is not None, f"{name}: missing collision element"

@test("military vehicle colors are olive/green range", "Gazebo Models")
def _():
    """Verify military vehicles use military colors (green-ish), not civilian."""
    for name in MILITARY_MODELS:
        sdf_path = GAZEBO_DIR / "models" / name / "model.sdf"
        tree = ET.parse(sdf_path)
        # Check first ambient color
        ambient = tree.find(".//ambient")
        if ambient is not None:
            rgba = [float(x) for x in ambient.text.strip().split()]
            r, g, b = rgba[0], rgba[1], rgba[2]
            # Military colors: green channel should be dominant or similar to red
            # and overall dark (all < 0.5)
            assert max(r, g, b) < 0.5, \
                f"{name}: color too bright ({r:.2f},{g:.2f},{b:.2f})"

@test("helicopter has rotor disk >= 10m", "Gazebo Models")
def _():
    sdf_path = GAZEBO_DIR / "models" / "helicopter_mi24" / "model.sdf"
    tree = ET.parse(sdf_path)
    for v in tree.findall(".//visual"):
        if "rotor" in (v.get("name", "") or "").lower():
            size = v.find(".//size")
            if size is not None:
                dims = [float(x) for x in size.text.split()]
                assert max(dims) >= 10.0, f"Rotor disk too small: {max(dims)}m"
                return
    assert False, "No rotor visual found"

@test("model proportions are realistic", "Gazebo Models")
def _():
    """Verify length > width > height for ground vehicles (not helo)."""
    for name in ["tank_t72", "apc_bmp2", "sp_artillery"]:
        sdf_path = GAZEBO_DIR / "models" / name / "model.sdf"
        tree = ET.parse(sdf_path)
        for v in tree.findall(".//visual"):
            if v.get("name") == "hull":
                dims = [float(x) for x in v.find(".//size").text.split()]
                l, w, h = dims[0], dims[1], dims[2]
                assert l > w > h, f"{name}: l={l}, w={w}, h={h} — not l>w>h"


# ═══════════════════════════════════════════════════════════════════
# Section 3: World File Validation
# ═══════════════════════════════════════════════════════════════════

@test("military_training.sdf exists and valid XML", "World")
def _():
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    assert path.exists()
    ET.parse(path)

@test("military world has 19 military vehicles", "World")
def _():
    """Count all <include> elements that reference military models."""
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    tree = ET.parse(path)
    includes = tree.findall(".//include")
    mil_models = {"tank_t72", "apc_bmp2", "military_truck",
                  "sp_artillery", "mlrs_grad", "helicopter_mi24"}
    count = 0
    for inc in includes:
        uri = inc.find("uri")
        if uri is not None:
            model_name = uri.text.replace("model://", "")
            if model_name in mil_models:
                count += 1
    assert count == 19, f"Expected 19 military vehicles, found {count}"

@test("military world vehicle distribution", "World")
def _():
    """Verify per-type counts match plan."""
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    tree = ET.parse(path)
    counts = {}
    for inc in tree.findall(".//include"):
        uri = inc.find("uri")
        if uri is not None:
            name = uri.text.replace("model://", "")
            counts[name] = counts.get(name, 0) + 1
    expected = {
        "tank_t72": 4,
        "apc_bmp2": 4,
        "military_truck": 4,
        "sp_artillery": 3,
        "mlrs_grad": 2,
        "helicopter_mi24": 2,
    }
    for model, exp_count in expected.items():
        actual = counts.get(model, 0)
        assert actual == exp_count, \
            f"{model}: expected {exp_count}, got {actual}"

@test("military world has drone model", "World")
def _():
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    tree = ET.parse(path)
    for inc in tree.findall(".//include"):
        uri = inc.find("uri")
        if uri is not None and "iris_with_camera" in uri.text:
            return
    assert False, "No iris_with_camera model in world"

@test("military world GPS matches drone.yaml", "World")
def _():
    import yaml
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    tree = ET.parse(path)
    lat_el = tree.find(".//latitude_deg")
    lon_el = tree.find(".//longitude_deg")
    assert lat_el is not None and lon_el is not None
    with open(CONFIG_DIR / "drone.yaml") as f:
        cfg = yaml.safe_load(f)
    assert abs(float(lat_el.text) - cfg["simulation"]["home_lat"]) < 0.001
    assert abs(float(lon_el.text) - cfg["simulation"]["home_lon"]) < 0.001

@test("vehicles have unique names", "World")
def _():
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    tree = ET.parse(path)
    names = []
    for inc in tree.findall(".//include"):
        name_el = inc.find("name")
        if name_el is not None:
            names.append(name_el.text)
    assert len(names) == len(set(names)), \
        f"Duplicate names: {[n for n in names if names.count(n) > 1]}"

@test("vehicles are spread across area (not stacked)", "World")
def _():
    path = GAZEBO_DIR / "worlds" / "military_training.sdf"
    tree = ET.parse(path)
    positions = []
    for inc in tree.findall(".//include"):
        pose = inc.find("pose")
        if pose is not None:
            parts = pose.text.strip().split()
            x, y = float(parts[0]), float(parts[1])
            positions.append((x, y))
    # Check no two vehicles are within 3m (would be overlapping)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            assert dist > 3.0, \
                f"Vehicles too close: {dist:.1f}m at {positions[i]} and {positions[j]}"


# ═══════════════════════════════════════════════════════════════════
# Section 4: Detector Unit Tests
# ═══════════════════════════════════════════════════════════════════

@test("detector.py imports cleanly", "Detector")
def _():
    import importlib.util
    spec = importlib.util.spec_from_file_location("detector", BACKEND_DIR / "detector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "VehicleDetector")
    assert hasattr(mod, "CLASS_NAMES")
    assert hasattr(mod, "CLASS_COLORS")

@test("CLASS_NAMES has 7 entries", "Detector")
def _():
    from detector import CLASS_NAMES
    assert len(CLASS_NAMES) == 7

@test("CLASS_COLORS has 7 entries", "Detector")
def _():
    from detector import CLASS_COLORS
    assert len(CLASS_COLORS) == 7
    for color in CLASS_COLORS:
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

@test("preprocessing: letterbox 640x480 → 640x640", "Detector")
def _():
    """Test that a 640x480 frame is correctly letterboxed to 640x640."""
    import numpy as np
    from detector import VehicleDetector
    # Create a mock detector without loading a model
    det = object.__new__(VehicleDetector)
    det.input_size = 640
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob, scale, pad = det.preprocess(frame)
    assert blob.shape == (1, 3, 640, 640), f"Shape: {blob.shape}"
    assert blob.dtype == np.float32
    assert 0 <= blob.min() and blob.max() <= 1.0
    # 640x480 → scale = min(640/640, 640/480) = 1.0
    # new_w=640, new_h=480, pad_x=0, pad_y=80
    assert pad == (0, 80), f"Pad: {pad}"

@test("preprocessing: letterbox 1920x1080 → 640x640", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    det = object.__new__(VehicleDetector)
    det.input_size = 640
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    blob, scale, pad = det.preprocess(frame)
    assert blob.shape == (1, 3, 640, 640)
    # scale = min(640/1920, 640/1080) = 0.333...
    # new_w=640, new_h=360 → pad_x=0, pad_y=140
    assert pad[0] == 0
    assert 138 <= pad[1] <= 142  # ~140

@test("preprocessing: square frame → no padding", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    det = object.__new__(VehicleDetector)
    det.input_size = 640
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    blob, scale, pad = det.preprocess(frame)
    assert pad == (0, 0)

@test("NMS: removes overlapping boxes", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    boxes = np.array([
        [100, 100, 200, 200],  # box A
        [110, 110, 210, 210],  # box B (overlaps A heavily)
        [300, 300, 400, 400],  # box C (no overlap)
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    keep = VehicleDetector._nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2, f"Expected 2 after NMS, got {len(keep)}"
    assert 0 in keep  # Box A (highest score) kept
    assert 2 in keep  # Box C (no overlap) kept
    assert 1 not in keep  # Box B suppressed

@test("NMS: keeps all non-overlapping boxes", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    boxes = np.array([
        [0, 0, 50, 50],
        [100, 100, 150, 150],
        [200, 200, 250, 250],
        [300, 300, 350, 350],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    keep = VehicleDetector._nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 4

@test("NMS: single box returns single box", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    scores = np.array([0.9])
    keep = VehicleDetector._nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0]

@test("postprocess: empty output returns empty list", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    det = object.__new__(VehicleDetector)
    det.conf_threshold = 0.35
    det.iou_threshold = 0.45
    det.num_classes = 7
    det.input_size = 640
    # YOLOv8 output: (1, 4+7, N) with all low scores
    fake_output = np.random.uniform(0, 0.1, (1, 11, 100)).astype(np.float32)
    result = det.postprocess(fake_output, (1.0, 1.0), (0, 80), (480, 640))
    assert result == []

@test("postprocess: high-confidence detection is returned", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector, CLASS_NAMES
    det = object.__new__(VehicleDetector)
    det.conf_threshold = 0.35
    det.iou_threshold = 0.45
    det.num_classes = 7
    det.input_size = 640
    # Craft a single detection: cx=320, cy=320, w=100, h=80, class 0 (tank) = 0.9
    # Shape: (1, 11, 1) — one detection
    data = np.zeros((1, 11, 1), dtype=np.float32)
    data[0, 0, 0] = 320  # cx
    data[0, 1, 0] = 320  # cy
    data[0, 2, 0] = 100  # w
    data[0, 3, 0] = 80   # h
    data[0, 4, 0] = 0.9  # class 0 score (tank)
    result = det.postprocess(data, (1.0, 1.0), (0, 80), (480, 640))
    assert len(result) == 1
    assert result[0]["class_name"] == "tank"
    assert abs(result[0]["confidence"] - 0.9) < 0.01
    # bbox should be: x1=270, y1=200 (320-80-80=200 after pad removal), x2=370, y2=280
    bbox = result[0]["bbox"]
    assert abs(bbox[0] - 270.0) < 1.0  # x1 = 320 - 50
    assert abs(bbox[2] - 370.0) < 1.0  # x2 = 320 + 50

@test("postprocess: bbox clipping to image bounds", "Detector")
def _():
    import numpy as np
    from detector import VehicleDetector
    det = object.__new__(VehicleDetector)
    det.conf_threshold = 0.1
    det.iou_threshold = 0.45
    det.num_classes = 7
    det.input_size = 640
    # Detection at edge of image — should clip
    data = np.zeros((1, 11, 1), dtype=np.float32)
    data[0, 0, 0] = 10   # cx near left edge
    data[0, 1, 0] = 10   # cy near top
    data[0, 2, 0] = 100  # w — extends past left
    data[0, 3, 0] = 100  # h — extends past top
    data[0, 4, 0] = 0.9
    result = det.postprocess(data, (1.0, 1.0), (0, 0), (480, 640))
    assert len(result) == 1
    bbox = result[0]["bbox"]
    assert bbox[0] >= 0, f"x1 should be >= 0, got {bbox[0]}"
    assert bbox[1] >= 0, f"y1 should be >= 0, got {bbox[1]}"


# ═══════════════════════════════════════════════════════════════════
# Section 5: Pipeline Script Validation
# ═══════════════════════════════════════════════════════════════════

ML_SCRIPTS = [
    "download_datasets.py",
    "prepare_dataset.py",
    "generate_synthetic.py",
    "train.py",
    "export_all.py",
    "evaluate.py",
    "benchmark_inference.py",
]

@test("all ML scripts exist", "Pipeline")
def _():
    for script in ML_SCRIPTS:
        path = ML_DIR / "scripts" / script
        assert path.exists(), f"Missing: {script}"

@test("all ML scripts have shebang", "Pipeline")
def _():
    for script in ML_SCRIPTS:
        path = ML_DIR / "scripts" / script
        first_line = path.read_text().split("\n")[0]
        assert first_line.startswith("#!"), f"{script}: missing shebang"

@test("all ML scripts are executable", "Pipeline")
def _():
    for script in ML_SCRIPTS:
        path = ML_DIR / "scripts" / script
        assert os.access(path, os.X_OK), f"{script}: not executable"

@test("all ML scripts compile without errors", "Pipeline")
def _():
    import py_compile
    for script in ML_SCRIPTS:
        path = ML_DIR / "scripts" / script
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            assert False, f"{script}: {e}"

@test("train.py supports two-phase training", "Pipeline")
def _():
    txt = (ML_DIR / "scripts" / "train.py").read_text()
    assert "warmup" in txt.lower() or "phase" in txt.lower()
    assert "freeze" in txt
    assert "YOLO" in txt

@test("export_all.py supports ONNX and TFLite", "Pipeline")
def _():
    txt = (ML_DIR / "scripts" / "export_all.py").read_text()
    assert "onnx" in txt.lower()
    assert "tflite" in txt.lower()

@test("prepare_dataset.py handles xView class mapping", "Pipeline")
def _():
    txt = (ML_DIR / "scripts" / "prepare_dataset.py").read_text()
    # Should reference xView class IDs
    assert "73" in txt or "xview" in txt.lower()

@test("launch scripts exist and are executable", "Pipeline")
def _():
    for name in ["start_military_world.sh", "start_camera_detect.sh"]:
        path = PROJECT_DIR / "scripts" / name
        assert path.exists(), f"Missing {name}"
        assert os.access(path, os.X_OK), f"{name} not executable"


# ═══════════════════════════════════════════════════════════════════
# Section 6: Integration Tests
# ═══════════════════════════════════════════════════════════════════

@test("camera_bridge.py has detection support", "Integration")
def _():
    txt = (BACKEND_DIR / "camera_bridge.py").read_text()
    assert "--detect" in txt
    assert "detection_results" in txt
    assert "VehicleDetector" in txt
    assert "capture_frame_raw" in txt

@test("app.py relays detection_results", "Integration")
def _():
    txt = (BACKEND_DIR / "app.py").read_text()
    assert "detection_results" in txt

@test("index.html has detection overlay canvas", "Integration")
def _():
    txt = (FRONTEND_DIR / "templates" / "index.html").read_text()
    assert "det-canvas" in txt
    assert "detection_results" in txt
    assert "drawDetections" in txt

@test("frontend has correct detection colors", "Integration")
def _():
    txt = (FRONTEND_DIR / "templates" / "index.html").read_text()
    for cls in EXPECTED_CLASSES:
        assert cls in txt, f"Missing class '{cls}' in frontend detection colors"

@test("frontend shows detection stats (count + latency)", "Integration")
def _():
    txt = (FRONTEND_DIR / "templates" / "index.html").read_text()
    assert "det-stats" in txt
    assert "latency" in txt.lower() or "detLatency" in txt

@test("camera_bridge backwards compatible (no --detect = no detection)", "Integration")
def _():
    """Ensure camera_bridge works without detection flag."""
    txt = (BACKEND_DIR / "camera_bridge.py").read_text()
    assert "_detector = None" in txt  # Global default is None
    assert "run_detection=False" in txt or "run_detection = False" in txt \
        or 'run_detection=detect_enabled' in txt


# ═══════════════════════════════════════════════════════════════════
# Section 7: KPI Acceptance Gates
# ═══════════════════════════════════════════════════════════════════

# These tests require a trained model. They define the acceptance
# criteria and will report SKIP if no model exists.

MODEL_SEARCH_PATHS = [
    ML_DIR / "models" / "mil_vehicle_v1" / "weights" / "best.onnx",
    ML_DIR / "models" / "mil_vehicle_v1" / "weights" / "best.pt",
    PROJECT_DIR / "best.onnx",
]

def find_model():
    """Find trained model, return path or None."""
    for p in MODEL_SEARCH_PATHS:
        if p.exists():
            return p
    return None

@test("KPI: ONNX model exists", "KPI")
def _():
    model = find_model()
    if model is None:
        return (False, "NO MODEL FOUND — train with: cd ml && python scripts/train.py")
    size_mb = model.stat().st_size / (1024 * 1024)
    return (True, f"Model: {model.name} ({size_mb:.1f}MB)", size_mb, "< 15MB")

@test("KPI: ONNX model size < 15MB (edge deployable)", "KPI")
def _():
    model = find_model()
    if model is None:
        return (False, "SKIP — no model", None, "< 15MB")
    size_mb = model.stat().st_size / (1024 * 1024)
    passed = size_mb < 15.0
    return (passed, f"{size_mb:.1f}MB", size_mb, 15.0)

@test("KPI: inference latency < 40ms on Apple Silicon", "KPI")
def _():
    model = find_model()
    if model is None or not str(model).endswith(".onnx"):
        return (False, "SKIP — no ONNX model", None, "< 40ms")
    try:
        from detector import VehicleDetector
        import numpy as np
        det = VehicleDetector(str(model))
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Warmup
        for _ in range(10):
            det.detect(frame)
        # Benchmark
        latencies = []
        for _ in range(50):
            det.detect(frame)
            latencies.append(det.latency_ms)
        median_ms = sorted(latencies)[len(latencies) // 2]
        passed = median_ms < 40.0
        return (passed, f"Median: {median_ms:.1f}ms", median_ms, 40.0)
    except Exception as e:
        return (False, f"Error: {e}", None, "< 40ms")

@test("KPI: FPS >= 25 on Apple Silicon", "KPI")
def _():
    model = find_model()
    if model is None or not str(model).endswith(".onnx"):
        return (False, "SKIP — no ONNX model", None, ">= 25 FPS")
    try:
        from detector import VehicleDetector
        import numpy as np
        det = VehicleDetector(str(model))
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            det.detect(frame)
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            det.detect(frame)
        elapsed = time.perf_counter() - t0
        fps = n / elapsed
        passed = fps >= 25.0
        return (passed, f"{fps:.1f} FPS", fps, 25.0)
    except Exception as e:
        return (False, f"Error: {e}", None, ">= 25 FPS")

@test("KPI: model outputs 7 classes", "KPI")
def _():
    model = find_model()
    if model is None or not str(model).endswith(".onnx"):
        return (False, "SKIP — no ONNX model", None, "7 classes")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model))
        output_shape = sess.get_outputs()[0].shape
        # YOLOv8n output: (1, 4+num_classes, N)
        num_classes = output_shape[1] - 4 if output_shape[1] else None
        if num_classes is None:
            return (False, "Dynamic output shape — cannot verify", None, 7)
        passed = num_classes == 7
        return (passed, f"{num_classes} classes", num_classes, 7)
    except Exception as e:
        return (False, f"Error: {e}", None, 7)

@test("KPI: detection produces valid bbox format", "KPI")
def _():
    model = find_model()
    if model is None or not str(model).endswith(".onnx"):
        return (False, "SKIP — no ONNX model", None, "valid bbox")
    try:
        from detector import VehicleDetector
        import numpy as np
        det = VehicleDetector(str(model), conf_threshold=0.01)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = det.detect(frame)
        # Even with noise, check format (may be empty with random input)
        for d in dets:
            assert "class_id" in d
            assert "class_name" in d
            assert "confidence" in d
            assert "bbox" in d
            assert len(d["bbox"]) == 4
            x1, y1, x2, y2 = d["bbox"]
            assert x2 >= x1, f"x2 < x1: {d['bbox']}"
            assert y2 >= y1, f"y2 < y1: {d['bbox']}"
            assert 0 <= d["class_id"] < 7
        return (True, f"Format valid ({len(dets)} dets from noise)", len(dets), None)
    except Exception as e:
        return (False, f"Error: {e}", None, "valid")


# ═══════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(results, output_path=None):
    """Print results and generate KPI JSON report."""
    # Group by section
    sections = {}
    for r in results:
        sections.setdefault(r.section, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print("\n" + "=" * 72)
    print("  MILITARY VEHICLE DETECTION — ML PIPELINE TEST REPORT")
    print("=" * 72)

    for section, tests in sections.items():
        sec_passed = sum(1 for t in tests if t.passed)
        sec_total = len(tests)
        indicator = "PASS" if sec_passed == sec_total else "FAIL"
        print(f"\n{'─' * 72}")
        print(f"  [{indicator}] {section} ({sec_passed}/{sec_total})")
        print(f"{'─' * 72}")
        for t in tests:
            icon = "PASS" if t.passed else "FAIL"
            msg = f"  [{icon}] {t.name}"
            if t.message:
                msg += f"  — {t.message}"
            if t.value is not None and t.target is not None:
                msg += f"  (measured: {t.value}, target: {t.target})"
            print(msg)

    print(f"\n{'=' * 72}")
    print(f"  TOTAL: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 72}")

    # KPI summary
    kpi_tests = [r for r in results if r.section == "KPI"]
    if kpi_tests:
        print(f"\n{'─' * 72}")
        print("  KPI DASHBOARD")
        print(f"{'─' * 72}")
        print(f"  {'KPI':<45} {'Value':>10} {'Target':>10} {'Status':>8}")
        print(f"  {'─'*45} {'─'*10} {'─'*10} {'─'*8}")
        for t in kpi_tests:
            val_str = f"{t.value}" if t.value is not None else "N/A"
            tgt_str = f"{t.target}" if t.target is not None else "—"
            status = "PASS" if t.passed else ("SKIP" if t.value is None else "FAIL")
            print(f"  {t.name:<45} {val_str:>10} {tgt_str:>10} {status:>8}")
        print()

    # JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{100*passed/total:.1f}%",
        },
        "sections": {},
        "kpis": {},
        "dataset": {
            "target_classes": EXPECTED_CLASSES,
            "num_classes": 7,
            "target_images": "~11K (6K real + 5K synthetic)",
            "real_sources": ["xView", "DOTA v2", "Roboflow"],
            "synthetic_source": "Gazebo military_training.sdf",
            "split": "70% train / 15% val / 15% test (synthetic in train+val only)",
        },
        "vehicles": {
            "gazebo_models": {
                "tank_t72": {
                    "type": "Main Battle Tank", "class": "tank",
                    "hull_lxwxh": "7.0 x 3.6 x 1.2m",
                    "features": "turret + gun barrel (5.0m)",
                    "color": "dark olive drab",
                    "instances_in_world": 4,
                },
                "apc_bmp2": {
                    "type": "APC / IFV", "class": "apc_ifv",
                    "hull_lxwxh": "6.7 x 3.1 x 1.1m",
                    "features": "small turret",
                    "color": "olive green",
                    "instances_in_world": 4,
                },
                "military_truck": {
                    "type": "6x6 Cargo Truck", "class": "military_truck",
                    "hull_lxwxh": "~8.0 x 2.4 x 2.0m (cab+cargo)",
                    "features": "cab + khaki cargo bed",
                    "color": "olive drab / khaki",
                    "instances_in_world": 4,
                },
                "sp_artillery": {
                    "type": "Self-Propelled Gun", "class": "sp_artillery",
                    "hull_lxwxh": "7.5 x 3.4 x 1.5m",
                    "features": "large turret + very long barrel (7.0m)",
                    "color": "dark military green",
                    "instances_in_world": 3,
                },
                "mlrs_grad": {
                    "type": "Multiple Launch Rocket System", "class": "mlrs",
                    "hull_lxwxh": "~8.0 x 2.4 x 2.0m (truck + launcher)",
                    "features": "angled launcher rack (4.0x2.0x1.5m, -20deg pitch)",
                    "color": "olive drab",
                    "instances_in_world": 2,
                },
                "helicopter_mi24": {
                    "type": "Attack Helicopter", "class": "helicopter",
                    "hull_lxwxh": "6.0 x 2.0 x 2.0m fuselage",
                    "features": "rotor disk (14m), tail boom (5m), landing skids",
                    "color": "dark camo green",
                    "instances_in_world": 2,
                },
            },
            "world_total": 19,
            "tactical_formations": [
                "Tank column (3x T-72 + 2x BMP-2) on E-W road",
                "Artillery battery (3x SP arty) NE field",
                "MLRS section (2x Grad) NW field",
                "Supply convoy (3x trucks + 1x APC) N-S road",
                "Helicopter LZ (2x Mi-24) SE clearing",
                "Mixed staging area (tank + APC + truck) near origin",
            ],
            "note_uav_class": "UAV class (6) has no Gazebo model yet — needs synthetic/real data only",
        },
        "performance_targets": {
            "apple_silicon_dev": {"precision": "FP32 ONNX", "target_fps": 25, "max_latency_ms": 40},
            "jetson_orin_nano": {"precision": "FP16 TensorRT", "target_fps": 30, "max_latency_ms": 33},
            "rpi5_hailo_8l": {"precision": "INT8 HEF", "target_fps": 25, "max_latency_ms": 40},
            "nxp_imx8m_plus": {"precision": "INT8 TFLite", "target_fps": 8, "max_latency_ms": 120},
        },
        "accuracy_targets": {
            "mAP_50_overall": ">= 0.50 on real-world test set",
            "per_class_recall_min": ">= 0.40 for each class",
            "zero_missed_tanks": "tank recall >= 0.60 (critical class)",
        },
    }

    for section, tests in sections.items():
        report["sections"][section] = {
            "passed": sum(1 for t in tests if t.passed),
            "total": len(tests),
            "tests": [
                {"name": t.name, "passed": t.passed, "message": t.message}
                for t in tests
            ],
        }

    for t in kpi_tests:
        report["kpis"][t.name] = {
            "passed": t.passed,
            "value": t.value,
            "target": t.target,
            "message": t.message,
        }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {output_path}")

    return report


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ML Pipeline Test Suite for Military Vehicle Detection")
    parser.add_argument("--quick", action="store_true",
                        help="Skip KPI tests (no model needed)")
    parser.add_argument("--kpi", action="store_true",
                        help="Run only KPI tests")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_DIR / "ml_pipeline_kpi_report.json"),
                        help="Output JSON report path")
    args = parser.parse_args()

    # Use registered tests
    all_tests = list(_all_tests)

    # Filter
    if args.quick:
        all_tests = [t for t in all_tests if t._test_section != "KPI"]
    elif args.kpi:
        all_tests = [t for t in all_tests if t._test_section == "KPI"]

    print(f"\nRunning {len(all_tests)} tests...\n")

    for test_fn in all_tests:
        test_fn()

    report = generate_report(results, args.output)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
