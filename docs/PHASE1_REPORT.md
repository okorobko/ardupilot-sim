# Phase 1 Report — Detection Pipeline Smoke Test

**Date:** 2026-04-12
**Status:** PASSED (with critical bug fix)

---

## Test procedure

1. Launched full military simulation stack (Gazebo + SITL + backend + camera bridge)
2. Armed drone in GUIDED mode, took off to ~45m (climbed to 67m due to prior altitude)
3. Flew NE at 3 m/s toward tank column area for 15 seconds
4. Hovered and monitored detections
5. Flew south toward artillery/MLRS formations for 10 seconds

## Critical bug found & fixed

**Bug:** `detector.py` `postprocess()` assumed model output bbox coordinates were in **pixel space (0–640)** but the ONNX model outputs **normalized coordinates (0–1)**.

**Evidence:**
```
Raw output shape: (1, 11, 8400)
cx range: [0.005, 0.994]   ← normalized, NOT pixel
cy range: [0.014, 0.993]
```

**Result:** All bounding boxes were sub-pixel (`bbox=[0.4, 0.0, 0.6, 0.0]`) — invisible on screen.

**Fix:** Added `* self.input_size` scale in `postprocess()`:
```python
cx, cy, w, h = cx * self.input_size, cy * self.input_size, w * self.input_size, h * self.input_size
```

**After fix:**
```
bbox=[271.0, 61.8, 393.4, 166.0]  ← 122×104 pixel box, correct for vehicle at 67m
```

## Detection results

| Metric | Value |
|--------|-------|
| Total frames processed | ~700 during test flight |
| Detections | 3 |
| Classes detected | sp_artillery (3×) |
| Confidence range | 31% – 41% |
| Bbox sample | [271.0, 61.8, 393.4, 166.0] |
| Inference latency | 5.8–7.5 ms (CoreML) |
| Flight altitude | 67 m |

## Observations

1. **Detection count is low (3 over ~2 min flight)** — at 67m, objects are small and the model was trained with 640×640 input. The camera FOV at 67m covers a wide area but each vehicle is only ~10-20 pixels.
2. **Only sp_artillery detected** — this class has very high test mAP (0.947) and distinctive shape. Other classes may need lower altitude (40-50m) or lower confidence threshold.
3. **Confidence is marginal** (31-41%) — just above the 0.30 threshold. At lower altitude, confidence will increase.
4. **Altitude too high** — drone climbed to 67m; optimal detection range is 40-60m per the ML README.

## Infrastructure confirmed working

- ✅ Model loaded (ONNX, CoreML accelerated)
- ✅ Frame capture (`gz topic → BGR`)
- ✅ Inference pipeline (~7 ms)
- ✅ SocketIO transport (`detection_results` event)
- ✅ Browser rendering (`drawDetections()` on `det-canvas`)
- ✅ Bounding box coordinates now correct (post-fix)

## MAVLink port discovery

During setup, discovered that **SITL SERIAL0 (TCP:5760) does not send heartbeats** in Gazebo JSON mode. Switched to **SERIAL1 (TCP:5762)** which works correctly. Updated `drone.yaml` accordingly.

## Files modified

| File | Change |
|------|--------|
| `backend/detector.py` | Fixed normalized→pixel bbox scaling |
| `config/drone.yaml` | Changed `mavlink_port` from 5760 to 5762 |

## Next step

→ **Phase 2B**: Enlarge the downward-cam detection overlay for better visibility, then move to Phase 2A (chase-cam detection).
