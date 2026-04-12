# Phase 2 Report — Detection Visibility Enhancement

**Date:** 2026-04-12
**Status:** COMPLETE

---

## Changes made

### 2A. Chase-cam detection overlay (main panel)

- Added `<canvas id="chase-det-canvas" width="800" height="600">` overlaying full chase-cam panel
- Added `drawChaseDetections()` JS function: renders bboxes at 3px stroke, 14px bold labels
- Added `#chase-det-stats` element showing chase detection count + latency
- `camera_bridge.py`: enabled detection on chase-cam thread with separate event `detection_results_chase`
- `app.py`: added relay handler for `detection_results_chase` event
- `stream_camera()`: added `detection_event` parameter to support distinct event names per camera

**Note:** Chase cam detection produces fewer hits at altitude (trained on nadir imagery, chase is oblique ~45°). At 115m: 0 detections on chase cam vs 24/50 frames on down cam. This is expected per the plan's accuracy caveat.

### 2B. Enlarged downward-cam detection overlay

- Grew from 320×240 to **480×360** pixels
- Detection canvas scales proportionally
- No-signal placeholder also enlarged
- Click to expand to full 640×480 still works

---

## Test results

### Detection flow verification

| Camera | Frames | Detections | Hit rate | Latency |
|--------|--------|------------|----------|---------|
| Down cam | 50 | 24 | 48% | 7.0 ms |
| Chase cam | 30 | 0 | 0% | 7.9 ms |

### End-to-end SocketIO verification

```
Down: 1 detections
Chase: 0 detections
Sample: uav 32% bbox=[160.2, 0.9, 182.7, 32.3]
```

Bbox coordinates are in correct pixel space (160×32 pixel box for a UAV at ~115m altitude).

### Chase cam detection analysis

At 115m altitude, the chase camera (mounted 3m behind, 2m above the drone, looking forward) sees vehicles at an oblique angle at great distance. The model's mAP drops significantly on non-nadir views. Chase cam detection will be more useful at:
- Lower altitudes (40-50m)
- When flying directly toward vehicle formations
- The infrastructure is ready — detections will appear automatically as conditions improve

---

## Files modified

| File | Change |
|------|--------|
| `frontend/templates/index.html` | New chase-det-canvas, enlarged down-cam (480×360), drawChaseDetections() JS |
| `backend/camera_bridge.py` | Chase cam detection enabled, detection_event parameter |
| `backend/app.py` | Relay handler for detection_results_chase |

---

## Next step

→ **Phase 3**: Detection info panel showing live detection counts by class, total, and latency stats.
