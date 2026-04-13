# Phase 4 Report — Temporal Smoothing (SimpleTracker)

**Date:** 2026-04-13
**Status:** COMPLETE

---

## Changes made

### SimpleTracker class (`backend/detector.py`)

IoU-based frame-to-frame tracker with these properties:
- **Greedy IoU matching**: each new frame's detections are matched to existing tracks by highest IoU (threshold 0.3)
- **Stable track IDs**: each track gets a monotonically increasing integer ID
- **min_hits = 2**: a track must be seen in 2+ consecutive frames before being displayed (kills one-frame false positives)
- **max_age = 8**: tracks persist for 8 frames without a match before removal (prevents flicker during brief occlusions)
- **Always-emit pattern**: the camera bridge emits `detection_results` every frame (even with 0 tracks), so the frontend always redraws and stale boxes disappear cleanly

### Integration

- `camera_bridge.py`: each camera stream gets its own `SimpleTracker` instance; raw detections are fed through `tracker.update()` before emission
- Frontend labels now show `#ID class conf%` format (e.g., `#2 sp_artillery 40%`)
- Stats labels changed from "det" to "trk" to reflect tracking

---

## Test results

| Metric | Before (Phase 1-3) | After (Phase 4) |
|--------|--------------------|-----------------| 
| Raw detections | 22 in ~20s | 22 in ~20s (same) |
| Displayed tracks | 22 independent boxes | **2 stable tracks** |
| False positive suppression | None | min_hits=2 filters single-frame noise |
| Flicker | Every frame is independent | Tracks persist up to 8 frames |
| Track IDs | N/A | #2, #3 (stable across frames) |

### Sample tracked detection
```json
{
    "track_id": 2,
    "class_id": 3,
    "class_name": "sp_artillery",
    "confidence": 0.405,
    "bbox": [214.6, 149.0, 375.7, 291.3]
}
```

---

## How it works

```
Frame N:   raw detect → [{sp_arty, bbox_A}]
           tracker.update() → match to track #2 (IoU > 0.3)
           → track #2 hit_count=5, age=0 → EMIT

Frame N+1: raw detect → [{sp_arty, bbox_A'}]  (slightly shifted)
           tracker.update() → match to track #2 (IoU > 0.3)
           → track #2 hit_count=6, bbox updated → EMIT (same ID, new position)

Frame N+2: raw detect → []  (occlusion/missed)
           tracker.update() → no match, track #2 age=1
           → track #2 still emitted (age < max_age=8)

Frame N+10: still no match → track #2 age=9 > max_age → REMOVED
```

---

## Files modified

| File | Change |
|------|--------|
| `backend/detector.py` | Added `SimpleTracker` class (IoU matching, track management) |
| `backend/camera_bridge.py` | Integrated tracker per camera stream, always-emit pattern |
| `frontend/templates/index.html` | Track ID in labels, "trk" stats label |

---

## Next step

→ **Phase 5**: Flight guidance (mini-map vehicle dots from SDF, SURVEY auto-fly button).
