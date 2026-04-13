# Real-Time Detection Bounding Boxes — Plan & Analysis

**Date:** 2026-04-11 (completed 2026-04-13)
**Status:** ALL PHASES COMPLETE
**Goal:** Live, in-browser bounding boxes around military vehicles (tanks, APCs, trucks, artillery, MLRS, helicopters, UAVs) coming from the YOLOv8n model running in real time on the drone camera streams.

---

## Executive summary

The full detection pipeline is **already built and running** end-to-end:
- Model loaded (`mil_vehicle_aerial_v3_full/weights/best.onnx`, test mAP50 = **0.773**)
- CoreML-accelerated inference at **~7 ms/frame**
- SocketIO transport (`detection_results` event) working
- Browser canvas overlay (`drawDetections()`) rendering colored boxes with labels

**Currently shows 0 detections** because the drone is parked on grass at the spawn point. The downward camera points at empty ground — no vehicles in view. Detections will appear as soon as the drone takes off and flies over the military formations in `military_training.sdf`.

This is therefore **not a "build from scratch" task** — it is a **verification + enhancement + polish** task. The biggest gains are in *visibility* (making the detections prominent on the main view) and *temporal stability* (no flicker).

---

## Current architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Gazebo /camera   →   camera_bridge.py   →   VehicleDetector.detect()│
│     (5 fps)            (capture raw BGR)       (ONNX+CoreML, 7ms)    │
│                             ↓                          ↓             │
│                     emit camera_frame          emit detection_results│
│                      (base64 JPEG)           (bbox, class, conf)     │
│                             ↓                          ↓             │
│                            Flask-SocketIO relay                      │
│                                     ↓                                │
│                       index.html  drawDetections()                   │
│                       (colored boxes on det-canvas overlay)          │
└──────────────────────────────────────────────────────────────────────┘
```

### Key components

| Component | File | Status |
|-----------|------|--------|
| Model (ONNX) | `ml/models/mil_vehicle_aerial_v3_full/weights/best.onnx` | ready |
| Detector wrapper | `backend/detector.py` (`VehicleDetector`) | ready |
| Camera bridge | `backend/camera_bridge.py` (`stream_camera`) | ready |
| SocketIO transport | `backend/app.py` relays `detection_results` | ready |
| Frontend renderer | `frontend/templates/index.html` (`drawDetections()`) | ready |
| Classes | 7: tank, apc_ifv, military_truck, sp_artillery, mlrs, helicopter, uav | ready |

### Current live stats (at time of writing)

```
[DOWN] 2150 frames, 0 w/detections, latency=7.3ms
Model: mil_vehicle_aerial_v3_full/weights/best.onnx
Confidence threshold: 0.35
CoreML: 235/238 nodes accelerated
```

---

## Problem analysis

| # | Problem | Severity | Why it matters |
|---|---------|----------|----------------|
| 1 | Nothing is detected until the drone flies over formations | High | User thinks it's broken; needs a verification flight |
| 2 | Detection boxes appear on the **small corner overlay**, not the main view | High | Visually weak — user misses them |
| 3 | Chase cam (the **big** view) has no detection at all | Medium | User watches that panel while flying |
| 4 | Detections flicker frame-to-frame (each frame is independent) | Medium | Looks unstable, cheap, hard to count targets |
| 5 | No detection list / summary — hard to see *what* is being seen | Medium | Can't count hits at a glance |
| 6 | User has no flight guidance to reach the detection zones | Low | Manual hunting for vehicles |

---

## Implementation plan — phases

### Phase 1 — Verify current pipeline (smoke test, no code)

**Goal:** confirm the existing pipeline works before changing anything.

1. `ARM` → `TAKEOFF 40` → fly NE toward the artillery battery (~50 m from home).
2. Watch the downward-cam overlay (top-right corner):
   - RED = tank
   - ORANGE = apc_ifv
   - YELLOW = military_truck
   - LIGHT-RED = sp_artillery
   - MAGENTA = mlrs
   - CYAN = helicopter
   - TEAL = uav
3. Check `camera_bridge.py` stderr for `w/detections > 0` and `latency ~ 7-15 ms`.
4. If pass → pipeline is sound; move to Phase 2.
   If fail → debug: confidence threshold, class mapping, image decoding, frame flow.

**Deliverable:** screenshot / log line showing ≥ 1 detection.

---

### Phase 2 — Make detections prominent (biggest UX win)

Two complementary approaches, both recommended:

#### 2A. Draw detection boxes on the main chase-cam view

- Enable `run_detection=True` on the chase-cam thread in `camera_bridge.py`.
- Emit a **separate** `detection_results_chase` event (do not clobber the downward one).
- In `index.html` add a new `<canvas id="chase-det-canvas">` overlaying `#chase-img` full-panel.
- Add `drawChaseDetections()` scaling 800×600 native coords to the rendered chase-cam size.
- **Caveat:** the v3 model is trained on nadir/aerial imagery. Chase cam is oblique third-person. Expect ~40-60 % of nadir mAP. Acceptable for "is something there?", not for precision targeting.
- **Compute cost:** 2× inference (14 ms total). Still real-time at 5 fps down + 3 fps chase.

#### 2B. Enlarge and re-position the downward-cam detection overlay

- Grow from 320×240 to ~480×360 (or make it resizable).
- Thicker border, animated glow when detections present.
- Keep it the *primary* surface since the nadir view is what the model is best at.

**Recommendation:** do both. Phase 2B is near-zero effort. Phase 2A adds the "wow factor" on the main view.

---

### Phase 3 — Detection info panel (text + stats)

New "Detections" section in the right-side commands panel or a new cell in the bottom strip:

```
┌─────────────────────────────┐
│ DETECTIONS (live)           │
├─────────────────────────────┤
│ ● tank         87 %  (× 2)  │
│ ● apc_ifv      73 %  (× 1)  │
│ ● sp_arty      91 %  (× 1)  │
│─────────────────────────────│
│ Total: 4  |  Latency: 8 ms  │
│ Det rate: 4.2 /s            │
└─────────────────────────────┘
```

- Counts by class, peak confidence per class
- Rolling 1-s detection rate
- Inference latency (p50 / p95)
- Click a class to focus/highlight only those boxes

Even when the boxes are visually small, the user always sees *what* is being detected.

---

### Phase 4 — Temporal smoothing (kill the flicker)

**Current behaviour:** each frame is independent → boxes pop in and out → looks cheap and unstable.

**Fix:** simple IoU-based tracking with N-frame persistence (no full ByteTrack).

Add a `SimpleTracker` class in `detector.py`:

- Per-track state: `(track_id, class_id, last_bbox, last_seen_frame, hit_count)`
- On each new frame:
  - Match detections to tracks by IoU > 0.3
  - Unmatched detection → new candidate track
  - Unmatched track → age +1
- Display rules:
  - Show a track only after `hit_count ≥ 3` (confirmed for 3 consecutive frames)
  - Remove a track after `max_age ≥ 10` unmatched frames
- Emit **tracks** (with stable IDs) instead of raw per-frame detections.

**Frontend side:** render with stable IDs → motion vectors possible, no flicker.

**Payoff:** boxes feel "anchored" to vehicles, user can count unique targets, false positives suppressed naturally.

---

### Phase 5 — Flight guidance (help user find vehicles)

User currently wanders around not knowing where vehicles are placed.

Options (pick one or more):

1. **Mini-map dots:** parse vehicle positions from `military_training.sdf` → plot on Leaflet map.
2. **"Next target" HUD arrow:** main-panel arrow pointing at nearest undetected formation.
3. **Auto-survey demo button:** new `SURVEY` button flying a preset pattern over all 5 formations (extends existing `demo_roundtrip` for the military world).

**Recommendation:** mini-map dots (~30 min) + SURVEY button (~1 hr, waypoint math).

---

### Phase 6 — Performance polish (only if needed)

Current headroom is huge (7 ms budget out of 200 ms/frame at 5 fps), but for completeness:

- **Async detection:** decouple capture thread from inference thread so slow frames never block camera streaming.
- **Frame skip on backlog:** if inference falls behind, drop frames instead of queueing.
- **Batch inference:** run both cameras in one ONNX call (`batch=2`) — ~30 % faster than two sequential calls.
- **INT8 TFLite fallback:** use `best_int8.tflite` (3.2 MB) for even lower latency.

Skip this phase unless something feels slow.

---

## Recommended order (quickest path to "wow")

| Order | Phase | Est. effort | Payoff |
|-------|-------|-------------|--------|
| 1 | Phase 1 — verify (fly now) | 5 min | confirms pipeline |
| 2 | Phase 2B — enlarge down-cam overlay | 20 min | immediate visibility |
| 3 | Phase 3 — detection info panel | 1 h | situational awareness |
| 4 | Phase 4 — temporal smoothing | 1.5 h | kills flicker → feels real |
| 5 | Phase 2A — chase-cam detection | 1 h | boxes on the main view |
| 6 | Phase 5 — mini-map dots + SURVEY button | 1.5 h | flight guidance |
| 7 | Phase 6 — performance polish | (skip unless needed) | — |

---

## Files that will change

| File | Change |
|------|--------|
| `backend/camera_bridge.py` | Add chase-cam detection + separate SocketIO event |
| `backend/detector.py` | Add `SimpleTracker` class |
| `backend/app.py` | Relay new `detection_results_chase` event |
| `frontend/templates/index.html` | New canvas for chase cam, detection panel, layout tweaks |
| `scripts/parse_military_vehicles.py` | (new) one-shot utility: SDF → JSON vehicle positions |

---

## Open questions

1. **Run detection on chase cam?** Accept ~50 % mAP drop for the visual "wow" factor, or keep nadir-only and just enlarge the down-cam overlay?
2. **Priority order** — do tracking (Phase 4) first for the smoothness win, or UI visibility fixes (Phase 2/3) first?
3. **Mini-map vehicle dots** — show them always (cheat sheet), or only after first detection of that class (discovery mode)?

---

## Reference: class color map

| ID | Class | BGR (detector.py) | Hex (frontend) |
|----|-------|-------------------|----------------|
| 0 | tank | (0, 0, 255) | `#ff0000` |
| 1 | apc_ifv | (0, 140, 255) | `#ff8c00` |
| 2 | military_truck | (0, 200, 200) | `#c8c800` |
| 3 | sp_artillery | (80, 80, 255) | `#ff5050` |
| 4 | mlrs | (255, 0, 128) | `#ff0080` |
| 5 | helicopter | (255, 200, 0) | `#00c8ff` |
| 6 | uav | (200, 200, 0) | `#00c8c8` |
