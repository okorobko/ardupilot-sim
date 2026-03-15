#!/usr/bin/env python3
"""UI structure tests for the ArduPilot Drone Simulator dashboard.

Verifies:
  1. HTML structure and required elements
  2. Map uses OpenStreetMap (light theme)
  3. No dark map background override
  4. Telemetry and attitude are side-by-side (CSS grid, not stacked)
  5. Coordinate overlay on map
  6. Console, telemetry, map all in bottom row
  7. Map receives position updates via SocketIO
  8. Coordinate overlay updates with drone position

Usage: python3 scripts/test_ui.py
Requires: Backend running on localhost:5001
"""

import json
import re
import sys
import time
import urllib.request

BASE = "http://localhost:5001"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  -- {detail}")


print("=" * 60)
print("  ArduPilot Simulator — UI Tests")
print("=" * 60)

# Fetch HTML
try:
    html = urllib.request.urlopen(f"{BASE}/", timeout=5).read().decode()
except Exception as e:
    print(f"ERROR: Cannot reach {BASE}: {e}")
    sys.exit(1)

# ── 1. Required HTML elements ────────────────────────────
print("\n[1/8] Required HTML Elements")
required_ids = [
    "three-canvas", "map", "console-log", "attitude-canvas",
    "t-alt", "t-spd", "t-hdg", "t-climb", "t-sats", "t-batt",
    "batt-bar", "hud-alt", "hud-spd", "hud-hdg",
    "dot-conn", "dot-armed", "lbl-mode",
    "btn-demo", "demo-status", "takeoff-alt",
    "map-coords", "mode-select",
]
for eid in required_ids:
    check(f'Element #{eid} exists', f'id="{eid}"' in html, "not found in HTML")


# ── 2. Map uses OpenStreetMap (light tiles) ──────────────
print("\n[2/8] Map Tile Provider")
check("Uses OpenStreetMap tiles", "tile.openstreetmap.org" in html, "OSM tile URL not found")
check("NOT using CartoDB dark tiles", "cartocdn.com/dark" not in html, "dark CartoDB tiles still present")


# ── 3. No dark map background ────────────────────────────
print("\n[3/8] Map Theme")
# Check leaflet-container background is NOT dark
dark_bg_match = re.search(r'leaflet-container\s*\{[^}]*background:\s*(#[0-9a-fA-F]+)', html)
if dark_bg_match:
    bg_color = dark_bg_match.group(1).lower()
    is_light = bg_color not in ("#0f1923", "#000000", "#111111", "#1a1a1a", "#0a0a0a")
    check("Leaflet background is light", is_light, f"background={bg_color}")
else:
    check("Leaflet background not forced dark", True)


# ── 4. Telemetry layout — grid with attitude side-by-side ─
print("\n[4/8] Telemetry Layout (grid, not stacked)")
# Check bottom-telem uses CSS grid
telem_css = re.search(r'\.bottom-telem\s*\{([^}]+)\}', html)
if telem_css:
    telem_style = telem_css.group(1)
    check("bottom-telem uses display:grid", "display: grid" in telem_style or "display:grid" in telem_style, telem_style[:80])
    check("grid-template-columns defined", "grid-template-columns" in telem_style, "no columns")
else:
    check("bottom-telem CSS found", False, "not found")

# Check attitude-cell spans grid column
check("attitude-cell in grid column 3", "grid-column: 3" in html, "attitude not in column 3")
check("attitude-cell spans rows 1-4", "grid-row: 1 / 4" in html, "attitude doesn't span rows")


# ── 5. Coordinate overlay on map ────────────────────────
print("\n[5/8] Map Coordinate Overlay")
check("map-coords div exists", 'id="map-coords"' in html)
check("map-coords has CSS class", 'class="map-coords"' in html)
# Check that updateMap writes to map-coords
check("JS updates map-coords", "map-coords" in html and "textContent" in html and "toFixed(6)" in html,
      "coordinate update code not found")


# ── 6. Bottom row layout ────────────────────────────────
print("\n[6/8] Bottom Row Layout")
# Check panel-bottom contains console, telem, map in order
bottom_section = html[html.find('panel-bottom'):html.find('</div>\n</div>\n\n<script')]
check("Bottom has console", "bottom-console" in bottom_section, "missing")
check("Bottom has telemetry", "bottom-telem" in bottom_section, "missing")
check("Bottom has map", "bottom-map" in bottom_section, "missing")

# Check order: console before telem before map
console_pos = html.find("bottom-console")
telem_pos = html.find("bottom-telem")
map_pos = html.find("bottom-map")
check("Order: console -> telem -> map",
      0 < console_pos < telem_pos < map_pos,
      f"positions: console={console_pos}, telem={telem_pos}, map={map_pos}")


# ── 7. Map receives position via SocketIO ────────────────
print("\n[7/8] Map Position Updates (SocketIO)")
try:
    import socketio
    sio = socketio.Client()
    positions = []

    @sio.on("telemetry")
    def on_t(data):
        if data.get("position"):
            positions.append(data["position"])

    sio.connect(BASE)
    time.sleep(3)
    sio.disconnect()

    check("Telemetry positions received", len(positions) > 5, f"count={len(positions)}")
    if positions:
        p = positions[-1]
        check("Position lat != 0", p["lat"] != 0, f"lat={p['lat']}")
        check("Position lon != 0", p["lon"] != 0, f"lon={p['lon']}")
        check("Position lat is valid float", isinstance(p["lat"], float), type(p["lat"]))
        check("Position lon is valid float", isinstance(p["lon"], float), type(p["lon"]))
except Exception as e:
    check("SocketIO position test", False, str(e))


# ── 8. Coordinate overlay updates ────────────────────────
print("\n[8/8] Coordinate Overlay JS Logic")
# Verify the JS code that updates map-coords is correct
check("updateMap writes lat.toFixed(6)", "lat.toFixed(6)" in html)
check("updateMap writes lon.toFixed(6)", "lon.toFixed(6)" in html)
check("updateMap writes alt", "alt.toFixed(1)" in html and "map-coords" in html)


# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
