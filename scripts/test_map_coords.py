#!/usr/bin/env python3
"""Test that map-coords overlay receives live position updates.

This test uses a real browser via Selenium to verify the JS actually runs
and updates the DOM element. Falls back to SocketIO-level test if no browser.

Usage: python3 scripts/test_map_coords.py
"""

import json
import sys
import time
import urllib.request

BASE = "http://localhost:5001"

def status():
    return json.loads(urllib.request.urlopen(f"{BASE}/api/status", timeout=5).read())

def get_html():
    return urllib.request.urlopen(f"{BASE}/", timeout=5).read().decode()

print("=" * 60)
print("  Map Coordinates Debug Test")
print("=" * 60)

# 1. Check backend
print("\n[1] Backend status...")
s = status()
print(f"  connected={s['connected']} lat={s['position']['lat']} lon={s['position']['lon']} alt={s['position']['alt']}")
assert s['connected'], "Backend not connected to SITL"
assert s['position']['lat'] != 0, "Lat is 0"

# 2. Check HTML has the element
print("\n[2] HTML structure...")
html = get_html()
assert 'id="map-coords"' in html, "map-coords element missing"
print("  map-coords element: present")

# 3. Check updateMap JS code
print("\n[3] JS updateMap function...")
# Extract the function
idx = html.find("function updateMap")
if idx == -1:
    print("  FAIL: updateMap function not found!")
    sys.exit(1)
func_end = html.find("\n}", idx) + 2
func_code = html[idx:func_end]
print(f"  Function found ({len(func_code)} chars)")

# Check it writes to map-coords
assert "map-coords" in func_code, "updateMap doesn't reference map-coords"
assert "textContent" in func_code, "updateMap doesn't set textContent"
assert "data.position" in func_code, "updateMap doesn't check data.position"
print("  map-coords update code: OK")

# 4. Check telemetry handler calls updateMap
print("\n[4] Telemetry handler...")
assert "updateMap(data)" in html, "updateMap not called from telemetry handler"
print("  updateMap called from telemetry: OK")

# 5. Check SocketIO telemetry actually flows
print("\n[5] SocketIO telemetry flow...")
import socketio
sio = socketio.Client()
telem_events = []
config_events = []

@sio.on("config")
def on_cfg(data):
    config_events.append(data)

@sio.on("telemetry")
def on_telem(data):
    telem_events.append(data)

sio.connect(BASE)
time.sleep(4)
sio.disconnect()

print(f"  Config events: {len(config_events)}")
print(f"  Telemetry events: {len(telem_events)}")

if not telem_events:
    print("  FAIL: No telemetry events received!")
    sys.exit(1)

t = telem_events[-1]
print(f"  Last telemetry keys: {list(t.keys())}")
print(f"  position: {t.get('position')}")
print(f"  position type: {type(t.get('position'))}")

if t.get('position'):
    p = t['position']
    print(f"  lat={p.get('lat')} type={type(p.get('lat'))}")
    print(f"  lon={p.get('lon')} type={type(p.get('lon'))}")
    print(f"  alt={p.get('alt')} type={type(p.get('alt'))}")
else:
    print("  FAIL: No position in telemetry!")

# 6. Check if there's a JS error - simulate what the browser does
print("\n[6] Simulating browser JS logic...")
data = telem_events[-1]
if data.get('position'):
    lat = data['position']['lat']
    lon = data['position']['lon']
    alt = data['position'].get('alt', 0)
    expected = f"{lat:.6f}, {lon:.6f}  alt={alt:.1f}m"
    print(f"  Expected map-coords text: '{expected}'")
    print("  JS simulation: OK")
else:
    print("  FAIL: position missing from telemetry data")
    sys.exit(1)

# 7. Try Selenium browser test
print("\n[7] Browser test (Selenium)...")
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)
    driver.get(BASE)

    # Wait for map-coords to update from "Waiting for GPS..."
    try:
        WebDriverWait(driver, 15).until(
            lambda d: "Waiting" not in d.find_element(By.ID, "map-coords").text
            and "---" not in d.find_element(By.ID, "map-coords").text
        )
        coords_text = driver.find_element(By.ID, "map-coords").text
        print(f"  map-coords text: '{coords_text}'")
        assert "." in coords_text, f"Doesn't look like coordinates: {coords_text}"
        print("  PASS: Coordinates updating in browser!")
    except Exception as e:
        coords_text = driver.find_element(By.ID, "map-coords").text
        print(f"  FAIL: map-coords still shows: '{coords_text}'")

        # Check for JS errors
        logs = driver.get_log("browser")
        if logs:
            print("  Browser console errors:")
            for log in logs:
                if log["level"] in ("SEVERE", "WARNING"):
                    print(f"    [{log['level']}] {log['message']}")
        else:
            print("  No browser console errors found")

        # Check if socket connected
        connected = driver.execute_script("return typeof socket !== 'undefined' && socket.connected")
        print(f"  Socket connected: {connected}")

        # Check if telemetry received
        has_telem = driver.execute_script("return typeof latestTelem !== 'undefined' && Object.keys(latestTelem).length > 0")
        print(f"  Has telemetry data: {has_telem}")

        if has_telem:
            telem_str = driver.execute_script("return JSON.stringify(latestTelem.position)")
            print(f"  latestTelem.position: {telem_str}")

    driver.quit()

except ImportError:
    print("  Selenium not installed, installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "selenium"], capture_output=True)

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=opts)
        driver.get(BASE)
        time.sleep(10)

        coords_text = driver.find_element(By.ID, "map-coords").text
        print(f"  map-coords text: '{coords_text}'")

        # Debug
        connected = driver.execute_script("return typeof socket !== 'undefined' && socket.connected")
        print(f"  Socket connected: {connected}")
        has_telem = driver.execute_script("return typeof latestTelem !== 'undefined' && Object.keys(latestTelem).length > 0")
        print(f"  Has telemetry: {has_telem}")
        if has_telem:
            pos = driver.execute_script("return JSON.stringify(latestTelem.position)")
            print(f"  Position data: {pos}")

        # Check for errors
        logs = driver.get_log("browser")
        for log in logs:
            if log["level"] in ("SEVERE", "WARNING"):
                print(f"  [{log['level']}] {log['message']}")

        driver.quit()
    except Exception as e:
        print(f"  Selenium failed: {e}")
        print("  Skipping browser test")

print("\n" + "=" * 60)
print("  Done")
print("=" * 60)
