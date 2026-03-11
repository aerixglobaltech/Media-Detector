"""
test_imou_local.py
─────────────────────────────────────────────────────────────────────────────
Test local connectivity to an IMOU (Dahua) camera WITHOUT using the cloud.

PREREQUISITES (do these ONCE on your phone before running this script):
  1. Open the Imou Life app → tap your camera → Settings (gear icon)
  2. Look for "ONVIF" or "Local Network"  → toggle it ON
     - Some models ask you to set a "Device Password" first — set one and
       remember it (this becomes the RTSP password).
  3. Note the camera's local IP shown in the app (if visible).

WHAT THIS SCRIPT DOES:
  Step 1 — Scans your local network to find all active devices
  Step 2 — Checks which devices have RTSP port 554 open
  Step 3 — Tries to pull a video frame from each candidate using OpenCV
  Step 4 — Saves a snapshot image if successful

USAGE:
    python test_imou_local.py
    python test_imou_local.py --ip 192.168.1.50
    python test_imou_local.py --ip 192.168.1.50 --password YOUR_PASSWORD
"""

from __future__ import annotations

import argparse
import socket
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2


# ─── Default credentials ─────────────────────────────────────────────────────
DEFAULT_USER     = "admin"
DEFAULT_PASSWORD = "L2BA7F0F"   # Safety code printed on the camera sticker

# IMOU / Dahua cameras commonly use these RTSP URL patterns
RTSP_PATHS = [
    "/cam/realmonitor?channel=1&subtype=0",        # Dahua / Imou main stream
    "/cam/realmonitor?channel=1&subtype=1",        # Dahua / Imou sub stream
    "/live/ch00_0",                                 # Imou alternate
    "/onvif1",                                      # ONVIF stream 1
    "/",                                            # Generic fallback
]

# Ports to check
RTSP_PORT  = 554
ONVIF_PORT = 80
DAHUA_PORT = 37777


# ─── Network Scanner ─────────────────────────────────────────────────────────

def get_local_subnet() -> str:
    """Detect the local subnet (e.g. '192.168.1')."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    parts = ip.split(".")
    subnet = ".".join(parts[:3])
    print(f"  Your IP address : {ip}")
    print(f"  Scanning subnet : {subnet}.0/24\n")
    return subnet


def check_port(ip: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if the given TCP port is open on ip."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        return True
    except Exception:
        return False


def scan_network(subnet: str) -> list[dict]:
    """
    Scan the entire /24 subnet for devices with camera-like ports open.
    Returns a list of dicts: { ip, ports: [list of open ports] }
    """
    print("  ── Step 1: Scanning network for active devices ──\n")
    candidates = []

    def probe(ip: str) -> dict | None:
        open_ports = []
        for port in (RTSP_PORT, ONVIF_PORT, DAHUA_PORT):
            if check_port(ip, port):
                open_ports.append(port)
        if open_ports:
            return {"ip": ip, "ports": open_ports}
        return None

    with ThreadPoolExecutor(max_workers=100) as pool:
        futures = {pool.submit(probe, f"{subnet}.{i}"): i for i in range(1, 255)}
        for future in as_completed(futures):
            result = future.result()
            if result:
                candidates.append(result)

    # Sort by IP for readability
    candidates.sort(key=lambda c: int(c["ip"].split(".")[-1]))
    return candidates


# ─── RTSP Tester ──────────────────────────────────────────────────────────────

def test_rtsp(ip: str, user: str, password: str) -> str | None:
    """
    Try all known RTSP URL patterns on the given IP.
    Returns the first working RTSP URL, or None.
    """
    for path in RTSP_PATHS:
        url = f"rtsp://{user}:{password}@{ip}:{RTSP_PORT}{path}"
        print(f"    Trying: {url} ... ", end="", flush=True)

        cap = cv2.VideoCapture(url)

        # Wait up to 5 seconds for the stream to open
        start = time.time()
        while not cap.isOpened() and (time.time() - start) < 5:
            time.sleep(0.2)

        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                print("✔ SUCCESS")
                return url
            else:
                print("opened but no frame")
        else:
            cap.release()
            print("timeout")

    return None


def save_snapshot(url: str, filename: str = "imou_snapshot.jpg") -> bool:
    """Capture a single frame from the RTSP URL and save it as a JPEG."""
    cap = cv2.VideoCapture(url)
    start = time.time()
    while not cap.isOpened() and (time.time() - start) < 5:
        time.sleep(0.2)

    if not cap.isOpened():
        cap.release()
        return False

    ok, frame = cap.read()
    cap.release()

    if ok and frame is not None:
        path = os.path.join(os.path.dirname(__file__), filename)
        cv2.imwrite(path, frame)
        print(f"\n  Snapshot saved to: {path}")
        return True
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test local IMOU camera RTSP connectivity")
    parser.add_argument("--ip",       type=str, default=None,            help="Camera IP address (skip network scan)")
    parser.add_argument("--user",     type=str, default=DEFAULT_USER,    help="RTSP username (default: admin)")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="RTSP password / safety code")
    parser.add_argument("--save",     action="store_true",               help="Save a snapshot JPEG on success")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  IMOU Local Camera Connectivity Test")
    print("=" * 60 + "\n")

    # ── Determine which IPs to test ──────────────────────────────────
    if args.ip:
        targets = [{"ip": args.ip, "ports": [RTSP_PORT]}]
        print(f"  Target IP: {args.ip}\n")
    else:
        subnet = get_local_subnet()
        targets = scan_network(subnet)

    if not targets:
        print("  No devices found on the network with camera ports open.")
        print("\n  Possible reasons:")
        print("    1. ONVIF / Local Network is NOT enabled in the Imou Life app")
        print("    2. The camera is on a different subnet or VLAN")
        print("    3. The camera is powered off or disconnected from Wi-Fi")
        print("\n  Fix: Open Imou Life app → Camera Settings → enable ONVIF")
        return

    # ── Show discovered devices ──────────────────────────────────────
    print(f"\n  Found {len(targets)} device(s) with camera-like ports:\n")
    for t in targets:
        port_str = ", ".join(str(p) for p in t["ports"])
        marker = " ← likely camera" if RTSP_PORT in t["ports"] else ""
        print(f"    {t['ip']:>15}   ports: {port_str}{marker}")

    # ── Test RTSP on each candidate ──────────────────────────────────
    print("\n  ── Step 2: Testing RTSP connectivity ──\n")
    working_url = None

    for t in targets:
        ip = t["ip"]
        print(f"  [{ip}]")

        if RTSP_PORT not in t["ports"]:
            print(f"    Port 554 closed — skipping RTSP test\n")
            continue

        url = test_rtsp(ip, args.user, args.password)
        if url:
            working_url = url
            break
        else:
            print(f"    No working RTSP path found on {ip}\n")

    # ── Results ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    if working_url:
        print(f"  ✔ CONNECTED SUCCESSFULLY!\n")
        print(f"  Working RTSP URL:")
        print(f"    {working_url}\n")
        print(f"  Add to your config.py:")
        print(f'    SOURCE = "{working_url}"\n')

        if args.save:
            save_snapshot(working_url)
    else:
        print("  ✘ Could NOT connect to any camera via RTSP.\n")
        print("  Troubleshooting checklist:")
        print("  ─────────────────────────")
        print("  [ ] 1. Open Imou Life app → Camera Settings → Enable ONVIF")
        print("  [ ] 2. Set a 'Device Password' if the app asks for one")
        print("  [ ] 3. Use the Device Password (not safety code) as --password")
        print("  [ ] 4. Check if the camera is on the same Wi-Fi network")
        print("  [ ] 5. Try restarting the camera (unplug power for 10 seconds)")
        print(f"\n  Re-run with a specific IP and password:")
        print(f"    python test_imou_local.py --ip <CAMERA_IP> --password <DEVICE_PASSWORD>")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
