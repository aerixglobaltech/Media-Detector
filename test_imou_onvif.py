"""
test_imou_onvif.py
─────────────────────────────────────────────────────────────────────────────
Test ONVIF connectivity to IMOU (Dahua) cameras on the local network.

ONVIF is an industry standard protocol for IP cameras that provides:
  - Device discovery and identification
  - Camera capabilities and configuration
  - Stream URI retrieval (RTSP URLs)
  - PTZ (Pan/Tilt/Zoom) control (if supported)
  - Event subscriptions

PREREQUISITES:
  1. Enable ONVIF in the Imou Life app (Camera Settings → ONVIF → ON)
  2. Know the camera's local IP (use test_imou_local.py to scan)

USAGE:
    python test_imou_onvif.py
    python test_imou_onvif.py --ip 192.168.1.38
    python test_imou_onvif.py --ip 192.168.1.38 --password YOUR_PASSWORD
"""

from __future__ import annotations

import argparse
import sys
import time

from onvif import ONVIFCamera

# ─── Default credentials ─────────────────────────────────────────────────────
DEFAULT_USER     = "admin"
DEFAULT_PASSWORD = "L2BA7F0F"
ONVIF_PORT       = 80        # Standard ONVIF port (some cameras use 8080)

# IPs discovered by test_imou_local.py
KNOWN_CAMERA_IPS = ["192.168.1.38", "192.168.1.108"]


def test_onvif(ip: str, port: int, user: str, password: str) -> bool:
    """
    Connect to a camera via ONVIF and retrieve device info + stream URIs.
    Returns True if connection was successful.
    """
    print(f"\n  ── Testing ONVIF: {ip}:{port} ──\n")

    # ── 1. Connect to the camera ─────────────────────────────────────
    print(f"  [1/5] Connecting to ONVIF device ... ", end="", flush=True)
    try:
        cam = ONVIFCamera(ip, port, user, password)
        print("OK")
    except Exception as e:
        print(f"FAILED\n        Error: {e}")
        # Try alternate port 8080
        if port != 8080:
            print(f"\n  Retrying on port 8080 ...")
            return test_onvif(ip, 8080, user, password)
        return False

    # ── 2. Get Device Information ────────────────────────────────────
    print(f"  [2/5] Fetching device information ... ", end="", flush=True)
    try:
        device_service = cam.create_devicemgmt_service()
        device_info    = device_service.GetDeviceInformation()
        print("OK")
        print(f"        Manufacturer : {device_info.Manufacturer}")
        print(f"        Model        : {device_info.Model}")
        print(f"        Firmware     : {device_info.FirmwareVersion}")
        print(f"        Serial       : {device_info.SerialNumber}")
        print(f"        Hardware ID  : {device_info.HardwareId}")
    except Exception as e:
        print(f"FAILED\n        Error: {e}")

    # ── 3. Get Network Interfaces ────────────────────────────────────
    print(f"  [3/5] Fetching network interfaces ... ", end="", flush=True)
    try:
        net_interfaces = device_service.GetNetworkInterfaces()
        print("OK")
        for iface in net_interfaces:
            if hasattr(iface, "IPv4") and iface.IPv4:
                config = iface.IPv4.Config
                if hasattr(config, "Manual") and config.Manual:
                    for addr in config.Manual:
                        print(f"        IPv4 Address : {addr.Address}")
                elif hasattr(config, "DHCP") and config.DHCP:
                    print(f"        IPv4 (DHCP)  : enabled")
    except Exception as e:
        print(f"FAILED\n        Error: {e}")

    # ── 4. Get Media Profiles & Stream URIs ──────────────────────────
    print(f"  [4/5] Fetching media profiles ... ", end="", flush=True)
    stream_uris = []
    try:
        media_service = cam.create_media_service()
        profiles      = media_service.GetProfiles()
        print(f"OK  ({len(profiles)} profile(s) found)")

        for i, profile in enumerate(profiles):
            name  = profile.Name
            token = profile.token
            print(f"\n        Profile [{i}]: {name} (token: {token})")

            # Video encoder config
            if hasattr(profile, "VideoEncoderConfiguration") and profile.VideoEncoderConfiguration:
                vec = profile.VideoEncoderConfiguration
                encoding   = vec.Encoding if hasattr(vec, "Encoding") else "?"
                resolution = f"{vec.Resolution.Width}x{vec.Resolution.Height}" if hasattr(vec, "Resolution") else "?"
                print(f"          Encoding   : {encoding}")
                print(f"          Resolution : {resolution}")

            # Get stream URI
            try:
                # Method 1: Named parameters (works with most onvif-zeep versions)
                stream_setup = media_service.create_type("GetStreamUri")
                stream_setup.StreamSetup = {
                    "Stream":    "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                }
                stream_setup.ProfileToken = token
                uri_response = media_service.GetStreamUri(stream_setup)
                uri = uri_response.Uri
                stream_uris.append(uri)
                print(f"          Stream URI : {uri}")
            except Exception as e1:
                # Method 2: Direct keyword args
                try:
                    uri_response = media_service.GetStreamUri({
                        "StreamSetup": {
                            "Stream":    "RTP-Unicast",
                            "Transport": {"Protocol": "RTSP"},
                        },
                        "ProfileToken": token,
                    })
                    uri = uri_response.Uri
                    stream_uris.append(uri)
                    print(f"          Stream URI : {uri}")
                except Exception as e2:
                    print(f"          Stream URI : FAILED ({e2})")

    except Exception as e:
        print(f"FAILED\n        Error: {e}")

    # ── 5. Get Snapshot URI ──────────────────────────────────────────
    print(f"\n  [5/5] Fetching snapshot URI ... ", end="", flush=True)
    try:
        if profiles:
            snap_uri = media_service.GetSnapshotUri(profiles[0].token)
            print(f"OK")
            print(f"        Snapshot URI : {snap_uri.Uri}")
    except Exception as e:
        print(f"FAILED\n        Error: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    if stream_uris:
        print(f"  ONVIF connection SUCCESSFUL for {ip}")
        print(f"\n  Available stream URIs:")
        for uri in stream_uris:
            print(f"    {uri}")
        return True
    else:
        print(f"  ONVIF connected but no stream URIs retrieved for {ip}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test ONVIF connectivity to IMOU cameras")
    parser.add_argument("--ip",       type=str, default=None,             help="Camera IP (default: test all known IPs)")
    parser.add_argument("--port",     type=int, default=ONVIF_PORT,       help="ONVIF port (default: 80)")
    parser.add_argument("--user",     type=str, default=DEFAULT_USER,     help="Username (default: admin)")
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD, help="Password / safety code")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  IMOU Camera — ONVIF Connectivity Test")
    print("=" * 60)

    ips = [args.ip] if args.ip else KNOWN_CAMERA_IPS
    results = {}

    for ip in ips:
        try:
            results[ip] = test_onvif(ip, args.port, args.user, args.password)
        except Exception as e:
            print(f"\n  Unexpected error on {ip}: {e}")
            results[ip] = False

    # ── Final Report ─────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  ONVIF Test Results")
    print("=" * 60 + "\n")

    for ip, ok in results.items():
        status = "PASS" if ok else "FAIL"
        icon   = "+" if ok else "x"
        print(f"  [{icon}]  {ip:>15}  →  {status}")

    any_pass = any(results.values())
    if not any_pass:
        print("\n  No cameras responded to ONVIF.")
        print("\n  Troubleshooting:")
        print("    1. Enable ONVIF in Imou Life app → Camera Settings → ONVIF → ON")
        print("    2. Some models require a separate 'Device Password' for ONVIF")
        print("    3. Try port 8080 instead:  python test_imou_onvif.py --port 8080")
        print("    4. The password might be different from the safety code")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
