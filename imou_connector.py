"""
imou_connector.py
─────────────────────────────────────────────────────────────────────────────
Imou Open Platform (Easy4ip) API connector.
Built from official docs: https://open.imoulife.com

What it does:
  1. Auto-detects the working Imou datacenter for your region
  2. Authenticates → gets access token
  3. Lists all cameras on the account
  4. Fetches the RTSP live stream URL for a chosen camera
  5. Lets you copy the URL directly into config.py

Usage:
    python imou_connector.py
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
import logging

import requests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ⚠  YOUR CREDENTIALS  (regenerate secret if this file is ever shared)
# ─────────────────────────────────────────────────────────────────────────────
APP_ID     = "lcfdd42a0e640d426e"
APP_SECRET = "37c55ce8752a427f876b3baff50288"
# ─────────────────────────────────────────────────────────────────────────────

# Official Imou datacenter endpoints (from open.imoulife.com docs)
_DATACENTERS = [
    "https://openapi-sg.easy4ip.com",   # South East Asia / Singapore ← India uses this
    "https://openapi-fk.easy4ip.com",   # Central Europe / Frankfurt
    "https://openapi-or.easy4ip.com",   # Western America / Oregon
    "https://openapi.easy4ip.com",      # Generic fallback
]


def _find_working_datacenter() -> str | None:
    """Try each datacenter and return the first one that responds."""
    print("  Detecting your Imou datacenter …")
    for url in _DATACENTERS:
        try:
            r = requests.get(url, timeout=6)
            print(f"  ✓  {url}")
            return url
        except Exception:
            print(f"  ✗  {url}")
    return None


# ─────────────────────────────────────────────────────────────────────────────


def _get_device_status(dev: dict) -> str:
    """
    Imou cameras of different models report online/offline under different
    field names. Checks all known variants and returns an emoji string.

    Known field names: status, deviceStatus, onlineStatus, online,
                       isOnline, netStatus, linkStatus, liveStatus
    """
    STATUS_KEYS   = ("status", "deviceStatus", "onlineStatus",
                     "online", "isOnline", "netStatus",
                     "linkStatus", "liveStatus")
    ONLINE_VALUES = {"1", "true", "on", "online", "connected", "normal"}

    for key in STATUS_KEYS:
        val = dev.get(key)
        if val is None:
            continue
        if str(val).lower() in ONLINE_VALUES:
            return "🟢 Online"
        if isinstance(val, (int, float)) and int(val) == 1:
            return "🟢 Online"
        return "🔴 Offline"

    # No status field found — still let the user try selecting it
    return "⚪ Unknown (try anyway)"


class ImouAPI:
    """Imou Open Platform API client."""


    def __init__(self, app_id: str, app_secret: str, base_url: str):
        self.app_id     = app_id
        self.app_secret = app_secret
        self.base_url   = base_url.rstrip("/")
        self._token: str | None = None

    # ── Signature helpers ─────────────────────────────────────────────

    @staticmethod
    def _nonce() -> str:
        return uuid.uuid4().hex[:16]

    @staticmethod
    def _ts() -> int:
        return int(time.time())

    def _sign(self, ts: int, nonce: str) -> str:
        """
        sign = MD5("time:{ts},nonce:{nonce},appSecret:{secret}")
        Returns 32-char lowercase hex string.
        """
        raw = f"time:{ts},nonce:{nonce},appSecret:{self.app_secret}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _system_block(self) -> dict:
        ts    = self._ts()
        nonce = self._nonce()
        return {
            "ver":   "1.0",
            "appId": self.app_id,
            "time":  ts,
            "nonce": nonce,
            "sign":  self._sign(ts, nonce),
        }

    # ── HTTP helper ───────────────────────────────────────────────────

    def _post(self, path: str, params: dict | None = None) -> dict:
        body: dict = {
            "id":     uuid.uuid4().hex,
            "system": self._system_block(),
            "params": dict(params or {}),
        }
        if self._token:
            body["params"]["token"] = self._token

        url  = f"{self.base_url}/openapi{path}"
        resp = requests.post(url, json=body, timeout=15)
        resp.raise_for_status()

        data   = resp.json()
        result = data.get("result", {})
        code   = result.get("code", "")

        if str(code) != "0":
            raise RuntimeError(f"[{code}] {result.get('msg', 'unknown error')}")

        return result.get("data", {})

    # ── Public API calls ──────────────────────────────────────────────

    def get_token(self) -> str:
        """Authenticate and cache the access token."""
        data         = self._post("/accessToken")
        self._token  = data["accessToken"]
        expires      = data.get("expireTime", "?")
        log.info("Token obtained (expires %s s)", expires)
        return self._token

    def list_devices(self) -> list[dict]:
        """
        Return all cameras linked to this account.
        Uses /listDeviceDetailsByPage (correct Imou endpoint name).
        """
        data    = self._post("/listDeviceDetailsByPage", {"pageSize": 50, "page": 1})
        devices = data.get("deviceList", [])
        return devices

    def bind_live(self, device_id: str, channel_id: str = "0", stream_id: int = 0) -> None:
        """
        Step 1 of getting a live URL — binds/activates the stream.
        streamId: 0 = HD Main stream, 1 = SD Sub stream
        """
        self._post("/bindDeviceLive", {
            "deviceId":  device_id,
            "channelId": channel_id,
            "streamId":  stream_id,
        })

    def get_live_stream_info(self, device_id: str, channel_id: str = "0") -> dict:
        """
        Step 2 of getting a live URL — retrieves HLS / RTSP URLs.
        Returns dict with 'hls', 'rtmp', 'rtsp' keys.
        """
        return self._post("/getLiveStreamInfo", {
            "deviceId":  device_id,
            "channelId": channel_id,
        })

    def get_rtsp(self, device_id: str, channel_id: str = "0") -> str | None:
        """
        Convenience: bind + get stream info and return a playable URL.
        Tries RTSP first, then HLS, then any URL field in the response.
        """
        try:
            self.bind_live(device_id, channel_id)
        except Exception as e:
            log.warning("bind_live optional step failed: %s", e)

        info = self.get_live_stream_info(device_id, channel_id)

        # Helper to scan any dict/list for URL-like values
        def _extract_url(obj) -> str | None:
            if isinstance(obj, str) and obj.startswith("http"):
                return obj
            if isinstance(obj, dict):
                # Priority order: rtsp > hls > rtmp > url > any string starting with http
                for key in ("rtsp", "hls", "rtmp", "url", "flv", "ws"):
                    if obj.get(key):
                        return obj[key]
                # Recurse into nested dicts
                for v in obj.values():
                    result = _extract_url(v)
                    if result:
                        return result
            if isinstance(obj, list):
                for item in obj:
                    result = _extract_url(item)
                    if result:
                        return result
            return None

        return _extract_url(info)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    print("\n" + "=" * 58)
    print("  Imou Open Platform  –  Camera Connector")
    print("=" * 58)

    # 0. Find working datacenter
    base_url = _find_working_datacenter()
    if not base_url:
        print("\n  ✗  Cannot reach any Imou datacenter. Check internet connection.")
        return

    api = ImouAPI(APP_ID, APP_SECRET, base_url)

    # 1. Authenticate
    print("\n  Authenticating …")
    try:
        api.get_token()
        print("  ✓  Token obtained successfully")
    except Exception as e:
        print(f"  ✗  Authentication failed: {e}")
        print("\n  Tip: Regenerate your App Secret at open.imoulife.com if expired.")
        return

    # 2. List cameras
    print("\n  Fetching camera list …")
    try:
        devices = api.list_devices()
    except Exception as e:
        print(f"  ✗  Could not list cameras: {e}")
        return

    if not devices:
        print("  ✗  No cameras linked to this App ID.")
        print("\n  Fix: Log into open.imoulife.com → your App → Bind Device")
        print("       Enter the serial number from your camera sticker.")
        return

    print(f"\n  Found {len(devices)} camera(s):\n")
    for i, dev in enumerate(devices):
        status_str = _get_device_status(dev)
        name   = (dev.get("name") or dev.get("deviceName") or "Unnamed")
        dev_id = dev.get("deviceId") or dev.get("deviceID") or "?"
        print(f"  [{i}]  {name}  ({status_str})")
        print(f"        DeviceID: {dev_id}")

    # 3. Pick a camera
    try:
        idx    = int(input("\n  Enter camera number: "))
        chosen = devices[idx]
    except (ValueError, IndexError):
        print("  ✗  Invalid selection.")
        return

    device_id  = chosen.get("deviceId") or chosen.get("deviceID")
    channel_id = chosen.get("channelId") or "0"
    print(f"\n  Getting stream URL for device: {device_id} …")

    # 4. Get RTSP URL
    try:
        rtsp_url = api.get_rtsp(device_id, channel_id)
    except Exception as e:
        print(f"  ✗  Could not get stream URL: {e}")
        print("\n  Make sure RTSP/ONVIF is enabled in the Imou Life app.")
        return

    print("\n" + "─" * 58)
    if rtsp_url:
        print(f"  ✓  Stream URL:\n\n     {rtsp_url}\n")
        print("  Add to config.py:")
        print(f'     SOURCE = "{rtsp_url}"')
    else:
        print("  ✗  No stream URL returned. Try enabling RTSP on the camera.")
        print("  Raw stream info dumped to console above (check logs).")
    print("─" * 58 + "\n")


if __name__ == "__main__":
    main()
