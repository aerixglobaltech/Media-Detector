from __future__ import annotations

import concurrent.futures
import ipaddress
import logging
import os
import re
import socket
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import cv2
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

log = logging.getLogger("camera_discovery")

_cache_lock = threading.Lock()
_detected_cache: list[dict] = []
_detected_cache_ts: float = 0.0

_COMMON_CAMERA_PORTS = (554, 8554, 80, 81, 8000, 8080, 8899)
_SCAN_TTL_SEC = 60.0
_SOCKET_TIMEOUT = 0.25
_MAX_WORKERS = 64
_MAX_HOSTS_PER_SUBNET = int(os.environ.get("CAMERA_SCAN_MAX_HOSTS", "254"))
_MAX_WEBCAM_INDEX = int(os.environ.get("CAMERA_SCAN_MAX_WEBCAM_INDEX", "8"))
_ONVIF_DISCOVERY_TIMEOUT = float(os.environ.get("CAMERA_ONVIF_DISCOVERY_TIMEOUT_SEC", "1.5"))
_ONVIF_DISCOVERY_PORT = 3702
_ONVIF_DISCOVERY_ADDR = "239.255.255.250"
_RTSP_PATH_HINTS = (
    "/Streaming/Channels/101",
    "/cam/realmonitor?channel=1&subtype=0",
    "/h264/ch1/main/av_stream",
    "/live/ch00_0",
    "/stream1",
)


def _is_private_ipv4(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def _get_local_private_ipv4s() -> list[str]:
    addresses: set[str] = set()

    try:
        host_info = socket.gethostbyname_ex(socket.gethostname())
        for ip in host_info[2]:
            if _is_private_ipv4(ip):
                addresses.add(ip)
    except socket.gaierror as exc:
        log.warning("Hostname lookup failed for camera scan: %s", exc)

    # UDP connect does not require internet, but gives active outbound interface IP.
    udp_targets = [("8.8.8.8", 80), ("1.1.1.1", 80)]
    for host, port in udp_targets:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((host, port))
            ip = sock.getsockname()[0]
            if _is_private_ipv4(ip):
                addresses.add(ip)
        except OSError:
            continue
        finally:
            sock.close()

    return sorted(addresses)


def _detect_system_cameras() -> list[dict]:
    cameras: list[dict] = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

    for index in range(_MAX_WEBCAM_INDEX + 1):
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if not ok:
                continue
            cameras.append(
                {
                    "id": f"det_sys_{index}",
                    "name": f"System Camera #{index}",
                    "type": "system",
                    "source": "local",
                    "ip": None,
                    "port": None,
                    "protocol": "direct",
                    "status": "🟢 Available",
                }
            )
        finally:
            cap.release()

    return cameras


def _probe_open_port(ip: str, port: int) -> tuple[str, int] | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_SOCKET_TIMEOUT)
    try:
        result = sock.connect_ex((ip, port))
        if result == 0:
            return ip, port
        return None
    except OSError:
        return None
    finally:
        sock.close()


def _scan_subnet_for_camera_ports(local_ip: str) -> dict[str, set[int]]:
    network = ipaddress.ip_network(f"{local_ip}/24", strict=False)
    host_limit = max(1, min(_MAX_HOSTS_PER_SUBNET, 254))
    hosts = [str(host) for host in network.hosts() if str(host) != local_ip][:host_limit]

    open_ports: dict[str, set[int]] = {}
    futures: list[concurrent.futures.Future] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        for ip in hosts:
            for port in _COMMON_CAMERA_PORTS:
                futures.append(executor.submit(_probe_open_port, ip, port))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if not result:
                continue
            ip, port = result
            if ip not in open_ports:
                open_ports[ip] = set()
            open_ports[ip].add(port)

    return open_ports


def _build_onvif_probe_message() -> bytes:
    message_id = f"uuid:{uuid.uuid4()}"
    payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>{message_id}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>"""
    return payload.encode("utf-8")


def _extract_xaddrs(xml_text: str) -> list[str]:
    matches = re.findall(r"<[^>]*XAddrs[^>]*>(.*?)</[^>]*XAddrs>", xml_text, flags=re.IGNORECASE | re.DOTALL)
    xaddrs: list[str] = []
    for raw in matches:
        for part in raw.split():
            if part.startswith(("http://", "https://")):
                xaddrs.append(part.strip())
    return xaddrs


def _host_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        return parsed.hostname
    except ValueError:
        return None


def _find_text_any_ns(root: ET.Element, tag_name: str) -> str:
    for elem in root.iter():
        if elem.tag.split("}")[-1] == tag_name and elem.text:
            return elem.text.strip()
    return ""


def fetch_onvif_device_info(
    ip: str,
    port: int = 80,
    username: str = "",
    password: str = "",
    timeout_sec: float = 3.0,
) -> dict:
    """
    Query ONVIF GetDeviceInformation from /onvif/device_service.
    Uses digest auth first, then basic auth fallback.
    """
    endpoint = f"http://{ip}:{port}/onvif/device_service"
    payload = """<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
  <s:Body>
    <tds:GetDeviceInformation xmlns:tds="http://www.onvif.org/ver10/device/wsdl"/>
  </s:Body>
</s:Envelope>"""
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
    }

    auth_methods = [HTTPDigestAuth(username, password), HTTPBasicAuth(username, password)]
    last_status = None
    last_error = None

    for auth in auth_methods:
        try:
            resp = requests.post(endpoint, data=payload, headers=headers, auth=auth, timeout=timeout_sec)
        except requests.RequestException as exc:
            last_error = str(exc)
            continue

        last_status = resp.status_code
        if resp.status_code == 401:
            continue
        if resp.status_code >= 400:
            return {"success": False, "error": f"ONVIF request failed: HTTP {resp.status_code}"}

        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            return {"success": False, "error": f"Invalid ONVIF XML response: {exc}"}

        manufacturer = _find_text_any_ns(root, "Manufacturer")
        model = _find_text_any_ns(root, "Model")
        firmware = _find_text_any_ns(root, "FirmwareVersion")
        serial = _find_text_any_ns(root, "SerialNumber")
        hardware = _find_text_any_ns(root, "HardwareId")

        if not (manufacturer or model):
            return {"success": False, "error": "ONVIF response missing manufacturer/model"}

        display_name = " ".join(part for part in [manufacturer, model] if part).strip()
        return {
            "success": True,
            "manufacturer": manufacturer,
            "model": model,
            "firmware": firmware,
            "serial": serial,
            "hardware_id": hardware,
            "display_name": display_name,
        }

    if last_status == 401:
        return {"success": False, "error": "Unauthorized (401). Check camera username/password."}
    if last_error:
        return {"success": False, "error": last_error}
    return {"success": False, "error": "ONVIF request failed"}


def _detect_onvif_cameras() -> list[dict]:
    probe = _build_onvif_probe_message()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.settimeout(_ONVIF_DISCOVERY_TIMEOUT)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    responses: dict[str, dict] = {}
    start = time.time()
    try:
        sock.sendto(probe, (_ONVIF_DISCOVERY_ADDR, _ONVIF_DISCOVERY_PORT))
        while (time.time() - start) < _ONVIF_DISCOVERY_TIMEOUT:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                break
            except OSError:
                break

            text = data.decode("utf-8", errors="ignore")
            xaddrs = _extract_xaddrs(text)
            if not xaddrs:
                continue
            host = _host_from_url(xaddrs[0])
            key = host or xaddrs[0]
            parsed = urlparse(xaddrs[0])
            onvif_port = parsed.port or (443 if parsed.scheme == "https" else 80)
            responses[key] = {
                "id": f"det_onvif_{key.replace(':', '_')}",
                "name": f"ONVIF Camera ({host or 'Unknown Host'})",
                "type": "network",
                "source": "onvif",
                "ip": host,
                "port": onvif_port,
                "onvif_port": onvif_port,
                "protocol": "onvif",
                "xaddrs": xaddrs,
                "status": "🟢 Discovered",
            }
    finally:
        sock.close()

    return list(responses.values())


def _port_to_protocol(port: int) -> str:
    if port in (554, 8554):
        return "rtsp"
    if port in (80, 81, 8000, 8080, 8899):
        return "http"
    return "tcp"


def _detect_network_cameras() -> list[dict]:
    local_ips = _get_local_private_ipv4s()
    if not local_ips:
        return []

    by_ip: dict[str, set[int]] = {}
    for local_ip in local_ips:
        for ip, ports in _scan_subnet_for_camera_ports(local_ip).items():
            if ip not in by_ip:
                by_ip[ip] = set()
            by_ip[ip].update(ports)

    detected: list[dict] = []
    for ip in sorted(by_ip):
        ports = sorted(by_ip[ip])
        preferred_port = 554 if 554 in ports else ports[0]
        protocol = _port_to_protocol(preferred_port)
        detected.append(
            {
                "id": f"det_net_{ip}_{preferred_port}",
                "name": f"Network Camera ({ip})",
                "type": "network",
                "source": "lan",
                "ip": ip,
                "port": preferred_port,
                "protocol": protocol,
                "open_ports": ports,
                "status": "🟡 Reachable",
            }
        )
    return detected


def _merge_detections(onvif_cameras: list[dict], network_cameras: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}

    for cam in onvif_cameras:
        key = cam.get("ip") or cam["id"]
        merged[key] = dict(cam)

    for cam in network_cameras:
        key = cam.get("ip") or cam["id"]
        if key not in merged:
            merged[key] = dict(cam)
            continue

        existing = merged[key]
        existing_ports = set(existing.get("open_ports", []))
        net_ports = set(cam.get("open_ports", []))
        all_ports = sorted(existing_ports.union(net_ports))
        if all_ports:
            existing["open_ports"] = all_ports

        if 554 in all_ports:
            existing["rtsp_port"] = 554
            if existing.get("protocol") == "onvif":
                existing["protocol"] = "onvif+rtsp"
        elif 8554 in all_ports:
            existing["rtsp_port"] = 8554

        if not existing.get("port"):
            existing["port"] = existing.get("onvif_port") or cam.get("port")

        existing["status"] = "🟢 Discovered"

    return sorted(merged.values(), key=lambda cam: (cam.get("type", ""), cam.get("name", "")))


def detect_cameras(force: bool = False) -> list[dict]:
    global _detected_cache, _detected_cache_ts

    with _cache_lock:
        cache_is_fresh = (time.time() - _detected_cache_ts) < _SCAN_TTL_SEC
        if not force and cache_is_fresh and _detected_cache:
            return list(_detected_cache)

    system_cameras = _detect_system_cameras()
    onvif_cameras = _detect_onvif_cameras()
    network_cameras = _detect_network_cameras()
    network_merged = _merge_detections(onvif_cameras, network_cameras)

    for cam in network_merged:
        rtsp_port = cam.get("rtsp_port")
        port_for_rtsp = rtsp_port if rtsp_port in (554, 8554) else cam.get("port")
        if port_for_rtsp in (554, 8554):
            cam["rtsp_hints"] = [f"rtsp://<user>:<pass>@{cam['ip']}:{port_for_rtsp}{path}" for path in _RTSP_PATH_HINTS]

    detected = sorted(system_cameras + network_merged, key=lambda cam: (cam["type"], cam["name"]))

    with _cache_lock:
        _detected_cache = detected
        _detected_cache_ts = time.time()
        return list(_detected_cache)
