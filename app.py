"""
app.py  –  AI CCTV Surveillance System  (Web UI)
─────────────────────────────────────────────────────────────────────────────
Flask web server on port 5000.
  /              → Camera selection + live view UI
  /api/cameras   → JSON list of all cameras (webcam + Imou cloud)
  /api/select    → POST {id} to start a camera
  /api/stop      → POST to stop current camera
  /api/stats     → JSON with live detection results (persons, actions, emotions)
  /video_feed    → MJPEG live stream
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import config as cfg
from modules.motion_detection import MotionDetector
from modules.object_detection import PersonDetector
from modules.tracking import PersonTracker
from modules.face_detection import FaceDetector
from modules.emotion_detection import EmotionDetector
from modules.action_detection import ActionDetector
from utils.stream import VideoStream
from utils.drawing import draw_person, draw_face, draw_status_bar
from utils.fps_counter import FPSCounter
from imou_connector import ImouAPI, _find_working_datacenter, _get_device_status

# ─── Imou credentials ────────────────────────────────────────────────────────
import os

# Manually load .env for local development without needing python-dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

APP_ID     = os.environ.get("IMOU_APP_ID", "")
APP_SECRET = os.environ.get("IMOU_APP_SECRET", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")


# ─── Data classes ─────────────────────────────────────────────────────────────

# Global Toggles for UI Control
AI_TOGGLES = {
    "person": True,
    "action": True,
    "emotion": True,
}

@dataclass
class TrackResult:
    track_id:      int
    bbox_full:     list[float]
    face_bbox_full: list[float] | None
    emotion:       str
    action:        str


@dataclass
class AIResult:
    motion:      bool              = False
    motion_mask: np.ndarray | None = None
    tracks:      list[TrackResult] = field(default_factory=list)
    timestamp:   float             = field(default_factory=time.monotonic)
    fps:         float             = 0.0

# ─── Scale helper ─────────────────────────────────────────────────────────────

def _scale_box(box, sx, sy):
    x1, y1, x2, y2 = box
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


# ─── AI Pipeline Thread ────────────────────────────────────────────────────────

class AIPipeline(threading.Thread):
    """Background AI thread: motion → YOLO → DeepSORT → action → emotion."""

    def __init__(self, in_queue, result_store, result_lock):
        super().__init__(daemon=True, name="AIPipeline")
        self.in_queue     = in_queue
        self.result_store = result_store
        self.result_lock  = result_lock
        self._stop_evt    = threading.Event()

        log.info("AI: loading models …")
        self.motion_det = MotionDetector(
            history=cfg.MOTION_HISTORY,
            var_threshold=cfg.MOTION_VAR_THRESHOLD,
            min_area=cfg.MOTION_MIN_AREA,
        )
        self.person_det = PersonDetector(
            model_path=cfg.YOLO_MODEL,
            conf_threshold=cfg.YOLO_CONF,
            device=cfg.YOLO_DEVICE,
        )
        self.tracker = PersonTracker(
            max_age=cfg.DEEPSORT_MAX_AGE,
            n_init=cfg.DEEPSORT_N_INIT,
            embedder=cfg.DEEPSORT_EMBEDDER,
        )
        self.face_det    = FaceDetector()
        self.emotion_det = EmotionDetector(
            skip_frames=cfg.EMOTION_SKIP_FRAMES,
            backend=cfg.EMOTION_BACKEND,
            min_face_pixels=cfg.EMOTION_MIN_FACE_PX,
        )
        self._emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")
        self._emotion_futures: dict[int, Future] = {}

        self.action_det = ActionDetector(
            enabled=cfg.ENABLE_ACTION,
            buffer_size=cfg.ACTION_BUFFER_SIZE,
            slow_stride=cfg.ACTION_SLOW_STRIDE,
            device=cfg.ACTION_DEVICE,
        )

        self._frame_idx            = 0
        self._last_detections: list = []
        self._last_objects: dict    = {"food": [], "phone": []}
        self._last_motion_time     = time.monotonic()
        self._full_w: int          = cfg.FRAME_WIDTH
        self._full_h: int          = cfg.FRAME_HEIGHT

    def stop(self):
        self._stop_evt.set()
        self.in_queue.put(None)

    def run(self):
        log.info("AI pipeline started.")
        while not self._stop_evt.is_set():
            try:
                ai_frame = self.in_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if ai_frame is None:
                break
            result = self._process(ai_frame, self._full_w, self._full_h)
            with self.result_lock:
                self.result_store[0] = result
        self._emotion_executor.shutdown(wait=False)
        log.info("AI pipeline stopped.")

    def _submit_emotion(self, crop: np.ndarray, tid: int):
        fut = self._emotion_futures.get(tid)
        if fut is not None and not fut.done():
            return
        self._emotion_futures[tid] = self._emotion_executor.submit(
            self.emotion_det.analyse, crop.copy(), tid
        )

    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int) -> AIResult:
        self._frame_idx += 1
        ai_h, ai_w = ai_frame.shape[:2]
        sx = full_w / ai_w
        sy = full_h / ai_h

        motion, mask = self.motion_det.detect(ai_frame)
        now = time.monotonic()
        if motion:
            self._last_motion_time = now

        in_cooldown = (now - self._last_motion_time) < cfg.MOTION_COOLDOWN_SEC
        if not motion and not in_cooldown:
            return AIResult(motion=False, motion_mask=mask, tracks=[])

        if AI_TOGGLES["person"]:
            if self._frame_idx % cfg.YOLO_SKIP_FRAMES == 0:
                self._last_detections = self.person_det.detect(ai_frame)
        else:
            self._last_detections = []

        if AI_TOGGLES["action"]:
            if self._frame_idx % 5 == 0:
                self._last_objects = self.person_det.detect_objects(ai_frame)
        else:
            self._last_objects = {"food": [], "phone": []}

        raw_tracks = self.tracker.update(self._last_detections, ai_frame)
        active_ids = {t.track_id for t in raw_tracks}
        if active_ids:
            self._last_motion_time = now

        track_results: list[TrackResult] = []

        for track in raw_tracks:
            tid  = track.track_id
            bbox = track.bbox
            bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
            bx2 = min(ai_w, bx2); by2 = min(ai_h, by2)

            action = ""
            emotion = ""
            face_bbox_full = None

            if AI_TOGGLES["action"]:
                bw = max(1, bx2 - bx1); bh = max(1, by2 - by1)
                ratio = bw / bh
                if ratio > 1.4:   geometry_label = "😴 sleeping"
                elif ratio > 0.85: geometry_label = "🪑 sitting"
                else:              geometry_label = "🧍 standing"

                person_crop = ai_frame[by1:by2, bx1:bx2]
                slowfast_label = self.action_det.update(tid, person_crop) if person_crop.size > 0 else ""

                def _overlaps(ob, pb, exp=40):
                    ox1,oy1,ox2,oy2=ob; px1,py1,px2,py2=pb
                    return ox1<px2+exp and ox2>px1-exp and oy1<py2+exp and oy2>py1-exp

                psb = [bx1*sx, by1*sy, bx2*sx, by2*sy]
                near_food  = any(_overlaps(f, psb) for f in self._last_objects.get("food", []))
                near_phone = any(_overlaps(p, psb) for p in self._last_objects.get("phone", []))

                TAGS = ("⚠", "🏃", "💪", "📖", "✍", "🍽", "💬", "💃", "🎵")
                if near_food:   action = "🍽 eating"
                elif near_phone: action = "💻 working"
                elif slowfast_label and any(t in slowfast_label for t in TAGS): action = slowfast_label
                else:           action = geometry_label

            if AI_TOGGLES["emotion"]:
                person_crop = ai_frame[by1:by2, bx1:bx2]
                if person_crop.size > 0:
                    self._submit_emotion(person_crop, tid)
                emotion = self.emotion_det._cache.get(tid, "")

            track_results.append(TrackResult(
                track_id=tid,
                bbox_full=_scale_box(bbox, sx, sy),
                face_bbox_full=face_bbox_full,
                emotion=emotion, action=action,
            ))

        if self._frame_idx % 120 == 0:
            self.emotion_det.purge(active_ids)
            self.action_det.purge(active_ids)
            for tid in list(self._emotion_futures):
                if tid not in active_ids:
                    del self._emotion_futures[tid]

        return AIResult(motion=motion, motion_mask=mask, tracks=track_results)


# ─── Camera Manager ────────────────────────────────────────────────────────────

class CameraManager:
    """
    Manages camera switching with ZERO model reloads.

    Architecture
    ────────────
    __init__()   → creates AIPipeline ONCE (loads YOLO / SlowFast / DeepFace)
    start()      → only replaces VideoStream + RenderThread  (< 1 second)
    stop_stream()→ stops video + render thread only (pipeline keeps running)
    stop_all()   → full shutdown used only on app exit
    """

    def __init__(self):
        self._result_store    = [AIResult()]
        self._result_lock     = threading.Lock()
        self._frame_queue     = queue.Queue(maxsize=2)
        self._jpeg_buf        = b""
        self._jpeg_lock       = threading.Lock()
        self._render_thread   = None
        self._render_stop_evt = threading.Event()
        self._fps_val         = 0.0
        self._active          = False
        self.camera_name      = ""
        self.stream_id        = 0

        # ── Load ALL models exactly ONCE here ──────────────────────────
        log.info("Loading AI models (one-time startup) …")
        self._pipeline = AIPipeline(
            self._frame_queue, self._result_store, self._result_lock
        )
        self._pipeline.start()
        log.info("AI models ready.")

    # ------------------------------------------------------------------
    def start(self, source, name: str = "Camera"):
        """Switch to a new camera source. Models are NOT reloaded."""
        self.stop_stream()                   # stop old video+render only

        log.info("Starting camera: %s  source=%s", name, source)
        self.camera_name      = name
        self._active          = True
        self._render_stop_evt = threading.Event()
        self.stream_id       += 1            # frontend detects the change

        # Drain stale frames from old camera so new camera starts clean
        while not self._frame_queue.empty():
            try: self._frame_queue.get_nowait()
            except queue.Empty: break

        # Reset AI result so ghost boxes don't appear on new camera
        with self._result_lock:
            self._result_store[0] = AIResult()

        stream = VideoStream(
            source=source,
            width=cfg.FRAME_WIDTH,
            height=cfg.FRAME_HEIGHT,
            target_fps=cfg.TARGET_FPS,
        )
        self._render_thread = threading.Thread(
            target=self._render_loop,
            args=(self._render_stop_evt, stream),   # stream passed directly
            daemon=True, name="Render",
        )
        self._render_thread.start()

    # ------------------------------------------------------------------
    def stop_stream(self):
        """Stop video capture + render thread. Pipeline keeps running."""
        self._active = False
        # Signal render thread to exit
        self._render_stop_evt.set()
        # Render thread will exit its loop; join it
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=4)
            self._render_thread = None
        # Clear stale jpeg
        with self._jpeg_lock:
            self._jpeg_buf = b""
        self._fps_val = 0.0

    # ------------------------------------------------------------------
    def stop_all(self):
        """Full shutdown — used only when the app exits."""
        self.stop_stream()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.join(timeout=8)
            self._pipeline = None

    # ------------------------------------------------------------------
    def _render_loop(self, stop_evt: threading.Event, stream):
        """
        Runs in its own thread.
        `stream` is a local reference — safe even if the manager
        reassigns its own attributes during a camera switch.
        """
        fps_ctr = FPSCounter(window=30)
        cap_idx = 0
        try:
            while not stop_evt.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    if stop_evt.is_set():
                        break
                    time.sleep(0.02)
                    continue

                cap_idx += 1
                self._fps_val = fps_ctr.tick()

                if cap_idx % cfg.AI_THREAD_FRAME_SKIP == 0:
                    fh, fw = frame.shape[:2]
                    ai_f   = cv2.resize(frame, (cfg.AI_FRAME_WIDTH, cfg.AI_FRAME_HEIGHT))
                    self._pipeline._full_w = fw
                    self._pipeline._full_h = fh
                    try:
                        self._frame_queue.put_nowait(ai_f)
                    except queue.Full:
                        pass

                with self._result_lock:
                    result = self._result_store[0]

                age    = time.monotonic() - result.timestamp
                tracks = result.tracks if age < cfg.RESULT_MAX_AGE_SEC else []

                for tr in tracks:
                    draw_person(frame, tr.bbox_full, tr.track_id,
                                emotion=tr.emotion, action=tr.action)
                    if tr.face_bbox_full:
                        draw_face(frame, tr.face_bbox_full, tr.track_id)

                draw_status_bar(
                    frame,
                    motion=result.motion if age < cfg.RESULT_MAX_AGE_SEC else False,
                    n_persons=len(tracks),
                    fps=self._fps_val,
                )

                ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok2:
                    with self._jpeg_lock:
                        self._jpeg_buf = jpg.tobytes()
        finally:
            # Always release the stream when this thread exits
            try: stream.release()
            except Exception: pass
            log.info("Render thread exited for: %s", self.camera_name)

    # ------------------------------------------------------------------
    def get_jpeg(self) -> bytes:
        with self._jpeg_lock:
            return self._jpeg_buf

    def get_stats(self) -> dict:
        with self._result_lock:
            r = self._result_store[0]
        age    = time.monotonic() - r.timestamp
        tracks = r.tracks if age < cfg.RESULT_MAX_AGE_SEC else []
        return {
            "active":  self._active,
            "camera":  self.camera_name,
            "fps":     round(self._fps_val, 1),
            "motion":  r.motion,
            "persons": len(tracks),
            "tracks": [
                {"id": t.track_id,
                 "emotion": t.emotion or "–",
                 "action":  t.action  or "–"}
                for t in tracks
            ],
        }



# ─── Imou camera fetcher ───────────────────────────────────────────────────────

def _fetch_imou_cameras():
    """Return list of Imou camera dicts with {id, name, status, stream_url}."""
    try:
        base = _find_working_datacenter()
        if not base:
            return []
        api = ImouAPI(APP_ID, APP_SECRET, base)
        api.get_token()
        devices = api.list_devices()
        result = []
        for i, dev in enumerate(devices):
            dev_id   = dev.get("deviceId") or dev.get("deviceID") or f"dev{i}"
            name     = dev.get("name") or dev.get("deviceName") or "Imou Camera"
            status   = _get_device_status(dev)
            result.append({
                "id":     f"imou_{dev_id}",
                "name":   name,
                "status": status,
                "type":   "imou",
                "dev_id": dev_id,
                "base":   base,
            })
        return result
    except Exception as e:
        log.warning("Imou fetch failed: %s", e)
        return []


# Cache cameras 60 s so repeated UI requests don't re-auth every time
_camera_cache: list[dict] = []
_camera_cache_ts: float   = 0.0
_camera_lock = threading.Lock()

def get_all_cameras(force=False):
    global _camera_cache, _camera_cache_ts
    with _camera_lock:
        if not force and (time.time() - _camera_cache_ts) < 60 and _camera_cache:
            return _camera_cache
        cameras = [{"id": "webcam", "name": "Webcam (Built-in)", "status": "🟢 Online", "type": "webcam"}]
        cameras.extend(_fetch_imou_cameras())
        _camera_cache    = cameras
        _camera_cache_ts = time.time()
        return cameras


# ─── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)
cam_mgr = CameraManager()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cameras")
def api_cameras():
    cameras = get_all_cameras()
    return jsonify(cameras)


@app.route("/api/select", methods=["POST"])
def api_select():
    data = request.get_json(force=True)
    cam_id = data.get("id", "webcam")
    cameras = get_all_cameras()
    cam = next((c for c in cameras if c["id"] == cam_id), None)
    if cam is None:
        return jsonify({"error": "Camera not found"}), 404

    if cam["type"] == "webcam":
        cam_mgr.start(0, cam["name"])
    elif cam["type"] == "imou":
        try:
            base   = cam["base"]
            api    = ImouAPI(APP_ID, APP_SECRET, base)
            api.get_token()
            stream_url = api.get_rtsp(cam["dev_id"])
            if not stream_url:
                return jsonify({"error": "Could not get stream URL from Imou"}), 500
            cam_mgr.start(stream_url, cam["name"])
        except Exception as e:
            log.error("Imou start error: %s", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "camera": cam["name"]})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    cam_mgr.stop_stream()
    return jsonify({"success": True})


@app.route("/api/stream_id")
def api_stream_id():
    """Returns current stream ID so frontend can detect camera switches."""
    return jsonify({"stream_id": cam_mgr.stream_id, "active": cam_mgr._active})


@app.route("/api/stats")
def api_stats():
    return jsonify(cam_mgr.get_stats())


@app.route("/api/toggles", methods=["GET", "POST"])
def api_toggles():
    if request.method == "POST":
        data = request.get_json(force=True)
        if "person" in data: AI_TOGGLES["person"] = bool(data["person"])
        if "action" in data: AI_TOGGLES["action"] = bool(data["action"])
        if "emotion" in data: AI_TOGGLES["emotion"] = bool(data["emotion"])
    return jsonify(AI_TOGGLES)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = cam_mgr.get_jpeg()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import atexit
    import os
    from waitress import serve
    
    atexit.register(cam_mgr.stop_all)
    
    # Get port from environment variable (Useful for Railway)
    port = int(os.environ.get("PORT", 5000))
    
    log.info(f"Starting MACHINE CONTROLLER Web UI on http://localhost:{port}")
    serve(app, host="0.0.0.0", port=port)
