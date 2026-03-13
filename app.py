"""
app.py  –  AI CCTV Surveillance System  (Web UI)
─────────────────────────────────────────────────────────────────────────────
Flask web server on port 5000.
  /              → Camera selection + live view UI
  /api/cameras   → JSON list of all cameras (Local RTSP + Webcam)
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
from datetime import datetime
import json
import os

# ─── Environment Configuration (Load .env early for config.py) ────────────────
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

import cv2
import numpy as np
import werkzeug.security
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from werkzeug.utils import secure_filename
from functools import wraps
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for

import config as cfg
from modules.motion_detection import MotionDetector
from modules.object_detection import PersonDetector
from modules.tracking import PersonTracker
from modules.face_detection import FaceDetector
from modules.emotion_detection import EmotionDetector
from modules.action_detection import ActionDetector
from modules.face_recognition import FaceRecognizer
from utils.stream import VideoStream
from utils.drawing import draw_person, draw_face, draw_status_bar, draw_tripwire
from utils.fps_counter import FPSCounter
from modules.notifier import TelegramNotifier




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")
notifier = TelegramNotifier()

total_motion_events = 0

# ─── Data classes ─────────────────────────────────────────────────────────────

# Global Toggles for UI Control
AI_TOGGLES = {
    "person": True,
    "action": False,
    "emotion": False,
}

@dataclass
class TrackResult:
    track_id:      int
    bbox_full:     list[float]
    face_bbox_full: list[float] | None
    emotion:       str
    action:        str
    identity:      str = ""


@dataclass
class AIResult:
    motion:      bool              = False
    motion_mask: np.ndarray | None = None
    tracks:      list[TrackResult] = field(default_factory=list)
    timestamp:   float             = field(default_factory=time.monotonic)
    fps:         float             = 0.0
    entries:     int               = 0
    exits:       int               = 0

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
            max_iou_distance=cfg.DEEPSORT_MAX_IOU,
            embedder=cfg.DEEPSORT_EMBEDDER,
        )
        self.face_det    = None
        self.emotion_det = None
        self._emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")
        self._emotion_futures: dict[int, Future] = {}

        self.action_det = None

        # Database-only Face Recognition (no local files used for matching)
        self.face_rec = None
        self._rec_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rec")
        self._rec_futures: dict[int, Future] = {}
        self._notified_identities: set[int] = set()

        self._frame_idx            = 0
        self._last_detections: list = []
        self._last_objects: dict    = {"food": [], "phone": []}
        self._last_motion_time     = time.monotonic()
        self._full_w: int          = cfg.FRAME_WIDTH
        self._full_h: int          = cfg.FRAME_HEIGHT
        self.camera_name: str      = "Camera"
        self.notifier             = TelegramNotifier()
        
        # Detection-based Counting
        self._seen_ids: set[int] = set()
        self._active_ids: set[int] = set()
        self.entries = 0
        self.exits   = 0
        self._entry_best_frames: dict[int, np.ndarray] = {}  # Best shot near entry
        self._exit_best_frames:  dict[int, np.ndarray] = {}  # Best shot near exit (continuous)
        self._best_entry_scores: dict[int, float]      = {}
        self._best_exit_scores:  dict[int, float]      = {}
        self._track_start_time:  dict[int, float]      = {}  # Records when an ID first appeared
        self._exit_taken: set[int] = set()                  # Track IDs that already have an exit shot
        self._db_sessions: dict[int, int] = {}             # Tracks DB row ID for each person session
        self._exit_filenames: dict[int, str] = {}          # Stores filenames for DB exit logging
        os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)

    def _calculate_clarity(self, frame, bbox=None):
        """Calculates frame clarity using Laplacian variance. Higher = Sharp/Clear."""
        try:
            if bbox is not None:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: return 0.0
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0.0

    def _save_snapshot(self, frame, tid, mode="ENTRY"):
        """Saves a forensic snapshot of a person entry or exit."""
        if not cfg.ENABLE_SNAPSHOTS or frame is None:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{mode.lower()}_{self.camera_name.replace(' ', '_')}_ID{tid}_{timestamp}.jpg"
            filepath = os.path.join(cfg.SNAPSHOT_DIR, filename)
            
            # Create a forensic copy with metadata overlay
            snap_frame = frame.copy()
            # Draw metadata bar (Entry = Green, Exit = Red)
            color = (0, 255, 0) if mode == "ENTRY" else (0, 0, 255)
            text = f"{mode}: {self.camera_name} | ID: {tid} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.rectangle(snap_frame, (0, 0), (snap_frame.shape[1], 35), (0, 0, 0), -1)
            cv2.putText(snap_frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imwrite(filepath, snap_frame)
            log.info(f"📸 {mode} SNAPSHOT SAVED: {filepath} (ID: {tid})")
            return snap_frame, filename
        except Exception as e:
            log.error(f"Failed to save {mode} snapshot: {e}")
            return None, None

    def _save_merged(self, entry_f, exit_f, tid):
        """Creates a side-by-side Merged Forensic Masterpiece."""
        if entry_f is None or exit_f is None: return
        try:
            h = 480
            w_e = int(entry_f.shape[1] * (h / entry_f.shape[0]))
            w_x = int(exit_f.shape[1] * (h / exit_f.shape[0]))
            merged = np.hstack((cv2.resize(entry_f, (w_e, h)), cv2.resize(exit_f, (w_x, h))))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"merged_{self.camera_name.replace(' ', '_')}_ID{tid}_{timestamp}.jpg"
            cv2.imwrite(os.path.join(cfg.SNAPSHOT_DIR, filename), merged)
            log.info(f"🔗 MERGED LOG SAVED: {filename}")
            return filename
        except Exception as e:
            log.error(f"Merge error: {e}")
            return None

    def _db_log_entry(self, tid, filename):
        """Creates a forensic entry record in the database."""
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO member_time_stamp (person_id, camera_name, entry_image, entry_time)
                    VALUES (%s, %s, %s, %s) RETURNING id
                """, (tid, self.camera_name, filename, datetime.now()))
                self._db_sessions[tid] = cur.fetchone()['id']
            conn.commit(); conn.close()
        except Exception as e:
            log.error(f"DB Entry log error: {e}")

    def _db_log_exit(self, tid, exit_file, merged_file=None):
        """Finalizes a forensic session in the database with exit and merged files."""
        if tid not in self._db_sessions: return
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE member_time_stamp 
                    SET exit_image = %s, merged_image = %s, exit_time = %s
                    WHERE id = %s
                """, (exit_file, merged_file, datetime.now(), self._db_sessions[tid]))
            conn.commit(); conn.close()
        except Exception as e:
            log.error(f"DB Exit log error: {e}")

    def stop(self):
        self._stop_evt.set()
        self.in_queue.put(None)

    def run(self):
        log.info("AI pipeline started.")
        import traceback
        while not self._stop_evt.is_set():
            try:
                # MANDATORY ZERO-LATENCY FLUSH: 
                # Discard all old buffered frames so AI always works on the ABSOLUTE LATEST frame.
                # This ensures the box moves with the person in "milli seconds".
                while self.in_queue.qsize() > 1:
                    try: self.in_queue.get_nowait()
                    except: break

                ai_frame = self.in_queue.get(timeout=1.0)
                if ai_frame is None: break
                
                result = self._process(ai_frame, self._full_w, self._full_h)
                with self.result_lock:
                    self.result_store[0] = result
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"⚠️ AI PIPELINE CRASH: {e}")
                traceback.print_exc()
                time.sleep(1) # prevent infinite fast-loop crash
                
        self._emotion_executor.shutdown(wait=False)
        self._rec_executor.shutdown(wait=False)
        log.info("AI pipeline stopped.")

    def _submit_emotion(self, crop: np.ndarray, tid: int):
        fut = self._emotion_futures.get(tid)
        if fut is not None and not fut.done():
            return
        self._emotion_futures[tid] = self._emotion_executor.submit(
            self.emotion_det.analyse, crop.copy(), tid
        )

    def _submit_rec(self, ai_frame: np.ndarray, bbox: list[float], tid: int):
        """Asynchronously detect face AND recognize identity."""
        fut = self._rec_futures.get(tid)
        if fut is not None and not fut.done():
            return
            
        def _task():
            try:
                face_bbox = self.face_det.detect_in_crop(ai_frame, bbox)
                if face_bbox:
                    fx1, fy1, fx2, fy2 = (int(v) for v in face_bbox)
                    face_crop = ai_frame[max(0,fy1):min(ai_frame.shape[0],fy2), max(0,fx1):min(ai_frame.shape[1],fx2)]
                    if face_crop.size > 0:
                        name = self.face_rec.recognize(face_crop, tid)
                        self.face_rec._cache[tid] = name
                        return name
            except Exception as e:
                log.debug(f"Rec task error: {e}")
            return "Unknown"

        self._rec_futures[tid] = self._rec_executor.submit(_task)

    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int) -> AIResult:
        self._frame_idx += 1
        if self._frame_idx % 30 == 0:
            log.info(f"AI: Processing frame #{self._frame_idx} (Camera: {self.camera_name})")
            
        ai_h, ai_w = ai_frame.shape[:2]
        sx = full_w / ai_w
        sy = full_h / ai_h

        motion, mask = self.motion_det.detect(ai_frame)
        now = time.monotonic()
        if motion:
            if (now - self._last_motion_time) > cfg.MOTION_COOLDOWN_SEC:
                global total_motion_events
                total_motion_events += 1
            self._last_motion_time = now

        # Always detect people if toggle is on (don't rely on motion)
        if AI_TOGGLES["person"]:
            skip = max(1, cfg.YOLO_SKIP_FRAMES)
            if self._frame_idx % skip == 0:
                self._last_detections = self.person_det.detect(ai_frame)
        else:
            self._last_detections = []

        if AI_TOGGLES["action"]:
            if self.action_det is None:
                log.info("AI: Lazy-loading Action Detector...")
                self.action_det = ActionDetector(
                    enabled=True,
                    buffer_size=cfg.ACTION_BUFFER_SIZE,
                    slow_stride=cfg.ACTION_SLOW_STRIDE,
                    device=cfg.ACTION_DEVICE,
                )
            if self._frame_idx % 5 == 0:
                self._last_objects = self.person_det.detect_objects(ai_frame)
        else:
            self._last_objects = {"food": [], "phone": []}

        raw_tracks = self.tracker.update(self._last_detections, ai_frame)
        active_ids = {t.track_id for t in raw_tracks}
        
        # 1. New Entrants
        for tid in active_ids:
            if tid not in self._seen_ids:
                self.entries += 1
                self._seen_ids.add(tid)
                self._track_start_time[tid] = now
                log.info(f"COUNT: New person entering frame (ID: {tid}). Total Entries: {self.entries}")

        # 2. Exits: IDs that WERE in active_ids but are now gone
        for tid in list(self._active_ids):
            if tid not in active_ids:
                self.exits += 1
                log.info(f"COUNT: Person left (ID: {tid}). Total Exits: {self.exits}")
                
                # Retrieve the BEST shots for Entry and Exit
                entry_f = self._entry_best_frames.get(tid)
                exit_f  = self._exit_best_frames.get(tid)
                
                # Save Exit snapshot (Fallback if edge trigger didn't hit)
                exit_file = self._exit_filenames.get(tid)
                if tid not in self._exit_taken:
                    exit_f_processed, exit_file = self._save_snapshot(exit_f, tid, mode="EXIT")
                else:
                    exit_f_processed = exit_f # Already contains the edge shot
                
                # Save Merged snapshot
                merged_file = None
                if entry_f is not None and exit_f_processed is not None:
                    merged_file = self._save_merged(entry_f, exit_f_processed, tid)
                
                # Finalize DB Record
                self._db_log_exit(tid, exit_file, merged_file)
                
                # Cleanup local track memory
                for d in [self._entry_best_frames, self._exit_best_frames, self._best_entry_scores, self._best_exit_scores, self._track_start_time, self._db_sessions, self._exit_filenames]:
                    if tid in d: del d[tid]
                if tid in self._exit_taken: self._exit_taken.remove(tid)
        
        # 3. Best-Shot Monitoring (During Active Stay)
        for track in raw_tracks:
            tid = track.track_id
            score = self._calculate_clarity(ai_frame, track.bbox)
            
            # --- Entry Shot (Capture first sharp frame and LOCK) ---
            if tid not in self._best_entry_scores:
                # After a short stabilization or a high-clarity score, take the shot
                if score > 45 or self._frame_idx % 12 == 0:
                    self._best_entry_scores[tid] = score
                    snap, filename = self._save_snapshot(ai_frame, tid, mode="ENTRY")
                    if snap is not None: 
                        self._entry_best_frames[tid] = snap
                        self._db_log_entry(tid, filename)

            # Priority 1: Proactive Edge Trigger (Capture and COMMIT to DB immediately)
            # GUARD: Only allow EXIT snapshot if the person has been in frame for at least 3 seconds
            stay_duration = now - self._track_start_time.get(tid, now)
            
            bx1, by1, bx2, by2 = track.bbox
            is_leaving = (bx1 < 15 or by1 < 15 or bx2 > (ai_w - 15) or by2 > (ai_h - 15))
            
            if is_leaving and stay_duration > 3.0 and tid not in self._exit_taken and self._frame_idx % 3 == 0:
                log.info(f"📸 PROACTIVE EXIT TRIGGER: ID {tid} is leaving. stay={stay_duration:.1f}s")
                snap, exit_file = self._save_snapshot(ai_frame, tid, mode="EXIT")
                if snap is not None:
                    self._exit_best_frames[tid] = snap
                    self._exit_filenames[tid] = exit_file
                    self._exit_taken.add(tid)
                    
                    # COMMIT TO DB NOW (Forensic Instant Persistence)
                    entry_f = self._entry_best_frames.get(tid)
                    merged_file = self._save_merged(entry_f, snap, tid) if entry_f is not None else None
                    self._db_log_exit(tid, exit_file, merged_file)

            # Priority 2: Continuous Buffer (Sharpest frame so far)
            if tid not in self._exit_taken:
                if tid not in self._best_exit_scores or score >= (self._best_exit_scores[tid] * 0.40):
                    self._best_exit_scores[tid] = max(score, self._best_exit_scores.get(tid, 0))
                    self._exit_best_frames[tid] = ai_frame.copy()
            
            # Fallback: If it's the absolute last frame they are seen, we MUST update to capture 'The End'
            # (Handled by the fact that the loop runs for every visible frame)

        self._active_ids = active_ids
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
            identity_name = ""
            face_bbox_full = None

            if AI_TOGGLES["action"]:
                if self.action_det is not None:
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
                else:
                    action = "Loading AI..."

            if AI_TOGGLES["emotion"]:
                if self.emotion_det is None or self.face_det is None or self.face_rec is None:
                    log.info("AI: Lazy-loading Face & Emotion models...")
                    self.face_det = FaceDetector()
                    self.emotion_det = EmotionDetector(
                        skip_frames=cfg.EMOTION_SKIP_FRAMES,
                        backend=cfg.EMOTION_BACKEND,
                        min_face_pixels=cfg.EMOTION_MIN_FACE_PX,
                    )
                    self.face_rec = FaceRecognizer(
                        skip_frames=15, 
                        backend=cfg.EMOTION_BACKEND,
                    )

                # Emotion
                person_crop = ai_frame[by1:by2, bx1:bx2]
                if person_crop.size > 0:
                    self._submit_emotion(person_crop, tid)
                emotion = self.emotion_det._cache.get(tid, "")

                # Recognition
                cached_identity = self.face_rec._cache.get(tid, "")
                is_unknown = (cached_identity == "" or cached_identity == "Unknown")
                
                active_rec_tasks = sum(1 for f in self._rec_futures.values() if not f.done())
                if (is_unknown or (self._frame_idx % 300 == 0)) and active_rec_tasks < 2:
                    self._submit_rec(ai_frame, bbox, tid)
                
                identity_name = cached_identity
                
                if identity_name and identity_name != "Unknown" and tid not in self._notified_identities:
                    self.notifier.send_message(f"✅ *MATCH DETECTED*: Staff member **{identity_name}** recognized on {self.camera_name}.")
                    self._notified_identities.add(tid)

            track_results.append(TrackResult(
                track_id=tid,
                bbox_full=_scale_box(bbox, sx, sy),
                face_bbox_full=face_bbox_full,
                emotion=emotion, action=action,
                identity=identity_name,
            ))
            
            if AI_TOGGLES["person"]:
                self.notifier.notify_person(tid, self.camera_name, action, cooldown=600)

        if self._frame_idx % 120 == 0:
            if self.emotion_det: self.emotion_det.purge(active_ids)
            if self.action_det: self.action_det.purge(active_ids)
            if self.face_rec: self.face_rec.purge(active_ids)
            # Cleanup seen set periodically to prevent memory growth (if session is very long)
            if len(self._seen_ids) > 1000:
                # Keep only active ones in seen if it's getting too big
                self._seen_ids = self._seen_ids.intersection(active_ids)
            for tid in list(self._emotion_futures):
                if tid not in active_ids:
                    del self._emotion_futures[tid]
            for tid in list(self._rec_futures):
                if tid not in active_ids:
                    del self._rec_futures[tid]
            for tid in list(self._notified_identities):
                if tid not in active_ids:
                    self._notified_identities.remove(tid)

        if self._frame_idx % 30 == 0:
            log.info(f"AI Pipeline: Returning {len(track_results)} tracks (Frame #{self._frame_idx})")
            
        return AIResult(motion=motion, motion_mask=mask, tracks=track_results, entries=self.entries, exits=self.exits)


# ─── Camera Manager ────────────────────────────────────────────────────────────

class CameraManager:
    """Manages switching camera sources without reloading AI models."""

    def __init__(self):
        self._result_store    = [AIResult()]
        self._result_lock     = threading.Lock()
        self._frame_queue     = queue.Queue(maxsize=1)  # size 1 = ZERO LAG
        self._jpeg_buf        = b""
        self._jpeg_lock       = threading.Lock()
        self._render_thread   = None
        self._render_stop_evt = threading.Event()
        self._fps_val         = 0.0
        self._active          = False
        self.camera_name      = ""
        self.stream_id        = 0

        log.info("Loading AI models (one-time startup) …")
        self._pipeline = AIPipeline(
            self._frame_queue, self._result_store, self._result_lock
        )
        self._pipeline.start()
        log.info("AI models ready.")

    def start(self, source, name: str = "Camera"):
        self.stop_stream()
        log.info("Starting camera: %s  source=%s", name, source)
        self.camera_name = name
        if self._pipeline:
            self._pipeline.camera_name = name
        self._active = True
        self._render_stop_evt = threading.Event()
        self.stream_id += 1

        while not self._frame_queue.empty():
            try: self._frame_queue.get_nowait()
            except queue.Empty: break

        with self._result_lock:
            self._result_store[0] = AIResult()
            # Reset pipeline counts for new camera
            self._pipeline.entries = 0
            self._pipeline.exits = 0
            self._pipeline._seen_ids = set()
            self._pipeline._active_ids = set()
            self._pipeline._entry_best_frames = {}
            self._pipeline._exit_best_frames = {}
            self._pipeline._best_entry_scores = {}
            self._pipeline._best_exit_scores = {}
            self._pipeline._track_start_time = {}
            self._pipeline._db_sessions = {}
            self._pipeline._exit_filenames = {}

        stream = VideoStream(
            source=source,
            width=cfg.FRAME_WIDTH,
            height=cfg.FRAME_HEIGHT,
            target_fps=cfg.TARGET_FPS,
        )
        self._render_thread = threading.Thread(
            target=self._render_loop,
            args=(self._render_stop_evt, stream),
            daemon=True, name="Render",
        )
        self._render_thread.start()

    def stop_stream(self):
        self._active = False
        self._render_stop_evt.set()
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=4)
            self._render_thread = None
        with self._jpeg_lock:
            self._jpeg_buf = b""
        self._fps_val = 0.0

    def stop_all(self):
        self.stop_stream()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.join(timeout=8)
            self._pipeline = None

    def _render_loop(self, stop_evt: threading.Event, stream):
        fps_ctr = FPSCounter(window=30)
        cap_idx = 0
        try:
            while not stop_evt.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    if stop_evt.is_set(): break
                    time.sleep(0.02); continue

                cap_idx += 1
                self._fps_val = fps_ctr.tick()

                if cap_idx % cfg.AI_THREAD_FRAME_SKIP == 0:
                    fh, fw = frame.shape[:2]
                    ai_f = cv2.resize(frame, (cfg.AI_FRAME_WIDTH, cfg.AI_FRAME_HEIGHT))
                    self._pipeline._full_w = fw
                    self._pipeline._full_h = fh
                    # Zero-Latency: Always clear the queue so AI only gets the latest frame
                    while not self._frame_queue.empty():
                        try: self._frame_queue.get_nowait()
                        except: break
                        
                    try:
                        self._frame_queue.put_nowait(ai_f)
                    except queue.Full: pass

                with self._result_lock:
                    result = self._result_store[0]

                age = time.monotonic() - result.timestamp
                tracks = result.tracks if age < cfg.RESULT_MAX_AGE_SEC else []

                if cfg.SHOW_TRIPWIRE:
                    draw_tripwire(frame, cfg.LINE_POSITION, cfg.LINE_DIRECTION, cfg.COLOR_ENTRY, cfg.COLOR_EXIT)

                for tr in tracks:
                    draw_person(frame, tr.bbox_full, tr.track_id, emotion=tr.emotion, action=tr.action, identity=tr.identity)
                    if tr.face_bbox_full:
                        draw_face(frame, tr.face_bbox_full, tr.track_id)

                draw_status_bar(
                    frame, 
                    motion=result.motion if age < cfg.RESULT_MAX_AGE_SEC else False, 
                    n_persons=len(tracks), 
                    fps=self._fps_val,
                    entries=result.entries,
                    exits=result.exits
                )

                ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok2:
                    with self._jpeg_lock:
                        self._jpeg_buf = jpg.tobytes()
        finally:
            try: stream.release()
            except: pass
            log.info("Render thread exited for: %s", self.camera_name)

    def get_jpeg(self) -> bytes:
        with self._jpeg_lock:
            return self._jpeg_buf

    def get_stats(self) -> dict:
        with self._result_lock:
            r = self._result_store[0]
        age = time.monotonic() - r.timestamp
        # Log every 5 seconds roughly (assuming called at ~1fps for UI polling)
        if int(time.time()) % 5 == 0:
            log.info(f"STATS: Age={age:.2f}s, Tracks={len(r.tracks)}")
        
        tracks = r.tracks if age < cfg.RESULT_MAX_AGE_SEC else []
        return {
            "active": self._active,
            "camera": self.camera_name,
            "fps": round(self._fps_val, 1),
            "motion": r.motion,
            "persons": len(tracks),
            "entries": r.entries,
            "exits": r.exits,
            "tracks": [{"id": t.track_id, "emotion": t.emotion or "–", "action": t.action or "–", "identity": t.identity or ""} for t in tracks],
        }

# ─── Camera Fetching Logic ───────────────────────────────────────────────────

def _fetch_imou_cameras():
    """Return local RTSP camera from environment variables."""
    rtsp_url = os.environ.get("IMOU_RTSP_URL")
    if not rtsp_url:
        return []
    return [{
        "id": "imou_local",
        "name": "Imou Camera (Local RTSP)",
        "status": "🟢 Online",
        "type": "imou_local",
        "url": rtsp_url
    }]

_camera_cache: list[dict] = []
_camera_cache_ts: float   = 0.0
_camera_lock = threading.Lock()

def get_all_cameras_legacy(force=False, user_email=None):
    global _camera_cache, _camera_cache_ts
    with _camera_lock:
        if force or (time.time() - _camera_cache_ts) >= 60 or not _camera_cache:
            base_cams = [{"id": "webcam", "name": "Webcam (Built-in)", "status": "🟢 Online", "type": "webcam"}]
            base_cams.extend(_fetch_imou_cameras())
            _camera_cache = base_cams
            _camera_cache_ts = time.time()
        cameras = list(_camera_cache)
    
# ─── Camera list ───────────────────────────────────────────────────────────────

def get_all_cameras(force=False, user_email=None):
    """Return all cameras: static RTSP list from config + webcam + DB local cameras."""
    cameras = [{"id": "webcam", "name": "Webcam (Built-in)", "status": "🟢 Online", "type": "webcam"}]

    for cam in cfg.CAMERAS:
        cameras.append({
            "id":     cam["id"],
            "name":   cam["name"],
            "status": "🟢 Online",
            "type":   "rtsp",
            "url":    cam["url"],
        })

    if user_email:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT id, name, brand, ip_address FROM local_cameras WHERE owner_email = %s", (user_email,))
            local_cams = cur.fetchall()
            cur.close(); conn.close()
            for row in local_cams:
                cameras.append({
                    "id": f"local_{row['id']}",
                    "name": row['name'],
                    "status": "🟢 Online",
                    "type": "local",
                    "brand": row['brand'],
                    "ip": row['ip_address'],
                    "id":     f"local_{row['id']}",
                    "name":   row['name'],
                    "status": "🟢 Online",
                    "type":   "local",
                    "brand":  row['brand'],
                    "ip":     row['ip_address'],
                })
        except Exception as e:
            log.error(f"Error fetching local cameras: {e}")
    return cameras

# ─── Flask App & Routes ───────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "super_secret_mission_control_key_xyz"
# Hidden folder for AI processing only (not public)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ai_cache")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cctv_logs")
    return psycopg2.connect(db_url, cursor_factory=RealDictCursor)

def init_db():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, name TEXT NOT NULL, company TEXT NOT NULL, password_hash TEXT NOT NULL)")
        cur.execute("CREATE TABLE IF NOT EXISTS local_cameras (id SERIAL PRIMARY KEY, name TEXT, brand TEXT, ip_address TEXT, port INTEGER, username TEXT, password TEXT, stream_path TEXT, owner_email TEXT)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id SERIAL PRIMARY KEY,
                staff_name TEXT NOT NULL,
                staff_number TEXT NOT NULL,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                face_encoding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS member_time_stamp (
                id SERIAL PRIMARY KEY,
                person_id INTEGER,
                camera_name TEXT,
                entry_image TEXT,
                exit_image TEXT,
                merged_image TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
            
        conn.commit()
        cur.close()
        log.info("Database initialized safely (Forensic Vault activated).")
    except Exception as e:
        log.error(f"DB init failed: {e}")
    finally:
        if conn: conn.close()

init_db()
cam_mgr = CameraManager()

# Synchronize Face Recognizer from DB on startup
def sync_ai_knowledge():
    try:
        conn = get_db_connection(); cur = conn.cursor()
        
        # 1. Load all staff images to check encoding validity
        cur.execute("SELECT id, staff_name, face_encoding, image_data FROM uploaded_images")
        rows = cur.fetchall()
        
        knowledge = []
        updates = []
        
        for r in rows:
            needs_update = False
            enc = None
            
            if r['face_encoding']:
                try:
                    enc = json.loads(r['face_encoding'])
                    # If we switched models (e.g. to Facenet 512), re-encode
                    if not isinstance(enc, list) or len(enc) != 512:
                        needs_update = True
                except:
                    needs_update = True
            else:
                needs_update = True
                
            if needs_update:
                try:
                    log.info(f"AI: Re-encoding signature for {r['staff_name']} (Model Migration)...")
                    new_enc = cam_mgr._pipeline.face_rec.extract_embedding(r['image_data'])
                    if new_enc:
                        enc = new_enc
                        updates.append((json.dumps(new_enc), r['id']))
                except Exception as ex:
                    log.warning(f"AI: Failed to re-encode {r['id']}: {ex}")
                    continue
            
            if enc:
                knowledge.append({"name": r['staff_name'], "encoding": enc})
        
        # Save updates back to DB
        if updates:
            for u in updates:
                cur.execute("UPDATE uploaded_images SET face_encoding = %s WHERE id = %s", u)
            conn.commit()
            log.info(f"AI: Updated {len(updates)} staff signatures in database.")
        
        if cam_mgr._pipeline and cam_mgr._pipeline.face_rec:
            cam_mgr._pipeline.face_rec.set_known_faces(knowledge)
            log.info(f"AI loaded {len(knowledge)} face signatures into memory.")
        else:
            log.info("AI: Known faces loaded to disk. Will sync to memory when Emotion Mode is activated.")
        
        cur.close(); conn.close()
    except Exception as e:
        log.error(f"AI Sync failed: {e}")

# Call sync after initialization
sync_ai_knowledge()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session: return redirect(url_for('route_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=["GET", "POST"])
def route_login():
    if request.method == "POST":
        email, password = request.form.get("email"), request.form.get("password")
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = %s", (email,))
        rec = cur.fetchone(); cur.close(); conn.close()
        if rec and werkzeug.security.check_password_hash(rec['password_hash'], password):
            session['user'] = email
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def route_signup():
    if request.method == "POST":
        email, name, company, password = request.form.get("email"), request.form.get("username"), request.form.get("company"), request.form.get("password")
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close(); conn.close()
            return render_template("signup.html", error="Email already registered")
        pw_hash = werkzeug.security.generate_password_hash(password)
        cur.execute("INSERT INTO users (email, name, company, password_hash) VALUES (%s, %s, %s, %s)", (email, name, company, pw_hash))
        conn.commit(); cur.close(); conn.close()
        session['user'] = email
        return redirect(url_for("index"))
    return render_template("signup.html")

@app.route("/logout")
def route_logout():
    session.pop('user', None)
    return redirect(url_for("route_login"))

@app.route("/")
@login_required
def index(): return render_template("dashboard.html")

@app.route("/live")
@login_required
def route_live(): return render_template("live.html")

@app.route("/settings")
@login_required
def route_settings(): return render_template("settings.html")

@app.route("/api/user_profile_legacy")
@login_required
def api_user_profile_legacy():
    email = session.get('user')
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT name, email, company FROM users WHERE email = %s", (email,))
        row = cur.fetchone(); cur.close(); conn.close()
        if not row: return jsonify({"error": "User not found"}), 404
        return jsonify(row)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/cameras")
def api_cameras():
    cameras = get_all_cameras(user_email=session.get('user'))
    return jsonify(cameras)

@app.route("/api/select", methods=["POST"])
def api_select():
    data = request.get_json(force=True)
    cam_id = data.get("id", "imou_local")
    cameras = get_all_cameras(user_email=session.get('user'))
    cam = next((c for c in cameras if c["id"] == cam_id), None)
    if not cam: return jsonify({"error": "Camera not found"}), 404

    if cam["type"] == "webcam":
        cam_mgr.start(0, cam["name"])
    elif cam["type"] == "imou_local":
        cam_mgr.start(cam["url"], cam["name"])
    elif cam["type"] == "rtsp":
        cam_mgr.start(cam["url"], cam["name"])
    elif cam["type"] == "local":
        try:
            db_id = cam_id.replace("local_", "")
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute("SELECT ip_address, port, username, password, stream_path FROM local_cameras WHERE id = %s", (db_id,))
            row = cur.fetchone(); cur.close(); conn.close()
            if not row: return jsonify({"error": "Local camera not found"}), 404
            rtsp_url = f"rtsp://{row['username']}:{row['password']}@{row['ip_address']}:{row['port']}{row['stream_path']}" if row['username'] else f"rtsp://{row['ip_address']}:{row['port']}{row['stream_path']}"
            cam_mgr.start(rtsp_url, cam["name"])
        except Exception as e: return jsonify({"error": str(e)}), 500
    return jsonify({"success": True, "camera": cam["name"]})

@app.route("/api/local_cameras", methods=["GET", "POST"])
@login_required
def api_local_cameras():
    email = session.get('user')
    if request.method == "POST":
        data = request.get_json(force=True)
        try:
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute("INSERT INTO local_cameras (name, brand, ip_address, port, username, password, stream_path, owner_email) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (data.get('name'), data.get('brand', 'Generic'), data.get('ip'), data.get('port', 554), data.get('username', ''), data.get('password', ''), data.get('path', ''), email))
            conn.commit(); cur.close(); conn.close()
            return jsonify({"success": True})
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500
    
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT id, name, brand, ip_address, port, username, password, stream_path FROM local_cameras WHERE owner_email = %s", (email,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return jsonify([{"id": r['id'], "name": r['name'], "brand": r['brand'], "ip": r['ip_address'], "port": r['port'], "username": r['username'], "path": r['stream_path']} for r in rows])
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/local_cameras/<int:cam_id>", methods=["DELETE"])
@login_required
def api_delete_local_camera(cam_id):
    email = session.get('user')
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("DELETE FROM local_cameras WHERE id = %s AND owner_email = %s", (cam_id, email))
        conn.commit(); cur.close(); conn.close()
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/upload_staff", methods=["POST"])
@login_required
def api_upload_staff():
    staff_name = request.form.get('staff_name')
    staff_number = request.form.get('staff_number')
    if not staff_name or not staff_number:
        return jsonify({"success": False, "error": "Staff name and number are required"}), 400

    if 'photos' not in request.files:
        return jsonify({"success": False, "error": "No photos provided"}), 400

    files = request.files.getlist('photos')
    
    try:
        conn = get_db_connection(); cur = conn.cursor()
        
        # Check current count
        cur.execute("SELECT COUNT(*) as count FROM uploaded_images WHERE staff_number = %s", (staff_number,))
        current_count = cur.fetchone()['count']
        
        if current_count + len(files) > 10:
            cur.close(); conn.close()
            return jsonify({"success": False, "error": f"Upload limit exceeded. Max 10 images per staff. Current: {current_count}"}), 400

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                img_data = file.read()
                
                # Pre-calculate embedding for the DB (Memory-only matching)
                encoding_json = None
                try:
                    if cam_mgr._pipeline and cam_mgr._pipeline.face_rec:
                        encoding = cam_mgr._pipeline.face_rec.extract_embedding(img_data)
                        if encoding:
                            encoding_json = json.dumps(encoding)
                    else:
                        log.info("AI: Face recognizer not active. Skipping embedding generation for this upload.")
                except Exception as ex:
                    log.warning(f"Could not encode {filename}: {ex}")

                # DB
                cur.execute("INSERT INTO uploaded_images (staff_name, staff_number, filename, image_data, face_encoding) VALUES (%s, %s, %s, %s, %s)",
                            (staff_name, staff_number, filename, psycopg2.Binary(img_data), encoding_json))

        conn.commit(); cur.close(); conn.close()
        # Refresh AI knowledge immediately
        sync_ai_knowledge()
        return jsonify({"success": True, "count": len(files)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/staff_profiles", methods=["GET"])
@login_required
def api_staff_profiles():
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT DISTINCT staff_name, staff_number FROM uploaded_images")
        staff_list = cur.fetchall()
        
        profiles = []
        for s in staff_list:
            # Get first image ID as thumbnail
            cur.execute("SELECT id FROM uploaded_images WHERE staff_number = %s LIMIT 1", (s['staff_number'],))
            img = cur.fetchone()
            thumb_url = f"/api/image/{img['id']}" if img else ""
            
            cur.execute("SELECT COUNT(*) as count FROM uploaded_images WHERE staff_number = %s", (s['staff_number'],))
            count = cur.fetchone()['count']
            
            profiles.append({
                "name": s['staff_name'],
                "number": s['staff_number'],
                "photo_count": count,
                "thumbnail": thumb_url
            })
        
        cur.close(); conn.close()
        return jsonify({"profiles": profiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/image/<int:img_id>")
def serve_image(img_id):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT image_data FROM uploaded_images WHERE id = %s", (img_id,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row: return "Not found", 404
        
        data = row['image_data']
        if not isinstance(data, bytes): data = data.tobytes()
        return Response(data, mimetype='image/jpeg')
    except Exception as e: return str(e), 500

@app.route("/api/staff_profiles/<staff_number>", methods=["GET"])
@login_required
def api_staff_photos(staff_number):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT staff_name, id FROM uploaded_images WHERE staff_number = %s", (staff_number,))
        rows = cur.fetchall()
        cur.close(); conn.close()
        
        if not rows: return jsonify({"error": "Staff not found"}), 404
        
        staff_name = rows[0]['staff_name']
        urls = [{"url": f"/api/image/{r['id']}"} for r in rows]
        
        return jsonify({"staff": staff_name, "number": staff_number, "photos": urls})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/staff_profiles/<staff_number>", methods=["DELETE"])
@login_required
def api_delete_staff(staff_number):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        # Get name for FS cleanup
        cur.execute("SELECT staff_name FROM uploaded_images WHERE staff_number = %s LIMIT 1", (staff_number,))
        row = cur.fetchone()
        if row:
            import shutil
            staff_dir = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(row['staff_name']))
            if os.path.exists(staff_dir): shutil.rmtree(staff_dir)
            
            cur.execute("DELETE FROM uploaded_images WHERE staff_number = %s", (staff_number,))
            conn.commit()
            
        cur.close(); conn.close()
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/stop", methods=["POST"])
def api_stop():
    cam_mgr.stop_stream()
    return jsonify({"success": True})

@app.route("/api/stream_id")
def api_stream_id():
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


@app.route("/api/notify_manual", methods=["POST"])
def api_notify_manual():
    msg = "📢 *MANUAL ALERT* from MISSION CONTROL UI\n\nA user has triggered a manual notification from the control panel."
    success = notifier.send_message(msg)
    return jsonify({"success": success})


@app.route("/api/dashboard_info")
def api_dashboard_info():
    unique_people = 0
    try:
        if cam_mgr._pipeline and hasattr(cam_mgr._pipeline, "tracker"):
            ds = cam_mgr._pipeline.tracker.tracker
            unique_people = max(0, ds.tracker._next_id - 1)
    except Exception:
        pass
        
    user_email = session.get('user')
    cams = get_all_cameras(user_email=user_email)
    online_count = sum(1 for c in cams if "Online" in c.get("status", ""))
    offline_count = sum(1 for c in cams if "Offline" in c.get("status", ""))
    
    return jsonify({
        "total_motion_events": total_motion_events,
        "unique_people": unique_people,
        "online_cameras": online_count,
        "offline_cameras": offline_count
    })


@app.route("/api/settings_info")
def api_settings_info():
    return jsonify({
        "tg_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        "tg_chats": os.environ.get("TELEGRAM_CHAT_ID", "")
    })

@app.route("/api/user_profile")
@login_required
def api_user_profile():
    email = session.get('user')
    if not email:
        return jsonify({"error": "Not logged in"}), 401
        
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT email, name, company FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    
    if user:
        return jsonify({
            "email": user["email"],
            "name": user["name"],
            "company": user["company"]
        })
    return jsonify({"error": "User not found"}), 404


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = cam_mgr.get_jpeg()
            if frame: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import atexit
    from waitress import serve
    atexit.register(cam_mgr.stop_all)
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting MISSION CONTROL Web UI on http://localhost:{port}")
    serve(app, host="0.0.0.0", port=port, threads=32)
