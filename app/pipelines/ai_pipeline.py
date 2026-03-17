"""
core/pipeline.py  –  AI Pipeline Thread
──────────────────────────────────────────────────────────────────────────────
Runs in a background daemon thread.  Drains a frame queue and produces
AIResult objects at full AI throughput, completely decoupled from the
render / web serving threads.

Processing stages per frame
───────────────────────────
  1. Motion gate (MOG2 subtractor) — cheap, skips idle scenes
  2. YOLOv8 person detection        — every cfg.YOLO_SKIP_FRAMES
  3. DeepSORT tracker               — every frame
  4. Per track:
       • SlowFast action recognition  (ThreadPoolExecutor, non-blocking)
       • DeepFace emotion detection   (ThreadPoolExecutor, non-blocking)
       • DeepFace face recognition    (ThreadPoolExecutor, non-blocking)
  5. Telegram notification on new person / staff match

Design notes
────────────
• Models are loaded ONCE in __init__; switching cameras does NOT reload them.
• Emotion and recognition run in separate 1-worker executors so they never
  block the YOLO / tracker loop.
• AI_TOGGLES are read from ``app.extensions`` at runtime, allowing live
  on/off switching from the web UI without restarting anything.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import os
import cv2
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from app.core import config as cfg
from app.core.data_types import AIResult, TrackResult, scale_box
from app.db.session import get_db_connection
from app.services.attendance_service import (
    log_movement,
    log_person,
    track_staff_attendance,
    update_movement_classification,
)
from app.services.ai.action_detection import ActionDetector
from app.services.ai.emotion_detection import EmotionDetector
from app.services.ai.face_detection import FaceDetector
from app.services.ai.face_recognition import FaceRecognizer
from app.services.ai.motion_detection import MotionDetector
from app.services.ai.notifier import TelegramNotifier
from app.services.ai.object_detection import PersonDetector
from app.services.ai.tracking import PersonTracker
from app.services.attendance_tracker import AttendanceTracker

log = logging.getLogger("pipeline")

# Motion events counter — exposed to the API via core.camera_manager
_total_motion_events: int = 0


class AIPipeline(threading.Thread):
    """
    Background AI thread: motion → YOLO → DeepSORT → action → emotion → recognition.

    Parameters
    ----------
    in_queue     : queue.Queue fed by the render thread (resized AI-res frames)
    result_store : list[AIResult] – single-element list, shared with the render thread
    result_lock  : threading.Lock protecting result_store
    upload_folder: path to staff face DB (for face recognition)
    """

    def __init__(
        self,
        in_queue:      "queue.Queue[np.ndarray | None]",
        result_store:  list,
        result_lock:   threading.Lock,
        upload_folder: str = "",
    ):
        super().__init__(daemon=True, name="AIPipeline")
        self.in_queue     = in_queue
        self.result_store = result_store
        self.result_lock  = result_lock
        self._stop_evt    = threading.Event()

        # ── Load all models ONCE ──────────────────────────────────────────────
        log.info("AI pipeline: loading models …")

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
        self.face_det = FaceDetector()
        self.emotion_det = EmotionDetector(
            skip_frames=cfg.EMOTION_SKIP_FRAMES,
            backend=cfg.EMOTION_BACKEND,
            min_face_pixels=cfg.EMOTION_MIN_FACE_PX,
        )
        self.action_det = ActionDetector(
            enabled=cfg.ENABLE_ACTION,
            buffer_size=cfg.ACTION_BUFFER_SIZE,
            slow_stride=cfg.ACTION_SLOW_STRIDE,
            device=cfg.ACTION_DEVICE,
        )
        self.face_rec = FaceRecognizer(
            db_path=upload_folder or cfg.__dict__.get("UPLOAD_FOLDER", "static/uploads"),
            skip_frames=15,
            backend=cfg.EMOTION_BACKEND,
        )

        self.camera_name = "Camera"  # Safe default to avoid AttributeError

        # Single-worker executors keep heavy models off the tracker loop
        self._emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")
        self._rec_executor     = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rec")
        self._emotion_futures: dict[int, Future] = {}
        self._rec_futures:     dict[int, Future] = {}

        from app.services.ai.notifier import TelegramNotifier
        self.notifier = TelegramNotifier()
        self._notified_identities: set[int] = set()
        # Smart attendance tracker — handles IN/OUT cycles, movement log, EOD
        self.attendance_tracker = AttendanceTracker(notifier=self.notifier)

        self._frame_idx        = 0
        self._last_detections: list = []
        self._last_objects:    dict = {"food": [], "phone": []}
        self._last_motion_time: float = time.monotonic()

        self._full_w: int = cfg.FRAME_WIDTH
        self._full_h: int = cfg.FRAME_HEIGHT

        # Detection-based Counting & Forensic Logs
        self._seen_ids: set[int] = set()
        self._active_ids: set[int] = set()
        self.entries = 0
        self.exits   = 0
        self._entry_best_frames: dict[int, np.ndarray] = {}
        self._exit_best_frames:  dict[int, np.ndarray] = {}
        self._best_entry_scores: dict[int, float]      = {}
        self._best_exit_scores:  dict[int, float]      = {}
        self._track_start_time:  dict[int, float]      = {}
        self._exit_taken: set[int] = set()
        self._db_sessions: dict[int, int] = {}
        self._entry_filenames: dict[int, str] = {}
        self._exit_filenames: dict[int, str] = {}
        self._track_att_ids: dict[int, int] = {}
        self._current_frame_motion_id = None
        self._current_frame_motion_img = None
        self._last_motion_img = None  # Keep last motion image for logging persons across frames
        self._last_member_log_mono: dict[int, float] = {}
        self._last_face_reload = time.monotonic()
        os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)

        log.info("AI pipeline: all models ready.")

    # ── Forensic Log Helpers ──────────────────────────────────────────────────
    
    def _calculate_clarity(self, frame, bbox=None) -> float:
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
        except Exception:
            return 0.0


    # ── Thread control ────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_evt.set()
        self.in_queue.put(None)

    def run(self) -> None:
        log.info("AI pipeline started.")
        while not self._stop_evt.is_set():
            try:
                # ZERO-LATENCY FLUSH: Discard all old buffered frames so AI always works on the ABSOLUTE LATEST frame.
                while self.in_queue.qsize() > 1:
                    try: self.in_queue.get_nowait()
                    except Exception: break

                ai_frame = self.in_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if ai_frame is None:
                break
            result = self._process(ai_frame, self._full_w, self._full_h)
            with self.result_lock:
                self.result_store[0] = result
        self._emotion_executor.shutdown(wait=False)
        self._rec_executor.shutdown(wait=False)
        log.info("AI pipeline stopped.")

    # ── Non-blocking executor helpers ─────────────────────────────────────────

    def _submit_emotion(self, crop: np.ndarray, tid: int) -> None:
        fut = self._emotion_futures.get(tid)
        if fut is not None and not fut.done():
            return
        self._emotion_futures[tid] = self._emotion_executor.submit(
            self.emotion_det.analyse, crop.copy(), tid
        )

    def _submit_rec(self, ai_frame: np.ndarray, bbox: list[float], tid: int) -> None:
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
                        res = self.face_rec.recognize(face_crop, tid)
                        self.face_rec._cache[tid] = res
                        return res["name"]
            except Exception as e:
                log.debug("Rec task error: %s", e)
            return "Unknown"

        self._rec_futures[tid] = self._rec_executor.submit(_task)

    # ── Main processing cycle ─────────────────────────────────────────────────

    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int) -> AIResult:
        global _total_motion_events

        try:
            try:
                from app.core.extensions import AI_TOGGLES
                if self._frame_idx % 60 == 0:
                    log.info("AI: Current Toggles: %s", AI_TOGGLES)
            except ImportError:
                AI_TOGGLES = {"person": True, "action": True, "emotion": True}

            self._frame_idx += 1
            now = time.monotonic()
            
            # 0. Periodic Face Reload (Every 2 minutes)
            if now - self._last_face_reload > 120:
                log.info("AI: Reloading staff faces from folder...")
                self.face_rec.load_from_folder(self.face_rec.db_path)
                self._last_face_reload = now

            ai_h, ai_w = ai_frame.shape[:2]
            sx = full_w / ai_w
            sy = full_h / ai_h

            if self._frame_idx % 10 == 0:
                log.info("AI: Processing frame #%d", self._frame_idx)

            # 1) Movement stage: store manual motion snapshot if MOG2 triggers
            motion, mask = self.motion_det.detect(ai_frame)
            now = time.monotonic()
            movement_id = None
            movement_label = "unknown"
            movement_conf = 0.0
            force_person_scan = False
            
            # Cooldown logic for MOG2-based motion log entries
            if motion and (now - self._last_motion_time) > 2.0:
                primitive_log("MOG2 Motion triggered.")
                motion_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                motion_filename = f"motion_{self.camera_name.replace(' ', '_')}_{motion_ts}.jpg"
                motion_dir = os.path.join("static", "uploads", "movement")
                os.makedirs(motion_dir, exist_ok=True)
                motion_path = os.path.join(motion_dir, motion_filename)
                
                cv2.imwrite(motion_path, ai_frame)
                primitive_log(f"Saved motion image: {motion_path}")
                
                self._current_frame_motion_img = f"static/uploads/movement/{motion_filename}"
                self._last_motion_img = self._current_frame_motion_img
                
                movement_id = log_movement(self.camera_name, self._current_frame_motion_img)
                movement_label, movement_conf = self.person_det.classify_motion(ai_frame)
                force_person_scan = movement_label == "human"
                if movement_id:
                    update_movement_classification(movement_id, movement_label, movement_conf)
                self._last_motion_time = now

            # 2) Human detection/tracking
            if AI_TOGGLES.get("person", True):
                if force_person_scan or self._frame_idx % max(1, cfg.YOLO_SKIP_FRAMES) == 0:
                    self._last_detections = self.person_det.detect(ai_frame)
                    if len(self._last_detections) > 0:
                        log.info("AI: YOLO found %d persons", len(self._last_detections))
            else:
                self._last_detections = []

            if AI_TOGGLES.get("action", True) and self._frame_idx % 5 == 0:
                self._last_objects = self.person_det.detect_objects(ai_frame)
            elif not AI_TOGGLES.get("action", True):
                self._last_objects = {"food": [], "phone": []}

            # 3) Tracker
            raw_tracks = self.tracker.update(self._last_detections, ai_frame)
            active_ids = {t.track_id for t in raw_tracks}
            if len(active_ids) > 0 or len(self._last_detections) > 0:
                log.info("AI: Tracker active_ids: %s", list(active_ids))
            
            # 1. New Entrants
            for tid in active_ids:
                if tid not in self._seen_ids:
                    self.entries += 1
                    self._seen_ids.add(tid)
                    self._track_start_time[tid] = now
                    log.info(f"COUNT: New person entering frame (ID: {tid}). Total Entries: {self.entries}")

            # 2. Exits: IDs that WERE active but are now gone
            for tid in list(self._active_ids):
                if tid not in active_ids:
                    self.exits += 1
                    log.info(f"COUNT: Person left (ID: {tid}). Total Exits: {self.exits}")
                    
                    for d in [self._entry_best_frames, self._exit_best_frames, self._best_entry_scores, 
                              self._best_exit_scores, self._track_start_time, self._db_sessions, 
                              self._entry_filenames, self._exit_filenames, self._track_att_ids]:
                        if tid in d: del d[tid]
                    if tid in self._exit_taken: self._exit_taken.remove(tid)

            # 4) Promotion stage: movement_log -> member_timestamp
            # FALLBACK: If we have persons but no motion image yet, capture one NOW.
            if not self._last_motion_img and raw_tracks:
                log.info("DEBUG: Detections present but no motion image. Capturing fallback snapshot...")
                motion_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                motion_filename = f"det_fallback_{self.camera_name.replace(' ', '_')}_{motion_ts}.jpg"
                motion_dir = os.path.join("static", "uploads", "movement")
                os.makedirs(motion_dir, exist_ok=True)
                motion_path = os.path.join(motion_dir, motion_filename)
                cv2.imwrite(motion_path, ai_frame)
                self._last_motion_img = f"static/uploads/movement/{motion_filename}"
                log.info(f"DEBUG: Fallback snapshot saved: {self._last_motion_img}")

            if self._last_motion_img and raw_tracks:
                for track in raw_tracks:
                    tid = track.track_id
                    if (now - self._last_member_log_mono.get(tid, 0.0)) < 1.5:
                        continue

                    bbox = track.bbox
                    bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                    bx2 = min(ai_w, bx2)
                    by2 = min(ai_h, by2)
                    person_crop = ai_frame[by1:by2, bx1:bx2]
                    if person_crop.size == 0:
                        continue

                    rec = self.face_rec._cache.get(tid, {"id": None, "name": "Unknown", "display_id": ""})
                    if rec.get("name", "Unknown") == "Unknown":
                        try:
                            rec = self.face_rec.recognize(person_crop, tid)
                        except Exception as e:
                            log.debug("Direct recognition failed for tid=%s: %s", tid, e)

                    identity = rec.get("name", "Unknown")
                    staff_id = rec.get("id")
                    
                    # FINAL HUMAN VERIFICATION
                    # If staff, it's definitely a human. If not staff, we MUST verify it's a "human" and not a false positive (shadow/light).
                    if staff_id and identity != "Unknown":
                        person_type = "staff"
                    else:
                        # Double-check classification for unknown detections
                        v_label, v_conf = self.person_det.classify_motion(person_crop)
                        if v_label != "human":
                            log.debug(f"DEBUG: Skipping promotion for TID {tid} - classify result: {v_label} ({v_conf:.2f})")
                            continue
                        person_type = "unknown"

                    confidence = self._calculate_clarity(ai_frame, bbox)
                    
                    if (now - self._track_start_time.get(tid, now)) < 0.5:
                        continue

                    log.info(f"DEBUG: Promoting track {tid} ({person_type}) to member_timestamp")
                    member_id = log_person(
                        self.camera_name,
                        person_type,
                        staff_id,
                        self._last_motion_img,
                        confidence,
                        staff_name=identity if person_type == "staff" else None,
                    )
                    if member_id:
                        self._last_member_log_mono[tid] = now

                    if person_type == "staff" and staff_id:
                        log.info(f"DEBUG: STAFF FOUND! Calling track_staff_attendance for {identity} (ID:{staff_id})")
                        attendance = track_staff_attendance(
                            staff_id=staff_id,
                            staff_name=identity,
                            entry_image=self._last_motion_img,
                        )
                        log.info(f"DEBUG: track_staff_attendance result: {attendance}")
                        if attendance.get("is_first_entry"):
                            self.notifier.send_message(
                                f"✅ *ATTENDANCE*: {identity} first detected today on {self.camera_name}."
                            )
                    else:
                        log.info(f"DEBUG: Person type is {person_type}, staff_id is {staff_id} - NOT calling attendance")
                    
                    log.info(
                        "Flow: snapshot=%s -> member_timestamp person_type=%s tid=%s",
                        self._last_motion_img, person_type, tid,
                    )

            self._active_ids = active_ids
            # MOVED: self._last_motion_time = now  <-- This was blocking the motion block if someone was in frame!

            # 4. Per-track processing for UI
            track_results: list[TrackResult] = []
            for track in raw_tracks:
                tid  = track.track_id
                bbox = track.bbox
                bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                bx2 = min(ai_w, bx2); by2 = min(ai_h, by2)

                action   = ""
                emotion  = ""
                identity = ""

                if AI_TOGGLES.get("action", True):
                    bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
                    ratio = bw / bh
                    if ratio > 1.4:    geometry_label = "😴 sleeping"
                    elif ratio > 0.85: geometry_label = "🪑 sitting"
                    else:              geometry_label = "🧍 standing"

                    person_crop    = ai_frame[by1:by2, bx1:bx2]
                    slowfast_label = self.action_det.update(tid, person_crop) if person_crop.size > 0 else ""

                    def _overlaps(ob, pb, exp=40):
                        ox1, oy1, ox2, oy2 = ob
                        px1, py1, px2, py2 = pb
                        return ox1 < px2 + exp and ox2 > px1 - exp and oy1 < py2 + exp and oy2 > py1 - exp

                    psb        = [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy]
                    near_food  = any(_overlaps(f, psb) for f in self._last_objects.get("food", []))
                    near_phone = any(_overlaps(p, psb) for p in self._last_objects.get("phone", []))

                    TAGS = ("⚠", "🏃", "💪", "📖", "✍", "🍽", "💬", "💃", "🎵")
                    if near_food:    action = "🍽 eating"
                    elif near_phone: action = "💻 working"
                    elif slowfast_label and any(t in slowfast_label for t in TAGS): action = slowfast_label
                    else:            action = geometry_label

                if AI_TOGGLES.get("emotion", True):
                    person_crop = ai_frame[by1:by2, bx1:bx2]
                    if person_crop.size > 0: self._submit_emotion(person_crop, tid)
                    emotion = self.emotion_det._cache.get(tid, "")

                if AI_TOGGLES.get("person", True):
                    cached_res = self.face_rec._cache.get(tid, {"name": "Unknown", "display_id": ""})
                    identity = cached_res["name"]
                    display_id = cached_res["display_id"]
                    
                    try:
                        is_unknown = (identity == "" or identity == "Unknown")
                        active_rec_tasks = sum(1 for f in self._rec_futures.values() if not f.done())
                        if (is_unknown or (self._frame_idx % 300 == 0)) and active_rec_tasks < 2:
                            self._submit_rec(ai_frame, bbox, tid)
                        
                        self.notifier.notify_person(tid, self.camera_name, action)
                    except Exception as e:
                        log.error("Notification error: %s", e)

                track_results.append(TrackResult(
                    track_id=tid,
                    bbox_full=scale_box(bbox, sx, sy),
                    face_bbox_full=None,
                    emotion=emotion, action=action, identity=identity,
                    display_id=display_id if 'display_id' in locals() else ""
                ))

            # 5. Cache cleanup
            if self._frame_idx % 120 == 0:
                self.emotion_det.purge(active_ids)
                self.action_det.purge(active_ids)
                self.face_rec.purge(active_ids)
                for d in (self._emotion_futures, self._rec_futures):
                    for tid in list(d):
                        if tid not in active_ids: del d[tid]
                for tid in list(self._notified_identities):
                    if tid not in active_ids:
                        self._notified_identities.discard(tid)
                for tid in list(self._last_member_log_mono):
                    if tid not in active_ids:
                        self._last_member_log_mono.pop(tid, None)
                if len(self._seen_ids) > 1000:
                    self._seen_ids = self._seen_ids.intersection(active_ids)

            return AIResult(motion=motion, motion_mask=mask, tracks=track_results, 
                        entries=self.entries, exits=self.exits)

        except Exception as e:
            log.error("CRITICAL AI PIPELINE ERROR: %s", e, exc_info=True)
            return AIResult(motion=False, entries=self.entries, exits=self.exits)
