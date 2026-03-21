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
        self.camera_roles = []       # Active roles for current camera (entry, exit, general)

        # Multi-worker executors to handle many people in parallel
        self._emotion_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="emotion")
        self._rec_executor     = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rec")
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

        # Detection-based Counting & Forensic Logs (Refactored to Track-Based)
        self.track_data: dict[int, dict] = {} # {tid: {"frames": [], "recognized": bool, "identity": dict, "best_frame": ndarray, ...}}
        self._seen_ids: set[int] = set()
        self._active_ids: set[int] = set()
        self.entries = 0
        self.exits   = 0
        self._exit_taken: set[int] = set()
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
            
        def _task():
            try:
                res = self.emotion_det.analyse(crop, tid)
                if tid in self.track_data:
                    self.track_data[tid]["emotion"] = res
                return res
            except Exception as e:
                log.debug("Emotion task error: %s", e)
                return ""
                
        self._emotion_futures[tid] = self._emotion_executor.submit(_task)

    def _submit_rec(self, ai_frame: np.ndarray, bbox: list[float], tid: int) -> None:
        """Asynchronously detect face AND recognize identity."""
        fut = self._rec_futures.get(tid)
        if fut is not None and not fut.done():
            return
            
        def _task():
            try:
                # Local re-detection of face inside the person box
                face_bbox = self.face_det.detect_in_crop(ai_frame, bbox)
                
                # FALLBACK: If standard detector fails, try a lower threshold for side-profiles (0.3)
                if not face_bbox:
                    face_bbox = self.face_det.detect_in_crop(ai_frame, bbox, threshold=0.3)
                if face_bbox:
                    fx1, fy1, fx2, fy2 = (int(v) for v in face_bbox)
                    face_crop = ai_frame[max(0,fy1):min(ai_frame.shape[0],fy2), max(0,fx1):min(ai_frame.shape[1],fx2)]
                    if face_crop.size > 0:
                        # PROACTIVE: Allow first attempt even if quality is just "okay"
                        # We use 15.0 as a floor for complete garbage images
                        if not self.face_det.is_high_quality(face_crop, min_sharpness=15.0):
                            log.debug(f"AI: Recognition skipped for Track {tid} - Image too poor")
                            return "Unknown"

                        res = self.face_rec.recognize(face_crop, tid)
                        
                        # UPDATE: Identity Voting Mechanism
                        if res and res.get("name") != "Unknown":
                            if tid in self.track_data:
                                td = self.track_data[tid]
                                votes = td.get("id_votes", {})
                                name = res["name"]
                                votes[name] = votes.get(name, 0) + 1
                                td["id_votes"] = votes
                                
                                # IDENTITY OVERRIDE: If a name gets 2+ votes OR is high confidence, update it
                                is_better = (votes.get(name, 0) > votes.get(td.get("identity"), 0))
                                if votes[name] >= 2 or is_better:
                                    old_id = td.get("identity", "Unknown")
                                    td["identity"] = name
                                    td["staff_db_id"] = res.get("id")
                                    td["display_id"] = f"{name} (#{tid})"
                                    td["recognized"] = True

                                    # UPDATE DATABASE (LATE SYNC)
                                    if td.get("db_id") and (old_id != name):
                                        from app.services.attendance_service import update_person_identity
                                        log.info(f"AI: [AUTO-CORRECT] {old_id} -> {name} for Track {tid}")
                                        update_person_identity(member_id=td["db_id"], staff_id=td["staff_db_id"], staff_name=name)
                        
                        return res.get("name", "Unknown")
            except Exception as e:
                log.debug("Rec task error: %s", e)
            return "Unknown"

        self._rec_futures[tid] = self._rec_executor.submit(_task)

    # ── Main processing cycle ─────────────────────────────────────────────────

    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int) -> AIResult:
        global _total_motion_events
        motion = False
        mask = None
        track_results = []

        try:
            try:
                from app.core.extensions import AI_TOGGLES
                if self._frame_idx % 60 == 0:
                    log.info("AI: Current Toggles: %s", AI_TOGGLES)
            except ImportError:
                AI_TOGGLES = {"person": True, "action": True, "emotion": True}

            self._frame_idx += 1
            now = time.monotonic()
            
            # 0. Periodic Face Reload (Every 2 minutes) - Async so it doesn't stop YOLO
            if now - self._last_face_reload > 120:
                log.info("AI: Triggering background reload of staff faces...")
                self._rec_executor.submit(self.face_rec.load_from_folder, self.face_rec.db_path)
                self._last_face_reload = now

            ai_h, ai_w = ai_frame.shape[:2]
            sx = full_w / ai_w
            sy = full_h / ai_h

            if self._frame_idx % 10 == 0:
                log.info("AI: Processing frame #%d", self._frame_idx)

            # --- 1. Movement stage ---
            motion, mask = self.motion_det.detect(ai_frame)
            if motion:
                _total_motion_events += 1
            
            # --- 2. YOLO Detection & Tracking (Every Frame) ---
            self._last_detections = self.person_det.track(ai_frame, persist=True)
            yolo_human = len(self._last_detections) > 0
            
            if yolo_human:
                self._last_yolo_human_time = now
                if self._frame_idx % 30 == 0: 
                    log.info(f"AI: YOLO found {len(self._last_detections)} persons")

            # Snapshot Logic: Capture for new tracks or if first time seeing human in 5s
            trigger_snapshot = yolo_human and (now - self._last_motion_time) > 5.0
            
            if trigger_snapshot:
                log.info(f"Movement Triggered: Watchdog_YOLO_Human={yolo_human}")
                motion_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                motion_filename = f"motion_{self.camera_name.replace(' ', '_')}_{motion_ts}.jpg"
                motion_dir = os.path.join("static", "uploads", "movement")
                os.makedirs(motion_dir, exist_ok=True)
                motion_path = os.path.join(motion_dir, motion_filename)
                
                cv2.imwrite(motion_path, ai_frame)
                log.info(f"Saved movement snapshot: {motion_path}")
                
                self._current_frame_motion_img = f"static/uploads/movement/{motion_filename}"
                self._last_motion_img = self._current_frame_motion_img
                
                self._current_frame_motion_id = log_movement(
                    camera_id=self.camera_name, 
                    image_path=self._current_frame_motion_img, 
                    track_id=None, 
                    event_type='MOTION_YOLO'
                )
                if self._current_frame_motion_id:
                    update_movement_classification(self._current_frame_motion_id, 'human', 1.0)
                self._last_motion_time = now

            # 1.5) Object detection (Phone/Food) for rule-based actions
            if self._frame_idx % 10 == 0:
                self._last_objects = self.person_det.detect_objects(ai_frame)

            # 2) Human detection/tracking (Already scanned in Watchdog logic above)
            pass

            # 3) Tracker Update (ByteTrack)
            raw_tracks = self.tracker.update(self._last_detections)
            
            # --- SPATIAL DEDUPLICATION (Fix for multiple boxes on one person) ---
            # Sort by track_id (older tracks first) to prioritize stability
            raw_tracks = sorted(raw_tracks, key=lambda t: t.track_id)
            filtered_tracks = []
            for t in raw_tracks:
                keep = True
                for ft in filtered_tracks:
                    iou = self._calculate_iou(t.bbox, ft.bbox)
                    if iou > 0.5: # 50% overlap means it's likely the same person
                        keep = False
                        break
                if keep:
                    filtered_tracks.append(t)
            
            raw_tracks = filtered_tracks
            active_ids = {t.track_id for t in raw_tracks}
            
            # ── PER-TRACK STATE PROCESSING ──
            for track in raw_tracks:
                tid = track.track_id
                bbox = track.bbox
                
                # Initialize track data if new
                if tid not in self.track_data:
                    self.track_data[tid] = {
                        "frames": [],           
                        "recognized": False,    
                        "logged": False,        
                        "identity": "Unknown",  
                        "staff_id": None,       
                        "best_frame": None,     
                        "best_full": None,      
                        "best_clarity": -1.0,   
                        "last_log_time": 0,     
                        "start_time": now,
                        "db_id": None,          
                        "missing_count": 0,     # GRACE PERIOD
                        "is_accounted": False   # Has it been counted in self.entries?
                    }
                
                td = self.track_data[tid]
                td["missing_count"] = 0 # Reset grace period
                
                # Increment entries ONLY once per unique STABLE track lifecycle
                if tid > 0 and not td.get("is_accounted"):
                    self.entries += 1 
                    td["is_accounted"] = True
                    log.info(f"COUNT: Person entering (ID: {tid}). Total: {self.entries}")
                
                current_id = td.get("identity", "Unknown")
                
                # --- PRE-ID POLLING (Always try to get identity before logging/HUD) ---
                cached = self.face_rec._cache.get(tid)
                if cached:
                    new_id = cached.get("name", "Unknown")
                    if new_id != "Unknown":
                        td["identity"] = new_id
                        current_id = new_id
                        td["staff_db_id"] = cached.get("id")
                        td["display_id"] = f"{new_id} (#{tid})"
                        
                        # Late Update: If already logged as Unknown (or wrong ID), fix it now across all logs
                        if td["logged"] and td.get("last_synced_id") != new_id:
                            from app.services.attendance_service import update_person_identity
                            log.info(f"AI: [IDENTITY SYNC] Synchronizing Track {tid} -> {new_id} in all logs")
                            update_person_identity(
                                member_id=td.get("db_id"), 
                                staff_id=td["staff_db_id"], 
                                staff_name=new_id,
                                track_id=tid
                            )
                            td["last_synced_id"] = new_id

                # A. Frame Collection...
                
                # A. Frame Collection (collect up to 50 for better selection and 30-frame log delay)
                td["track_frame_count"] = td.get("track_frame_count", 0) + 1
                
                # NEW: High-frequency movement logging (Every 5 frames)
                if td["track_frame_count"] % 5 == 0:
                    mv_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
                    mv_filename = f"move_{tid}_{mv_ts}.jpg"
                    mv_path = os.path.join("static", "uploads", "movement", mv_filename)
                    cv2.imwrite(mv_path, ai_frame)
                    
                    # Log movement (using global imports)
                    identity_name = td.get("identity", "Unknown")
                    s_id = td.get("staff_db_id")
                    p_type = "staff" if identity_name != "Unknown" else "unknown"
                    
                    mv_id = log_movement(
                        camera_id=self.camera_name,
                        image_path=f"static/uploads/movement/{mv_filename}",
                        track_id=tid,
                        event_type='MOVE_TRACK',
                        staff_id=s_id,
                        staff_name=identity_name if identity_name != "Unknown" else None,
                        person_type=p_type
                    )
                    if mv_id:
                        # Auto-classify as human since it's from a person track
                        update_movement_classification(mv_id, 'human', 1.0)

                if len(td["frames"]) < 50:
                    clarity = self._calculate_clarity(ai_frame, bbox)
                    bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                    bx2, by2 = min(ai_w, bx2), min(ai_h, by2)
                    crop = ai_frame[by1:by2, bx1:bx2]
                    
                    if crop.size > 0:
                        td["frames"].append({"clarity": clarity})
                        if clarity > td["best_clarity"]:
                            td["best_clarity"] = clarity
                            td["best_frame"] = crop.copy() # Face/Person crop
                            # Store full frame with a box for context
                            full_snap = ai_frame.copy()
                            cv2.rectangle(full_snap, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                            td["best_full"] = full_snap
                            td["best_bbox"] = bbox

                # B. Recognition Trigger (PROACTIVE: Try early, try often)
                is_much_better = td["best_clarity"] > (td.get("last_trigger_clarity", 0) * 1.5)
                
                num_frames = len(td["frames"])
                should_trigger = not td["recognized"] and (
                    num_frames == 1 or            # IMMEDIATE: Try once as soon as person enters
                    num_frames % 5 == 0 or        # PERIODIC: Retry every 5 frames if still unknown
                    td["best_clarity"] > 400      # HIGH QUALITY: Force retry if we found a super sharp frame
                )
                
                # Throttle: Only trigger again if we haven't tried in 2s OR if we found a MUCH sharper frame
                last_try_time = td.get("last_rec_time", 0)
                if should_trigger and (now - last_try_time > 2.0 or is_much_better):
                    if td["best_frame"] is not None:
                        td["last_rec_time"] = now
                        td["last_trigger_clarity"] = td["best_clarity"]
                        self._submit_rec(ai_frame.copy(), td["best_bbox"], tid)
                        if AI_TOGGLES.get("action", True):
                            self._submit_emotion(td["best_frame"], tid)
                        log.info(f"AI: Triggering recognition for Track {tid} (Clarity: {td['best_clarity']:.1f}, Try #{td.get('id_votes', {}).get('attempts', 0) + 1})")
                        
                        # Increment attempts
                        votes = td.get("id_votes", {})
                        votes["attempts"] = votes.get("attempts", 0) + 1
                        td["id_votes"] = votes

                # B. Recognition Trigger...

                # D. Entry Logging (IMMEDIATE: Log first, name later)
                # We log as soon as we have 1 stable frame to ensure the table is updated first
                if not td["logged"] and len(td["frames"]) >= 1:
                    # STRICT CLARITY: User wants clear photos, but 100 is a safe floor
                    if td["best_clarity"] < 100:
                        log.debug(f"AI: Skipping Entry log for Track {tid} - Low Clarity ({td['best_clarity']:.1f})")
                    else:
                        person_type = "staff" if current_id != "Unknown" else "unknown"
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Save FULL FRAME evidence (Context)
                        full_filename = f"entry_full_{tid}_{ts}.jpg"
                        full_path = os.path.join("static", "uploads", "movement", full_filename)
                        
                        # FALLBACK: If best_full is not yet captured (likely due to frame 5 trigger), use current frame
                        snap_to_save = td.get("best_full")
                        if snap_to_save is None:
                            snap_to_save = ai_frame.copy()
                            cv2.rectangle(snap_to_save, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        
                        cv2.imwrite(full_path, snap_to_save)
                        
                        # Save FACE CROP (Forensics)
                        crop_filename = f"entry_crop_{tid}_{ts}.jpg"
                        crop_path = os.path.join("static", "uploads", "movement", crop_filename)
                        
                        crop_to_save = td.get("best_frame")
                        if crop_to_save is None:
                            bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                            crop_to_save = ai_frame[by1:by2, bx1:bx2]
                            
                        if crop_to_save.size > 0:
                            cv2.imwrite(crop_path, crop_to_save)

                        from app.services.attendance_service import log_person, get_recent_sighting
                        
                        # DEDUPLICATION: Avoid row spam for the same staff member
                        s_id = td.get("staff_db_id")
                        existing_visit_id = None
                        if s_id:
                            existing_visit_id = get_recent_sighting(s_id, self.camera_name)
                        
                        if existing_visit_id:
                            log.info(f"AI: [DEDUPLICATED] Track {tid} ({current_id}) linked to recent visit {existing_visit_id}")
                            td["db_id"] = existing_visit_id
                        else:
                            res_id = log_person(
                                camera_id=self.camera_name,
                                person_type=person_type,
                                staff_id=s_id,
                                image_path=f"static/uploads/movement/{full_filename}", # UI shows context
                                confidence=td["best_clarity"],
                                staff_name=current_id if person_type == "staff" else None,
                                track_id=tid,
                                event_type='ENTRY'
                            )
                            td["db_id"] = res_id
                        
                        td["logged"] = True
                        log.info(f"AI: Logged {person_type} Entry for Track {tid} ({current_id})")

            # ── EXIT HANDLING (With 30-frame Grace Period) ──
            # self._active_ids contains IDs from PREVIOUS frame
            for tid in list(self._active_ids):
                if tid not in active_ids:
                    td = self.track_data.get(tid)
                    if not td: continue
                    
                    td["missing_count"] += 1
                    # Only confirm exit after 15 frames of silence (approx 1s)
                    # Lowered from 30 to 15 to handle rapid in/out testing better
                    if td["missing_count"] > 15:
                        self.exits += 1
                        if td["logged"]:
                            # ROLE CHECK: Only log EXIT if the camera has the 'exit' role
                            is_exit_cam = "exit" in [r.lower() for r in self.camera_roles]
                            
                            exit_image_path = ""
                            best_full = td.get("best_full")
                            if best_full is not None:
                                exit_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                exit_filename = f"exit_full_{tid}_{exit_ts}.jpg"
                                exit_image_path = os.path.join("static", "uploads", "movement", exit_filename)
                                cv2.imwrite(exit_image_path, best_full)
                                exit_image_path = f"static/uploads/movement/{exit_filename}"

                            if is_exit_cam:
                                from app.services.attendance_service import update_exit_logs
                                update_exit_logs(
                                    member_id=td.get("db_id"),
                                    movement_id=None, 
                                    exit_image=exit_image_path,
                                    merged_image="",
                                    track_id=tid
                                )
                                log.info(f"AI: Logged Formal Exit for Track {tid} (Evidence: {exit_image_path}) on Exit Camera")
                            else:
                                log.info(f"AI: Track {tid} disappeared, but Camera {self.camera_name} is NOT an Exit role. Skipping formal logs.")
                        
                        # Cleanup memory
                        del self.track_data[tid]
            
            # Combine current detection IDs + grace-period IDs for persistence
            persistence_ids = set(active_ids)
            for tid, td in self.track_data.items():
                if td.get("missing_count", 0) > 0:
                    persistence_ids.add(tid)
            self._active_ids = persistence_ids

            # 4. Per-track results for UI/HUD
            track_results: list[TrackResult] = []
            for track in raw_tracks:
                tid  = track.track_id
                bbox = track.bbox
                td = self.track_data.get(tid, {})
                
                # A. Identity & Display
                identity = td.get("identity", "Unknown")
                
                # Update Attendance Tracker (Heartbeat)
                if identity != "Unknown":
                    # Only send heartbeats if 'general' role is assigned (default to True if roles empty for backward compatibility?)
                    # Requirement says: "Assign one camera to all roles... Assign different cameras for different roles"
                    # We'll treat a camera as a General Monitoring camera if it has 'general' role OR NO roles assigned (legacy).
                    is_general = (not self.camera_roles) or ("general" in [r.lower() for r in self.camera_roles])
                    if is_general:
                        self.attendance_tracker.heartbeat(identity, self.camera_name, roles=self.camera_roles)
                    
                clarity = td.get("best_clarity", 0)
                if identity == "Unknown":
                    if not td["recognized"]:
                        display_id = f"collecting... (Q:{clarity:.0f})"
                    else:
                        display_id = f"Unknown (Q:{clarity:.0f})"
                else:
                    display_id = f"{identity} (#{tid}) (Q:{clarity:.0f})"
                
                # B. Emotion (Polled from track_data updated by async task)
                emotion = td.get("emotion", "")
                
                # C. Action (SlowFast Update + Rule-Based Fallback)
                action = ""
                if AI_TOGGLES.get("action", True):
                    # 1. SlowFast (Deep Learning)
                    bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                    bx2, by2 = min(ai_w, bx2), min(ai_h, by2)
                    person_crop = ai_frame[by1:by2, bx1:bx2]
                    if person_crop.size > 0:
                        action = self.action_det.update(tid, person_crop)
                    
                    # 2. Proximity Rules (Phones/Food) - Phase 1 Addition
                    # We check if a phone is detected within the person's bbox (upper part)
                    if not action or action == "standing":
                        phones = self._last_objects.get("phone", [])
                        for pbx in phones:
                            px1, py1, px2, py2 = pbx
                            # Simple intersection check with person box
                            if (px1 < bx2 and px2 > bx1 and py1 < by2 and py2 > by1):
                                # If phone is in the upper 40% of person box
                                if py1 < (by1 + (by2 - by1) * 0.4):
                                    action = "📱 using phone"
                                    break
                
                # 3. Geometric Fallback (Sleeping/Sitting)
                if not action:
                    bx1, by1, bx2, by2 = bbox
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
                        # Re-verify every 100 frames (approx 3-5 seconds) even if already recognized
                        if (is_unknown or (self._frame_idx % 100 == 0)) and active_rec_tasks < 2:
                            self._submit_rec(ai_frame, bbox, tid)
                        
                        # NON-BLOCKING TELEGRAM: Run in executor so YOLO doesn't stop
                        self._rec_executor.submit(self.notifier.notify_person, tid, self.camera_name, action)
                    except Exception as e:
                        log.error("Sync error: %s", e)

                track_results.append(TrackResult(
                    track_id=tid,
                    bbox_full=scale_box(bbox, sx, sy),
                    face_bbox_full=None,
                    emotion=emotion, 
                    action=action, 
                    identity=identity,
                    display_id=display_id
                ))

            # 5. Periodic cleanup of trackers & futures
            if self._frame_idx % 150 == 0:
                self.emotion_det.purge(active_ids)
                self.action_det.purge(active_ids)
                self.face_rec.purge(active_ids)
                for d in (self._emotion_futures, self._rec_futures):
                    for tid in list(d):
                        if tid not in active_ids: d.pop(tid, None)

            return AIResult(motion=motion, motion_mask=mask, tracks=track_results, 
                        entries=self.entries, exits=self.exits)

        except Exception as e:
            log.error("CRITICAL AI PIPELINE ERROR: %s", e, exc_info=True)
            return AIResult(motion=False, entries=self.entries, exits=self.exits)

    def reload_faces(self):
        """Trigger face recognizer to re-scan the uploads folder."""
        if hasattr(self, 'face_rec'):
            log.info("AI Pipeline: Reloading face signatures...")
            self.face_rec.load_from_folder(self.face_rec.db_path)

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3); yi1 = max(y1, y3)
        xi2 = min(x2, x4); yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _calculate_clarity(self, frame, bbox):
        """Calculate image sharpess using Laplacian variance."""
        try:
            bx1, by1, bx2, by2 = (int(v) for v in bbox)
            crop = frame[by1:by2, bx1:bx2]
            if crop.size == 0: return 0
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0
