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
        # Global models (Shared across camera views)
        self.face_det = FaceDetector()
        # ... (rest of models initialization)
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
        # Multi-Camera State Containers (Isolated per camera name)
        self._camera_states: dict[str, dict] = {} 
        self._state_lock = threading.Lock()
        self._last_face_reload = time.monotonic()
        os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)

        log.info("AI pipeline: all models ready.")

    def _get_state(self, cam_name: str) -> dict:
        """Returns the isolated tracking state for a specific camera."""
        with self._state_lock:
            if cam_name not in self._camera_states:
                self._camera_states[cam_name] = {
                    "tracker": PersonTracker(
                        max_age=cfg.DEEPSORT_MAX_AGE,
                        n_init=cfg.DEEPSORT_N_INIT,
                        embedder=cfg.DEEPSORT_EMBEDDER,
                    ),
                    "track_data": {},
                    "seen_ids": set(),
                    "active_ids": set(),
                    "entries": 0,
                    "exits": 0,
                    "exit_taken": set(),
                    "entry_best_frames": {},
                    "exit_best_frames": {},
                    "best_entry_scores": {},
                    "best_exit_scores": {},
                    "track_start_time": {},
                    "db_sessions": {},
                    "exit_filenames": {},
                    # Watchdog & Last State (Isolated for Multi-Cam)
                    "last_detections": [],
                    "last_objects": {},
                    "last_motion_time": 0,
                    "last_yolo_human_time": 0,
                    "current_frame_motion_img": "",
                    "current_frame_motion_id": None,
                    "frame_idx": 0
                }
            return self._camera_states[cam_name]

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
            # Normalize Laplacian variance to [0.0, 1.0]. A value of 500+ is typically very sharp.
            var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return var
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
                # Multiplexed Ingest: We process one frame at a time from the shared queue.
                # Frame-skipping at the source (MonitoringNode) ensures the queue stays manageable.
                item = self.in_queue.get(timeout=0.1)
                
                if isinstance(item, tuple):
                    # Format: (ai_f, name, roles, (original_w, original_h))
                    ai_frame, cam_name, cam_roles, dims = item
                    full_w, full_h = dims
                else:
                    # Legacy support
                    ai_frame = item
                    cam_name = self.camera_name # Fallback
                    cam_roles = self.camera_roles # Fallback
                    full_w, full_h = self._full_w, self._full_h
                
                state = self._get_state(cam_name)
            except queue.Empty:
                continue
            if ai_frame is None:
                break
            result = self._process(ai_frame, full_w, full_h, state, cam_name, cam_roles)
            with self.result_lock:
                self.result_store[0] = result
        self._emotion_executor.shutdown(wait=False)
        self._rec_executor.shutdown(wait=False)
        log.info("AI pipeline stopped.")

    # ── Non-blocking executor helpers ─────────────────────────────────────────

    def _submit_emotion(self, crop: np.ndarray, tid: int, cam_name: str) -> None:
        future_key = (cam_name, tid)
        fut = self._emotion_futures.get(future_key)
        if fut is not None and not fut.done():
            return
            
        def _task():
            try:
                res = self.emotion_det.analyse(crop, tid)
                state = self._get_state(cam_name)
                if tid in state['track_data']:
                    state['track_data'][tid]["emotion"] = res
                return res
            except Exception as e:
                log.debug("Emotion task error: %s", e)
                return ""
                
        self._emotion_futures[future_key] = self._emotion_executor.submit(_task)

    def _submit_rec(self, ai_frame: np.ndarray, bbox: list[float], tid: int, cam_name: str) -> None:
        """Asynchronously detect face AND recognize identity."""
        future_key = (cam_name, tid)
        fut = self._rec_futures.get(future_key)
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
                            state = self._get_state(cam_name)
                            if tid in state['track_data']:
                                td = state['track_data'][tid]
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
                                        
                                        # PARALLEL TELEGRAM: Notify upon successful identity match
                                        self.notifier.notify_person(tid, cam_name, identity=name, action=td.get("action", ""), image_path=td.get("entry_image_path"))
                        
                        return res.get("name", "Unknown")
            except Exception as e:
                log.debug("Rec task error: %s", e)
            return "Unknown"

        self._rec_futures[future_key] = self._rec_executor.submit(_task)

    # ── Main processing cycle ─────────────────────────────────────────────────

    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int, state: dict, cam_name: str, cam_roles: list) -> AIResult:
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

            state['frame_idx'] += 1

            if state['frame_idx'] % 10 == 0:
                log.info("AI: Processing frame #%d for %s", state['frame_idx'], cam_name)

            # --- 1. Movement stage ---
            motion, mask = self.motion_det.detect(ai_frame)
            if motion:
                _total_motion_events += 1
            
            # --- 2. YOLO Detection & Tracking (Every Frame) ---
            state['last_detections'] = self.person_det.track(ai_frame, persist=True)
            yolo_any = len(state['last_detections']) > 0
            
            if yolo_any:
                state['last_yolo_human_time'] = now
                if state['frame_idx'] % 30 == 0: 
                    log.info(f"AI: YOLO found {len(state['last_detections'])} persons on {cam_name}")

            # Snapshot Logic: Capture for new tracks or if first time seeing activity in 5s
            trigger_snapshot = yolo_any and (now - state['last_motion_time']) > 5.0
            
            if trigger_snapshot:
                log.info(f"Movement Triggered: Watchdog_YOLO_Any={yolo_any}")
                motion_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                motion_filename = f"motion_{cam_name.replace(' ', '_')}_{motion_ts}.jpg"
                motion_dir = os.path.join("static", "uploads", "movement")
                os.makedirs(motion_dir, exist_ok=True)
                motion_path = os.path.join(motion_dir, motion_filename)
                
                cv2.imwrite(motion_path, ai_frame)
                log.info(f"Saved movement snapshot: {motion_path}")
                
                state['current_frame_motion_img'] = f"static/uploads/movement/{motion_filename}"
                # Global last motion img is fine to keep for simple HUD feedback, 
                # but let's keep it isolated for logging
                state['current_frame_motion_id'] = log_movement(
                    camera_id=cam_name, 
                    image_path=state['current_frame_motion_img'], 
                    track_id=None, 
                    event_type='MOTION_YOLO'
                )
                if state['current_frame_motion_id']:
                    # Determine best classification for the snapshot
                    snapshot_label = "human"
                    if state['last_detections']:
                        # If any human is in the frame, prioritize 'human', otherwise use the class of the first detection
                        has_human = any(d[6] == self.person_det.PERSON_CLASS_ID for d in state['last_detections'])
                        if not has_human:
                            first_cls = state['last_detections'][0][6]
                            snapshot_label = "animal" if first_cls in self.person_det.ANIMAL_CLASS_IDS else "unknown"
                    
                    update_movement_classification(state['current_frame_motion_id'], snapshot_label, 1.0)
                state['last_motion_time'] = now

            # 1.5) Object detection (Phone/Food) for rule-based actions
            if state['frame_idx'] % 10 == 0:
                state['last_objects'] = self.person_det.detect_objects(ai_frame)

            # 2) Human detection/tracking (Already scanned in Watchdog logic above)
            pass

            # 3) Tracker Update (ByteTrack)
            raw_tracks = state['tracker'].update(state['last_detections'])
            
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
                if tid not in state['track_data']:
                    state['track_data'][tid] = {
                        "frames": [],           
                        "recognized": False,    
                        "logged": False,        
                        "identity": "Unknown",  
                        "db_id": None,          
                        "missing_count": 0,     # GRACE PERIOD
                        "is_accounted": False,  # Has it been counted in self.entries?
                        "obj_type": "human" if track.cls_id == self.person_det.PERSON_CLASS_ID else "animal",
                        "start_time": now,
                        "best_clarity": -1.0,
                        "best_frame": None,
                        "best_full": None,
                        "last_log_time": 0,
                        "track_frame_count": 0,
                        "id_votes": {"attempts": 0}
                    }
                
                td = state['track_data'][tid]
                td["missing_count"] = 0 # Reset grace period

                # RE-EVALUATE CLASSIFICATION (Improve robustness against initial frame flicker)
                # If we see it's an animal in ANY of the first 15 frames, mark as animal
                if td.get("track_frame_count", 0) < 15 and track.cls_id != self.person_det.PERSON_CLASS_ID:
                    if td.get("obj_type") == "human":
                        log.info(f"AI: Re-classifying Track {tid} from human to animal (Confidence Update)")
                        td["obj_type"] = "animal"
                
                # Increment entries ONLY once per unique STABLE track lifecycle
                if tid > 0 and not td.get("is_accounted"):
                    if td["obj_type"] == "human":
                        state['entries'] += 1 
                        td["is_accounted"] = True
                        log.info(f"COUNT: Person entering (ID: {tid}). Total: {state['entries']}")
                    else:
                        td["is_accounted"] = True # Mark animals as accounted but don't increment human entries
                        log.info(f"AI: Animal entering (ID: {tid})")
                
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
                        # Also sync if the name is correct but we found a better photo/clarity
                        should_sync_id = td.get("last_synced_id") != new_id
                        should_sync_img = td["best_clarity"] > (td.get("last_synced_clarity", 0) + 50) # Sync only if significantly better
                        
                        if td["logged"] and (should_sync_id or should_sync_img):
                            from app.services.attendance_service import update_person_identity
                            # Save the best crop to a file if it's the one we're syncing
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            sync_crop_filename = f"sync_crop_{tid}_{ts}.jpg"
                            sync_crop_path = os.path.join("static", "uploads", "movement", sync_crop_filename)
                            if td.get("best_frame") is not None:
                                cv2.imwrite(sync_crop_path, td["best_frame"])
                            
                            log.info(f"AI: [VISUAL SYNC] Updating Track {tid} -> {new_id} (Clarity: {td['best_clarity']:.1f})")
                            update_person_identity(
                                member_id=td.get("db_id"), 
                                staff_id=td["staff_db_id"], 
                                staff_name=new_id,
                                track_id=tid,
                                image_path=f"static/uploads/movement/{sync_crop_filename}" if td.get("best_frame") is not None else None,
                                confidence=td["best_clarity"]
                            )
                            td["last_synced_id"] = new_id
                            td["last_synced_clarity"] = td["best_clarity"]

                            # PARALLEL TELEGRAM: Notify when officially recognized
                            self._rec_executor.submit(self.notifier.notify_person, tid, cam_name, identity=new_id, action=td.get("action", ""), image_path=td.get("entry_image_path"))

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
                    p_type = td["obj_type"] if identity_name == "Unknown" else "staff"
                    
                    mv_id = log_movement(
                        camera_id=cam_name,
                        image_path=f"static/uploads/movement/{mv_filename}",
                        track_id=tid,
                        event_type='MOVE_TRACK',
                        staff_id=s_id,
                        staff_name=identity_name if identity_name != "Unknown" else None,
                        person_type=p_type
                    )
                    if mv_id:
                        td["movement_id"] = mv_id
                        # Use the track's object type for classification
                        update_movement_classification(mv_id, td["obj_type"], 1.0)

                if len(td["frames"]) < 50:
                    clarity = self._calculate_clarity(ai_frame, bbox)
                    bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                    bx2, by2 = min(ai_w, bx2), min(ai_h, by2)
                    crop = ai_frame[by1:by2, bx1:bx2]
                    
                    if crop.size > 0:
                        td["frames"].append({"clarity": clarity})
                        best_c = td.get("best_clarity", -1.0)
                        if clarity > best_c:
                            td["best_clarity"] = clarity
                            td["best_frame"] = crop.copy() # Face/Person crop
                            # Store full frame with a box for context
                            full_snap = ai_frame.copy()
                            cv2.rectangle(full_snap, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                            td["best_full"] = full_snap
                            td["best_bbox"] = bbox

                # B. Recognition Trigger (PROACTIVE: Try early, try often)
                is_much_better = td.get("best_clarity", -1.0) > (td.get("last_trigger_clarity", 0) * 1.5)
                
                num_frames = len(td.get("frames", []))
                should_trigger = not td.get("recognized") and (
                    num_frames == 1 or            # IMMEDIATE: Try once as soon as person enters
                    num_frames % 5 == 0 or        # PERIODIC: Retry every 5 frames if still unknown
                    td.get("best_clarity", -1.0) > 50      # Quality Threshold
                )
                
                # Throttle: Only trigger again if we haven't tried in 2s OR if we found a MUCH sharper frame
                last_try_time = td.get("last_rec_time", 0)
                if should_trigger and (now - last_try_time > 2.0 or is_much_better):
                    if td["best_frame"] is not None:
                        td["last_rec_time"] = now
                        td["last_trigger_clarity"] = td["best_clarity"]
                        self._submit_rec(ai_frame.copy(), td["best_bbox"], tid, cam_name)
                        if AI_TOGGLES.get("action", True):
                            self._submit_emotion(td["best_frame"], tid, cam_name)
                        log.info(f"AI: Triggering recognition for Track {tid} on {cam_name} (Clarity: {td['best_clarity']:.1f})")
                        
                        # Increment attempts
                        votes = td.get("id_votes", {})
                        votes["attempts"] = votes.get("attempts", 0) + 1
                        td["id_votes"] = votes

                # B. Recognition Trigger...

                # D. Entry Logging (IMMEDIATE: Log first, name later)
                # We log as soon as we have 1 stable frame to ensure the table is updated first
                # ONLY log in Member Logs (log_person) if it's a HUMAN track
                if not td["logged"] and len(td["frames"]) >= 1 and td["obj_type"] == "human":
                    # STRICT CLARITY: User wants clear photos, but 30 is a safe floor for entry
                    if td["best_clarity"] < 30:
                        log.debug(f"AI: Skipping Entry log for Track {tid} - Low Clarity ({td['best_clarity']:.1f})")
                    else:
                        person_type = "staff" if current_id != "Unknown" else "unknown"
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Save FULL FRAME evidence (Context)
                        c_safe = cam_name.replace(' ', '_')
                        full_filename = f"entry_full_{c_safe}_{tid}_{ts}.jpg"
                        full_path = os.path.join("static", "uploads", "movement", full_filename)
                        
                        # FALLBACK: If best_full is not yet captured (likely due to frame 5 trigger), use current frame
                        snap_to_save = td.get("best_full")
                        if snap_to_save is None:
                            snap_to_save = ai_frame.copy()
                            cv2.rectangle(snap_to_save, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        
                        cv2.imwrite(full_path, snap_to_save)
                        
                        # Save FACE CROP (Forensics)
                        crop_filename = f"entry_crop_{c_safe}_{tid}_{ts}.jpg"
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
                            existing_visit_id = get_recent_sighting(s_id, cam_name)
                        
                        if existing_visit_id:
                            log.info(f"AI: [DEDUPLICATED] Track {tid} ({current_id}) linked to recent visit {existing_visit_id}")
                            td["db_id"] = existing_visit_id
                        else:
                            # ROLE-BASED LOGGING:
                            role_list = [r.lower() for r in cam_roles]
                            is_exit_only = ('exit' in role_list and 'entry' not in role_list and 'general' not in role_list)
                            
                            e_time = None if is_exit_only else datetime.now()
                            x_time = datetime.now() if is_exit_only else None
                            e_type = 'EXIT' if is_exit_only else 'ENTRY'
                            
                            res_id = log_person(
                                camera_id=cam_name,
                                person_type=person_type,
                                staff_id=s_id,
                                image_path=f"static/uploads/movement/{full_filename}", 
                                confidence=td["best_clarity"],
                                staff_name=current_id if person_type == "staff" else None,
                                track_id=tid,
                                event_type=e_type,
                                entry_time=e_time,
                                exit_time=x_time,
                                roles=cam_roles
                            )
                            td["db_id"] = res_id
                            td["logged"] = True
                        # Ensure absolute path for Telegram reliability
                        td["entry_image_path"] = os.path.abspath(full_path)
                        log.info(f"AI: Logged {person_type} Entry for Track {tid} ({current_id})")
                        # PARALLEL TELEGRAM: Notify upon initial entry log
                        self._rec_executor.submit(self.notifier.notify_person, tid, cam_name, identity=current_id, action=td.get("action", ""), image_path=td.get("entry_image_path"))

            # ── EXIT HANDLING (With 15-frame Grace Period) ──
            # state['active_ids'] contains IDs from PREVIOUS frame for this camera
            for tid in list(state['active_ids']):
                if tid not in active_ids:
                    td = state['track_data'].get(tid)
                    if not td: continue
                    
                    td["missing_count"] += 1
                    if td["missing_count"] > 15:
                        state['exits'] += 1
                        if td["logged"]:
                            # ROLE CHECK: Only log EXIT if the camera has the 'exit' role OR is 'general'
                            role_list = [r.lower() for r in cam_roles]
                            is_exit_cam = ("exit" in role_list) or ("general" in role_list) or (not role_list)
                            
                            exit_image_path = ""
                            best_full = td.get("best_full")
                            if best_full is not None:
                                exit_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                c_safe = cam_name.replace(' ', '_')
                                exit_filename = f"exit_full_{c_safe}_{tid}_{exit_ts}.jpg"
                                exit_image_path = os.path.join("static", "uploads", "movement", exit_filename)
                                cv2.imwrite(exit_image_path, best_full)
                                exit_image_path = f"static/uploads/movement/{exit_filename}"

                            if is_exit_cam:
                                from app.services.attendance_service import update_exit_logs
                                # Role-Based: Finalize the exit record with the best exit image
                                update_exit_logs(
                                    member_id=td.get("db_id"),
                                    movement_id=td.get("movement_id"), 
                                    exit_image=exit_image_path,
                                    merged_image="",
                                    track_id=tid,
                                    exit_camera_name=cam_name,
                                    exit_camera_id=cam_name
                                )
                                log.info(f"AI: Logged Formal Exit for Track {tid} on Exit Camera")
                            else:
                                log.info(f"AI: Track {tid} disappeared, Camera {cam_name} is NOT an Exit role.")
                        
                        del state['track_data'][tid]
            
            # Combine current detection IDs + grace-period IDs for persistence
            persistence_ids = set(active_ids)
            for tid, td in state['track_data'].items():
                if td.get("missing_count", 0) > 0:
                    persistence_ids.add(tid)
            state['active_ids'] = persistence_ids

            # 4. Per-track results for UI/HUD
            track_results: list[TrackResult] = []
            for track in raw_tracks:
                tid  = track.track_id
                bbox = track.bbox
                td = state['track_data'].get(tid, {})
                
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
                        phones = state['last_objects'].get("phone", [])
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
                    bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                    bx2, by2 = min(ai_w, bx2), min(ai_h, by2)
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
                    near_food  = any(_overlaps(f, psb) for f in state['last_objects'].get("food", []))
                    near_phone = any(_overlaps(p, psb) for p in state['last_objects'].get("phone", []))

                    TAGS = ("⚠", "🏃", "💪", "📖", "✍", "🍽", "💬", "💃", "🎵")
                    if near_food:    action = "🍽 eating"
                    elif near_phone: action = "💻 working"
                    elif slowfast_label and any(t in slowfast_label for t in TAGS): action = slowfast_label
                    else:            action = geometry_label

                if AI_TOGGLES.get("emotion", True):
                    bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                    bx2, by2 = min(ai_w, bx2), min(ai_h, by2)
                    person_crop = ai_frame[by1:by2, bx1:bx2]
                    if person_crop.size > 0: self._submit_emotion(person_crop, tid, self.camera_name)
                    emotion = self.emotion_det._cache.get(tid, "")

                if AI_TOGGLES.get("person", True):
                    cached_res = self.face_rec._cache.get(tid, {"name": "Unknown", "display_id": ""})
                    identity = cached_res["name"]
                    display_id = cached_res["display_id"]
                    
                    try:
                        is_unknown = (identity == "" or identity == "Unknown")
                        active_rec_tasks = sum(1 for f in self._rec_futures.values() if not f.done())
                        # Re-verify every 100 frames (approx 3-5 seconds) even if already recognized
                        # ONLY for human tracks
                        if td["obj_type"] == "human" and (is_unknown or (state['frame_idx'] % 100 == 0)) and active_rec_tasks < 2:
                            self._submit_rec(ai_frame, bbox, tid, cam_name)
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
            if state['frame_idx'] % 150 == 0:
                self.emotion_det.purge(active_ids)
                self.action_det.purge(active_ids)
                self.face_rec.purge(active_ids)
                for d in (self._emotion_futures, self._rec_futures):
                    for tid in list(d):
                        if tid not in active_ids: d.pop(tid, None)

            # Final Step: Update state for next frame
            state['active_ids'] = active_ids

            return AIResult(motion=motion, motion_mask=mask, tracks=track_results, 
                        entries=state['entries'], exits=state['exits'])

        except Exception as e:
            log.error("CRITICAL AI PIPELINE ERROR: %s", e, exc_info=True)
            return AIResult(motion=False, entries=state['entries'], exits=state['exits'])

    def reload_faces(self):
        """Trigger face recognizer to re-scan the uploads folder in the background."""
        if hasattr(self, 'face_rec'):
            log.info("AI Pipeline: Triggering background reload of face signatures...")
            # Use the recognition executor so it doesn't block the main YOLO loop
            self._rec_executor.submit(self.face_rec.load_from_folder, self.face_rec.db_path)

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3); yi1 = max(y1, y3)
        xi2 = min(x2, x4); yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - inter_area
        return inter_area / union_area if union_area > 0 else 0
