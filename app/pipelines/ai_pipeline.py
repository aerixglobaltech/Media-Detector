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
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from app.core import config as cfg
from app.core.data_types import AIResult, TrackResult, scale_box
from app.services.ai.action_detection import ActionDetector
from app.services.ai.emotion_detection import EmotionDetector
from app.services.ai.face_detection import FaceDetector
from app.services.ai.face_recognition import FaceRecognizer
from app.services.ai.motion_detection import MotionDetector
from app.services.ai.notifier import TelegramNotifier
from app.services.ai.object_detection import PersonDetector
from app.services.ai.tracking import PersonTracker

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

        # Single-worker executors keep heavy models off the tracker loop
        self._emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")
        self._rec_executor     = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rec")
        self._emotion_futures: dict[int, Future] = {}
        self._rec_futures:     dict[int, Future] = {}

        self._notifier              = TelegramNotifier()
        self._notified_identities:  set[int] = set()

        self._frame_idx        = 0
        self._last_detections: list = []
        self._last_objects:    dict = {"food": [], "phone": []}
        self._last_motion_time: float = time.monotonic()

        self._full_w: int = cfg.FRAME_WIDTH
        self._full_h: int = cfg.FRAME_HEIGHT
        self.camera_name: str = "Camera"

        log.info("AI pipeline: all models ready.")

    # ── Thread control ────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_evt.set()
        self.in_queue.put(None)

    def run(self) -> None:
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

    def _submit_rec(self, crop: np.ndarray, tid: int) -> None:
        fut = self._rec_futures.get(tid)
        if fut is not None and not fut.done():
            return
        self._rec_futures[tid] = self._rec_executor.submit(
            self.face_rec.recognize, crop.copy(), tid
        )

    # ── Main processing cycle ─────────────────────────────────────────────────

    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int) -> AIResult:
        global _total_motion_events

        # Import AI_TOGGLES at call-time so changes from the UI take effect immediately
        try:
            from app.core.extensions import AI_TOGGLES
        except ImportError:
            AI_TOGGLES = {"person": True, "action": True, "emotion": True}

        self._frame_idx += 1
        ai_h, ai_w = ai_frame.shape[:2]
        sx = full_w / ai_w
        sy = full_h / ai_h

        # 1. Motion gate
        motion, mask = self.motion_det.detect(ai_frame)
        now = time.monotonic()
        if motion:
            if (now - self._last_motion_time) > cfg.MOTION_COOLDOWN_SEC:
                _total_motion_events += 1
            self._last_motion_time = now

        in_cooldown = (now - self._last_motion_time) < cfg.MOTION_COOLDOWN_SEC
        if not motion and not in_cooldown:
            return AIResult(motion=False, motion_mask=mask, tracks=[])

        # 2. YOLO person detection
        if AI_TOGGLES["person"]:
            if self._frame_idx % cfg.YOLO_SKIP_FRAMES == 0:
                self._last_detections = self.person_det.detect(ai_frame)
        else:
            self._last_detections = []

        # 2b. Object detection for food / phone context
        if AI_TOGGLES["action"] and self._frame_idx % 5 == 0:
            self._last_objects = self.person_det.detect_objects(ai_frame)
        elif not AI_TOGGLES["action"]:
            self._last_objects = {"food": [], "phone": []}

        # 3. Tracker
        raw_tracks = self.tracker.update(self._last_detections, ai_frame)
        active_ids = {t.track_id for t in raw_tracks}
        if active_ids:
            self._last_motion_time = now

        # 4. Per-track processing
        track_results: list[TrackResult] = []

        for track in raw_tracks:
            tid  = track.track_id
            bbox = track.bbox
            bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
            bx2 = min(ai_w, bx2)
            by2 = min(ai_h, by2)

            action   = ""
            emotion  = ""
            identity = ""

            # ── Action recognition ───────────────────────────────────────────
            if AI_TOGGLES["action"]:
                bw    = max(1, bx2 - bx1)
                bh    = max(1, by2 - by1)
                ratio = bw / bh

                if ratio > 1.4:    geometry_label = "😴 sleeping"
                elif ratio > 0.85: geometry_label = "🪑 sitting"
                else:              geometry_label = "🧍 standing"

                person_crop    = ai_frame[by1:by2, bx1:bx2]
                slowfast_label = (
                    self.action_det.update(tid, person_crop)
                    if person_crop.size > 0 else ""
                )

                def _overlaps(ob, pb, exp=40):
                    ox1, oy1, ox2, oy2 = ob
                    px1, py1, px2, py2 = pb
                    return ox1 < px2 + exp and ox2 > px1 - exp and oy1 < py2 + exp and oy2 > py1 - exp

                psb        = [bx1 * sx, by1 * sy, bx2 * sx, by2 * sy]
                near_food  = any(_overlaps(f, psb) for f in self._last_objects.get("food", []))
                near_phone = any(_overlaps(p, psb) for p in self._last_objects.get("phone", []))

                TAGS = ("⚠", "🏃", "💪", "📖", "✍", "🍽", "💬", "💃", "🎵")
                if near_food:
                    action = "🍽 eating"
                elif near_phone:
                    action = "💻 working"
                elif slowfast_label and any(t in slowfast_label for t in TAGS):
                    action = slowfast_label
                else:
                    action = geometry_label

            # ── Emotion detection ────────────────────────────────────────────
            if AI_TOGGLES["emotion"]:
                person_crop = ai_frame[by1:by2, bx1:bx2]
                if person_crop.size > 0:
                    self._submit_emotion(person_crop, tid)
                emotion = self.emotion_det._cache.get(tid, "")

            # ── Face recognition ─────────────────────────────────────────────
            if AI_TOGGLES["person"]:
                person_crop = ai_frame[by1:by2, bx1:bx2]
                if person_crop.size > 0:
                    self._submit_rec(person_crop, tid)
                identity = self.face_rec._cache.get(tid, "")

                if identity and identity != "Unknown" and tid not in self._notified_identities:
                    self._notifier.send_message(
                        f"✅ *MATCH DETECTED*: Staff member **{identity}**"
                        f" recognized on {self.camera_name}."
                    )
                    self._notified_identities.add(tid)

                # Telegram alert for new person detection
                self._notifier.notify_person(tid, self.camera_name, action)

            track_results.append(TrackResult(
                track_id=tid,
                bbox_full=scale_box(bbox, sx, sy),
                face_bbox_full=None,
                emotion=emotion,
                action=action,
                identity=identity,
            ))

        # 5. Periodic cache cleanup
        if self._frame_idx % 120 == 0:
            self.emotion_det.purge(active_ids)
            self.action_det.purge(active_ids)
            self.face_rec.purge(active_ids)
            for futures_dict in (self._emotion_futures, self._rec_futures):
                for tid in list(futures_dict):
                    if tid not in active_ids:
                        del futures_dict[tid]
            for tid in list(self._notified_identities):
                if tid not in active_ids:
                    self._notified_identities.discard(tid)

        return AIResult(motion=motion, motion_mask=mask, tracks=track_results)
