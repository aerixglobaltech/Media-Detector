"""
main.py  –  AI CCTV Surveillance System
─────────────────────────────────────────────────────────────────────────────

Thread architecture
───────────────────
  CAPTURE     →  frame_queue  →  AI PIPELINE THREAD
  (main loop)                     ├─ Motion detect
                                  ├─ YOLOv8 (every N frames)
                                  ├─ DeepSORT tracker
                                  ├─ RetinaFace face detect
                                  └─ submits face crops → EMOTION EXECUTOR
                                       (ThreadPoolExecutor, 1 worker)
                                       └─ DeepFace (non-blocking)

  DISPLAY (main loop)
    ├─ reads latest AIResult (timestamped)
    ├─ if result older than RESULT_MAX_AGE_SEC → hide boxes (ghost-box fix)
    └─ draws overlays at full camera FPS

Bug fixes in this version
──────────────────────────
1. GHOST BOXES  – AIResult now carries a timestamp; display thread skips
                  drawing if the result is stale (person left & AI is busy).
2. EMOTION HUNG – DeepFace runs in a ThreadPoolExecutor (1 worker) so it
                  never blocks the YOLO/tracker loop.
3. DOUBLE FACE  – face_det called once, result reused for draw.

Press 'q' to quit · 'm' to toggle motion mask
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

import config as cfg
from modules.motion_detection import MotionDetector
from modules.object_detection import PersonDetector
from modules.tracking import PersonTracker
from modules.face_detection import FaceDetector
from modules.emotion_detection import EmotionDetector
from modules.action_detection import ActionDetector
from utils.stream import VideoStream
from utils.drawing import draw_person, draw_face, draw_status_bar, draw_motion_mask
from utils.fps_counter import FPSCounter
from imou_connector import ImouAPI, _find_working_datacenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ─── Shared data structures ───────────────────────────────────────────────

@dataclass
class TrackResult:
    track_id:      int
    bbox_full:     list[float]
    face_bbox_full: list[float] | None
    emotion:       str
    action:        str


@dataclass
class AIResult:
    motion:      bool               = False
    motion_mask: np.ndarray | None  = None
    tracks:      list[TrackResult]  = field(default_factory=list)
    timestamp:   float              = field(default_factory=time.monotonic)  # ← NEW


# ─── Scale helpers ────────────────────────────────────────────────────────

def _scale_box(box: list[float], sx: float, sy: float) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


# ─── AI Pipeline thread ───────────────────────────────────────────────────

class AIPipeline(threading.Thread):
    """
    Background thread: drains frame_queue → runs AI → stores result.
    DeepFace runs in a separate ThreadPoolExecutor so it never blocks tracking.
    """

    def __init__(
        self,
        in_queue:     "queue.Queue[np.ndarray | None]",
        result_store: list,
        result_lock:  threading.Lock,
    ):
        super().__init__(daemon=True, name="AIPipeline")
        self.in_queue     = in_queue
        self.result_store = result_store
        self.result_lock  = result_lock
        self._stop_evt    = threading.Event()

        # ── Load models ────────────────────────────────────────────────
        log.info("AI thread: loading motion detector …")
        self.motion_det = MotionDetector(
            history=cfg.MOTION_HISTORY,
            var_threshold=cfg.MOTION_VAR_THRESHOLD,
            min_area=cfg.MOTION_MIN_AREA,
        )
        log.info("AI thread: loading YOLOv8 (%s) …", cfg.YOLO_MODEL)
        self.person_det = PersonDetector(
            model_path=cfg.YOLO_MODEL,
            conf_threshold=cfg.YOLO_CONF,
            device=cfg.YOLO_DEVICE,
        )
        log.info("AI thread: loading DeepSORT …")
        self.tracker = PersonTracker(
            max_age=cfg.DEEPSORT_MAX_AGE,
            n_init=cfg.DEEPSORT_N_INIT,
            embedder=cfg.DEEPSORT_EMBEDDER,
        )
        log.info("AI thread: loading RetinaFace …")
        self.face_det = FaceDetector()

        log.info("AI thread: loading DeepFace (emotion executor) …")
        self.emotion_det = EmotionDetector(
            skip_frames=cfg.EMOTION_SKIP_FRAMES,
            backend=cfg.EMOTION_BACKEND,
            min_face_pixels=cfg.EMOTION_MIN_FACE_PX,
        )
        # Single-worker executor keeps DeepFace off the tracker loop
        self._emotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="emotion")
        # track_id → (Future, last submitted time)
        self._emotion_futures: dict[int, Future] = {}

        log.info("AI thread: action recogniser (enabled=%s) …", cfg.ENABLE_ACTION)
        self.action_det = ActionDetector(
            enabled=cfg.ENABLE_ACTION,
            buffer_size=cfg.ACTION_BUFFER_SIZE,
            slow_stride=cfg.ACTION_SLOW_STRIDE,
            device=cfg.ACTION_DEVICE,
        )

        self._frame_idx       = 0
        self._last_detections: list  = []
        self._last_objects: dict     = {"food": [], "phone": []}
        self._last_motion_time: float = time.monotonic()
        # Real frame dimensions — updated each tick by the capture loop.
        # Initialise to config defaults so _process() is safe before first frame.
        self._full_w: int = cfg.FRAME_WIDTH
        self._full_h: int = cfg.FRAME_HEIGHT

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_evt.set()
        self.in_queue.put(None)

    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("AI pipeline thread started.")
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
        log.info("AI pipeline thread stopped.")

    # ------------------------------------------------------------------
    def _submit_emotion(self, face_crop: np.ndarray, track_id: int) -> None:
        """Submit a DeepFace job to the executor (non-blocking)."""
        # Don't queue a new job if one is already running for this track
        fut = self._emotion_futures.get(track_id)
        if fut is not None and not fut.done():
            return
        crop_copy = face_crop.copy()
        fut = self._emotion_executor.submit(
            self.emotion_det.analyse, crop_copy, track_id
        )
        self._emotion_futures[track_id] = fut

    # ------------------------------------------------------------------
    def _process(self, ai_frame: np.ndarray, full_w: int, full_h: int) -> AIResult:
        self._frame_idx += 1
        ai_h, ai_w = ai_frame.shape[:2]
        # Use ACTUAL captured frame size (not config constant) so Imou cameras
        # with different native resolutions scale bboxes correctly.
        sx = full_w / ai_w
        sy = full_h / ai_h

        # 1. Motion gate with cooldown
        motion, mask = self.motion_det.detect(ai_frame)
        now = time.monotonic()
        if motion:
            self._last_motion_time = now   # real movement seen

        in_cooldown = (now - self._last_motion_time) < cfg.MOTION_COOLDOWN_SEC

        if not motion and not in_cooldown:
            # Truly idle → skip AI pipeline
            return AIResult(motion=False, motion_mask=mask, tracks=[])

        # Motion OR still within cooldown window → run full AI pipeline

        # 2. YOLO detection (skip frames)
        if self._frame_idx % cfg.YOLO_SKIP_FRAMES == 0:
            self._last_detections = self.person_det.detect(ai_frame)

        # 2b. Object detection (food, phone etc.) — every 5 frames
        if self._frame_idx % 5 == 0 and cfg.DETECTION_MODE == "action_only":
            self._last_objects = self.person_det.detect_objects(ai_frame)


        # 3. Tracker
        raw_tracks = self.tracker.update(self._last_detections, ai_frame)
        active_ids = {t.track_id for t in raw_tracks}

        # ── KEY FIX: keep pipeline alive as long as anyone is tracked ──
        # This prevents flickering when a person sits still for >30 seconds.
        # The pipeline only idles when NO tracks exist AND motion has stopped.
        if active_ids:
            self._last_motion_time = now


        # 4. Per-track processing
        track_results: list[TrackResult] = []
        mode = cfg.DETECTION_MODE   # "yolo_only" | "action_only" | "full"

        for track in raw_tracks:
            tid  = track.track_id
            bbox = track.bbox   # ai_frame coords

            # ── YOLO-only: fastest, no extra models ────────────────────────
            if mode == "yolo_only":
                track_results.append(TrackResult(
                    track_id=tid,
                    bbox_full=_scale_box(bbox, sx, sy),
                    face_bbox_full=None,
                    emotion="",
                    action="",
                ))
                continue

            # ── Action-only: posture + SlowFast, NO face/emotion ─────────
            if mode == "action_only":
                bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
                bx2 = min(ai_w, bx2); by2 = min(ai_h, by2)
                bw  = max(1, bx2 - bx1)
                bh  = max(1, by2 - by1)

                # Geometry posture (instant, every frame)
                ratio = bw / bh
                if ratio > 1.4:
                    geometry_label = "😴 sleeping / lying down"
                elif ratio > 0.85:
                    geometry_label = "🪑 sitting"
                else:
                    geometry_label = "🧍 standing"

                # SlowFast in background for fine-grained activity
                person_crop    = ai_frame[by1:by2, bx1:bx2]
                slowfast_label = (
                    self.action_det.update(tid, person_crop)
                    if person_crop.size > 0 else ""
                )

                # ── Object proximity check (instant: food → eating, phone → working) ─
                def _overlaps(obj_box, p_box, expand=40):
                    """Check if obj_box is inside or near person bbox."""
                    ox1,oy1,ox2,oy2 = obj_box
                    px1,py1,px2,py2 = p_box
                    px1 -= expand; py1 -= expand
                    px2 += expand; py2 += expand
                    return ox1 < px2 and ox2 > px1 and oy1 < py2 and oy2 > py1

                person_screen_box = [bx1*sx, by1*sy, bx2*sx, by2*sy]
                near_food  = any(_overlaps(f, person_screen_box)
                                 for f in self._last_objects.get("food", []))
                near_phone = any(_overlaps(p, person_screen_box)
                                 for p in self._last_objects.get("phone", []))

                # Priority: object detection > SlowFast specific > geometry
                SPECIFIC_TAGS = ("⚠", "🏃", "💪", "📖", "✍", "🍽", "💬", "💃", "🎵")
                if near_food:
                    action = "🍽 eating"
                elif near_phone:
                    action = "💻 working / using phone"
                elif slowfast_label and any(t in slowfast_label for t in SPECIFIC_TAGS):
                    action = slowfast_label
                else:
                    action = geometry_label

                # ── Emotion: pass full person crop to DeepFace directly ────
                # DeepFace has its own built-in face detector — no RetinaFace
                # needed. Works even when the face is at an angle or partially
                # visible, since enforce_detection=False is set inside analyse().
                if person_crop.size > 0:
                    self._submit_emotion(person_crop, tid)
                emotion = self.emotion_det._cache.get(tid, "")

                track_results.append(TrackResult(
                    track_id=tid,
                    bbox_full=_scale_box(bbox, sx, sy),
                    face_bbox_full=None,   # face box not needed in action_only mode
                    emotion=emotion,
                    action=action,
                ))
                continue

            # ── Full mode: face + emotion + action ─────────────────────────
            face_bb_ai = self.face_det.detect_in_crop(ai_frame, bbox)

            if face_bb_ai is not None:
                fx1, fy1, fx2, fy2 = (max(0, int(v)) for v in face_bb_ai)
                fx2 = min(ai_w, fx2); fy2 = min(ai_h, fy2)
                face_crop = ai_frame[fy1:fy2, fx1:fx2]
                if face_crop.size > 0:
                    self._submit_emotion(face_crop, tid)

            emotion = self.emotion_det._cache.get(tid, "")

            bx1, by1, bx2, by2 = (max(0, int(v)) for v in bbox)
            bx2 = min(ai_w, bx2); by2 = min(ai_h, by2)
            person_crop = ai_frame[by1:by2, bx1:bx2]
            action = self.action_det.update(tid, person_crop) if person_crop.size > 0 else "N/A"

            bbox_full    = _scale_box(bbox,       sx, sy)
            face_bb_full = _scale_box(face_bb_ai, sx, sy) if face_bb_ai else None

            track_results.append(TrackResult(
                track_id=tid,
                bbox_full=bbox_full,
                face_bbox_full=face_bb_full,
                emotion=emotion,
                action=action,
            ))


        # Periodic cleanup
        if self._frame_idx % 120 == 0:
            self.emotion_det.purge(active_ids)
            self.action_det.purge(active_ids)
            for tid in list(self._emotion_futures):
                if tid not in active_ids:
                    self._emotion_futures.pop(tid, None)

        return AIResult(motion=True, motion_mask=mask, tracks=track_results)


# ─── Imou Camera Selector (runs before the AI starts) ───────────────────────

def camera_select_startup() -> str | None:
    """
    Ask the user whether to use the Webcam or an Imou camera.
    If Imou, show the camera list and return the stream URL.
    If Webcam, return None (main() will use config.py SOURCE = 0).
    """
    # ── Hardcoded credentials ─────────────────────────────────────────
    APP_ID     = "lcfdd42a0e640d426e"
    APP_SECRET = "37c55ce8752a427f876b3baff50288"
    # ─────────────────────────────────────────────────────────────────

    print()
    print("=" * 60)
    print("  AI CCTV Surveillance System")
    print("=" * 60)
    print()
    print("  Select video source:")
    print("  [1]  Webcam (built-in / USB camera)")
    print("  [2]  Imou CCTV Camera (cloud-connected)")
    print()

    source_choice = input("  Your choice [1 or 2]: ").strip()

    if source_choice == "1":
        print("\n  ✓  Using Webcam. Starting AI …\n")
        return None   # main() will use SOURCE = 0

    if source_choice != "2":
        print("\n  Invalid choice – defaulting to Webcam.\n")
        return None

    # ── Imou Cloud path ───────────────────────────────────────────────
    print("\n  Connecting to Imou Cloud …")
    base_url = _find_working_datacenter()
    if not base_url:
        print("  ✗  Cannot reach Imou servers. Falling back to Webcam.")
        return None

    api = ImouAPI(APP_ID, APP_SECRET, base_url)

    try:
        api.get_token()
    except Exception as e:
        print(f"  ✗  Auth failed: {e}")
        return None

    try:
        devices = api.list_devices()
    except Exception as e:
        print(f"  ✗  Could not list cameras: {e}")
        return None

    if not devices:
        print("  ✗  No cameras found on this account. Falling back to Webcam.")
        return None

    # ── Show camera table ─────────────────────────────────────────────
    print(f"\n  {'#':<4} {'Camera Name':<25} {'Status':<22} Device ID")
    print("  " + "─" * 65)
    for i, dev in enumerate(devices):
        from imou_connector import _get_device_status
        status_str = _get_device_status(dev)
        name       = (dev.get("name") or dev.get("deviceName") or "Unnamed")[:24]
        dev_id     = dev.get("deviceId") or dev.get("deviceID") or "?"
        print(f"  [{i}] {name:<25} {status_str:<22} {dev_id}")

    print()

    # ── Camera selection ──────────────────────────────────────────────
    try:
        choice = input(f"  Which camera do you want to view? [0-{len(devices)-1}]: ").strip()
        idx    = int(choice)
        chosen = devices[idx]
    except (ValueError, IndexError):
        print("  ✗  Invalid selection – falling back to Webcam.")
        return None

    device_id  = chosen.get("deviceId") or chosen.get("deviceID")
    channel_id = chosen.get("channelId") or "0"
    cam_name   = chosen.get("name") or chosen.get("deviceName") or device_id

    print(f"\n  Connecting to '{cam_name}' …")

    try:
        url = api.get_rtsp(device_id, channel_id)
    except Exception as e:
        print(f"  ✗  Could not get stream URL: {e}")
        return None

    if url:
        print(f"  ✓  Stream ready. Launching AI on: {cam_name}\n")
        return url
    else:
        print("  ✗  No stream URL returned (camera may be offline).")
        return None


# ─── Main / Display loop ──────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 62)
    log.info("  AI CCTV Surveillance System  (threaded v2)")
    log.info("=" * 62)

    # ── Camera selection via Imou Cloud API ───────────────────────────
    selected_url = camera_select_startup()
    source = selected_url if selected_url else cfg.SOURCE

    frame_queue:  "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=2)
    result_store: list[AIResult]                   = [AIResult()]
    result_lock   = threading.Lock()

    ai_pipeline = AIPipeline(frame_queue, result_store, result_lock)
    ai_pipeline.start()

    log.info("Opening video source: %s", source)
    stream = VideoStream(
        source=source,
        width=cfg.FRAME_WIDTH,
        height=cfg.FRAME_HEIGHT,
        target_fps=cfg.TARGET_FPS,
    )

    fps_counter = FPSCounter(window=30)
    show_mask   = cfg.SHOW_MOTION_MASK
    capture_idx = 0

    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    log.info("Pipeline ready. Press 'q' to quit, 'm' for motion mask.")

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            capture_idx += 1
            current_fps  = fps_counter.tick()

            # Feed AI thread every N frames — pass real frame dimensions for correct scaling
            if capture_idx % cfg.AI_THREAD_FRAME_SKIP == 0:
                full_h, full_w = frame.shape[:2]
                ai_frame = cv2.resize(frame, (cfg.AI_FRAME_WIDTH, cfg.AI_FRAME_HEIGHT))
                # Attach real frame size so AI can scale bboxes accurately
                ai_pipeline._full_w = full_w
                ai_pipeline._full_h = full_h
                try:
                    frame_queue.put_nowait(ai_frame)
                except queue.Full:
                    pass

            # Read latest result
            with result_lock:
                result: AIResult = result_store[0]

            # ── Ghost-box fix: only draw if result is fresh ────────────
            result_age = time.monotonic() - result.timestamp
            tracks_to_draw = result.tracks if result_age < cfg.RESULT_MAX_AGE_SEC else []

            # ── Overlays ───────────────────────────────────────────────
            if show_mask and result.motion_mask is not None:
                big_mask = cv2.resize(
                    result.motion_mask,
                    (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
                draw_motion_mask(frame, big_mask)

            for tr in tracks_to_draw:
                draw_person(
                    frame, tr.bbox_full, tr.track_id,
                    emotion=tr.emotion, action=tr.action,
                )
                if tr.face_bbox_full is not None:
                    draw_face(frame, tr.face_bbox_full, tr.track_id)

            draw_status_bar(
                frame,
                motion=result.motion if result_age < cfg.RESULT_MAX_AGE_SEC else False,
                n_persons=len(tracks_to_draw),
                fps=current_fps,
            )

            cv2.imshow(cfg.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show_mask = not show_mask

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        log.info("Shutting down …")
        ai_pipeline.stop()
        ai_pipeline.join(timeout=5)
        stream.release()
        cv2.destroyAllWindows()
        log.info("Done.")


if __name__ == "__main__":
    main()
