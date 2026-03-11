"""
core/camera_manager.py  –  Camera Manager
──────────────────────────────────────────────────────────────────────────────
Manages camera switching with ZERO model reloads.

Architecture
────────────
  __init__()    → creates AIPipeline ONCE (loads all AI models)
  start()       → replaces VideoStream + RenderThread only (< 1 second)
  stop_stream() → stops video + render thread (pipeline keeps running)
  stop_all()    → full shutdown, used only on app exit

The render thread reads frames from the VideoStream and:
  1. Downscales every Nth frame → feeds to AI pipeline queue
  2. Draws AI result overlays at full camera FPS
  3. JPEG-encodes the annotated frame → stored in _jpeg_buf

The Flask /video_feed route reads _jpeg_buf in a streaming response.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import cv2
import numpy as np

from app.core import config as cfg
from app.core.data_types import AIResult
from app.pipelines.ai_pipeline import AIPipeline
from app.utils.drawing import draw_face, draw_motion_mask, draw_person, draw_status_bar
from app.utils.fps_counter import FPSCounter
from app.utils.stream import VideoStream

log = logging.getLogger("camera_manager")

# Shared motion events counter (read by /api/dashboard_info)
_total_motion_events: int = 0


class CameraManager:
    """
    Manages the complete video → AI → display pipeline.
    AI models are loaded exactly once and reused across camera switches.
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

        # Determine upload folder for face recognition DB
        upload_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static", "uploads",
        )

        log.info("Loading AI models (one-time startup) …")
        self._pipeline = AIPipeline(
            in_queue=self._frame_queue,
            result_store=self._result_store,
            result_lock=self._result_lock,
            upload_folder=upload_folder,
        )
        self._pipeline.start()
        log.info("AI models ready.")

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start(self, source, name: str = "Camera") -> None:
        """Switch to a new camera source — models are NOT reloaded."""
        self.stop_stream()     # stop old video + render thread only

        log.info("Starting camera: %s  source=%s", name, source)
        self.camera_name = name
        if self._pipeline:
            self._pipeline.camera_name = name
        self._active          = True
        self._render_stop_evt = threading.Event()
        self.stream_id       += 1   # frontend polls this to detect camera switches

        # Drain stale frames so new camera starts clean
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        # Reset AI result to prevent ghost boxes appearing on the new camera
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
            args=(self._render_stop_evt, stream),
            daemon=True,
            name="Render",
        )
        self._render_thread.start()

    def stop_stream(self) -> None:
        """Stop video capture + render thread.  AI pipeline keeps running."""
        self._active = False
        self._render_stop_evt.set()
        if self._render_thread and self._render_thread.is_alive():
            self._render_thread.join(timeout=4)
            self._render_thread = None
        with self._jpeg_lock:
            self._jpeg_buf = b""
        self._fps_val = 0.0

    def stop_all(self) -> None:
        """Full shutdown — used only when the Flask app exits."""
        self.stop_stream()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.join(timeout=8)
            self._pipeline = None

    # ── Render Loop ───────────────────────────────────────────────────────────

    def _render_loop(self, stop_evt: threading.Event, stream: VideoStream) -> None:
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

                cap_idx      += 1
                self._fps_val = fps_ctr.tick()

                # Feed the AI thread every N frames
                if cap_idx % cfg.AI_THREAD_FRAME_SKIP == 0:
                    fh, fw = frame.shape[:2]
                    ai_f   = cv2.resize(frame, (cfg.AI_FRAME_WIDTH, cfg.AI_FRAME_HEIGHT))
                    self._pipeline._full_w = fw
                    self._pipeline._full_h = fh
                    try:
                        self._frame_queue.put_nowait(ai_f)
                    except queue.Full:
                        pass

                # Read latest AI result
                with self._result_lock:
                    result = self._result_store[0]

                age    = time.monotonic() - result.timestamp
                tracks = result.tracks if age < cfg.RESULT_MAX_AGE_SEC else []

                # Draw overlays
                for tr in tracks:
                    draw_person(
                        frame, tr.bbox_full, tr.track_id,
                        emotion=tr.emotion,
                        action=tr.action,
                        identity=tr.identity,
                    )
                    if tr.face_bbox_full:
                        draw_face(frame, tr.face_bbox_full, tr.track_id)

                draw_status_bar(
                    frame,
                    motion=result.motion if age < cfg.RESULT_MAX_AGE_SEC else False,
                    n_persons=len(tracks),
                    fps=self._fps_val,
                )

                # JPEG encode and store for the MJPEG route
                ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok2:
                    with self._jpeg_lock:
                        self._jpeg_buf = jpg.tobytes()

        finally:
            try:
                stream.release()
            except Exception:
                pass
            log.info("Render thread exited for: %s", self.camera_name)

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_jpeg(self) -> bytes:
        """Return the most recently encoded JPEG frame."""
        with self._jpeg_lock:
            return self._jpeg_buf

    def get_stats(self) -> dict:
        """Return live detection stats as a JSON-serializable dict."""
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
                {
                    "id":       t.track_id,
                    "emotion":  t.emotion  or "–",
                    "action":   t.action   or "–",
                    "identity": t.identity or "",
                }
                for t in tracks
            ],
        }
