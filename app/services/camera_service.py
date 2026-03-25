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
        self._monitoring_nodes: dict[str, dict] = {} # {name: {"stream": VideoStream, "stop_evt": Event, "last_frame": ndarray}}
        self._monitoring_lock = threading.Lock()

        # Determine upload folder for face recognition DB
        upload_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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

    def start(self, source, name: str = "Camera", **kwargs) -> None:
        """Switch to a new camera instantly by promoting its background monitoring node."""
        log.info("Switching view to: %s", name)
        
        roles = kwargs.get("roles", [])
        
        # Ensure the camera is being monitored in the background
        # If it's already monitoring, this just returns True without doing anything
        self.start_monitoring(source, name, roles=roles)
        
        self.camera_name = name
        if self._pipeline:
            self._pipeline.camera_name = name
            self._pipeline.camera_roles = roles
            
        self._active = True
        self.stream_id += 1 

        # Start the global render thread if not already running
        if self._render_thread is None or not self._render_thread.is_alive():
            self._render_stop_evt = threading.Event()
            self._render_thread = threading.Thread(
                target=self._render_loop,
                args=(self._render_stop_evt,),
                daemon=True,
                name="GlobalRender",
            )
            self._render_thread.start()

    def reload_faces(self) -> None:
        """Tell the AI pipeline to reload known face signatures."""
        if self._pipeline:
            self._pipeline.reload_faces()

    def stop_stream(self) -> None:
        """Mute the main UI stream. Background AI monitoring continues."""
        self._active = False
        with self._jpeg_lock:
            self._jpeg_buf = b""
        self._fps_val = 0.0

    def start_monitoring(self, source, name: str, roles: list[str]) -> bool:
        """Start a background stream for AI processing without changing the UI view."""
        with self._monitoring_lock:
            if name in self._monitoring_nodes:
                return True # Already monitoring
            
            log.info(f"AI: Starting background monitoring for '{name}' (source={source})")
            stop_evt = threading.Event()
            try:
                stream = VideoStream(source=source, width=cfg.FRAME_WIDTH, height=cfg.FRAME_HEIGHT, target_fps=5.0)
                thread = threading.Thread(
                    target=self._monitoring_loop,
                    args=(stop_evt, stream, name, roles),
                    daemon=True,
                    name=f"Monitor_{name}"
                )
                thread.start()
                self._monitoring_nodes[name] = {
                    "stream": stream, 
                    "stop_evt": stop_evt, 
                    "thread": thread, 
                    "last_frame": None,
                    "roles": roles
                }
                return True
            except Exception as e:
                log.error(f"AI: Failed to start monitoring for '{name}': {e}")
                return False

    def stop_monitoring(self, name: str) -> None:
        """Stop background monitoring for a specific camera."""
        with self._monitoring_lock:
            if name in self._monitoring_nodes:
                node = self._monitoring_nodes.pop(name)
                node["stop_evt"].set()
                # Thread will exit on next read

    def stop_all(self) -> None:
        """Full shutdown — used only when the Flask app exits."""
        self.stop_stream()
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline.join(timeout=8)
            self._pipeline = None

    # ── Render Loop ───────────────────────────────────────────────────────────

    def _render_loop(self, stop_evt: threading.Event) -> None:
        """
        Global render thread. Pulls frames from the currently active monitoring node.
        """
        fps_ctr = FPSCounter(window=30)
        while not stop_evt.is_set():
            if not self._active or not self.camera_name:
                time.sleep(0.1)
                continue

            # Find the monitoring node for the active camera
            node = None
            with self._monitoring_lock:
                node = self._monitoring_nodes.get(self.camera_name)
            
            if not node or node.get("last_frame") is None:
                time.sleep(0.01)
                continue

            frame = node["last_frame"].copy()
            self._fps_val = fps_ctr.tick()

            # Read latest AI result
            with self._result_lock:
                result = self._result_store[0]
            
            age = time.monotonic() - result.timestamp
            tracks = result.tracks if age < cfg.RESULT_MAX_AGE_SEC else []

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
                entries=result.entries,
                exits=result.exits
            )

            # JPEG encode for the MJPEG route
            ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                with self._jpeg_lock:
                    self._jpeg_buf = jpg.tobytes()

            time.sleep(1.0 / cfg.TARGET_FPS)

    def _monitoring_loop(self, stop_evt: threading.Event, stream: VideoStream, name: str, roles: list[str]):
        """Runs in background for AI processing of non-active cameras."""
        cap_idx = 0
        try:
            while not stop_evt.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    if stop_evt.is_set(): break
                    time.sleep(0.1); continue

                # Update last frame for the main renderer
                with self._monitoring_lock:
                    if name in self._monitoring_nodes:
                        self._monitoring_nodes[name]["last_frame"] = frame

                # Feed the AI thread every N frames
                if cap_idx % cfg.AI_THREAD_FRAME_SKIP == 0:
                    fh, fw = frame.shape[:2]
                    ai_f = cv2.resize(frame, (cfg.AI_FRAME_WIDTH, cfg.AI_FRAME_HEIGHT))
                    try:
                        self._frame_queue.put_nowait((ai_f, name, roles, (fw, fh)))
                    except queue.Full: pass

                cap_idx += 1
                time.sleep(0.02)
        finally:
            try: stream.release()
            except: pass
            log.info(f"AI: Monitoring thread exited for: {name}")

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
            "entries": r.entries,
            "exits":   r.exits,
            "tracks": [
                {
                    "id":       t.track_id,
                    "emotion":  t.emotion  or "–",
                    "action":   t.action   or "–",
                    "identity": t.identity or "",
                    "staff_id": t.display_id or "",
                }
                for t in tracks
            ],
        }
