"""
utils/stream.py
─────────────────────────────────────────────────────────────────────────────
Video stream management helpers.

Supports:
    - Webcam (integer index)
    - RTSP / HTTP CCTV streams (URL string)
    - Video files (path string)
"""

from __future__ import annotations

import time
import logging
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoStream:
    """
    Thin OpenCV VideoCapture wrapper with:
      * automatic reconnect on RTSP streams
      * frame skipping to target a desired FPS
      * basic frame validation
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 25.0,
        reconnect_delay: float = 3.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay

        self._cap: cv2.VideoCapture | None = None
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._latest_frame = None
        self._ok = False
        
        self._open()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        if self._cap is not None:
            self.release()

        src = self.source
        is_url = isinstance(src, str) and (
            src.startswith("rtsp://")
            or src.startswith("http://")
            or src.startswith("https://")
            or ".m3u8" in src
        )

        if is_url:
            self._cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self._cap = cv2.VideoCapture(src)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # absolute lowest latency
            
            # Start the frame-purging background thread
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._update, daemon=True, name="VideoStream")
            self._thread.start()
            
            log.info("Stream opened (Threaded): %s", src)
        else:
            log.warning("Failed to open stream: %s", src)

    # ------------------------------------------------------------------
    def _update(self):
        """Background thread that constantly reads frames to prevent lagging behind."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                break
            # Blocking read inside the thread
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                self._latest_frame = frame
            # If the camera drops, standard read logic handles reconnect
            if not ok:
                break
                
    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Grab the absolute most recent frame dynamically."""
        with self._lock:
            ok, frame = self._ok, self._latest_frame

        if not ok or frame is None:
            # If it's a webcam (int) that failed, do not aggressively spam the logs trying to reconnect
            delay = 30.0 if isinstance(self.source, int) else self.reconnect_delay
            log.warning("Lost connection to %s – attempting reconnect in %ss", self.source, delay)
            self.release()
            time.sleep(delay)
            self._open()
            return False, None

        return True, frame.copy()

    # ------------------------------------------------------------------
    def release(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return float(self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps)
        return self.target_fps

    # ------------------------------------------------------------------
    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *_) -> None:
        self.release()
