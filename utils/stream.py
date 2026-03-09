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
        self._open()

    # ------------------------------------------------------------------
    def _open(self) -> None:
        if self._cap is not None:
            self._cap.release()

        src = self.source

        # For HLS (.m3u8) and RTSP streams, force the FFMPEG backend.
        # Without this, Windows picks a backend that cannot decode video.
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
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)   # low latency
            log.info("Stream opened: %s", src)
        else:
            log.warning("Failed to open stream: %s", src)


    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read one frame, attempting reconnect if the stream dropped.

        Returns
        -------
        ok    : bool
        frame : BGR ndarray or None
        """
        if self._cap is None or not self._cap.isOpened():
            log.info("Reconnecting to %s …", self.source)
            time.sleep(self.reconnect_delay)
            self._open()
            return False, None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            log.warning("Lost frame from %s – attempting reconnect", self.source)
            time.sleep(self.reconnect_delay)
            self._open()
            return False, None

        return True, frame

    # ------------------------------------------------------------------
    def release(self) -> None:
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
