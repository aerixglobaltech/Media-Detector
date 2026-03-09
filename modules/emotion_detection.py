"""
modules/emotion_detection.py
─────────────────────────────────────────────────────────────────────────────
Emotion classification using DeepFace.

Fixes applied:
  - Minimum face crop size check (avoids garbage results on tiny crops).
  - Face crop is contrast-enhanced before analysis.
  - Cache is returned immediately on skip frames (no blocking).
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from deepface import DeepFace as _DF
    _DEEPFACE_OK = True
except Exception:
    _DEEPFACE_OK = False


class EmotionDetector:
    """
    Per-track emotion cache with configurable refresh rate.

    Parameters
    ----------
    skip_frames     : int  – re-analyse after this many calls per track
    backend         : str  – DeepFace detector backend ('opencv' is fastest)
    min_face_pixels : int  – ignore face crops smaller than this (px on shortest side)
    """

    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(
        self,
        skip_frames: int = 20,
        backend: str = "opencv",
        min_face_pixels: int = 40,
    ):
        self.skip_frames = skip_frames
        self.backend = backend
        self.min_face_pixels = min_face_pixels
        self._cache: dict[int, str] = {}
        self._counters: dict[int, int] = {}
        self._available = _DEEPFACE_OK

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(face_crop: np.ndarray) -> np.ndarray:
        """Enhance face crop for better DeepFace accuracy."""
        # Convert to LAB, equalise lightness, convert back
        try:
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        except Exception:
            return face_crop

    # ------------------------------------------------------------------
    def analyse(self, face_crop: np.ndarray, track_id: int) -> str:
        """
        Return the dominant emotion for *track_id*.

        Parameters
        ----------
        face_crop : BGR image of the face region
        track_id  : unique person track ID

        Returns
        -------
        emotion string, e.g. 'happy', 'neutral', 'angry'
        """
        if not self._available:
            return "unknown"

        counter = self._counters.get(track_id, 0)
        self._counters[track_id] = counter + 1

        # Return cached result between refresh intervals
        if counter % self.skip_frames != 0 and track_id in self._cache:
            return self._cache[track_id]

        # Validate crop size
        if face_crop is None or face_crop.size == 0:
            return self._cache.get(track_id, "unknown")

        h, w = face_crop.shape[:2]
        if min(h, w) < self.min_face_pixels:
            return self._cache.get(track_id, "unknown")

        # Resize to a standard 224×224 for consistent accuracy
        try:
            resized = cv2.resize(face_crop, (224, 224))
            enhanced = self._preprocess(resized)

            result = _DF.analyze(
                img_path=enhanced,
                actions=["emotion"],
                detector_backend=self.backend,
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            emotion: str = result.get("dominant_emotion", "unknown")
        except Exception:
            emotion = self._cache.get(track_id, "unknown")

        self._cache[track_id] = emotion
        return emotion

    # ------------------------------------------------------------------
    def purge(self, active_ids: set[int]) -> None:
        """Remove cached entries for disappeared tracks."""
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                self._cache.pop(tid, None)
                self._counters.pop(tid, None)
