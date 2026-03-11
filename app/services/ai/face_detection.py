"""
modules/face_detection.py
─────────────────────────────────────────────────────────────────────────────
Face detection using RetinaFace, restricted to a person's bounding box.

For each person crop we run RetinaFace and return the largest detected
face so that downstream emotion analysis gets the most prominent face.
"""

from __future__ import annotations

import numpy as np

try:
    from retinaface import RetinaFace as _RF
    _RETINAFACE_OK = True
except Exception:
    _RETINAFACE_OK = False


class FaceDetector:
    """
    Detects faces within person bounding boxes using RetinaFace.
    Falls back to center-crop when RetinaFace is unavailable.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self._available = _RETINAFACE_OK

    # ------------------------------------------------------------------
    def detect_in_crop(
        self,
        frame: np.ndarray,
        person_bbox: list[float],
    ) -> list[float] | None:
        """
        Detect the largest face inside *person_bbox* on *frame*.

        Parameters
        ----------
        frame       : full BGR frame
        person_bbox : [x1, y1, x2, y2] (person bounding box)

        Returns
        -------
        face_bbox_abs : [x1, y1, x2, y2] in **frame** coordinates, or None
        """
        h_frame, w_frame = frame.shape[:2]

        px1 = max(0, int(person_bbox[0]))
        py1 = max(0, int(person_bbox[1]))
        px2 = min(w_frame, int(person_bbox[2]))
        py2 = min(h_frame, int(person_bbox[3]))

        if px2 <= px1 or py2 <= py1:
            return None

        crop = frame[py1:py2, px1:px2]

        if not self._available:
            # Fallback: upper-third of the person crop
            ch = crop.shape[0]
            return [px1, py1, px2, py1 + ch // 3]

        try:
            faces = _RF.detect_faces(crop, threshold=self.threshold)
        except Exception:
            return None

        if not isinstance(faces, dict) or len(faces) == 0:
            return None

        # Pick the face with the largest area
        best: list[float] | None = None
        best_area = 0
        for face_data in faces.values():
            fx1, fy1, fx2, fy2 = face_data["facial_area"]
            area = (fx2 - fx1) * (fy2 - fy1)
            if area > best_area:
                best_area = area
                best = [px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2]

        return best
