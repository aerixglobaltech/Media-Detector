from __future__ import annotations
import os
import cv2
import numpy as np

try:
    from deepface import DeepFace as _DF
    _DEEPFACE_OK = True
except Exception:
    _DEEPFACE_OK = False

class FaceRecognizer:
    def __init__(self, db_path: str, skip_frames: int = 15, backend: str = "opencv", model_name="VGG-Face"):
        self.db_path = db_path
        self.skip_frames = skip_frames
        self.backend = backend
        self.model_name = model_name
        self._cache: dict[int, str] = {}
        self._counters: dict[int, int] = {}
        self._available = _DEEPFACE_OK

        os.makedirs(db_path, exist_ok=True)

    def recognize(self, face_crop: np.ndarray, track_id: int) -> str:
        if not self._available or face_crop is None or face_crop.size == 0:
            return ""

        counter = self._counters.get(track_id, 0)
        self._counters[track_id] = counter + 1

        if counter % self.skip_frames != 0 and track_id in self._cache:
            return self._cache[track_id]

        h, w = face_crop.shape[:2]
        if min(h, w) < 40:
            return self._cache.get(track_id, "")

        # Check if database has any valid photos first
        has_images = False
        for root, dirs, files in os.walk(self.db_path):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                has_images = True
                break
        
        if not has_images:
            return self._cache.get(track_id, "")

        try:
            enhanced = cv2.resize(face_crop, (224, 224))
            # DeepFace.find expects an image path or a BGR numpy array
            result = _DF.find(
                img_path=enhanced,
                db_path=self.db_path,
                detector_backend=self.backend,
                model_name=self.model_name,
                enforce_detection=False,
                silent=True,
            )
            
            if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                df = result[0]
                best_match_path = df.iloc[0]["identity"]
                norm_path = os.path.normpath(best_match_path)
                parts = norm_path.split(os.sep)
                if len(parts) >= 2:
                    staff_name = parts[-2]
                else:
                    staff_name = "Unknown"
                
                self._cache[track_id] = staff_name
                return staff_name
        except Exception as e:
            pass

        return self._cache.get(track_id, "")

    def purge(self, active_ids: set[int]) -> None:
        """Remove cached entries for disappeared tracks."""
        for tid in list(self._cache.keys()):
            if tid not in active_ids:
                self._cache.pop(tid, None)
                self._counters.pop(tid, None)
