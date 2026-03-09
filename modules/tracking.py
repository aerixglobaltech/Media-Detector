"""
modules/tracking.py
─────────────────────────────────────────────────────────────────────────────
Person tracking using DeepSORT (deep-sort-realtime library).

Converts YOLOv8 detections into the format expected by DeepSORT and
returns standardized Track objects.
"""

from __future__ import annotations

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class Track:
    """Lightweight container for a confirmed track."""

    __slots__ = ("track_id", "bbox")

    def __init__(self, track_id: int, bbox: list[float]):
        self.track_id = track_id          # unique integer ID
        self.bbox = bbox                  # [x1, y1, x2, y2]

    def __repr__(self) -> str:
        return f"Track(id={self.track_id}, bbox={[round(v) for v in self.bbox]})"


class PersonTracker:
    """Wraps deep-sort-realtime for person tracking."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        embedder: str = "mobilenet",  # 'mobilenet' or 'torchreid'
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            half=False,
            bgr=True,
        )

    # ------------------------------------------------------------------
    def update(
        self,
        detections: list[list[float]],
        frame: np.ndarray,
    ) -> list[Track]:
        """
        Parameters
        ----------
        detections : list of [x1, y1, x2, y2, confidence]
        frame      : BGR image used by DeepSORT's appearance embedder

        Returns
        -------
        tracks : list[Track] – only confirmed (active) tracks
        """
        # deep-sort-realtime expects: list of ([left, top, w, h], confidence, class)
        ds_input: list[tuple] = []
        for x1, y1, x2, y2, conf in detections:
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], conf, "person"))

        raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        tracks: list[Track] = []
        for t in raw_tracks:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()  # [x1, y1, x2, y2]
            tracks.append(Track(track_id=int(t.track_id), bbox=list(ltrb)))

        return tracks
