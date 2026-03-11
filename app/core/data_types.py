"""
core/data_types.py  –  Shared AI Result Data Classes
──────────────────────────────────────────────────────────────────────────────
Defines the data structures shared between the AI pipeline thread
and the render/display threads.

  TrackResult  – bounding box + emotion + action + identity for one person
  AIResult     – full output of one pipeline processing cycle
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackResult:
    """Result data for a single tracked person."""
    track_id:       int
    bbox_full:      list[float]           # [x1, y1, x2, y2] in full-frame coords
    face_bbox_full: list[float] | None    # face crop coords, or None
    emotion:        str                   # e.g. "happy", "neutral", ""
    action:         str                   # e.g. "🧍 standing", "🏃 running"
    identity:       str = ""              # recognized staff name, or ""


@dataclass
class AIResult:
    """Aggregate output from one run of the AI pipeline."""
    motion:      bool               = False
    motion_mask: np.ndarray | None  = None
    tracks:      list[TrackResult]  = field(default_factory=list)
    timestamp:   float              = field(default_factory=time.monotonic)
    fps:         float              = 0.0


def scale_box(box: list[float], sx: float, sy: float) -> list[float]:
    """Scale a bounding box [x1, y1, x2, y2] by (sx, sy) ratios."""
    x1, y1, x2, y2 = box
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
