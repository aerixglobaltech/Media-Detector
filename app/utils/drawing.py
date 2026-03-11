"""
utils/drawing.py
─────────────────────────────────────────────────────────────────────────────
Overlay rendering helpers: bounding boxes, labels, status banners.
"""

from __future__ import annotations

import cv2
import numpy as np

# Colour palette – one colour per track ID (cycles)
_PALETTE = [
    (255,  80,  80),
    ( 80, 200, 255),
    ( 80, 255, 120),
    (255, 200,  80),
    (200,  80, 255),
    (255, 255,  80),
    ( 80, 255, 255),
    (255, 150, 150),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------

def draw_person(
    frame: np.ndarray,
    bbox: list[float],
    track_id: int,
    emotion: str = "",
    action: str = "",
    identity: str = "",
) -> None:
    """Draw a bounding box and text labels for one tracked person."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = _track_color(track_id)

    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents (visual polish)
    corner_len = 14
    thickness = 3
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1),
        (x2, y1, -1,  1),
        (x2, y2, -1, -1),
        (x1, y2,  1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

    # Label block above the box
    lines = [
        f"Staff: {identity}" if (identity and identity != "Unknown") else f"Person #{track_id}",
        f"Emotion: {emotion}" if emotion else None,
        f"Action: {action}" if action else None,
    ]
    lines = [l for l in lines if l]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    line_height = 18

    # Background rectangle
    pad = 4
    block_h = len(lines) * line_height + pad * 2
    block_y1 = max(0, y1 - block_h)
    block_y2 = y1

    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_w = max(max_w, tw)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, block_y1), (x1 + max_w + pad * 2, block_y2), color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, line in enumerate(lines):
        ty = block_y1 + pad + (i + 1) * line_height - 2
        cv2.putText(
            frame, line,
            (x1 + pad, ty),
            font, font_scale, (255, 255, 255), font_thickness,
            cv2.LINE_AA,
        )


def draw_face(
    frame: np.ndarray,
    face_bbox: list[float],
    track_id: int,
) -> None:
    """Draw a thin face bounding box."""
    fx1, fy1, fx2, fy2 = (int(v) for v in face_bbox)
    color = _track_color(track_id)
    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 1)


def draw_status_bar(
    frame: np.ndarray,
    motion: bool,
    n_persons: int,
    fps: float,
) -> None:
    """Top-left status overlay."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Motion: {'YES' if motion else 'NO'}",
        f"Persons: {n_persons}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    fthick = 1
    lh = 22
    pad = 8
    max_w = max(cv2.getTextSize(l, font, fscale, fthick)[0][0] for l in lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (max_w + pad * 2, len(lines) * lh + pad * 2),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        color = (0, 255, 100) if ("YES" in line or "Motion" not in line) else (0, 120, 255)
        if "Motion" in line and not motion:
            color = (180, 180, 180)
        cv2.putText(
            frame, line,
            (pad, pad + (i + 1) * lh - 4),
            font, fscale, color, fthick, cv2.LINE_AA,
        )


def draw_motion_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.15,
) -> None:
    """Tint moving pixels green for debugging."""
    tint = np.zeros_like(frame)
    tint[mask > 0] = (0, 255, 80)
    cv2.addWeighted(tint, alpha, frame, 1 - alpha, 0, frame)
