"""
utils/fps_counter.py
─────────────────────────────────────────────────────────────────────────────
Lightweight rolling-average FPS counter.
"""

import time
from collections import deque


class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30):
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Call once per processed frame.  Returns current FPS."""
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
