"""
modules/motion_detection.py
─────────────────────────────────────────────────────────────────────────────
Motion detection using OpenCV's MOG2 background subtractor.

The detector returns True when the pixel area of detected movement exceeds
`min_area`.  This acts as a cheap gate before the expensive AI pipeline.
"""

import cv2
import numpy as np


class MotionDetector:
    """
    Wraps cv2.createBackgroundSubtractorMOG2 and exposes a simple
    detect(frame) → (bool, mask) interface.
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_area: int = 1500,
        blur_kernel: int = 21,
    ):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.blur_kernel = blur_kernel

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Parameters
        ----------
        frame : BGR image (H, W, 3)

        Returns
        -------
        motion_detected : bool
        mask            : binary mask (H, W) after morphological clean-up
        """
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        fg_mask = self.subtractor.apply(blurred)

        # Remove shadows (grey pixels → 127) and noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours to measure total moving area
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        motion_area = sum(cv2.contourArea(c) for c in contours)

        return motion_area >= self.min_area, fg_mask
