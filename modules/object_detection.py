"""
modules/object_detection.py
─────────────────────────────────────────────────────────────────────────────
Person detection using YOLOv8 (Ultralytics).

Only boxes whose class label == 'person' and whose confidence
exceeds `conf_threshold` are returned.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


# COCO class IDs for food / eating-related objects
FOOD_CLASS_IDS = {
    39,  # bottle
    40,  # wine glass
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
}

# COCO class IDs for phone / working-related objects
PHONE_CLASS_IDS = {
    66,  # keyboard
    67,  # cell phone
    63,  # laptop
    64,  # mouse
    73,  # book
}


class PersonDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model."""

    PERSON_CLASS_ID = 0  # COCO class index for 'person'

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.40,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """
        Run YOLOv8 on *frame* and return detected person bounding boxes.

        Returns list of [x1, y1, x2, y2, confidence]
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        boxes: list[list[float]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append([x1, y1, x2, y2, conf])

        return boxes

    # ------------------------------------------------------------------
    def detect_objects(self, frame: np.ndarray) -> dict[str, list[list[float]]]:
        """
        Detect non-person objects (food, phone, book etc.) in *frame*.

        Returns dict:
          {
            "food":  [[x1,y1,x2,y2], ...],   # food/eating objects found
            "phone": [[x1,y1,x2,y2], ...],   # phone/laptop/book found
          }
        """
        all_ids = FOOD_CLASS_IDS | PHONE_CLASS_IDS
        results = self.model(
            frame,
            conf=0.30,
            classes=list(all_ids),
            device=self.device,
            verbose=False,
        )

        food_boxes:  list[list[float]] = []
        phone_boxes: list[list[float]] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [x1, y1, x2, y2]
                if cls_id in FOOD_CLASS_IDS:
                    food_boxes.append(bbox)
                elif cls_id in PHONE_CLASS_IDS:
                    phone_boxes.append(bbox)

        return {"food": food_boxes, "phone": phone_boxes}
