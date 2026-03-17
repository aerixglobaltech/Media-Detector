"""
modules/object_detection.py
─────────────────────────────────────────────────────────────────────────────
Person detection using YOLOv8 (Ultralytics).

Only boxes whose class label == 'person' and whose confidence
exceeds `conf_threshold` are returned.
"""

from __future__ import annotations

import numpy as np
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    log.warning("ultralytics module not found. AI detection will be disabled.")
    HAS_YOLO = False
    class YOLO:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return []


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


import logging
log = logging.getLogger("object_det")

class PersonDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model."""

    PERSON_CLASS_ID = 0  # COCO class index for 'person'
    ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.40,
        device: str = "cpu",
    ):
        log.info("Loading YOLO model: %s on %s", model_path, device)
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
            verbose=True,        # Enable Ultralytics' own logging
            agnostic_nms=True,
            iou=0.4,
        )

        boxes: list[list[float]] = []
        for result in results:
            if len(result.boxes) > 0:
                log.info("YOLO: Found %d persons (conf > %.2f)", len(result.boxes), self.conf_threshold)
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                # 1. Kill "Giant Ghost Boxes" (Likely background wall/floor errors)
                bw, bh = (x2 - x1), (y2 - y1)
                fw, fh = frame.shape[1], frame.shape[0]
                if bw > (fw * 0.98) or bh > (fh * 0.98):
                    log.debug("YOLO: Filtered giant box: %dx%d", int(bw), int(bh))
                    continue

                # 2. Filter out tiny artifacts (Noise)
                if bh < (fh * 0.05):
                    log.debug("YOLO: Filtered tiny box: %dx%d", int(bw), int(bh))
                    continue

                boxes.append([x1, y1, x2, y2, conf])

        if not boxes and len(results) > 0 and len(results[0].boxes) > 0:
            log.info("YOLO: All detections were filtered out by giant/tiny rules.")

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

    def classify_motion(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Classify a movement frame as human / animal / unknown.
        Returns: (label, confidence)
        """
        try:
            results = self.model(
                frame,
                conf=0.50,
                classes=[self.PERSON_CLASS_ID, *sorted(self.ANIMAL_CLASS_IDS)],
                device=self.device,
                verbose=False,
            )
            best_label = "unknown"
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if cls_id == self.PERSON_CLASS_ID and conf >= best_conf:
                        best_label = "human"
                        best_conf = conf
                    elif cls_id in self.ANIMAL_CLASS_IDS and conf >= best_conf and best_label != "human":
                        best_label = "animal"
                        best_conf = conf
            return best_label, best_conf
        except Exception as exc:
            log.warning("YOLO classify_motion failed: %s", exc)
            return "unknown", 0.0
