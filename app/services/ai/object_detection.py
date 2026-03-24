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
        self._untracked_count = 0 

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
                # Relaxed from 0.98 to 1.0 to allow subjects that fill the frame (e.g. very close to lens)
                if bw > (fw * 1.0) or bh > (fh * 1.0):
                    log.info("YOLO: Filtered giant box: %dx%d (Frame: %dx%d)", int(bw), int(bh), fw, fh)
                    continue

                # 2. Filter out tiny artifacts (Noise)
                # Relaxed from 0.05 to 0.02 to allow subjects very far away
                if bh < (fh * 0.02):
                    log.debug("YOLO: Filtered tiny box: %dx%d", int(bw), int(bh))
                    continue

                boxes.append([x1, y1, x2, y2, conf])

        if not boxes and len(results) > 0 and len(results[0].boxes) > 0:
            log.info("YOLO: All detections were filtered out by giant/tiny rules.")

        return boxes

    def track(self, frame: np.ndarray, persist: bool = True) -> list[list[float]]:
        """
        Run YOLOv8 tracking on *frame* (ByteTrack).
        Returns list of [x1, y1, x2, y2, confidence, track_id]
        """
        if not HAS_YOLO: return []
        
        classes = [self.PERSON_CLASS_ID, *sorted(self.ANIMAL_CLASS_IDS)]
        results = self.model.track(
            frame,
            persist=persist,
            classes=classes,
            conf=0.25, # Lowered from 0.30 to catch more animal signals during rapid movement
            iou=0.5,
            device=self.device,
            verbose=False
        )

        tracks: list[list[float]] = []
        for result in results:
            if result.boxes is None:
                continue
            
            # Log raw detections for debugging
            if len(result.boxes) > 0:
                log.info(f"YOLO: Found {len(result.boxes)} detections in frame (classes: {classes})")
            
            # Fallback for frames where tracker might not have assigned IDs yet
            # We use UNIQUE negative IDs to prevent cache collisions in FaceRecognizer
            if result.boxes.id is not None:
                ids = result.boxes.id.tolist()
            else:
                ids = []
                for _ in range(len(result.boxes)):
                    self._untracked_count += 1
                    ids.append(-(1000 + (self._untracked_count % 10000)))
            
            img_h = frame.shape[0]
            for box, tid, conf, cls in zip(result.boxes.xyxy, ids, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                bw, bh = x2 - x1, y2 - y1
                cls_id = int(cls)
                
                # REJECTION FILTERS TO BLOCK CEILING FANS / ARTIFACTS:
                # 1. Ceiling Zone: Ignore small objects in the top 15% of the frame (Fans)
                if y2 < (img_h * 0.15) and bh < (img_h * 0.20):
                    continue
                
                # 2. Min Height: Ignore tiny artifacts (< 30px)
                if bh < 30:
                    continue
                    
                # 3. Aspect Ratio: 
                # Humans (vertical) vs Animals (often horizontal/wide)
                ratio = bw / bh
                if cls_id == self.PERSON_CLASS_ID:
                    if ratio < 0.10 or ratio > 2.5: # Human aspect ratio guards
                        continue
                else:
                    if ratio < 0.05 or ratio > 4.5: # Relaxed for animals (e.g. dogs walking/running)
                        continue
                
                track_id = int(tid)
                tracks.append([x1, y1, x2, y2, float(conf), track_id, cls_id])

        return tracks

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
