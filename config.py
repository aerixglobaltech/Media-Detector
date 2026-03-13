"""
config.py  –  AI CCTV Surveillance System
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os

# ─── Video Source ──────────────────────────────────────────────────────────
SOURCE: int | str = os.environ.get("IMOU_CAM1_URL", 0)   # Default to Camera 1 if set, else webcam


# ─── Camera List (Web UI) ───────────────────────────────────────────────────
CAMERAS: list[dict] = [
    {
        "id":   "cam_1",
        "name": "IMOU Camera 1",
        "url":  os.environ.get("IMOU_CAM1_URL", "rtsp://admin:L2BA7F0F@192.168.1.40:554/cam/realmonitor?channel=1&subtype=0"),
    },
    {
        "id":   "cam_2",
        "name": "IMOU Camera 2",
        "url":  os.environ.get("IMOU_CAM2_URL", "rtsp://admin:L28DED5C@192.168.1.240:554/cam/realmonitor?channel=1&subtype=0"),
    },
]
FRAME_WIDTH:  int   = 1280
FRAME_HEIGHT: int   = 720
TARGET_FPS:   float = 30.0

# Professional Surveillance Resolution (YOLO Native 640x384)
AI_FRAME_WIDTH:  int = 640
AI_FRAME_HEIGHT: int = 384

# ─── Motion Detection ──────────────────────────────────────────────────────
MOTION_MIN_AREA:      int   = 800    # lower = catches subtle movement
MOTION_HISTORY:       int   = 300
MOTION_VAR_THRESHOLD: float = 12.0  # lower = more sensitive

# Once motion is detected, keep the AI pipeline running for this many seconds
# even if the person sits perfectly still (MOG2 would otherwise call it "background").
MOTION_COOLDOWN_SEC: float = 30.0

# ─── Person Detection (YOLOv8) ─────────────────────────────────────────────
YOLO_MODEL:       str   = "yolov8n.pt"  # Nano Model: Exact speed of reference videos
YOLO_CONF:        float = 0.30          # Webb-Presence Fix: Detects desk-sitting people reliably
YOLO_DEVICE:      str   = "cpu" 
YOLO_SKIP_FRAMES: int   = 0             # Direct movement processing

import os
# Detection mode:
#   yolo_only -> YOLO + DeepSORT only  (fast, stable)
#   full      -> + RetinaFace + DeepFace + Action
DETECTION_MODE: str = os.environ.get("DETECTION_MODE", "yolo_only")  # Optimized for CPU DEMO speed

# ─── Tracking (DeepSORT) ───────────────────────────────────────────────────
DEEPSORT_MAX_AGE:  int = 80             # High-Stability: Prevents ID flipping during occlusion
DEEPSORT_N_INIT:   int = 2              # Rapid confirmation
DEEPSORT_MAX_IOU:  float = 0.7          
DEEPSORT_EMBEDDER: str = "mobilenet"

# ─── Emotion Detection (DeepFace) ──────────────────────────────────────────
# DeepFace runs in its OWN thread (non-blocking).
# skip_frames = analyse emotion every N tracker updates per track.
EMOTION_SKIP_FRAMES: int = 8
EMOTION_BACKEND:     str = "opencv"
EMOTION_MIN_FACE_PX: int = 40           # skip if face crop is smaller than this

# ─── Action Recognition (SlowFast) ─────────────────────────────────────────
ENABLE_ACTION:      bool = True           # SlowFast runs in background thread (non-blocking)
ACTION_BUFFER_SIZE: int  = 16
ACTION_SLOW_STRIDE: int  = 8
ACTION_DEVICE:      str  = "cpu"

# ─── Threading ─────────────────────────────────────────────────────────────
AI_THREAD_FRAME_SKIP: int = 3            # Optimized for 640px on CPU

# Result age: match the AI thread speed (Clean-Surveillance Profile)
RESULT_MAX_AGE_SEC: float = 1.2

# ─── Counting (Tripwire) ──────────────────────────────────────────────────
ENABLE_COUNTING: bool = True
# Tripwire position (0.0 to 1.0 of frame height/width)
LINE_POSITION:  float = 0.5   
LINE_DIRECTION:   str = "horizontal" # "horizontal" or "vertical" 
COLOR_ENTRY:    tuple = (0, 255, 0)   # Green
COLOR_EXIT:     tuple = (0, 0, 255)   # Red

# ─── Display ───────────────────────────────────────────────────────────────
SHOW_MOTION_MASK: bool = False
WINDOW_NAME:      str  = "AI CCTV Surveillance"
SHOW_TRIPWIRE:    bool = False

# ─── Snapshots (Security Logs) ─────────────────────────────────────────────
ENABLE_SNAPSHOTS: bool = True
SNAPSHOT_DIR:     str  = "snapshots"
