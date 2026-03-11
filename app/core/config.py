"""
config.py  –  AI CCTV Surveillance System
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

# ─── Video Source ──────────────────────────────────────────────────────────
SOURCE: int | str = 0   # 0 = webcam; overridden at startup if Imou selected
FRAME_WIDTH:  int   = 1280
FRAME_HEIGHT: int   = 720
TARGET_FPS:   float = 30.0

# Smaller resolution fed into AI models (faster inference)
AI_FRAME_WIDTH:  int = 640
AI_FRAME_HEIGHT: int = 360

# ─── Motion Detection ──────────────────────────────────────────────────────
MOTION_MIN_AREA:      int   = 800    # lower = catches subtle movement
MOTION_HISTORY:       int   = 300
MOTION_VAR_THRESHOLD: float = 12.0  # lower = more sensitive

# Once motion is detected, keep the AI pipeline running for this many seconds
# even if the person sits perfectly still (MOG2 would otherwise call it "background").
MOTION_COOLDOWN_SEC: float = 30.0

# ─── Person Detection (YOLOv8) ─────────────────────────────────────────────
import os
YOLO_MODEL:       str   = os.path.join("data", "models", "yolov8n.pt")
YOLO_CONF:        float = 0.30          # lower = detects people in tricky lighting
YOLO_DEVICE:      str   = "cpu"          # change to "cuda" if GPU PyTorch installed
YOLO_SKIP_FRAMES: int   = 1              # run YOLO every frame for max stability

import os
# Detection mode:
#   yolo_only -> YOLO + DeepSORT only  (fast, stable)
#   full      -> + RetinaFace + DeepFace + Action
DETECTION_MODE: str = os.environ.get("DETECTION_MODE", "action_only")  # person box always on + SlowFast action (no emotion)

# ─── Tracking (DeepSORT) ───────────────────────────────────────────────────
DEEPSORT_MAX_AGE:  int = 20              # frames before a lost track is deleted
DEEPSORT_N_INIT:   int = 2
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
AI_THREAD_FRAME_SKIP: int = 2

# Max age (seconds) of an AI result before boxes are hidden from the display.
# Must be LONGER than the worst-case AI pipeline time on CPU (~1-2 sec).
# Ghost boxes are removed because DeepSORT stops outputting old tracks
# after DEEPSORT_MAX_AGE frames — not by this timer alone.
RESULT_MAX_AGE_SEC: float = 3.0

# ─── Display ───────────────────────────────────────────────────────────────
SHOW_MOTION_MASK: bool = False
WINDOW_NAME:      str  = "AI CCTV Surveillance"
