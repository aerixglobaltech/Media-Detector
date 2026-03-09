# AI CCTV Surveillance System (v2.0)

A high-performance, real-time AI monitoring system that transforms standard CCTV feeds into intelligent surveillance. It detects people, tracks their movement, identifies complex actions (like fighting or sleeping), and analyzes emotions — all while maintaining a smooth, non-blocking video stream.

---

## 🚀 Key Features

### 1. Dual-Source Input
- **Webcam:** Instantly use your built-in or USB camera.
- **Imou Cloud Integration:** Connect directly to your **Imou CCTV cameras** via the cloud. Interactive selection menu at startup with real-time camera status (Online/Offline).

### 2. Intelligent Detection Modes
- **`yolo_only`**: Ultra-fast YOLOv8 person detection + DeepSORT tracking. (Best for weak CPUs).
- **`action_only`** (Recommended): Person tracking + **Posture** (Sleeping/Sitting/Standing) + **Advanced Actions** (Running/Fighting/Falling) + **Emotions**.
- **`full`**: Everything above + High-precision RetinaFace detection.

### 3. Advanced Action Logic
- **Posture Detection (Instant):** Uses bounding-box geometry to instantly detect if someone is **sleeping (lying down)**, **sitting**, or **standing**.
- **Activity Recognition:** Uses the **SlowFast-R50** model in a background thread to identify complex behaviors like fighting, stumbling, or playing instruments.
- **Instant "Eating" Detection:** Uses YOLO to detect food items (bowl, spoon, pizza, etc.) near a person for immediate "Eating" alerts.

### 4. Non-Blocking Performance
- **Asynchronous Pipeline:** All heavy AI models (DeepFace, SlowFast) run in their own background threads. The main video feed never freezes or stutters.
- **CPU Optimized:** AI frames are dynamically resized (320x192) and processed with skip-frame logic to achieve 30+ FPS even without a GPU.

---

## 🛠 Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core Application Logic |
| **CV Engine** | OpenCV | Video Capture & Stream Rendering |
| **Object Detection** | YOLOv8 (Ultralytics) | Person & Food Item Detection |
| **Tracking** | DeepSORT | Stable ID Persistence across frames |
| **Action AI** | SlowFast (PyTorchVideo) | Complex movement analysis (Kinetics-400) |
| **Emotion AI** | DeepFace | Facial expression analysis |
| **CCTV Cloud** | Imou Open API | Imou Camera Stream Integration |

---

## 📦 Installation & Setup

### 1. Fast Install (Windows)
Run the provided automated script to set up everything:
```powershell
.\install.bat
```

### 2. Manual Setup
```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)

# Install core requirements
pip install -r requirements.txt
```

---

## 🚦 How to Run

1. **Activate Environment:**
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Start the System:**
   ```powershell
   python main.py
   ```
3. **Select Source:**
   - Type `1` for Webcam.
   - Type `2` for Imou CCTV (requires cloud credentials in code).

---

## 📝 Configuration (`config.py`)

You can tune the system performance by editing `config.py`:
- `DETECTION_MODE`: Switch between `"action_only"`, `"yolo_only"`, or `"full"`.
- `MOTION_COOLDOWN_SEC`: How long to keep AI running after person leaves (default 30s).
- `YOLO_SKIP_FRAMES`: Process every Nth frame for YOLO (default 1 for stability).
- `AI_FRAME_WIDTH`: Resolution for AI models (lower = faster).

---

## 🛡 Surveillance Labels
The system recognizes and displays:
- **Environment:** `🧍 standing`, `🪑 sitting`, `😴 sleeping / lying down`.
- **Activities:** `🍽 eating`, `💻 working`, `📖 reading`, `✍ writing`, `💬 talking`.
- **Alerts:** `⚠ fighting`, `🏃 running`, `⚠ falling`.
- **Emotions:** `happy`, `neutral`, `angry`, `surprise`, `fear`, `sad`.

---

## 🚧 Folder Structure
- `/modules`: Contains individual AI engines (YOLO, Tracker, Action, Emotion).
- `/utils`: Helper classes for FPS, drawing, and video streaming.
- `main.py`: The central pipeline and interactive UI.
- `imou_connector.py`: The cloud communication layer.
