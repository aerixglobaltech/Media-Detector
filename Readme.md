# AI CCTV Surveillance System (v2.0)

A high-performance, real-time AI monitoring system that transforms standard CCTV feeds into intelligent surveillance. It detects people, tracks their movement, identifies complex actions (like fighting or sleeping), and analyzes emotions — all while maintaining a smooth, non-blocking video stream.

---

## 🚀 Key Features

### 1. Unified Access (Monolithic Architecture)
The project is built as a **monolithic system** with a standard, easy-to-manage folder structure:
- All core AI modules are grouped into the `/modules` directory.
- Utility scripts (diagnostics, seeding, setup) are contained in the `/scripts` directory.
- Data is stored in **PostgreSQL** (set `DATABASE_URL` in `.env`).

### 2. Dual-Source Monitoring
- **Webcam:** Instantly use your built-in or USB camera.
- **Remote CCTV Access:** Connect directly to **Imou Cloud cameras** or office **Hik-Connect** systems via authorized API gateways.

### 3. Intelligent Detection Modes
- **`yolo_only`**: Ultra-fast YOLOv8 person detection + DeepSORT tracking.
- **`action_only`** (Recommended): Person tracking + **Posture Logic** + **SlowFast Activity Recognition**.
- **`full`**: Everything above + high-precision RetinaFace detection & Emotion Analysis.

---

## 🛠 Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core Application Logic |
| **Object Detection** | YOLOv8 (Ultralytics) | Person & Object Proximity Analysis |
| **Action AI** | SlowFast (PyTorchVideo) | Complex behavior analysis (Kinetics-400) |
| **Database** | PostgreSQL | Storage for Users, Cameras, and Logs |
| **Backend** | Flask/Waitress | Multi-threaded Web UI for remote monitoring |

---

## 📦 Installation (Windows)

The system uses **PostgreSQL**. Ensure `DATABASE_URL` is set in `.env` (the installer will prompt for it).

1.  **Open PowerShell** in the project directory.
2.  **Run the Installer**:
    ```powershell
    .\install_windows.ps1
    ```
    *   This script will automate virtual environment creation, pip installation, and database seeding.

### Default Admin Credentials:
*   **Email:** `admin@growmax.com`
*   **Password:** `Admin@123`

---

## 🚦 How to Run

1.  **Activate Environment:**
    ```powershell
    .\venv\Scripts\activate
    ```
2.  **Launch Web UI:**
    ```powershell
    python run.py
    ```
3.  **Access Dashboard:** Open your browser to `http://localhost:5000`

---

## 📂 Project Structure
- `/modules`: Core AI engines (YOLO, Tracker, Action Detector, Imou Connector).
- `/scripts`: Utility scripts like `seed_admin.py` and diagnostic tools.
- `/templates` & `/static`: Frontend assets for the Web Monitoring Dashboard.
- `app.py`: Main entry point for the Multi-Camera Web System.
- `main.py`: Legacy CLI entry point for local debugging.
- `.env`: Stores `DATABASE_URL` for PostgreSQL connection.

---

## 📝 Configuration (`config.py`)
Tweak the AI sensitivity by globally modifying `config.py`:
- `DETECTION_MODE`: Switch between `"action_only"`, `"yolo_only"`, or `"full"`.
- `MOTION_COOLDOWN_SEC`: Cooldown window before stopping AI processing.
- `YOLO_CONF`: Trust threshold for detection (0.0 to 1.0).
