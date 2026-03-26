# AI Personnel Management & Surveillance System

A professional-grade, real-time AI monitoring and attendance system. It utilizes YOLOv8 for person detection, DeepFace for high-precision recognition, and SlowFast for action analysis, providing a complete security and personnel tracking solution.

---

## :rocket: Key Features

### :date: AI Attendance & Identity Discovery
- **Automated Logging**: Automatically marks `IN` and `OUT` attendance for recognized staff.
- **Dynamic Identity Discovery**: Logs "Unknown" detections immediately and updates them automatically once a face is recognized.
- **Snapshot Verification**: Captures high-resolution snapshots for every entry and exit event.
- **Timezone Accuracy**: Standardized timestamping eliminates timezone offsets in the UI.

### :satellite: Intelligent Surveillance
- **Multi-Camera Support**: Connect via USB Webcams or Remote RTSP/API streams (Imou, Hikvision).
- **Behavior Analysis**: Detects actions like sitting, standing, sleeping, or running.
- **Real-time Notifications**: Sends instant match alerts and attendance logs to Telegram.

---

## :tools: Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **PostgreSQL** (Installed and running)
- **Visual Studio C++ Build Tools** (Required for some AI libraries)

---

## 🏗️ Professional Deployment (RECOMMENDED)

The best way to deliver this app to your clients is using our **Single-File Setup Installer**:

1. **Build**: Run the `installer.iss` script via **Inno Setup**.
2. **Setup**: The client runs `MediaDetector_Setup.exe`.
3. **Launch**: A desktop shortcut "Media Detector" is created. It automatically starts Postgres, the AI Backend, and opens the Dashboard.

*No technical knowledge or command prompt is required for the client.* 🚀

---

## :tools: Developer Quick Start

### 2. Installation
```powershell
# 1. Clone the repository
git clone <repository-url>
cd media

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/mediadetect
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
UPLOAD_FOLDER=static/uploads
```

---

## :vertical_traffic_light: Running the System

1. **Initialize Database**: The system creates all required tables on the first run. Ensure PostgreSQL is accessible via the `DATABASE_URL`.
2. **Launch Application**:
   ```powershell
   python run.py
   ```
3. **Access Dashboard**: Open `http://localhost:5000` in your browser.
   - **Default Admin**: `admin@aerix.com` / `AerixNova@2025`

---

## :open_file_folder: Project Structure
- `app/api/`: REST endpoints for dashboard, cameras, and attendance.
- `app/pipelines/`: The core `AIPipeline` logic for real-time processing.
- `app/services/`: AI model wrappers (Face, Action, Object Detection).
- `app/db/`: Database session management and schema initialization.
- `templates/`: Modern, responsive HTML5 dashboard.
- `static/uploads/`: Storage for staff face profiles and attendance snapshots.

---

## :wrench: Staff & Attendance Setup
1. Go to **Member Directory** in the Sidebar.
2. Create a profile and upload **5-10 clear face photos**.
3. Profiles are automatically reloaded by the AI every 2 minutes.
4. View the **Attendance** page for a live log of all identified and unidentified entries.

---

## :pencil: Maintenance
- **Logs**: Check `app_debug.log` for detailed system events.
- **Database**: Use `scripts/check_schema.py` to verify table integrity.
- **Performance**: Adjust `YOLO_SKIP_FRAMES` in `config.py` to optimize for lower-end hardware.
*Developed by Aerix Global Tech*
