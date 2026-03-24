; MISSION CONTROL - Inno Setup Script
; -----------------------------------------------------------------------------
; USE THIS FILE WITH "INNO SETUP" (FREE) TO BUILD THE FINAL SETUP.EXE
; -----------------------------------------------------------------------------

[Setup]
AppId={{C7A9D8E6-F1A2-4B3C-9D8E-6F5A4B3C2D1E}}
AppName=Media Detector - Mission Control
AppVersion=1.0
AppPublisher=Aerix Global Tech
DefaultDirName={autopf}\MediaDetector
DefaultGroupName=MediaDetector
DisableDirPage=no
DisableProgramGroupPage=no
OutputDir=C:\Users\testi\CV\dist
OutputBaseFilename=MediaDetector_Setup
SetupIconFile="{#SourcePath}\static\favicon.ico"
; WizardImageFile and WizardSmallImageFile removed because they must be BMP format.
; Enabeling default high-quality wizard visuals instead.
Compression=lzma
SolidCompression=yes
WizardStyle=modern
LicenseFile=C:\Users\testi\CV\LICENSE.txt
PrivilegesRequired=admin

[Types]
Name: "full"; Description: "Full installation (Recommended)"
Name: "compact"; Description: "Compact installation (No AI Models)"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "main"; Description: "Main Application Files"; Types: full compact custom; Flags: fixed
Name: "models"; Description: "Optional AI Models"; Types: full custom
Name: "models\yolo"; Description: "Object Detection (YOLOv8 - Essential)"; Types: full custom
Name: "models\facerec"; Description: "Face Recognition (DeepFace/Facenet)"; Types: full custom
Name: "models\emotion"; Description: "Emotion AI (DeepFace Lite)"; Types: full custom
Name: "models\action"; Description: "Action Recognition (SlowFast - 277MB)"; Types: full custom
Name: "postgres"; Description: "Portable PostgreSQL Database"; Types: full custom

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 1. Main Application (REQUIRED)
Source: "C:\Users\testi\CV\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: main; \
    Excludes: ".git*, venv\*, .idea\*, *.pyc, __pycache__*, static\uploads\**\*, *.log, *.db, yolov8n.pt, package\postgres\*"

; 2. AI Models (OPTIONAL)
; 2a. Object Detection
Source: "C:\Users\testi\CV\yolov8n.pt"; DestDir: "{app}"; Flags: ignoreversion; Components: models\yolo

; 2b. Face Recognition (DeepFace Weights)
Source: "C:\Users\testi\.deepface\weights\facenet512_weights.h5"; DestDir: "{app}\ai_models\.deepface\weights"; Flags: ignoreversion; Components: models\facerec
Source: "C:\Users\testi\.deepface\weights\retinaface.h5"; DestDir: "{app}\ai_models\.deepface\weights"; Flags: ignoreversion; Components: models\facerec

; 2c. Emotion AI (DeepFace Lite)
Source: "C:\Users\testi\.deepface\weights\facial_expression_model_weights.h5"; DestDir: "{app}\ai_models\.deepface\weights"; Flags: ignoreversion; Components: models\emotion

; 2d. Action Recognition (SlowFast Hub Cache)
Source: "C:\Users\testi\.cache\torch\hub\checkpoints\SLOWFAST_8x8_R50.pyth"; DestDir: "{app}\ai_models\hub\checkpoints"; Flags: ignoreversion; Components: models\action

; 3. Portable PostgreSQL Database (OPTIONAL)
Source: "C:\Users\testi\CV\package\postgres\*"; DestDir: "{app}\package\postgres"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: postgres

[Icons]
Name: "{group}\Media Detector"; Filename: "{app}\package\python\pythonw.exe"; Parameters: """{app}\launcher.py"""
Name: "{autodesktop}\Media Detector"; Filename: "{app}\package\python\pythonw.exe"; Parameters: """{app}\launcher.py"""; Tasks: desktopicon

[Run]
Filename: "{app}\package\python\pythonw.exe"; Parameters: """{app}\launcher.py"""; Description: "{cm:LaunchProgram,Media Detector}"; Flags: nowait postinstall skipifsilent
