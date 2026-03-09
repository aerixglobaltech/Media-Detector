@echo off
REM ─────────────────────────────────────────────────────────────
REM  install.bat  –  One-shot setup for AI CCTV Surveillance
REM  Run once from the project root in your activated venv.
REM ─────────────────────────────────────────────────────────────

echo.
echo  [1/6] Upgrading pip …
python -m pip install --upgrade pip

echo.
echo  [2/6] Installing core packages …
pip install opencv-python numpy scipy scikit-learn filterpy Pillow

echo.
echo  [3/6] Installing PyTorch (CPU) …
echo  TIP: For GPU support visit https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo  [4/6] Installing Ultralytics YOLOv8 …
pip install ultralytics

echo.
echo  [5/6] Installing DeepSORT, DeepFace, RetinaFace …
pip install deep-sort-realtime deepface retina-face tf-keras

echo.
echo  [6/6] Installing PyTorchVideo …
pip install pytorchvideo

echo.
echo  ─────────────────────────────────────────────────────────
echo  Installation complete.  Running setup_check.py …
echo  ─────────────────────────────────────────────────────────
python setup_check.py
