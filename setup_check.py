"""
setup_check.py
─────────────────────────────────────────────────────────────────────────────
Quick dependency checker.  Run this BEFORE main.py to verify every required
package is importable and print the detected device (CPU / CUDA).

Usage:
    python setup_check.py
"""

import sys
import importlib

CHECKS = [
    ("cv2",                      "OpenCV",           "opencv-python"),
    ("numpy",                    "NumPy",            "numpy"),
    ("torch",                    "PyTorch",          "torch"),
    ("torchvision",              "TorchVision",      "torchvision"),
    ("ultralytics",              "Ultralytics YOLO", "ultralytics"),
    ("deep_sort_realtime",       "DeepSORT RT",      "deep-sort-realtime"),
    ("deepface",                 "DeepFace",         "deepface"),
    ("retinaface",               "RetinaFace",       "retina-face"),
    ("pytorchvideo",             "PyTorchVideo",     "pytorchvideo"),
    ("scipy",                    "SciPy",            "scipy"),
    ("sklearn",                  "scikit-learn",     "scikit-learn"),
    ("filterpy",                 "filterpy",         "filterpy"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

print("\n" + "=" * 55)
print("  AI CCTV Surveillance – Dependency Check")
print("=" * 55)

all_ok = True
for module, label, pip_name in CHECKS:
    try:
        importlib.import_module(module)
        print(f"  {GREEN}✓{RESET}  {label}")
    except ImportError:
        print(f"  {RED}✗{RESET}  {label}  →  pip install {pip_name}")
        all_ok = False

# PyTorch device
try:
    import torch
    if torch.cuda.is_available():
        dev = f"CUDA – {torch.cuda.get_device_name(0)}"
    else:
        dev = "CPU only"
    print(f"\n  Device : {YELLOW}{dev}{RESET}")
except Exception:
    pass

print("=" * 55)
if all_ok:
    print(f"\n  {GREEN}All dependencies satisfied.  Ready to run app.py{RESET}\n")
else:
    print(f"\n  {RED}Some packages are missing.  See above.{RESET}\n")
    sys.exit(1)
