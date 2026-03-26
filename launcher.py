# launcher.py
# -----------------------------------------------------------------------------
# MISSION CONTROL - APP LAUNCHER
# This script starts the Database and Backend, then opens the Browser.
# -----------------------------------------------------------------------------

import os
import sys
import subprocess
import time
import webbrowser
import socket

# Windows-only flag for hiding console windows
CREATE_NO_WINDOW = 0x08000000

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    # Use absolute project root
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    # Ensuring necessary Folders exist (since we exclude them from the installer)
    os.makedirs(os.path.join(root, "static", "uploads", "movement"), exist_ok=True)

    # 1. Start Portable PostgreSQL if needed
    pg_bin_dir = os.path.join(root, "package", "postgres", "bin")
    pg_ctl = os.path.join(pg_bin_dir, "pg_ctl.exe")
    pg_init = os.path.join(pg_bin_dir, "initdb.exe")
    pg_data = os.path.join(root, "package", "postgres", "data")
    
    if os.path.exists(pg_ctl):
        # Auto-Initialize if data folder is missing
        if not os.path.exists(pg_data) or not os.listdir(pg_data):
            print(">>> Initializing Fresh Database...")
            subprocess.run([pg_init, "-D", pg_data, "--encoding=UTF8"], 
                           creationflags=CREATE_NO_WINDOW)
        
        print(">>> Starting Database...")
        subprocess.Popen([pg_ctl, "start", "-D", pg_data], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=CREATE_NO_WINDOW)
    
    # 2. Start Backend using Portable Python
    py_exe = os.path.join(root, "package", "python", "python.exe")
    if not os.path.exists(py_exe):
        print("ERROR: Portable Python not found!")
        input("Press Enter to exit...")
        return

    print(">>> Starting Backend...")
    # Add project root to PYTHONPATH for the child process
    env = os.environ.copy()
    env["PYTHONPATH"] = root
    
    # REDIRECT AI MODELS to the app's local folder (Resolves Inno Setup Warnings)
    env["DEEPFACE_HOME"] = os.path.join(root, "ai_models")
    env["TORCH_HOME"]    = os.path.join(root, "ai_models")
    
    backend = subprocess.Popen([py_exe, "run.py"], 
                               env=env,
                               creationflags=CREATE_NO_WINDOW)

    # 3. Wait for the server to be ready (Port 5000)
    print(">>> Waiting for Mission Control to start...")
    max_retries = 60 # Increased timeout for heavy model loading
    for i in range(max_retries):
        # ACTIVE MONITORING: Stop if the process died
        if backend.poll() is not None:
            print("\n!!! ERROR: Application failed to start (Process exited).")
            # Try to show the crash log if it exists
            if os.path.exists("CRASH.log"):
                with open("CRASH.log", "r") as f:
                    print("-" * 40)
                    print(f.read())
                    print("-" * 40)
            else:
                print("Check 'app_debug.log' for details.")
            input("Press Enter to exit...")
            return

        if is_port_open(5000):
            print("\n>>> System Ready! Opening Browser...")
            webbrowser.open("http://localhost:5000")
            break
        
        # Simple progress indicator
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(1)
    else:
        print("\nERROR: Server took too long to start.")
        input("Press Enter to exit...")

    # Keep the launcher alive as long as the backend is running
    backend.wait()

if __name__ == "__main__":
    main()
