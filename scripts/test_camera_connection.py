import cv2
import os
import time
import urllib.parse

def test_connection():
    print("--- CAMERA CONNECTION TEST (ENCODED) ---")
    print("This version AUTOMATICALLY handles special characters like @, #, etc.\n")
    
    # --- DEFAULT VALUES ---
    default_ip = "192.168.1.240"
    default_user = "testt"
    default_pass = "Admin123#"
    default_port = "554"
    default_path = "/cam/realmonitor?channel=1&subtype=0"
    # ----------------------

    ip = input(f"Enter Camera IP [{default_ip}]: ").strip() or default_ip
    user = input(f"Enter Username [{default_user}]: ").strip() or default_user
    password = input(f"Enter Password [{default_pass}]: ").strip() or default_pass
    port = input(f"Enter RTSP Port [{default_port}]: ").strip() or default_port
    path = input(f"Enter Stream Path [{default_path}]: ").strip() or default_path

    # Encoding the password automatically
    safe_pass = urllib.parse.quote(password, safe='')
    
    # Constructing the URL
    url = f"rtsp://{user}:{safe_pass}@{ip}:{port}{path}"
    
    print(f"\nConstructed URL: rtsp://{user}:*******@{ip}:{port}{path}")
    print("Connecting...")

    # Timeout settings
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|stimeout;5000000"
    
    start_time = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if cap.isOpened():
        ret, frame = cap.read()
        elapsed = time.time() - start_time
        if ret:
            print(f"\nSUCCESS! Camera connected and sent a frame in {elapsed:.2f}s.")
        else:
            print("\nPARTIAL SUCCESS: Linked, but no video frames received.")
    else:
        elapsed = time.time() - start_time
        print(f"\nFAILED: Could not open the stream (Time: {elapsed:.2f}s).")
    
    cap.release()

if __name__ == "__main__":
    test_connection()
