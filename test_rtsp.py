import cv2
import time
import sys

def test_stream(url, name):
    print(f"\n--- Testing {name} ---")
    print(f"URL: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"FAILED: Could not open {name}")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"SUCCESS: Captured frame from {name}! Size: {frame.shape}")
        cap.release()
        return True
    else:
        print(f"FAILED: Opened {name} but could not read a frame.")
        cap.release()
        return False

# EDIT THESE CREDENTIALS
USER = "maratik"
PASS = "admin12345"
IP = "192.168.0.83"

streams = [
    (f"rtsp://{USER}:{PASS}@{IP}:554/stream1", "HD Stream (Standard)"),
    (f"rtsp://{USER}:{PASS}@{IP}:554/stream2", "SD Stream (Low Res)"),
    (f"rtsp://{USER}:{PASS}@{IP}/stream1", "HD Stream (No Port)"),
]

print("Starting diagnostics...")
for url, name in streams:
    if test_stream(url, name):
        print("\n!!! SOLUTION FOUND !!!")
        print(f"Use this URL in ToBeLess: {url}")
        sys.exit(0)

print("\n--- ALL TESTS FAILED ---")
print("Possible causes:")
print("1. Camera Account logic (Username/Password) is still rejected by the camera.")
print("2. SD Card in camera might be blocking RTSP (Common in C210).")
print("3. ONVIF might be disabled in Advanced Settings.")
print("4. Windows Firewall is blocking Port 554.")
