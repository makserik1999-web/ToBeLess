# face_blur.py
import cv2
import numpy as np

def blur_faces(frame_bgr, face_recognizer, min_conf=0.25, blur_strength=25, expand=0.1):
    """
    Detect faces using face_recognizer.detect_faces and blur them in-place.
    - expand: fraction to expand bbox (0.1 -> enlarge by 10% each side)
    - blur_strength: kernel size (odd)
    Returns (out_frame, boxes) where boxes list is [(x1,y1,x2,y2,conf,name,score), ...]
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    boxes = face_recognizer.detect_faces(out, min_conf=min_conf)
    res = []
    for (x1,y1,x2,y2,conf) in boxes:
        # expand
        dw = int((x2-x1)*expand); dh = int((y2-y1)*expand)
        ax1 = max(0, x1-dw); ay1 = max(0, y1-dh); ax2 = min(w-1, x2+dw); ay2 = min(h-1, y2+dh)
        crop = out[ay1:ay2, ax1:ax2]
        # blur using gaussian with odd kernel
        k = max(1, int(blur_strength)|1)
        try:
            blurred = cv2.GaussianBlur(crop, (k,k), 0)
        except:
            blurred = cv2.blur(crop, (5,5))
        out[ay1:ay2, ax1:ax2] = blurred

        # attempt to identify name
        try:
            name, score = face_recognizer.identify(crop)
        except Exception:
            name, score = "Unknown", None
        res.append((ax1,ay1,ax2,ay2,conf,name,score))
    return out, res
