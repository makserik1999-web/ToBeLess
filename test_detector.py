from app import FightDetector
import cv2, sys

img_path = 'faces/images/Abil_1.png'
img = cv2.imread(img_path)
if img is None:
    print('Failed to load', img_path)
    sys.exit(1)

try:
    det = FightDetector()
except Exception as e:
    print('FightDetector init failed:', e)
    sys.exit(1)

try:
    out, metrics = det.process_frame(img, frame_count=1)
    print('Process done. Metrics:', metrics)
    cv2.imwrite('debug_out.jpg', out)
    print('Wrote debug_out.jpg')
except Exception as e:
    print('process_frame failed:', e)
    import traceback; traceback.print_exc()
