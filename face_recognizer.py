# face_recognizer.py
import json
import time
from pathlib import Path
import cv2
import numpy as np

# optional YOLO
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class FaceRecognizer:
    """
    Simple robust face recognizer WITHOUT heavy deps.
    Strategy:
      - Detect faces: YOLO (if available) -> cv2.dnn (if model provided) -> Haar cascade
      - Embedding: grayscale 112x112 + CLAHE -> flatten -> L2-normalize
      - Template matching: cosine distance
    Usage:
      fr = FaceRecognizer(yolo_model_path="yolov8n-face.pt", db_path="faces/embeddings.json", debug=True)
      fr.bulk_register_from_folder("faces/images")
      boxes = fr.detect_faces(frame)
      name,score = fr.identify(crop)
    """
    def __init__(self, yolo_model_path="yolov8n-face.pt", db_path="faces/embeddings.json", dnn_proto=None, dnn_model=None, debug=False, vec_size=(128,128), threshold=0.55):
        self.debug = debug
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vec_w, self.vec_h = vec_size
        self.threshold = float(threshold)

        # load YOLO face if available and model exists
        self.yolo = None
        if YOLO is not None and yolo_model_path:
            try:
                p = Path(yolo_model_path)
                if p.exists():
                    self.yolo = YOLO(str(p))
                    if self.debug: print(f"[FaceRecognizer] ✓ YOLO loaded from {p}")
                else:
                    if self.debug: print(f"[FaceRecognizer] YOLO model not found at {p}")
            except Exception as e:
                if self.debug: print("[FaceRecognizer] YOLO load error:", e)
                self.yolo = None

        # optional cv2.dnn (Caffe SSD)
        self.dnn_net = None
        if dnn_proto and dnn_model:
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
                if self.debug: print("[FaceRecognizer] ✓ DNN face detector loaded")
            except Exception as e:
                if self.debug: print("[FaceRecognizer] DNN load failed:", e)
                self.dnn_net = None

        # Haar fallback
        self.haar = None
        try:
            haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if haar_path.exists():
                self.haar = cv2.CascadeClassifier(str(haar_path))
                if self.haar.empty():
                    self.haar = None
                else:
                    if self.debug: print(f"[FaceRecognizer] ✓ Haar loaded from {haar_path}")
            else:
                # try local copy in project
                local = Path(__file__).parent / "haarcascade_frontalface_default.xml"
                if local.exists():
                    self.haar = cv2.CascadeClassifier(str(local))
                    if not self.haar.empty() and self.debug:
                        print(f"[FaceRecognizer] ✓ Haar loaded from local {local}")
        except Exception as e:
            if self.debug: print("[FaceRecognizer] Haar load error:", e)
            self.haar = None

        # memory DB: {name: [vecs]}
        self._mem_db = {}
        self._load_db()
        if self.debug:
            print(f"[FaceRecognizer] Initialization complete. Persons in DB: {len(self._mem_db)}")

    # ---------------- DB load/save ----------------
    def _load_db(self):
        self._mem_db = {}
        if self.db_path.exists():
            try:
                raw = json.loads(self.db_path.read_text(encoding="utf-8"))
                for name, vecs in raw.items():
                    arrs = []
                    for v in vecs:
                        a = np.array(v, dtype=np.float32)
                        n = np.linalg.norm(a)
                        if n > 1e-6:
                            arrs.append(a / n)
                    if arrs:
                        self._mem_db[name] = arrs
            except Exception as e:
                if self.debug: print("[FaceRecognizer] DB read error:", e)
                self._mem_db = {}

    def save_db(self):
        dump = {n: [v.tolist() for v in arr] for n, arr in self._mem_db.items()}
        try:
            self.db_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.debug: print(f"[FaceRecognizer] ✓ Saved DB ({len(dump)} persons)")
        except Exception as e:
            if self.debug: print("[FaceRecognizer] Save DB failed:", e)

    # ---------------- Embedding ----------------
    def _check_face_quality(self, img_bgr):
        """Check if face image quality is good enough for recognition"""
        if img_bgr is None or img_bgr.size == 0:
            return False
        h, w = img_bgr.shape[:2]
        # Minimum resolution check
        if w < 32 or h < 32:
            return False
        # Check if image is too blurry using Laplacian variance
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Too blurry
            if self.debug: print(f"[FaceRecognizer] Image too blurry: {laplacian_var:.2f}")
            return False
        return True

    def _img_to_vector(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return None
        # Quality check
        if not self._check_face_quality(img_bgr):
            return None
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (self.vec_w, self.vec_h), interpolation=cv2.INTER_CUBIC)
            # Enhanced preprocessing
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            proc = clahe.apply(resized)
            # Bilateral filter for edge-preserving smoothing
            proc = cv2.bilateralFilter(proc, 5, 50, 50)
            # Normalize histogram
            proc = cv2.equalizeHist(proc)
            v = proc.astype(np.float32).reshape(-1)
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                return None
            return v / norm
        except Exception as e:
            if self.debug: print("[FaceRecognizer] _img_to_vector error:", e)
            return None

    # ---------------- Detection ----------------
    def detect_faces(self, frame_bgr, min_conf=0.5):
        """
        Returns list of (x1,y1,x2,y2,conf)
        """
        h, w = frame_bgr.shape[:2]
        boxes = []

        # YOLO
        if self.yolo is not None:
            try:
                outs = self.yolo(frame_bgr, conf=min_conf, verbose=False, imgsz=640)
                for out in outs:
                    if getattr(out, "boxes", None) is None:
                        continue
                    for b in out.boxes:
                        xyxy = b.xyxy[0].cpu().numpy()
                        x1,y1,x2,y2 = [int(x) for x in xyxy]
                        conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 1.0
                        x1,y1 = max(0,x1), max(0,y1)
                        x2,y2 = min(w-1,x2), min(h-1,y2)
                        # Stricter minimum size
                        if (x2-x1) >= 40 and (y2-y1) >= 40:
                            boxes.append((x1,y1,x2,y2,conf))
                if boxes:
                    boxes = self._apply_nms(boxes, iou_threshold=0.3)
                    if self.debug: print(f"[FaceRecognizer] YOLO detected {len(boxes)} faces")
                    return boxes
            except Exception as e:
                if self.debug: print("[FaceRecognizer] YOLO detect error:", e)
                boxes = []

        # cv2.dnn
        if self.dnn_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300,300), (104.0,177.0,123.0))
                self.dnn_net.setInput(blob)
                dets = self.dnn_net.forward()
                for i in range(0, dets.shape[2]):
                    conf = float(dets[0,0,i,2])
                    if conf < min_conf: continue
                    box = dets[0,0,i,3:7] * np.array([w,h,w,h])
                    x1,y1,x2,y2 = box.astype("int")
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(w-1,x2), min(h-1,y2)
                    if (x2-x1) >= 16 and (y2-y1) >= 16:
                        boxes.append((x1,y1,x2,y2,conf))
                if boxes:
                    boxes = self._apply_nms(boxes, 0.4)
                    if self.debug: print(f"[FaceRecognizer] DNN detected {len(boxes)} faces")
                    return boxes
            except Exception as e:
                if self.debug: print("[FaceRecognizer] DNN detect error:", e)
                boxes = []

        # Haar (improved parameters)
        if self.haar is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                # Equalize histogram for better detection
                gray = cv2.equalizeHist(gray)
                # Better parameters: higher scaleFactor, more neighbors, larger minSize
                dets = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50,50), flags=cv2.CASCADE_SCALE_IMAGE)
                for (x,y,ww,hh) in dets:
                    boxes.append((int(x),int(y),int(x+ww),int(y+hh),0.8))
                if boxes:
                    boxes = self._apply_nms(boxes, iou_threshold=0.3)
                    if self.debug: print(f"[FaceRecognizer] Haar detected {len(boxes)} faces")
                return boxes
            except Exception as e:
                if self.debug: print("[FaceRecognizer] Haar detect error:", e)
        return []

    # ---------------- Matching ----------------
    def identify(self, crop_bgr):
        """
        Input: cropped face BGR
        Returns: (name, score) where score is cosine distance (0..2) -> smaller is better (we use 1-dot)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", None
        v = self._img_to_vector(crop_bgr)
        if v is None:
            return "Unknown", None

        best_name = None
        best_score = float("inf")
        # Use average distance across all templates for more robust matching
        name_scores = {}
        for name, refs in self._mem_db.items():
            scores = []
            for ref in refs:
                if len(ref) != len(v):
                    continue
                dot = float(np.dot(v, ref))
                dot = max(-1.0, min(1.0, dot))
                dist = 1.0 - dot
                scores.append(dist)
            if scores:
                # Use minimum distance (best match) from all templates
                avg_score = min(scores)
                name_scores[name] = avg_score
                if avg_score < best_score:
                    best_score = avg_score
                    best_name = name

        if best_name is not None and best_score <= self.threshold:
            if self.debug: print(f"[FaceRecognizer] identify -> {best_name} (score={best_score:.4f})")
            return best_name, float(best_score)
        if self.debug:
            if best_name is not None:
                print(f"[FaceRecognizer] identify -> Unknown (best {best_name} score={best_score:.4f}, threshold={self.threshold})")
            else:
                print("[FaceRecognizer] identify -> Unknown (no refs)")
        return "Unknown", (best_score if best_score != float("inf") else None)

    # ---------------- Registration / helpers ----------------
    def register_face(self, name, frame_bgr, use_best_box=True, persist_image=True):
        if frame_bgr is None or frame_bgr.size == 0:
            return False, "invalid image"
        boxes = self.detect_faces(frame_bgr)
        if not boxes:
            return False, "no face found"
        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True) if use_best_box else boxes
        x1,y1,x2,y2,_ = boxes[0]
        crop = frame_bgr[y1:y2, x1:x2]
        v = self._img_to_vector(crop)
        if v is None:
            return False, "bad quality"
        self._mem_db.setdefault(name, []).append(v)
        self.save_db()
        if persist_image:
            try:
                p = Path("faces/images"); p.mkdir(parents=True, exist_ok=True)
                fname = p / f"{name}_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), crop)
            except Exception:
                pass
        return True, "ok"

    def bulk_register_from_folder(self, folder="faces/images"):
        p = Path(folder)
        if not p.exists():
            if self.debug: print("[FaceRecognizer] bulk_register: folder not found", folder)
            return
        files = list(p.glob("*.*"))
        if self.debug: print(f"[FaceRecognizer] bulk_register: found {len(files)} files")
        for f in files:
            name = f.stem.split("_")[0]
            img = cv2.imread(str(f))
            if img is None:
                if self.debug: print(" - failed load", f); continue
            ok,msg = self.register_face(name, img, persist_image=False)
            if self.debug: print(f" - {f.name} -> {name}: {ok}/{msg}")

    def list_persons(self):
        return {n: len(v) for n, v in self._mem_db.items()}

    # ---------------- utils ----------------
    def _compute_iou(self, a,b):
        ax1,ay1,ax2,ay2 = a[:4]; bx1,by1,bx2,by2 = b[:4]
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        aarea = (ax2-ax1)*(ay2-ay1); barea = (bx2-bx1)*(by2-by1)
        union = aarea + barea - inter
        return inter/union if union>0 else 0.0

    def _apply_nms(self, boxes, iou_threshold=0.4):
        if not boxes: return []
        boxes = sorted(boxes, key=lambda x:x[4], reverse=True)
        keep = []
        for box in boxes:
            keep_flag = True
            for k in keep:
                if self._compute_iou(box,k) > iou_threshold:
                    keep_flag = False; break
            if keep_flag: keep.append(box)
        return keep

    def debug_one_frame(self, frame):
        boxes = self.detect_faces(frame, min_conf=0.2)
        print("DEBUG boxes:", boxes)
        for i,(x1,y1,x2,y2,conf) in enumerate(boxes):
            crop = frame[y1:y2, x1:x2]
            name, score = self.identify(crop)
            print(f"DEBUG box{i}: conf={conf:.2f} -> {name}, {score}")
            cv2.imwrite(f"debug_crop_{i}.jpg", crop)
