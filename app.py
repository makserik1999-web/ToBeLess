#!/usr/bin/env python3
"""
Fight detection with YOLO-pose and automatic face recognition from faces/images folder
"""
import os, time, uuid, json, threading, traceback
from pathlib import Path
from collections import deque
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import cv2, numpy as np
import math

try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

try:
    from bot import send_alert, send_photo
except Exception:
    def send_alert(*a, **k): pass
    def send_photo(*a, **k): pass

# ------------ Config ------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
FACES_DIR = Path("faces/images")
FACES_DIR.mkdir(parents=True, exist_ok=True)

ALERT_COOLDOWN = 8
ANALYTICS_SNAPSHOT_SIZE = 300
SKIP_FRAMES = 1
RESIZE_WIDTH = 640
SSE_INTERVAL = 0.5

try:
    import torch
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------- Replace the old FaceRecognizer with this (put in place of the old class) ----------
class FaceRecognizer:
    """
    Lightweight face recognition:
    - Uses YOLOv8-face for face DETECTION (if available)
    - Uses simple grayscale-resize template vectors for IDENTIFICATION (no extra libs)
    - Templates are stored in faces/embeddings.json as lists of floats per person
    """
    def __init__(self, db_path="faces/embeddings.json", face_model_path="yolov8n-face.pt", device=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.face_model_path = face_model_path
        # optional insightface FaceAnalysis instance and DB for deep embeddings
        self.fa = None
        self._insight_db = {}  # name -> list of numpy embedding arrays
        try:
            if 'FaceAnalysis' in globals() and FaceAnalysis is not None:
                try:
                    # instantiate with defaults; prepare later if needed
                    self.fa = FaceAnalysis()
                    try:
                        # prefer GPU if available (ctx_id=0) else CPU (ctx_id=-1)
                        ctx_id = 0 if (device is None and ((getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()) or torch.cuda.is_available())) else -1
                        self.fa.prepare(ctx_id=ctx_id)
                    except Exception:
                        # non-fatal if prepare fails; FaceAnalysis can still work in many configs
                        pass
                    print(f"[FaceRecognizer] ✓ insightface FaceAnalysis initialized")
                except Exception as e:
                    self.fa = None
                    print(f"[FaceRecognizer] ✗ insightface init failed: {e}")
        except Exception:
            self.fa = None
        # Face tracking: temporal consistency to avoid confusing same person
        self.face_tracking = {}  # person_id -> {'name': name, 'conf': score, 'frames': count}
        self.tracking_counter = 0

        # load DB (name -> list of vectors)
        if self.db_path.exists():
            try:
                self.db = json.loads(self.db_path.read_text(encoding='utf-8'))
                print(f"[FaceRecognizer] Loaded {len(self.db)} persons from DB")
            except Exception as e:
                print(f"[FaceRecognizer] Failed to load DB: {e}")
                self.db = {}
        else:
            self.db = {}

        # optional yolov8 face detector
        self.yolo_face = None
        try:
            if 'YOLO' in globals() and YOLO is not None and face_model_path and Path(face_model_path).exists():
                try:
                    self.yolo_face = YOLO(face_model_path)
                    print(f"[FaceRecognizer] ✓ YOLO face model loaded from {face_model_path}")
                except Exception as e:
                    print(f"[FaceRecognizer] ✗ Failed to load yolo face model: {e}")
            else:
                if face_model_path and not Path(face_model_path).exists():
                    print(f"[FaceRecognizer] ⚠ YOLO face model not found: {face_model_path}")
        except Exception as e:
            print(f"[FaceRecognizer] ✗ YOLO init error: {e}")

        # params for template vectors
        self.vec_w = 128
        self.vec_h = 128
        # identification threshold (L2 / cosine -> we use normalized vectors and cosine distance)
        self.identify_threshold = 0.45  # smaller = stricter; tune if needed

        # keep names -> list of numpy arrays in memory for speed
        self._mem_db = {}
        self._load_mem_db()

    def _img_to_vector(self, img_bgr):
        """Convert BGR face crop to normalized float vector (unit length)."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
        # Resize, equalize, convert float
        try:
            resized = cv2.resize(gray, (self.vec_w, self.vec_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        # equalize to reduce lighting effect
        resized = cv2.equalizeHist(resized)
        v = resized.astype(np.float32).reshape(-1)
        # normalize to unit vector (avoid zero)
        norm = np.linalg.norm(v)
        if norm <= 1e-6:
            return None
        v = v / norm
        return v

    def _load_mem_db(self):
        """Load self.db (lists) into numpy arrays for quick matching."""
        self._mem_db = {}
        if not self.db:
            return
        for name, lists in self.db.items():
            arrs = []
            for vec in lists:
                try:
                    a = np.array(vec, dtype=np.float32)
                    # ensure normalized
                    n = np.linalg.norm(a)
                    if n > 1e-6:
                        a = a / n
                        arrs.append(a)
                except Exception:
                    pass
            if arrs:
                self._mem_db[name] = arrs
        print(f"[FaceRecognizer] Memory DB ready: {len(self._mem_db)} persons")

    def save_db(self):
        try:
            # convert numpy to lists
            dump = {}
            for name, arrs in self._mem_db.items():
                dump[name] = [a.tolist() for a in arrs]
            self.db_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding='utf-8')
            # reload plain db as well
            self.db = dump
            print(f"[FaceRecognizer] ✓ Saved {len(dump)} persons to DB")
        except Exception as e:
            print("[FaceRecognizer] ✗ save_db failed:", e)

    def register_face(self, name, img_bgr, persist_image=True):
        """
        Register face crop (BGR numpy) under `name`.
        If persist_image=True, also save image into faces/images/ as timestamped file.
        """
        if img_bgr is None or img_bgr.size == 0:
            return False, "invalid image"
        v = self._img_to_vector(img_bgr)
        if v is None:
            return False, "could not process image"
        self._mem_db.setdefault(name, []).append(v)
        # if insightface is available, compute and store its embedding as well
        if self.fa is not None:
            try:
                faces = self.fa.get(img_bgr)
                if faces and len(faces) > 0 and hasattr(faces[0], 'embedding'):
                    emb = np.array(faces[0].embedding, dtype=np.float32)
                    n = np.linalg.norm(emb)
                    if n > 1e-6:
                        emb = emb / n
                        self._insight_db.setdefault(name, []).append(emb)
            except Exception:
                pass
        # also update self.db for future saving
        self.save_db()
        # optionally save raw image into faces/images
        try:
            if persist_image:
                Path("faces/images").mkdir(parents=True, exist_ok=True)
                fname = Path("faces/images") / f"{name}_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), img_bgr)
        except Exception:
            pass
        print(f"[FaceRecognizer] ✓ Registered face (name={name})")
        return True, "registered"

    def _best_match(self, vec):
        """Return (best_name, best_score) where score is cosine distance (0 = identical).
        
        Enhanced matching with:
        - Multiple reference vector comparison
        - Weighted confidence based on count of templates
        - L2 distance as fallback metric
        """
        best_name = None
        best_score = float('inf')
        if vec is None:
            return None, None
        
        for name, arrs in self._mem_db.items():
            name_scores = []
            for ref in arrs:
                # Cosine distance (normalized vectors)
                dot = float(np.dot(vec, ref))
                dot = max(-1.0, min(1.0, dot))
                cosine_dist = 1.0 - dot
                name_scores.append(cosine_dist)
            
            if name_scores:
                # Use minimum score (best match) weighted by consistency
                avg_score = np.mean(name_scores)
                min_score = np.min(name_scores)
                std_score = np.std(name_scores) if len(name_scores) > 1 else 0
                
                # Penalize high variance (inconsistent templates)
                weighted_score = min_score + (std_score * 0.05)
                
                if weighted_score < best_score:
                    best_score = weighted_score
                    best_name = name
        
        return best_name, best_score

    def _best_match_insight(self, emb):
        """Compare an insightface embedding against the insight DB (cosine distance).
        Return (best_name, best_score) or (None, None).
        """
        if emb is None:
            return None, None
        best_name = None
        best_score = float('inf')
        for name, arrs in self._insight_db.items():
            for ref in arrs:
                dot = float(np.dot(emb, ref))
                dot = max(-1.0, min(1.0, dot))
                cosine_dist = 1.0 - dot
                if cosine_dist < best_score:
                    best_score = cosine_dist
                    best_name = name
        return best_name, best_score

    def identify(self, crop_bgr):
        """Return (name, score) or ('Unknown', None).
        
        Enhanced with adaptive threshold based on database size.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", None
        
        # Check face quality before identification
        if not self._check_face_quality(crop_bgr):
            return "Unknown", None
        # If insightface available and we have insight DB, prefer deep embeddings
        if self.fa is not None and self._insight_db:
            try:
                faces = self.fa.get(crop_bgr)
                if faces and len(faces) > 0 and hasattr(faces[0], 'embedding'):
                    emb = np.array(faces[0].embedding, dtype=np.float32)
                    n = np.linalg.norm(emb)
                    if n > 1e-6:
                        emb = emb / n
                        name, score = self._best_match_insight(emb)
                        if name is not None and score is not None:
                            # slightly stricter threshold for deep embeddings
                            threshold = max(0.32, self.identify_threshold * 0.8)
                            if score <= threshold:
                                return name, float(score)
            except Exception:
                pass

        # Fallback to lightweight grayscale template matcher
        vec = self._img_to_vector(crop_bgr)
        if vec is None:
            return "Unknown", None
        name, score = self._best_match(vec)
        if name is not None and score is not None:
            # Adaptive threshold: stricter with more templates
            num_templates = len(self._mem_db.get(name, []))
            threshold = self.identify_threshold * (1.0 - min(0.15, num_templates * 0.01))
            if score <= threshold:
                return name, float(score)
        return "Unknown", float(score) if score is not None else None

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes (x1,y1,x2,y2)."""
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]
        
        ix_min = max(x1_min, x2_min)
        iy_min = max(y1_min, y2_min)
        ix_max = min(x1_max, x2_max)
        iy_max = min(y1_max, y2_max)
        
        if ix_max <= ix_min or iy_max <= iy_min:
            return 0.0
        
        inter = (ix_max - ix_min) * (iy_max - iy_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    def _apply_nms(self, boxes, iou_threshold=0.4):
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if not boxes:
            return []
        
        # Sort by confidence descending
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep = []
        
        for box in boxes:
            should_keep = True
            for kept_box in keep:
                iou = self._compute_iou(box, kept_box)
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(box)
        
        return keep
    
    def _check_face_quality(self, crop_bgr, min_size=30):
        """Check if face crop has acceptable quality (size, blur, lighting, anti-spoofing)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        
        h, w = crop_bgr.shape[:2]
        if w < min_size or h < min_size:
            return False
        
        # Check blur using Laplacian variance
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Too blurry
                return False
        except:
            pass
        
        # Check lighting (image should not be too dark or bright)
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            mean_intensity = gray.mean()
            if mean_intensity < 30 or mean_intensity > 220:  # Too dark or bright
                return False
        except:
            pass
        
        # Anti-spoofing: detect photos/printed faces (flat texture, no depth variation)
        try:
            # Check texture frequency content (real faces have more high-freq, photos are flat)
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            # Compute local standard deviation (texture complexity)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            texture_map = cv2.filter2D(gray, cv2.CV_32F, kernel)
            texture_var = np.var(texture_map)
            if texture_var < 50:  # Too flat = likely photo
                return False
            
            # Check for excessive saturation (printed photos are over-saturated)
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].astype(np.float32)
            sat_mean = saturation.mean()
            if sat_mean > 220:  # Over-saturated = likely fake
                return False
        except:
            pass
        
        return True

    def detect_faces_yolo(self, frame_bgr, min_conf=0.35):
        """Detect faces with YOLOv8-face -> returns list of boxes (x1,y1,x2,y2,conf).
        
        Improvements:
        - Higher min_conf threshold for better quality
        - Non-Maximum Suppression to remove duplicates
        - Quality checks on detected faces
        """
        if self.yolo_face is None:
            return []
        try:
            dets = self.yolo_face(frame_bgr, verbose=False, conf=min_conf)
            boxes = []
            
            for r in dets:
                if getattr(r, "boxes", None) is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < min_conf:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Clamp to frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_bgr.shape[1]-1, x2)
                    y2 = min(frame_bgr.shape[0]-1, y2)
                    
                    # Check minimum size
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                    
                    boxes.append((x1, y1, x2, y2, conf))
            
            # Apply NMS to remove overlapping detections
            boxes = self._apply_nms(boxes, iou_threshold=0.4)
            
            # Filter by quality
            quality_boxes = []
            for box in boxes:
                x1, y1, x2, y2, conf = box
                crop = frame_bgr[y1:y2, x1:x2]
                if self._check_face_quality(crop):
                    quality_boxes.append(box)
            
            return quality_boxes if quality_boxes else boxes  # Fallback to all if none pass quality
            
        except Exception as e:
            print(f"[FaceRecognizer] detect_faces_yolo error: {e}")
            return []

# ---------- Helper: load faces from faces/images/ into DB ----------
def load_faces_from_folder(face_recognizer: FaceRecognizer, folder_path="faces/images"):
    """
    Load images from faces/images/*.jpg and register them into recognizer.
    Filenames: Name_anything.jpg -> name 'Name' (split by '_' or first space)
    This function does NOT duplicate existing templates (it trains in-memory).
    """
    p = Path(folder_path)
    if not p.exists():
        print(f"[load_faces_from_folder] folder not found: {p}")
        return
    files = list(p.glob("*.*"))
    if not files:
        print("[load_faces_from_folder] no images found")
        return
    added = 0
    for f in files:
        if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue
        # derive name
        name = f.stem
        # try split by underscore/space to allow "Alex_1.jpg"
        if "_" in name:
            name = name.split("_")[0]
        elif " " in name:
            name = name.split(" ")[0]
        # read image
        try:
            img = cv2.imread(str(f))
            if img is None:
                continue
            # create vector and add to memory db (but avoid saving copy)
            v = face_recognizer._img_to_vector(img)
            if v is not None:
                face_recognizer._mem_db.setdefault(name, []).append(v)
                added += 1

            # If insightface is available, also compute deep embedding and store
            if getattr(face_recognizer, 'fa', None) is not None:
                try:
                    faces = face_recognizer.fa.get(img)
                    if faces and len(faces) > 0 and hasattr(faces[0], 'embedding'):
                        emb = np.array(faces[0].embedding, dtype=np.float32)
                        n = np.linalg.norm(emb)
                        if n > 1e-6:
                            emb = emb / n
                            face_recognizer._insight_db.setdefault(name, []).append(emb)
                except Exception:
                    pass
        except Exception as e:
            print(f"[load_faces_from_folder] skip {f}: {e}")
    # after loading, write to disk DB
    face_recognizer.save_db()
    print(f"[load_faces_from_folder] Added {added} templates to DB")



class FightDetector:
    def __init__(self, model_path="yolov8n-pose.pt", device=None):
        # --- Face recognition init ---
        try:
            self.face_recognizer = FaceRecognizer(db_path="faces/embeddings.json", face_model_path="yolov8n-face.pt")
        except Exception as e:
            print("[FightDetector] FaceRecognizer init failed:", e)
            self.face_recognizer = None
        self.last_recognition = {}
        
        # Pose smoothing: filter out jittery detections
        self.pose_buffer = deque(maxlen=3)  # Keep last 3 poses for smoothing
        self.min_pose_confidence = 0.5  # Only accept high-confidence poses
        
        try:
            if device is None:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
        except Exception:
            device = device or "cpu"
        self.device = device

        if YOLO is None:
            raise RuntimeError("ultralytics YOLO not available")

        print(f"[FightDetector] loading pose model '{model_path}' on {self.device}")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.body_proximity_threshold = 120.0
        self.limb_proximity_threshold = 50.0
        self.fight_hold_duration = 60

        self.fight_detected = False
        self.fight_start_time = 0
        self.last_fight_detection = 0
        self.pose_history = deque(maxlen=512)

        self.analytics = {
            'total_detections': 0,
            'fight_duration_history': [],
            'people_count_history': [],
            'detection_confidence_history': [],
            'fight_events': []
        }

        self.KP = {'nose':0,'left_eye':1,'right_eye':2,'left_ear':3,'right_ear':4,
                   'left_shoulder':5,'right_shoulder':6,'left_elbow':7,'right_elbow':8,
                   'left_wrist':9,'right_wrist':10,'left_hip':11,'right_hip':12,
                   'left_knee':13,'right_knee':14,'left_ankle':15,'right_ankle':16}

    def _dist(self, a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    
    def _smooth_keypoints(self, kp_current):
        """Smooth keypoints across frames to reduce jitter.
        Uses exponential moving average of last 3 frames.
        """
        if kp_current is None:
            return kp_current
        
        self.pose_buffer.append(kp_current)
        
        if len(self.pose_buffer) < 2:
            return kp_current
        
        # Exponential moving average (EMA)
        smoothed = np.array(kp_current, dtype=np.float32)
        alpha = 0.6  # Weight for current frame
        
        for prev_kp in list(self.pose_buffer)[:-1]:
            smoothed = alpha * smoothed + (1 - alpha) * np.array(prev_kp, dtype=np.float32)
            alpha *= 0.4  # Decrease weight for older frames
        
        return smoothed.astype(np.float32)
    
    def _filter_low_confidence_poses(self, poses, min_conf=0.5):
        """Filter out poses with low average keypoint confidence.
        
        Args:
            poses: List of pose dicts with 'keypoints' field
            min_conf: Minimum average confidence threshold
            
        Returns:
            Filtered list of poses
        """
        filtered = []
        for p in poses:
            if p['keypoints'] is None:
                continue
            # Calculate mean confidence of visible keypoints
            confidences = [kp[2] for kp in p['keypoints'] if kp[2] > 0]
            if confidences:
                mean_conf = np.mean(confidences)
                if mean_conf >= min_conf:
                    filtered.append(p)
        return filtered
    
    def _count_visible_keypoints(self, kp):
        """Count keypoints with confidence > 0.3 (visible)."""
        if kp is None:
            return 0
        return sum(1 for k in kp if k[2] > 0.3)
    
    def _is_heavily_occluded(self, kp, min_visible=8):
        """Check if person is heavily occluded (too few visible keypoints).
        
        Args:
            kp: Keypoints array (17 points for COCO-style)
            min_visible: Minimum visible keypoints needed (default 8 of 17)
            
        Returns:
            True if occluded, False if acceptable
        """
        visible_count = self._count_visible_keypoints(kp)
        return visible_count < min_visible

    def get_person_center(self, kp):
        torso = [self.KP['left_shoulder'], self.KP['right_shoulder'], self.KP['left_hip'], self.KP['right_hip']]
        pts = []
        for k in torso:
            if k < len(kp) and kp[k][2] > 0.5:
                pts.append(kp[k][:2])
        if pts:
            return np.mean(np.array(pts), axis=0)
        return None

    def get_bbox(self, kp):
        pts = [p[:2] for p in kp if p[2] > 0.5]
        if not pts:
            return None
        arr = np.array(pts)
        x1, y1 = arr.min(axis=0)
        x2, y2 = arr.max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)

    def _skeleton_lines(self, kp):
        conns = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        lines = []
        for a, b in conns:
            if a < len(kp) and b < len(kp) and kp[a][2] > 0.5 and kp[b][2] > 0.5:
                lines.append(((int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1]))))
        return lines

    def _line_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def check_limb_crossings(self, kp1, kp2):
        l1 = self._skeleton_lines(kp1)
        l2 = self._skeleton_lines(kp2)
        cnt = 0
        for a1, b1 in l1:
            for a2, b2 in l2:
                if self._line_intersect(a1, b1, a2, b2):
                    cnt += 1
        return cnt

    def check_close_limbs(self, kp1, kp2):
        limb_keys = [self.KP['left_wrist'], self.KP['right_wrist'], self.KP['left_elbow'], self.KP['right_elbow'],
                     self.KP['left_knee'], self.KP['right_knee'], self.KP['left_ankle'], self.KP['right_ankle']]
        contacts = 0
        min_d = float('inf')
        for a in limb_keys:
            if a < len(kp1) and kp1[a][2] > 0.5:
                for b in limb_keys:
                    if b < len(kp2) and kp2[b][2] > 0.5:
                        d = self._dist(kp1[a][:2], kp2[b][:2])
                        min_d = min(min_d, d)
                        if d < self.limb_proximity_threshold:
                            contacts += 1
        return contacts, (min_d if min_d != float('inf') else None)

    def detect_fight(self, poses, frame_count):
        if len(poses) < 2:
            return False, [], {'body_distances': [], 'limb_crossings': 0, 'close_contacts': 0, 'confidence': 0}
        
        detected = False
        areas = []
        metrics = {'body_distances': [], 'limb_crossings': 0, 'close_contacts': 0, 'confidence': 0}
        
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                k1 = poses[i]['keypoints']
                k2 = poses[j]['keypoints']
                c1 = self.get_person_center(k1)
                c2 = self.get_person_center(k2)
                body_close = False
                
                if c1 is not None and c2 is not None:
                    d = self._dist(c1, c2)
                    metrics['body_distances'].append(d)
                    if d < self.body_proximity_threshold:
                        body_close = True
                
                crosses = self.check_limb_crossings(k1, k2)
                metrics['limb_crossings'] += crosses
                close_contacts, min_limb = self.check_close_limbs(k1, k2)
                metrics['close_contacts'] += close_contacts
                
                if body_close or crosses > 0 or close_contacts > 0:
                    detected = True
                    b1 = self.get_bbox(k1)
                    b2 = self.get_bbox(k2)
                    if b1 and b2:
                        x1 = max(0, min(b1[0], b2[0]) - 20)
                        y1 = max(0, min(b1[1], b2[1]) - 20)
                        x2 = max(b1[2], b2[2]) + 20
                        y2 = max(b1[3], b2[3]) + 20
                        areas.append((x1, y1, x2, y2))
        
        conf = 0.0
        if metrics['body_distances']:
            avg_d = float(np.mean(metrics['body_distances']))
            conf += max(0.0, (self.body_proximity_threshold - avg_d) / self.body_proximity_threshold * 40.0)
        conf += min(metrics['limb_crossings'] * 20.0, 30.0)
        conf += min(metrics['close_contacts'] * 10.0, 30.0)
        metrics['confidence'] = min(conf, 100.0)
        
        if detected:
            self.analytics['total_detections'] += 1
            self.analytics['detection_confidence_history'].append({
                'frame': frame_count,
                'confidence': metrics['confidence'],
                'timestamp': datetime.now().isoformat()
            })
        
        return detected, areas, metrics

    def draw_skeleton(self, frame, kps, color=(0, 255, 0)):
        conns = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        for kp in kps:
            if kp[2] > 0.5:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)
        for a, b in conns:
            if a < len(kps) and b < len(kps) and kps[a][2] > 0.5 and kps[b][2] > 0.5:
                p1 = (int(kps[a][0]), int(kps[a][1]))
                p2 = (int(kps[b][0]), int(kps[b][1]))
                cv2.line(frame, p1, p2, color, 2)

    def process_frame(self, frame, frame_count):
        proc = frame
        if RESIZE_WIDTH:
            h, w = frame.shape[:2]
            if w != RESIZE_WIDTH:
                scale = RESIZE_WIDTH / float(w)
                new_h = int(h * scale)
                proc = cv2.resize(frame, (RESIZE_WIDTH, new_h))
        
        try:
            results = self.model(proc, verbose=False)
        except Exception as e:
            return frame.copy(), {}
        
        poses = []
        for r in results:
            if getattr(r, "keypoints", None) is not None:
                for kp in r.keypoints.data:
                    arr = kp.cpu().numpy().copy()
                    if RESIZE_WIDTH:
                        sx = frame.shape[1] / float(proc.shape[1])
                        sy = frame.shape[0] / float(proc.shape[0])
                        arr[:, 0] *= sx
                        arr[:, 1] *= sy
                    poses.append({'keypoints': arr})
        
        # ---- Face identification for current frame ----
        identified_names = []
        face_boxes = []
        
        # Get fast face detections via yolo (if available)
        if hasattr(self, 'face_recognizer') and self.face_recognizer is not None:
            try:
                yolo_face_boxes = self.face_recognizer.detect_faces_yolo(frame)
            except Exception:
                yolo_face_boxes = []
        else:
            yolo_face_boxes = []

        # For each person, try to find overlapping face box
        for p in poses:
            name_for_person = "Unknown"
            person_bbox = self.get_bbox(p['keypoints'])
            face_crop = None
            
            # 1) Try matching YOLO-face bbox inside person_bbox
            if person_bbox and yolo_face_boxes:
                px1, py1, px2, py2 = person_bbox
                best_box = None
                best_iou = 0.0
                
                for (fx1, fy1, fx2, fy2, conf) in yolo_face_boxes:
                    # Check center of face inside person bbox
                    cx = (fx1 + fx2) // 2
                    cy = (fy1 + fy2) // 2
                    if cx >= px1 and cx <= px2 and cy >= py1 and cy <= py2:
                        best_box = (fx1, fy1, fx2, fy2)
                        break
                    
                    # Fallback IoU
                    ix1 = max(px1, fx1)
                    iy1 = max(py1, fy1)
                    ix2 = min(px2, fx2)
                    iy2 = min(py2, fy2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    union = (px2 - px1) * (py2 - py1) + (fx2 - fx1) * (fy2 - fy1) - iw * ih
                    iou = (iw * ih) / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_box = (fx1, fy1, fx2, fy2)
                
                if best_box is not None:
                    fx1, fy1, fx2, fy2 = best_box
                    fx1 = max(0, fx1)
                    fy1 = max(0, fy1)
                    fx2 = min(frame.shape[1] - 1, fx2)
                    fy2 = min(frame.shape[0] - 1, fy2)
                    face_crop = frame[fy1:fy2, fx1:fx2].copy()

            # 2) If no face box found, try approximate head crop from keypoints
            # IMPROVED: Use multiple facial keypoints for better ROI
            if face_crop is None and p['keypoints'] is not None:
                kp = p['keypoints']
                try:
                    # Collect confident facial keypoints
                    facial_points = []
                    facial_idx_map = {
                        'nose': 0,
                        'left_eye': 1, 
                        'right_eye': 2,
                        'left_ear': 3,
                        'right_ear': 4
                    }
                    
                    for name, idx in facial_idx_map.items():
                        if idx < len(kp) and kp[idx][2] > 0.3:  # Lowered threshold for more robustness
                            facial_points.append((int(kp[idx][0]), int(kp[idx][1])))
                    
                    if facial_points:
                        # Find bounding box of all facial keypoints
                        xs = [pt[0] for pt in facial_points]
                        ys = [pt[1] for pt in facial_points]
                        
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        
                        # Expand ROI with adaptive padding
                        width = max_x - min_x
                        height = max_y - min_y
                        
                        # Adaptive expansion based on detected keypoint area
                        pad_x = max(int(width * 0.4), 30)
                        pad_y = max(int(height * 0.5), 40)
                        
                        x1 = max(0, min_x - pad_x)
                        y1 = max(0, min_y - pad_y)
                        x2 = min(frame.shape[1] - 1, max_x + pad_x)
                        y2 = min(frame.shape[0] - 1, max_y + pad_y)
                        
                        # Ensure minimum face size
                        if (x2 - x1) >= 30 and (y2 - y1) >= 40:
                            face_crop = frame[y1:y2, x1:x2].copy()
                except Exception as e:
                    print(f"[FightDetector] keypoint-based face crop error: {e}")
                    face_crop = None

            # 3) Identify using face_recognizer with temporal tracking
            if face_crop is not None and hasattr(self, 'face_recognizer') and self.face_recognizer is not None:
                try:
                    nm, dist = self.face_recognizer.identify(face_crop)
                    
                    # TEMPORAL TRACKING: avoid confusing same person detected in consecutive frames
                    person_id = len(identified_names)  # Current person index
                    
                    # If face recognized, check if it's consistent with last frames
                    if nm is not None and nm != "Unknown":
                        # Track this identification
                        if person_id not in self.face_recognizer.face_tracking:
                            self.face_recognizer.face_tracking[person_id] = {
                                'name': nm,
                                'conf': float(dist) if dist else 1.0,
                                'frames': 1,
                                'last_frame': frame_count
                            }
                        else:
                            # Update tracking: use EMA for confidence
                            tracked = self.face_recognizer.face_tracking[person_id]
                            if tracked['name'] == nm:  # Same person
                                tracked['frames'] += 1
                                tracked['conf'] = 0.7 * tracked['conf'] + 0.3 * (float(dist) if dist else 1.0)
                            else:  # Different name detected - require higher confidence
                                if float(dist) if dist else 1.0 > 0.35:
                                    nm = tracked['name']  # Keep previous identification
                                else:
                                    tracked['name'] = nm
                            tracked['last_frame'] = frame_count
                        
                        # Clean old tracking entries (not seen for 30 frames)
                        to_remove = [pid for pid, track in self.face_recognizer.face_tracking.items()
                                    if frame_count - track['last_frame'] > 30]
                        for pid in to_remove:
                            del self.face_recognizer.face_tracking[pid]
                    
                    name_for_person = nm
                    face_boxes.append({'name': nm, 'dist': dist, 'crop_exists': True})
                    
                    # If recognized (not Unknown) -> send Telegram once per ALERT_COOLDOWN per name
                    if nm is not None and nm != "Unknown":
                        now = time.time()
                        last = self.last_recognition.get(nm, 0)
                        if now - last > ALERT_COOLDOWN:
                            try:
                                _, buf = cv2.imencode('.jpg', face_crop)
                                b = buf.tobytes()
                                send_alert(f"👤 Распознан: {nm} (d={dist:.3f})")
                                send_photo(b, caption=f"Распознан: {nm}")
                                self.last_recognition[nm] = now
                            except Exception as e:
                                # ignore telegram failures, but don't change fight logic
                                print("[FightDetector] telegram send error:", e)
                except Exception as e:
                    print(f"[FightDetector] face identification error: {e}")
                    name_for_person = "Unknown"
            else:
                name_for_person = "Unknown"
                face_boxes.append({'name': name_for_person, 'dist': None, 'crop_exists': False})

            identified_names.append(name_for_person)

        self.pose_history.append(poses)
        self.analytics['people_count_history'].append({
            'frame': frame_count,
            'count': int(len(poses)),  # Force integer, no decimals
            'timestamp': datetime.now().isoformat()
        })
        
        fight, areas, metrics = self.detect_fight(poses, frame_count)
        
        if fight:
            if not self.fight_detected:
                self.fight_start_time = frame_count
                self.analytics['fight_events'].append({
                    'start_frame': frame_count,
                    'start_time': datetime.now().isoformat(),
                    'confidence': metrics.get('confidence', 0)
                })
            self.fight_detected = True
            self.last_fight_detection = frame_count
        else:
            if self.fight_detected:
                frames_since = frame_count - self.fight_start_time
                frames_last = frame_count - self.last_fight_detection
                if frames_since >= self.fight_hold_duration and frames_last > 10:
                    dur_s = frames_since / 30.0
                    self.analytics['fight_duration_history'].append(dur_s)
                    if self.analytics['fight_events']:
                        self.analytics['fight_events'][-1]['duration'] = dur_s
                    self.fight_detected = False
        
        out = frame.copy()
        
        # Draw skeletons and names
        for idx, p in enumerate(poses):
            color = (0, 0, 255) if self.fight_detected else (0, 255, 0)
            self.draw_skeleton(out, p['keypoints'], color=color)
            
            # Draw name above bbox if available
            name = identified_names[idx] if idx < len(identified_names) else "Unknown"
            bbox = self.get_bbox(p['keypoints'])
            if bbox:
                x1, y1, x2, y2 = bbox
                txt = name
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx = x1
                ty = max(10, y1 - 10)
                cv2.rectangle(out, (tx, ty - th - 6), (tx + tw + 8, ty + 2), (0, 0, 0), -1)
                cv2.putText(out, txt, (tx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if self.fight_detected:
            dur = (frame_count - self.fight_start_time) / 30.0
            cv2.putText(out, f"FIGHT DETECTED ({dur:.1f}s)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            for a in areas:
                cv2.rectangle(out, (a[0], a[1]), (a[2], a[3]), (0, 0, 255), 3)
        else:
            cv2.putText(out, "Normal", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(out, f"Persons: {len(poses)}", (30, out.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        metrics['people_names'] = identified_names
        return out, metrics


# ------------- Server globals -------------
app = Flask(__name__, static_folder="static", template_folder="templates")
JOBS = {}
JOBS_LOCK = threading.Lock()

detector = None
video_cap = None
proc_thread = None
frame_lock = threading.Lock()
current_frame = None
last_annot = None
stream_active = False

analytics_buffer = deque(maxlen=4000)
latest_stats = {'people': 0, 'fights': 0, 'fps': 0, 'confidence': 0.0, 'timestamp': None}
latest_stats_lock = threading.Lock()
last_alert_time = 0


def write_job_result(job_id, payload):
    path = UPLOAD_DIR / f"job_{job_id}_result.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        print("Failed to write job result:", traceback.format_exc())


def _send_alert_nonblocking(text, frame_path=None, caption=None):
    def job():
        try:
            send_alert(text)
        except Exception:
            pass
        if frame_path:
            try:
                send_photo(frame_path, caption or "")
            except Exception:
                pass
    t = threading.Thread(target=job, daemon=True)
    t.start()


# ------------- Processing loop -------------
def processing_loop(source_is_file=False, job_id=None):
    global detector, video_cap, current_frame, last_annot, stream_active, last_alert_time
    frame_count = 0
    processed = 0
    t0 = time.time()
    
    try:
        while stream_active and video_cap and video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                if source_is_file:
                    break
                time.sleep(0.02)
                continue
            
            frame_count += 1
            do_proc = (SKIP_FRAMES <= 1) or (frame_count % SKIP_FRAMES == 0)
            out_frame = None
            metrics = {}
            
            if do_proc:
                try:
                    out_frame, metrics = detector.process_frame(frame, frame_count)
                    last_annot = out_frame.copy()
                except Exception as e:
                    out_frame = frame.copy()
                    metrics = {}
            else:
                out_frame = last_annot.copy() if last_annot is not None else frame.copy()

            # Alerting
            if detector.fight_detected and (time.time() - last_alert_time > ALERT_COOLDOWN):
                snap = UPLOAD_DIR / f"alert_{int(time.time())}.jpg"
                try:
                    cv2.imwrite(str(snap), out_frame)
                    snap_path = str(snap)
                except Exception:
                    snap_path = None
                
                names = metrics.get('people_names', []) or []
                if names:
                    name_str = ", ".join([n for n in names if n and n != "Unknown"])
                    if not name_str:
                        name_str = ", ".join(names[:3])
                    txt = f"🚨 Fight detected! conf={metrics.get('confidence','N/A')} people={len(detector.pose_history[-1]) if detector.pose_history else 0} | {name_str}"
                else:
                    txt = f"🚨 Fight detected! conf={metrics.get('confidence','N/A')} people={len(detector.pose_history[-1]) if detector.pose_history else 0}"

                _send_alert_nonblocking(txt, snap_path, caption=txt)
                last_alert_time = time.time()

            with frame_lock:
                current_frame = out_frame

            # Push to analytics buffer
            try:
                people = len(detector.pose_history[-1]) if detector.pose_history else 0
                fight_flag = bool(detector.fight_detected)
                
                processed += 1
                now = time.time()
                elapsed = now - t0
                if elapsed >= 0.5:
                    fps_est = int(round(processed / elapsed)) if elapsed > 0 else latest_stats.get('fps', 0)
                    processed = 0
                    t0 = now
                else:
                    fps_est = latest_stats.get('fps', 0)

                snap = {
                    'frame': frame_count,
                    'fight': fight_flag,
                    'people': people,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                analytics_buffer.append(snap)

                # Update stats
                with latest_stats_lock:
                    latest_stats['people'] = people
                    latest_stats['fights'] = detector.analytics.get('total_detections', 0)
                    latest_stats['fps'] = fps_est
                    conf = float(metrics.get('confidence', 0.0))
                    if conf <= 0 and detector.analytics.get('detection_confidence_history'):
                        conf = float(detector.analytics['detection_confidence_history'][-1].get('confidence', 0.0))
                    latest_stats['confidence'] = conf
                    latest_stats['timestamp'] = snap['timestamp']
            except Exception as e:
                print(f"[analytics error] {e}")
    
    except Exception as e:
        print("[processing_loop] fatal:", e, traceback.format_exc())
    finally:
        stream_active = False
        try:
            if video_cap:
                video_cap.release()
        except Exception:
            pass
        if source_is_file and job_id:
            try:
                res = {
                    'job_id': job_id,
                    'analytics': detector.analytics if detector else {},
                    'ended_at': datetime.now().isoformat()
                }
                write_job_result(job_id, res)
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]['status'] = 'finished'
                        JOBS[job_id]['result_path'] = str(UPLOAD_DIR / f"job_{job_id}_result.json")
            except Exception as e:
                print("[warn] finalize job write failed:", e)


# ------------- MJPEG generator -------------
def frame_generator():
    global current_frame, stream_active
    while True:
        with frame_lock:
            frm = current_frame.copy() if current_frame is not None else None
        if frm is None:
            if not stream_active:
                time.sleep(0.1)
                continue
            time.sleep(0.03)
            continue
        try:
            _, buff = cv2.imencode(".jpg", frm)
            b = buff.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b + b'\r\n')
        except Exception:
            time.sleep(0.03)
            continue


# ------------- Flask routes -------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/start_stream', methods=['POST'])
def start_stream():
    global detector, video_cap, proc_thread, stream_active
    
    if detector is None:
        try:
            detector = FightDetector()
            # Автозагрузка лиц из папки faces/images
            if hasattr(detector, 'face_recognizer') and detector.face_recognizer:
                load_faces_from_folder(detector.face_recognizer, folder_path="faces/images")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to init detector: {e}'})
    
    source_is_file = False
    job_id = None
    
    if 'file' in request.files and request.files['file'].filename != '':
        f = request.files['file']
        fn = secure_filename(f.filename)
        saved = UPLOAD_DIR / f"{int(time.time())}_{fn}"
        try:
            f.save(str(saved))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed save: {e}'})
        source = str(saved)
        source_is_file = True
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {
                'status': 'running',
                'video': source,
                'started_at': time.time(),
                'result_path': None
            }
    else:
        data = request.get_json(silent=True) or request.form or request.values
        source = data.get('source', '0')

    try:
        if stream_active:
            stream_active = False
            if proc_thread and proc_thread.is_alive():
                proc_thread.join(timeout=1.0)
        
        if isinstance(source, str) and source.isdigit():
            idx = int(source)
            video_cap = cv2.VideoCapture(idx)
        else:
            video_cap = cv2.VideoCapture(source)
        
        if not video_cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open source'})
        
        try:
            video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        stream_active = True
        proc_thread = threading.Thread(target=processing_loop, args=(source_is_file, job_id), daemon=True)
        proc_thread.start()
        
        return jsonify({'success': True, 'streaming': True, 'stream_url': '/video_feed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_active, proc_thread, video_cap
    stream_active = False
    if proc_thread and proc_thread.is_alive():
        proc_thread.join(timeout=1.0)
    try:
        if video_cap:
            video_cap.release()
    except Exception:
        pass
    return jsonify({'success': True})


@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analytics')
def analytics():
    recent = list(analytics_buffer)[-ANALYTICS_SNAPSHOT_SIZE:]
    with latest_stats_lock:
        ls = dict(latest_stats)
    analytics_copy = detector.analytics.copy() if detector else {}
    return jsonify({
        'success': True,
        'streaming': bool(stream_active),
        'recent_data': recent,
        'analytics': analytics_copy,
        'latest_stats': ls
    })


@app.route('/stats_stream')
def stats_stream():
    def gen():
        while True:
            with latest_stats_lock:
                payload = dict(latest_stats)
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(SSE_INTERVAL)
    return Response(gen(), mimetype='text/event-stream')


@app.route('/job/<job_id>')
def job_status(job_id):
    with JOBS_LOCK:
        meta = JOBS.get(job_id)
    if not meta:
        p = UPLOAD_DIR / f"job_{job_id}_result.json"
        if p.exists():
            try:
                return jsonify({
                    'success': True,
                    'status': 'finished',
                    'analysis': json.loads(p.read_text(encoding='utf-8'))
                })
            except Exception:
                pass
        return jsonify({'success': False, 'error': 'Unknown job_id'}), 404
    
    p = UPLOAD_DIR / f"job_{job_id}_result.json"
    res = None
    if p.exists():
        try:
            res = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            res = None
    return jsonify({'success': True, 'status': meta.get('status'), 'job': meta, 'analysis': res})


@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(str(UPLOAD_DIR.resolve()), filename, as_attachment=False)


@app.route('/settings', methods=['POST'])
def settings():
    global detector
    if not detector:
        return jsonify({'success': False, 'error': 'No detector running'})
    data = request.get_json(silent=True) or request.form or request.values
    try:
        if 'body_proximity_threshold' in data:
            detector.body_proximity_threshold = float(data['body_proximity_threshold'])
        if 'limb_proximity_threshold' in data:
            detector.limb_proximity_threshold = float(data['limb_proximity_threshold'])
        if 'fight_hold_duration' in data:
            detector.fight_hold_duration = int(data['fight_hold_duration'])
    except Exception:
        pass
    return jsonify({'success': True})


@app.route('/add_face', methods=['POST'])
def add_face():
    global detector
    if detector is None or not hasattr(detector, 'face_recognizer') or detector.face_recognizer is None:
        return jsonify({'success': False, 'error': 'face_recognizer not initialized'})
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'success': False, 'error': 'provide file and name'})
    
    f = request.files['file']
    name = request.form['name']
    tmp = UPLOAD_DIR / f"tmp_{int(time.time())}_{secure_filename(f.filename)}"
    f.save(str(tmp))
    img = cv2.imread(str(tmp))
    ok, msg = detector.face_recognizer.register_face(name, img)
    try:
        tmp.unlink()
    except Exception:
        pass
    return jsonify({'success': ok, 'msg': msg})


@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    global detector
    if detector is None or not hasattr(detector, 'face_recognizer'):
        return jsonify({'success': False, 'error': 'Detector not initialized'})
    
    try:
        load_faces_from_folder(detector.face_recognizer, folder_path="faces/images")
        return jsonify({
            'success': True,
            'database': list(detector.face_recognizer.db.keys()),
            'count': len(detector.face_recognizer.db)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == "__main__":
    print("=" * 60)
    print("Fight Detection System with Face Recognition")
    print("=" * 60)
    print(f"Faces folder: {FACES_DIR.resolve()}")
    print(f"Upload folder: {UPLOAD_DIR.resolve()}")
    print("Add face images to 'faces/images/' folder")
    print("File name = person name (e.g., Alex.png → 'Alex')")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8080, debug=True)
