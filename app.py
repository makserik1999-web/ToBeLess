#!/usr/bin/env python3
"""
Fight detection with YOLO-pose and automatic face recognition from faces/images folder
Extended with: weapon detection, fall detection, scream detection, and report generation
"""
from face_recognizer import FaceRecognizer
from face_blur import blur_faces
from hybrid_fight_detector import HybridFightDetector
from video_source import RealTimeCapture
from weapon_detector import WeaponDetector
from fall_detector import FallDetector
from scream_detector import ScreamDetector
from report_generator import ReportGenerator
from video_source import RealTimeCapture
import cv2

# инициализация (один раз при старте)
face_rec = FaceRecognizer(yolo_model_path="yolov8n-face.pt", db_path="faces/embeddings.json", debug=False)
# если DB пустая — автоматически заполнить из faces/images (имена: Alex_1.jpg -> Alex)
face_rec.bulk_register_from_folder("faces/images")
face_blur_enabled = False
face_recognition_enabled = False  # DISABLED by default for performance (enable via API)

import os, time, uuid, json, threading, traceback
from pathlib import Path
try:
    import face_recognition
except Exception:
    face_recognition = None
    print("[WARNING] 'face_recognition' package not installed. dlib-based recognition disabled.")
    print("Install with: pip install face_recognition dlib -- or use InsightFace (pip install insightface onnxruntime)")
from collections import deque
from datetime import datetime, timedelta
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

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
        print("[WARNING] OPENAI_API_KEY not found in environment variables. Chatbot disabled.")
except Exception as e:
    openai_client = None
    print(f"[WARNING] OpenAI not available: {e}")

# ------------ Config ------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
FACES_DIR = Path("faces/images")
FACES_DIR.mkdir(parents=True, exist_ok=True)

ALERT_COOLDOWN = 8
ANALYTICS_SNAPSHOT_SIZE = 300
SKIP_FRAMES = 2          # Process every 2nd frame for 2x speedup
RESIZE_WIDTH = 416       # STABLE BASELINE: 416 for balance of speed & accuracy
SSE_INTERVAL = 0.5

# ------------ Performance Profiling ------------
PROFILING_ENABLED = True  # Toggle profiling on/off
PROFILE_INTERVAL = 30     # Print stats every N frames

class PerformanceProfiler:
    """Performance profiler for video processing pipeline"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all timing statistics"""
        self.timings = {
            'hybrid_detection': [],    # Full hybrid detector (pose + action)
            'weapon_detection': [],
            'fall_detection': [],
            'scream_detection': [],
            'frame_update': [],        # Frame buffer and stats update
            'total_frame': []          # Total frame processing time
        }
        self.frame_count = 0
        self.start_time = time.time()

    def time_function(self, name: str):
        """Context manager for timing a function"""
        return TimingContext(self, name)

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement"""
        if name in self.timings:
            self.timings[name].append(duration)

    def print_stats(self, hybrid_detector=None):
        """Print performance statistics"""
        if not PROFILING_ENABLED:
            return

        elapsed = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "="*70)
        print(f"[PROFILER] Performance Statistics (last {PROFILE_INTERVAL} frames)")
        print("="*70)

        total_time = 0
        bottleneck = ("none", 0)

        for name, times in self.timings.items():
            if times:
                avg_ms = (sum(times) / len(times)) * 1000
                max_ms = max(times) * 1000
                total_time += avg_ms

                # Track bottleneck
                if avg_ms > bottleneck[1]:
                    bottleneck = (name, avg_ms)

                print(f"  {name:20s}: avg={avg_ms:6.1f}ms  max={max_ms:6.1f}ms  calls={len(times)}")

        print("-"*70)
        print(f"  {'TOTAL':20s}: avg={total_time:6.1f}ms")
        print(f"  Actual FPS: {actual_fps:.1f}")
        print(f"  Theoretical max FPS: {1000/total_time:.1f}" if total_time > 0 else "  Theoretical max FPS: N/A")
        print(f"  🔴 BOTTLENECK: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")

        # Print hybrid detector sub-timings if available
        if hybrid_detector and hasattr(hybrid_detector, 'get_profiling_stats'):
            hs = hybrid_detector.get_profiling_stats()
            print("\n  [HYBRID BREAKDOWN]")
            print(f"    YOLO-Pose : avg={hs['yolo_pose']['avg_ms']:6.1f}ms  max={hs['yolo_pose']['max_ms']:6.1f}ms  ({hs['breakdown_pct']['yolo_pct']:.0f}%)")
            print(f"    SlowFast  : avg={hs['slowfast']['avg_ms']:6.1f}ms  max={hs['slowfast']['max_ms']:6.1f}ms  ({hs['breakdown_pct']['slowfast_pct']:.0f}%) [runs every {self.frame_count//(hs['slowfast']['calls'] or 1)} frames]" if hs['slowfast']['calls'] > 0 else f"    SlowFast  : not run yet")
            print(f"    Fusion    : avg={hs['fusion']['avg_ms']:6.1f}ms")
            print(f"    🎯 Sub-bottleneck: {hs['summary']['bottleneck']}")
            hybrid_detector.reset_profiling_stats()

        print("="*70 + "\n")

        # Reset for next interval
        self.timings = {k: [] for k in self.timings}
        self.frame_count = 0
        self.start_time = time.time()

class TimingContext:
    """Context manager for timing code blocks"""

    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        self.profiler.add_timing(self.name, duration)

# Global profiler instance
profiler = PerformanceProfiler()

def toggle_profiling(enabled: bool = None) -> bool:
    """Toggle or set profiling state"""
    global PROFILING_ENABLED
    if enabled is not None:
        PROFILING_ENABLED = enabled
    else:
        PROFILING_ENABLED = not PROFILING_ENABLED
    print(f"[Profiler] Profiling {'ENABLED' if PROFILING_ENABLED else 'DISABLED'}")
    return PROFILING_ENABLED

try:
    import torch
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------- Replace the old FaceRecognizer with this (put in place of the old class) ----------

def load_faces_from_folder(face_recognizer, folder_path="faces/images"):
    """Helper function to load faces from a folder into the recognizer"""
    if face_recognizer is None:
        print("[load_faces_from_folder] No face recognizer provided")
        return
    try:
        face_recognizer.bulk_register_from_folder(folder_path)
        print(f"[load_faces_from_folder] Loaded faces from {folder_path}")
    except Exception as e:
        print(f"[load_faces_from_folder] Error: {e}")


class FightDetector:
    def __init__(self, model_path="yolov8n-pose.pt", device=None):
        # --- Face recognition init (device is resolved later, patched in start_stream) ---
        self.face_rec = None          # set after device is known
        self.last_recognition = {}

        # ============ GPU DIAGNOSTICS ============
        print("\n" + "="*50)
        print("[GPU DIAGNOSTICS]")
        print("="*50)
        try:
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  CUDA version: {torch.version.cuda}")
                print(f"  GPU count: {torch.cuda.device_count()}")
                print(f"  GPU name: {torch.cuda.get_device_name(0)}")
                print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            if getattr(torch.backends, "mps", None):
                print(f"  MPS available: {torch.backends.mps.is_available()}")
        except Exception as e:
            print(f"  GPU diagnostics error: {e}")
        print("="*50 + "\n")

        # Device selection with forced CUDA
        try:
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"[FightDetector] ✓ Using CUDA (GPU acceleration)")
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                    print(f"[FightDetector] ✓ Using MPS (Apple Silicon)")
                else:
                    device = "cpu"
                    print(f"[FightDetector] ⚠️ Using CPU (no GPU available)")
        except Exception as e:
            device = device or "cpu"
            print(f"[FightDetector] Device detection failed: {e}, using {device}")
        self.device = device

        if YOLO is None:
            raise RuntimeError("ultralytics YOLO not available")

        print(f"[FightDetector] Loading pose model '{model_path}' on {self.device}")

        # Load and force model to device
        try:
            self.model = YOLO(model_path)
            # Force to CUDA if available
            if self.device == 'cuda':
                self.model.to('cuda')
                # Verify model is on GPU
                actual_device = next(self.model.model.parameters()).device
                print(f"[FightDetector] ✓ Model loaded on: {actual_device}")
                if 'cuda' not in str(actual_device):
                    print(f"[FightDetector] ⚠️ WARNING: Model not on CUDA! Forcing...")
                    self.model.model.cuda()
            else:
                self.model.to(self.device)
        except Exception as e:
            print(f"[FightDetector] ⚠️ Failed to move model to {self.device}: {e}")
            print(f"[FightDetector] Falling back to default device")
            self.model = YOLO(model_path)

        # ============ WARMUP INFERENCE ============
        # First inference is always slow due to CUDA kernel compilation
        self._use_half = (self.device == 'cuda')
        try:
            print("[FightDetector] Running warmup inference...")
            dummy = np.zeros((RESIZE_WIDTH, RESIZE_WIDTH, 3), dtype=np.uint8)
            with torch.no_grad():
                _ = self.model(dummy, imgsz=RESIZE_WIDTH, verbose=False,
                               half=self._use_half)
            print(f"[FightDetector] ✓ Warmup complete (FP{'16' if self._use_half else '32'})")
        except Exception as e:
            print(f"[FightDetector] Warmup failed: {e}")

        self.body_proximity_threshold = 70.0   # Tightened: closer proximity required (was 80)
        self.limb_proximity_threshold = 25.0   # Tightened: closer contact required (was 30)
        self.fight_hold_duration = 60

        self.fight_detected = False
        self.fight_start_time = 0
        self.last_fight_detection = 0
        self.last_fight_event_time = 0  # Cooldown for incident counting
        self.pose_history = deque(maxlen=512)

        # Cached poses from last detection (drawn on every fresh frame)
        self.last_poses = []
        self.last_fight_info = {'confidence': 0, 'fight_detected': False, 'fight_areas': []}
        self.skip_counter = 0  # Internal skip counter for YOLO inference

        # Face recognition throttle — run InsightFace every N frames only
        self.face_rec_interval = 20   # adjust: lower = more accurate, higher = faster
        self._face_rec_counter = 0
        self._last_face_data = []     # cached results reused between recognition frames

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

    def _bbox_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes to detect duplicate detections"""
        if box1 is None or box2 is None:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _is_same_person(self, kp1, kp2):
        """Check if two detections are the same person (duplicate / ghost skeleton)."""
        b1 = self.get_bbox(kp1)
        b2 = self.get_bbox(kp2)

        # Method 1: IoU — lowered from 0.5 to 0.3 to catch partial overlaps
        iou = self._bbox_iou(b1, b2)
        if iou > 0.3:
            return True, f"IoU={iou:.2f}"

        # Method 2: Box inclusion — one box nearly contained within the other
        # Catches ghost skeletons split off from a partially occluded person
        if b1 and b2:
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            if x2 > x1 and y2 > y1:
                inter = (x2 - x1) * (y2 - y1)
                area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
                area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                inclusion = inter / min(area1, area2)
                if inclusion > 0.8:  # One box is 80%+ inside the other
                    return True, f"inclusion={inclusion:.2f}"

        # Method 3: Centers very close (less than 50 pixels)
        c1 = self.get_person_center(kp1)
        c2 = self.get_person_center(kp2)
        if c1 is not None and c2 is not None:
            center_dist = self._dist(c1, c2)
            if center_dist < 50:
                return True, f"center_dist={center_dist:.0f}"

        return False, None

    def detect_fight(self, poses, frame_count):
        if len(poses) < 2:
            return False, [], {'confidence': 0}
        detected = False
        areas = []
        metrics = {'body_distances': [], 'limb_crossings': 0, 'close_contacts': 0, 'confidence': 0,
                   'pairs_checked': 0, 'duplicates_skipped': 0}

        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                k1 = poses[i]['keypoints']
                k2 = poses[j]['keypoints']

                # ===== DUPLICATE DETECTION CHECK =====
                # Skip if these are the same person detected twice
                is_same, reason = self._is_same_person(k1, k2)
                if is_same:
                    metrics['duplicates_skipped'] += 1
                    # Uncomment for debug: print(f"[FightDetector] Skipping duplicate: person {i} vs {j} ({reason})")
                    continue

                metrics['pairs_checked'] += 1

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

                # Only trigger if MULTIPLE indicators (reduce false positives)
                indicators = sum([body_close, crosses > 0, close_contacts > 1])
                if indicators >= 2:  # Require at least 2 indicators
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
            # OPTIMIZATION: Move fight counting to HybridFightDetector to respect Veto
            # self.analytics['total_detections'] += 1 (Disabled here)
            
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


    def process_frame(self, frame, frame_count, run_detection=True):
        """
        Process a frame. Always returns the FRESH frame with overlays.

        Args:
            frame: Current video frame (BGR)
            frame_count: Frame number
            run_detection: If True, run YOLO inference. If False, reuse last poses.

        Returns:
            (annotated_frame, metrics_dict)
        """
        global face_blur_enabled, face_recognition_enabled

        poses = self.last_poses  # Default: reuse last known poses

        if run_detection:
            # ---- Resize for YOLO ----
            proc = frame
            if RESIZE_WIDTH:
                h, w = frame.shape[:2]
                if w != RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / float(w)
                    new_h = int(h * scale)
                    proc = cv2.resize(frame, (RESIZE_WIDTH, new_h))

            # ---- YOLO inference (single frame, no batching) ----
            try:
                with torch.no_grad():
                    results = self.model(
                        proc,
                        imgsz=RESIZE_WIDTH,
                        verbose=False,
                        conf=0.5,
                        device=self.device,
                        half=self._use_half,
                        augment=False,
                        visualize=False
                    )
            except Exception as e:
                results = []

            # ---- Extract poses and scale back to original resolution ----
            poses = []
            for r in (results if results else []):
                if getattr(r, "keypoints", None) is not None:
                    for kp in r.keypoints.data:
                        arr = kp.cpu().numpy().copy()
                        # OPTIMIZATION: Ignore ghost skeletons (e.g. just limbs) that have < 5 keypoints
                        valid_kps = sum(1 for pt in arr if pt[2] > 0.5)
                        if valid_kps >= 5:
                            if RESIZE_WIDTH:
                                sx = frame.shape[1] / float(proc.shape[1])
                                sy = frame.shape[0] / float(proc.shape[0])
                                arr[:, 0] *= sx
                                arr[:, 1] *= sy
                            poses.append({'keypoints': arr})

            # Cache poses for reuse on skipped frames
            self.last_poses = poses

            # ---- Fight detection (only on detection frames) ----
            fight_detected, fight_areas, metrics = self.detect_fight(poses, frame_count)

            # Update fight state
            if fight_detected:
                if not self.fight_detected:
                    self.fight_start_time = frame_count
                self.fight_detected = True
                self.last_fight_detection = frame_count
            else:
                if self.fight_detected and (frame_count - self.last_fight_detection > self.fight_hold_duration):
                    self.fight_detected = False

            # Cache fight info
            self.last_fight_info = {
                'confidence': metrics.get('confidence', 0),
                'fight_detected': fight_detected,
                'fight_areas': fight_areas if fight_detected else [],
            }

            # Store pose history
            self.pose_history.append(poses)
        else:
            # Reuse cached fight info for skipped frames
            metrics = {
                'confidence': self.last_fight_info.get('confidence', 0),
                'body_distances': [],
                'limb_crossings': 0,
                'close_contacts': 0,
                'pairs_checked': 0,
                'duplicates_skipped': 0,
            }
            fight_detected = self.last_fight_info.get('fight_detected', False)
            fight_areas = self.last_fight_info.get('fight_areas', [])

        # ==== RENDER: Always draw on the FRESH frame ====

        # Face processing
        identified_names = []
        if not face_blur_enabled and not face_recognition_enabled:
            out = frame  # No copy needed
            identified_names = ["—"] * len(poses)
        else:
            out = frame.copy()
            if face_blur_enabled and hasattr(self, 'face_rec') and self.face_rec is not None:
                try:
                    out, _ = blur_faces(out, self.face_rec, min_conf=0.5, blur_strength=25, expand=0.1)
                    if not face_recognition_enabled:
                        identified_names = ["Blurred"] * len(poses)
                except Exception as e:
                    print(f"[FightDetector] face blur error: {e}")

            if face_recognition_enabled and self.face_rec is not None:
                try:
                    self._face_rec_counter += 1
                    if self._face_rec_counter >= self.face_rec_interval:
                        # Run InsightFace inference and refresh cache
                        self._face_rec_counter = 0
                        self._last_face_data = self.face_rec.get_face_data(out, min_conf=0.5)
                    face_data = self._last_face_data
                    identified_names = [fd['name'] for fd in face_data] if face_data else []
                    # Draw face bboxes + name labels from cache
                    for fd in face_data:
                        fx1, fy1, fx2, fy2 = fd['bbox']
                        name = fd['name']
                        score = fd.get('score')
                        color = (0, 220, 0) if name != 'Unknown' else (0, 140, 255)
                        cv2.rectangle(out, (fx1, fy1), (fx2, fy2), color, 2)
                        label = name if score is None else f"{name} ({1 - score:.2f})"
                        cv2.putText(out, label, (fx1, fy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"[FightDetector] face recognition error: {e}")
                    identified_names = []

            if not identified_names:
                identified_names = ["—"] * len(poses)

        # Draw skeletons (always green - hybrid will redraw red if confirmed)
        for p in poses:
            self.draw_skeleton(out, p['keypoints'], (0, 255, 0))

        # Draw names on pose bboxes only when face recognition is OFF
        # (when it's ON, names are already drawn on face bboxes above)
        if not face_recognition_enabled:
            for i, p in enumerate(poses):
                if i < len(identified_names):
                    name = identified_names[i]
                    bbox = self.get_bbox(p['keypoints'])
                    if bbox and name and name != "—":
                        x1, y1, x2, y2 = bbox
                        cv2.putText(out, name, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Build return metrics
        metrics['people_count'] = len(poses)
        metrics['people_names'] = identified_names
        metrics['fight_detected'] = fight_detected
        metrics['fight_areas'] = fight_areas
        metrics['poses'] = poses

        return out, metrics


# ------------- Server globals -------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# CORS configuration for React frontend
from flask_cors import CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

JOBS = {}
JOBS_LOCK = threading.Lock()

detector = None
weapon_detector = None
fall_detector = None
scream_detector = None
report_gen = ReportGenerator(output_dir="reports", company_name="ToBeLess AI Security", debug=True)

# Detection toggles
weapon_detection_enabled = True
fall_detection_enabled = True
scream_detection_enabled = True   # Enabled by default (requires microphone)

# Event storage for reports
detection_events = []
detection_events_lock = threading.Lock()

video_cap = None
proc_thread = None
frame_lock = threading.Lock()
current_frame = None
stream_active = False

analytics_buffer = deque(maxlen=4000)
latest_stats = {
    'people': 0,
    'fights': 0,
    'weapons': 0,
    'falls': 0,
    'screams': 0,
    'fps': 0,
    'confidence': 0.0,
    'timestamp': None
}
latest_stats_lock = threading.Lock()
last_alert_time = 0
last_weapon_alert_time = 0
last_fall_alert_time = 0
last_scream_alert_time = 0


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
    global detector, weapon_detector, fall_detector, scream_detector
    global video_cap, current_frame, stream_active
    global last_alert_time, last_weapon_alert_time, last_fall_alert_time, last_scream_alert_time
    global detection_events, profiler

    frame_count = 0
    processed = 0
    t0 = time.time()

    # Detection counters for this session
    weapon_count = 0
    fall_count = 0
    scream_count = 0

    # Per-person face-sighting cooldown {name: last_alert_timestamp}
    face_seen_times: dict = {}
    # Weapon detection runs every N frames to keep FPS up
    weapon_frame_counter = 0

    # Reset profiler for new session
    profiler.reset()

    try:
        while stream_active and video_cap and video_cap.isOpened():
            frame_start = time.perf_counter()

            ret, frame = video_cap.read()
            if not ret:
                if source_is_file:
                    print("[Video] End of file reached, looping...")
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                    ret, frame = video_cap.read()  # Try reading again
                    if not ret:
                        print("[Video] ERROR: Cannot read even after loop, stopping")
                        break
                    print("[Video] Loop successful, continuing...")
                    # Reset frame counter for clean loop
                    frame_count = 0
                    profiler.reset()
                    if detector:
                        detector.reset_statistics()
                else:
                    # RealTimeCapture handles reconnection internally;
                    # just wait briefly for the next frame to arrive.
                    time.sleep(0.05)
                    continue

            frame_count += 1
            profiler.frame_count += 1

            # YOLO runs on selected frames; every frame goes to hybrid (for SlowFast buffer)
            run_yolo = (SKIP_FRAMES <= 1) or (frame_count % SKIP_FRAMES == 0)

            # Debug every 60 frames
            if frame_count % 60 == 0:
                print(f"[Loop] Frame {frame_count}: stream_active={stream_active}, "
                      f"cap.isOpened()={video_cap.isOpened() if video_cap else False}")

            out_frame = None
            metrics = {}

            # EVERY frame goes to hybrid detector (for temporal buffer + rendering)
            # YOLO only runs when run_yolo=True
            with profiler.time_function('hybrid_detection'):
                try:
                    out_frame, metrics = detector.process_frame(frame, frame_count, run_yolo=run_yolo)
                except Exception as e:
                    out_frame = frame.copy()
                    metrics = {}

            # Fight alerting
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

                # Store fight event for reports
                with detection_events_lock:
                    detection_events.append({
                        'type': 'fight',
                        'confidence': metrics.get('confidence', 0) / 100 if metrics.get('confidence', 0) > 1 else metrics.get('confidence', 0),
                        'timestamp': datetime.now().isoformat(),
                        'details': f"People involved: {len(detector.pose_history[-1]) if detector.pose_history else 0}"
                    })

            # Face-sighting alerts (known persons, per-person cooldown)
            face_alert_cooldown = 60  # seconds between alerts for the same person
            known_names = [
                n for n in metrics.get('people_names', []) or []
                if n and n not in ('Unknown', '—', '')
            ]
            for name in set(known_names):
                if time.time() - face_seen_times.get(name, 0) > face_alert_cooldown:
                    _send_alert_nonblocking(f"👤 Known person detected: {name}")
                    face_seen_times[name] = time.time()

            # Weapon detection (every 5 frames to save GPU budget)
            if weapon_detection_enabled and weapon_detector is not None and out_frame is not None:
                weapon_frame_counter += 1
                if weapon_frame_counter >= 5:
                    weapon_frame_counter = 0
                    try:
                        w_dets = weapon_detector.detect(out_frame)
                        if w_dets:
                            out_frame = weapon_detector.draw_detections(out_frame, w_dets)
                            if time.time() - last_weapon_alert_time > ALERT_COOLDOWN:
                                worst = max(w_dets, key=lambda d: d['confidence'])
                                txt = f"🔫 Weapon: {worst['class_name']} ({worst['confidence']:.0%})"
                                snap_w = UPLOAD_DIR / f"weapon_{int(time.time())}.jpg"
                                try:
                                    cv2.imwrite(str(snap_w), out_frame)
                                    snap_w_path = str(snap_w)
                                except Exception:
                                    snap_w_path = None
                                _send_alert_nonblocking(txt, snap_w_path, caption=txt)
                                last_weapon_alert_time = time.time()
                                with detection_events_lock:
                                    detection_events.append({
                                        'type': 'weapon',
                                        'confidence': worst['confidence'],
                                        'timestamp': datetime.now().isoformat(),
                                        'details': f"{worst['class_name']} (danger: {worst['danger_level']})"
                                    })
                    except Exception as e:
                        print(f"[weapon] {e}")

            # Scream detection events — drain the audio queue
            if scream_detector is not None:
                try:
                    while True:
                        evt = scream_detector.detection_queue.get_nowait()
                        if time.time() - last_scream_alert_time > ALERT_COOLDOWN:
                            score = evt.get('score', 0)
                            txt = f"🔊 Scream detected! score={score:.2f}"
                            _send_alert_nonblocking(txt)
                            last_scream_alert_time = time.time()
                        with detection_events_lock:
                            detection_events.append({
                                'type': 'scream',
                                'confidence': evt.get('score', 0),
                                'timestamp': evt.get('timestamp', datetime.now().isoformat()),
                                'details': f"volume: {evt.get('volume', 0):.2f}"
                            })
                except Exception:
                    pass  # queue.Empty — no events

            # Frame update and stats (profiled)
            with profiler.time_function('frame_update'):
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
                        latest_stats['weapons'] = weapon_detector.stats['total_detections'] if weapon_detector else 0
                        latest_stats['falls'] = fall_detector.stats['total_falls_detected'] if fall_detector else 0
                        latest_stats['screams'] = scream_detector.stats['total_screams_detected'] if scream_detector else 0
                        latest_stats['fps'] = fps_est
                        # Only show confidence from current frame detection
                        # Don't use stale historical confidence when no fight is detected
                        latest_stats['confidence'] = float(metrics.get('confidence', 0.0))
                        latest_stats['timestamp'] = snap['timestamp']
                except Exception as e:
                    print(f"[analytics error] {e}")

            # Total frame time
            frame_duration = time.perf_counter() - frame_start
            profiler.add_timing('total_frame', frame_duration)

            # Print profiling stats every PROFILE_INTERVAL frames
            if PROFILING_ENABLED and frame_count % PROFILE_INTERVAL == 0:
                profiler.print_stats(hybrid_detector=detector)
    
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
def _create_loading_frame():
    """Create a loading placeholder frame"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background

    # Add loading text - brighter so it's clearly visible
    text = "Initializing..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
    x = (640 - text_size[0]) // 2
    y = (480 + text_size[1]) // 2
    cv2.putText(frame, text, (x, y), font, 1.2, (200, 200, 200), 2)

    # Add subtext
    subtext = "Please wait..."
    sub_size = cv2.getTextSize(subtext, font, 0.7, 1)[0]
    cv2.putText(frame, subtext, ((640 - sub_size[0]) // 2, y + 40), font, 0.7, (150, 150, 150), 1)

    return frame

_loading_frame = _create_loading_frame()

def frame_generator():
    global current_frame, stream_active
    while True:
        with frame_lock:
            frm = current_frame.copy() if current_frame is not None else None
        if frm is None:
            # Show loading frame instead of black screen
            if stream_active:
                frm = _loading_frame
            else:
                time.sleep(0.1)
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
    global detector, weapon_detector, fall_detector, scream_detector, video_cap, proc_thread, stream_active

    if detector is None:
        try:
            # Create pose detector first
            pose_detector = FightDetector()
            # OPTIMIZED: Wrap with hybrid detector (background SlowFast thread)
            detector = HybridFightDetector(
                pose_detector=pose_detector,
                device='cuda' if pose_detector.device == 'cuda' else 'cpu',
                require_both=False,  # Relaxed: RWF model is smart enough to work without pose
                action_weight=0.85,  # Stronger trust in SlowFast (was 0.7)
                violence_threshold=0.6,  # Raised: require 60% confidence to fire
                buffer_size=32  # Standard R50 requires 32 frames
            )

            # ============ POST-INIT DIAGNOSTICS ============
            print("\n" + "="*50)
            print("[DETECTOR DEVICE STATUS]")
            print("="*50)
            print(f"  Pose detector device: {pose_detector.device}")
            try:
                yolo_device = next(pose_detector.model.model.parameters()).device
                print(f"  YOLO model actual device: {yolo_device}")
            except Exception as e:
                print(f"  YOLO device check failed: {e}")
            print(f"  Hybrid detector device: {detector.device if hasattr(detector, 'device') else 'N/A'}")
            if hasattr(detector, 'slowfast_detector'):
                print(f"  SlowFast device: {detector.slowfast_detector.device}")
            print("="*50 + "\n")

            # Init FaceRecognizer now that we know the device (CUDA / CPU)
            try:
                fr_device = pose_detector.device  # 'cuda' or 'cpu'
                pose_detector.face_rec = FaceRecognizer(
                    db_path="faces/embeddings.json",
                    device=fr_device,
                    debug=False,
                )
                print(f"[App] FaceRecognizer initialised on {fr_device} (backend={pose_detector.face_rec.backend})")
                load_faces_from_folder(pose_detector.face_rec, folder_path="faces/images")
            except Exception as e:
                print(f"[App] FaceRecognizer init failed: {e}")
                pose_detector.face_rec = None
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to init detector: {e}'})

    if weapon_detector is None and weapon_detection_enabled:
        try:
            dev = detector.device if detector and hasattr(detector, 'device') else 'cpu'
            weapon_detector = WeaponDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.5,
                device=dev,
                debug=False
            )
            print(f"[App] ✓ Weapon detector initialized on {dev}")
        except Exception as e:
            print(f"[App] Weapon detector init failed: {e}")

    # STABLE BASELINE: Fall detector DISABLED for stability
    # if fall_detector is None and fall_detection_enabled:
    #     try:
    #         fall_detector = FallDetector(
    #             model_path="yolov8n-pose.pt",
    #             fall_angle_threshold=45.0,
    #             confirmation_frames=5,
    #             device='cuda' if detector and hasattr(detector, 'device') else 'cpu',
    #             debug=True
    #         )
    #         print("[App] ✓ Fall detector initialized")
    #     except Exception as e:
    #         print(f"[App] Fall detector init failed: {e}")
    print("[App] Fall detector DISABLED (stable baseline)")

    # Initialize scream detector (if enabled)
    if scream_detector is None and scream_detection_enabled:
        try:
            scream_detector = ScreamDetector(
                volume_threshold=0.3,
                cooldown_seconds=2.0,
                debug=True
            )
            if scream_detector.start():
                print("[App] ✓ Scream detector initialized and started")
            else:
                print("[App] Scream detector failed to start (no microphone?)")
        except Exception as e:
            print(f"[App] Scream detector init failed: {e}")
    
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
        
        if source_is_file:
            # File source: use plain cv2.VideoCapture so we can loop-back
            video_cap = cv2.VideoCapture(source)
            try:
                video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        else:
            # Live source (webcam / RTSP): use RealTimeCapture for zero-lag
            # frames and automatic reconnection on stream drops.
            src = int(source) if isinstance(source, str) and source.isdigit() else source
            video_cap = RealTimeCapture(src, reconnect=True)

        if not video_cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open source'})

        # Reset detector stats for fresh start
        if detector:
            detector.reset_statistics()
        if weapon_detector:
            weapon_detector.stats = {'total_detections': 0}
        if fall_detector:
            fall_detector.stats = {'total_falls_detected': 0}
        if scream_detector:
            scream_detector.stats = {'total_screams_detected': 0}

        stream_active = True
        proc_thread = threading.Thread(target=processing_loop, args=(source_is_file, job_id), daemon=True)
        proc_thread.start()
        
        return jsonify({'success': True, 'streaming': True, 'stream_url': '/video_feed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_active, proc_thread, video_cap, current_frame, latest_stats, detector
    stream_active = False
    if proc_thread and proc_thread.is_alive():
        proc_thread.join(timeout=2.0)
    try:
        if video_cap:
            video_cap.release()
    except Exception:
        pass
    # Stop SlowFast background thread and reset detector for fresh start
    if detector and hasattr(detector, 'stop'):
        try:
            detector.stop()
        except Exception:
            pass
    detector = None  # Force re-creation on next start_stream

    # Clear the current frame so old video doesn't persist
    with frame_lock:
        current_frame = None

    # Reset stats
    with latest_stats_lock:
        latest_stats = {
            'people': 0, 'fights': 0, 'weapons': 0, 'falls': 0, 'screams': 0,
            'fps': 0, 'confidence': 0, 'timestamp': None
        }

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


@app.route('/toggle_profiling', methods=['POST'])
def toggle_profiling_endpoint():
    """Toggle performance profiling on/off"""
    global PROFILING_ENABLED
    data = request.get_json(silent=True) or {}
    enabled = data.get('enabled')  # Optional: explicitly set True/False

    if enabled is not None:
        PROFILING_ENABLED = bool(enabled)
    else:
        PROFILING_ENABLED = not PROFILING_ENABLED

    profiler.reset()  # Reset stats on toggle

    return jsonify({
        'success': True,
        'profiling_enabled': PROFILING_ENABLED
    })


@app.route('/profiling_stats', methods=['GET'])
def get_profiling_stats():
    """Get current profiling statistics including hybrid detector breakdown"""
    global detector

    result = {
        'enabled': PROFILING_ENABLED,
        'frame_count': profiler.frame_count,
        'timings': {
            name: {
                'avg_ms': (sum(times) / len(times)) * 1000 if times else 0,
                'max_ms': max(times) * 1000 if times else 0,
                'count': len(times)
            }
            for name, times in profiler.timings.items()
        }
    }

    # Add hybrid detector sub-timings if available
    if detector and hasattr(detector, 'get_profiling_stats'):
        result['hybrid_breakdown'] = detector.get_profiling_stats()

    return jsonify(result)


@app.route('/add_face', methods=['POST'])
def add_face():
    global detector
    # Access face_rec through pose_detector since we're using HybridFightDetector
    face_rec = detector.pose_detector.face_rec if hasattr(detector, 'pose_detector') else getattr(detector, 'face_rec', None)
    if detector is None or face_rec is None:
        return jsonify({'success': False, 'error': 'face_recognizer not initialized'})
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'success': False, 'error': 'provide file and name'})

    f = request.files['file']
    name = request.form['name']
    tmp = UPLOAD_DIR / f"tmp_{int(time.time())}_{secure_filename(f.filename)}"
    f.save(str(tmp))
    img = cv2.imread(str(tmp))
    ok, msg = face_rec.register_face(name, img)
    try:
        tmp.unlink()
    except Exception:
        pass
    return jsonify({'success': ok, 'msg': msg})


@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    global detector
    # Access face_rec through pose_detector since we're using HybridFightDetector
    face_rec = detector.pose_detector.face_rec if hasattr(detector, 'pose_detector') else getattr(detector, 'face_rec', None)
    if detector is None or face_rec is None:
        return jsonify({'success': False, 'error': 'Detector not initialized'})

    try:
        load_faces_from_folder(face_rec, folder_path="faces/images")
        return jsonify({
            'success': True,
            'database': list(face_rec._mem_db.keys()),
            'count': len(face_rec._mem_db)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



@app.route('/toggle_face_blur', methods=['POST'])
def toggle_face_blur():
    global face_blur_enabled
    data = request.get_json(silent=True) or request.form or request.values
    enabled = data.get('enabled', 'false').lower() == 'true'
    face_blur_enabled = enabled
    return jsonify({'success': True, 'face_blur_enabled': face_blur_enabled})

@app.route('/toggle_face_recognition', methods=['POST'])
def toggle_face_recognition():
    global face_recognition_enabled
    data = request.get_json(silent=True) or request.form or request.values
    enabled = data.get('enabled', 'false').lower() == 'true'
    face_recognition_enabled = enabled
    return jsonify({'success': True, 'face_recognition_enabled': face_recognition_enabled})

@app.route('/feature_status', methods=['GET'])
def feature_status():
    return jsonify({
        'success': True,
        'face_blur_enabled': face_blur_enabled,
        'face_recognition_enabled': face_recognition_enabled
    })

@app.route('/heatmap')
def heatmap():
    """Generate heatmap visualization from fight event locations"""
    global detector

    if detector is None:
        # Return blank image if no detector
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    analytics_copy = detector.analytics.copy()
    fight_events = analytics_copy.get('fight_events', [])

    if not fight_events:
        # Return blank heatmap
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', blank)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

    # Create heatmap from fight locations
    heatmap_img = np.zeros((480, 640), dtype=np.float32)

    for event in fight_events:
        # Extract location from event metadata
        # Default to center if no location data
        center_x = 320
        center_y = 240

        # Add Gaussian blob at fight location (50px radius)
        y, x = np.ogrid[-center_y:480-center_y, -center_x:640-center_x]
        mask = x*x + y*y <= 50*50
        heatmap_img[mask] += 1.0

    # Normalize to 0-255
    if heatmap_img.max() > 0:
        heatmap_img = (heatmap_img / heatmap_img.max() * 255).astype(np.uint8)
    else:
        heatmap_img = heatmap_img.astype(np.uint8)

    # Apply color map for visualization
    heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

    ret, buffer = cv2.imencode('.jpg', heatmap_colored)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/hotspots')
def hotspots():
    """Return JSON with hotspot coordinates and intensity"""
    global detector

    if detector is None:
        return jsonify({'success': True, 'hotspots': []})

    analytics_copy = detector.analytics.copy()
    fight_events = analytics_copy.get('fight_events', [])

    if not fight_events:
        return jsonify({'success': True, 'hotspots': []})

    # Cluster fight locations into hotspots using grid-based clustering
    # Divide into 4x4 grid for spatial grouping
    grid = {}

    for event in fight_events:
        # Use grid cell as key (simplified spatial clustering)
        cell_x = 1  # Default center-left
        cell_y = 1  # Default center-top
        key = f"{cell_x}_{cell_y}"

        if key not in grid:
            grid[key] = []
        grid[key].append(event)

    # Convert grid to hotspots list
    hotspots_list = []
    for key, events in grid.items():
        if len(events) > 0:
            cell_x, cell_y = map(int, key.split('_'))
            hotspots_list.append({
                'x': cell_x * 160 + 80,  # Grid cell center X
                'y': cell_y * 120 + 60,  # Grid cell center Y
                'intensity': min(1.0, len(events) / 10.0),
                'events': len(events)
            })

    return jsonify({'success': True, 'hotspots': hotspots_list})

# ============ Chatbot Routes ============
@app.route('/chatbot')
def chatbot_page():
    """Render chatbot page"""
    return render_template('chatbot.html')

@app.route('/chatbot/send', methods=['POST'])
def chatbot_send():
    """Handle chatbot message and get AI response"""
    if openai_client is None:
        return jsonify({
            'success': False,
            'error': 'Chatbot service is not configured. Please set OPENAI_API_KEY in environment variables.'
        })

    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        history = data.get('history', [])

        if not user_message:
            return jsonify({'success': False, 'error': 'Message is empty'})

        # Build conversation messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": """You are a Safety Assistant AI for ToBeLess, a violence detection and prevention system.
Your role is to:
- Provide helpful information about violence prevention strategies
- Explain safety best practices in various environments (schools, workplaces, public spaces)
- Discuss de-escalation techniques
- Offer guidance on security measures and emergency response
- Answer questions about the ToBeLess AI detection system
- Provide educational information about conflict resolution

Be professional, empathetic, and focused on safety and prevention.
If asked about harmful activities, redirect to constructive safety solutions.
Keep responses concise and actionable."""
            }
        ]

        # Add conversation history (limit to last 10 messages to manage context)
        for msg in history[-10:]:
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })

        # Get response from OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        return jsonify({
            'success': True,
            'response': assistant_message
        })

    except Exception as e:
        print(f"[Chatbot Error] {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to get response: {str(e)}'
        })


# ============ Report Generation Endpoints ============

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate a security report (PDF, Excel, or JSON)"""
    global detection_events, report_gen

    data = request.get_json(silent=True) or {}
    report_format = data.get('format', 'pdf').lower()

    # Collect statistics
    stats = {
        'fight_detections': detector.analytics.get('total_detections', 0) if detector else 0,
        'weapon_detections': weapon_detector.stats['total_detections'] if weapon_detector else 0,
        'fall_detections': fall_detector.stats['total_falls_detected'] if fall_detector else 0,
        'scream_detections': scream_detector.stats['total_screams_detected'] if scream_detector else 0
    }

    # Get events
    with detection_events_lock:
        events = detection_events.copy()

    try:
        if report_format == 'excel':
            filepath = report_gen.generate_excel_report(
                events=events,
                statistics=stats,
                start_time=datetime.now() - timedelta(hours=24),
                end_time=datetime.now()
            )
        elif report_format == 'json':
            filepath = report_gen.generate_json_report(
                events=events,
                statistics=stats,
                start_time=datetime.now() - timedelta(hours=24),
                end_time=datetime.now()
            )
        else:  # PDF
            filepath = report_gen.generate_pdf_report(
                events=events,
                statistics=stats,
                start_time=datetime.now() - timedelta(hours=24),
                end_time=datetime.now()
            )

        if filepath:
            return jsonify({
                'success': True,
                'filepath': filepath,
                'filename': Path(filepath).name,
                'format': report_format
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to generate {report_format.upper()} report. Check if required packages are installed.'
            })

    except Exception as e:
        print(f"[Report Error] {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/download_report/<filename>')
def download_report(filename):
    """Download a generated report"""
    reports_dir = Path("reports")
    filepath = reports_dir / secure_filename(filename)

    if filepath.exists():
        return send_from_directory(str(reports_dir), filename, as_attachment=True)
    else:
        return jsonify({'success': False, 'error': 'Report not found'}), 404


@app.route('/list_reports')
def list_reports():
    """List available reports"""
    reports = report_gen.get_available_reports()
    return jsonify({'success': True, 'reports': reports})


@app.route('/detection_events')
def get_detection_events():
    """Get all stored detection events"""
    with detection_events_lock:
        events = detection_events.copy()
    return jsonify({'success': True, 'events': events, 'count': len(events)})


@app.route('/clear_events', methods=['POST'])
def clear_events():
    """Clear all stored detection events"""
    global detection_events
    with detection_events_lock:
        detection_events = []
    return jsonify({'success': True, 'message': 'Events cleared'})


# ============ Detection Toggle Endpoints ============

@app.route('/toggle_weapon_detection', methods=['POST'])
def toggle_weapon_detection():
    """Toggle weapon detection on/off"""
    global weapon_detection_enabled
    weapon_detection_enabled = not weapon_detection_enabled
    return jsonify({
        'success': True,
        'weapon_detection_enabled': weapon_detection_enabled
    })


@app.route('/toggle_fall_detection', methods=['POST'])
def toggle_fall_detection():
    """Toggle fall detection on/off"""
    global fall_detection_enabled
    fall_detection_enabled = not fall_detection_enabled
    return jsonify({
        'success': True,
        'fall_detection_enabled': fall_detection_enabled
    })


@app.route('/toggle_scream_detection', methods=['POST'])
def toggle_scream_detection():
    """Toggle scream detection on/off"""
    global scream_detection_enabled, scream_detector
    scream_detection_enabled = not scream_detection_enabled

    # Start/stop the scream detector
    if scream_detection_enabled and scream_detector:
        scream_detector.start()
    elif scream_detector:
        scream_detector.stop()

    return jsonify({
        'success': True,
        'scream_detection_enabled': scream_detection_enabled
    })


@app.route('/detection_status')
def detection_status():
    """Get status of all detection modules"""
    return jsonify({
        'success': True,
        'fight_detection': True,  # Always enabled
        'weapon_detection': weapon_detection_enabled,
        'fall_detection': fall_detection_enabled,
        'scream_detection': scream_detection_enabled,
        'statistics': {
            'fights': detector.analytics.get('total_detections', 0) if detector else 0,
            'weapons': weapon_detector.stats['total_detections'] if weapon_detector else 0,
            'falls': fall_detector.stats['total_falls_detected'] if fall_detector else 0,
            'screams': scream_detector.stats['total_screams_detected'] if scream_detector else 0
        }
    })


if __name__ == "__main__":
    print("")
    print("=" * 60)
    print("  OPTIMIZED MODE - ToBeLess AI v2.1")
    print("=" * 60)
    print("  Fight Detection: YOLO-Pose + SlowFast (background thread)")
    print("  Weapon Detection: DISABLED")
    print("  Fall Detection: DISABLED")
    print("  Scream Detection: DISABLED")
    print("-" * 60)
    print(f"  RESIZE_WIDTH: {RESIZE_WIDTH}")
    print(f"  SKIP_FRAMES: {SKIP_FRAMES} (YOLO only; every frame to SlowFast)")
    print(f"  SlowFast: runs continuously in background thread")
    print(f"  Buffer size: 16 frames")
    print(f"  No batch accumulation (fresh frame every time)")
    print("-" * 60)
    print(f"  Faces folder: {FACES_DIR.resolve()}")
    print(f"  Upload folder: {UPLOAD_DIR.resolve()}")
    print("=" * 60)
    print("  Server starting on http://0.0.0.0:8080")
    print("=" * 60)
    print("")
    app.run(host="0.0.0.0", port=8080, debug=True)
