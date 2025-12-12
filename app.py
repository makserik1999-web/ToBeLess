#!/usr/bin/env python3
"""
Fight detection with YOLO-pose and automatic face recognition from faces/images folder
"""
from face_recognizer import FaceRecognizer
from face_blur import blur_faces
import cv2

# инициализация (один раз при старте)
face_rec = FaceRecognizer(yolo_model_path="yolov8n-face.pt", db_path="faces/embeddings.json", debug=True)
# если DB пустая — автоматически заполнить из faces/images (имена: Alex_1.jpg -> Alex)
face_rec.bulk_register_from_folder("faces/images")
face_blur_enabled = False
face_recognition_enabled = True  # default on

import os, time, uuid, json, threading, traceback
from pathlib import Path
try:
    import face_recognition
except Exception:
    face_recognition = None
    print("[WARNING] 'face_recognition' package not installed. dlib-based recognition disabled.")
    print("Install with: pip install face_recognition dlib -- or use InsightFace (pip install insightface onnxruntime)")
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
        # --- Face recognition init ---
        try:
            self.face_rec = FaceRecognizer(
                yolo_model_path="yolov8n-face.pt",
                db_path="faces/embeddings.json",
                debug=True
            )
        except Exception as e:
            print("[FightDetector] FaceRecognizer init failed:", e)
            self.face_recognizer = None
        self.last_recognition = {}

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
            return False, [], {'confidence': 0}
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
        global face_blur_enabled, face_recognition_enabled
        
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
        
        # ---- Face processing (blur and/or recognition) ----
        identified_names = []
        working_frame = frame.copy()
        
        # Apply face blur if enabled
        if face_blur_enabled and hasattr(self, 'face_rec') and self.face_rec is not None:
            try:
                working_frame, blur_results = blur_faces(working_frame, self.face_rec,
                                                         min_conf=0.5, blur_strength=25, expand=0.1)
                # If recognition is disabled, just show "Blurred" for all faces
                if not face_recognition_enabled:
                    identified_names = ["Blurred"] * len(poses)
            except Exception as e:
                print(f"[FightDetector] face blur error: {e}")
        
        # Face recognition (only if enabled and blur is off, or we want names despite blur)
        if face_recognition_enabled and hasattr(self, 'face_rec') and self.face_rec is not None:
            face_boxes = []
            
            # Get YOLO face detections
            try:
                yolo_face_boxes = self.face_rec.detect_faces(frame)
            except Exception:
                yolo_face_boxes = []
    
            for p in poses:
                name_for_person = "Unknown"
                person_bbox = self.get_bbox(p['keypoints'])
                face_crop = None
                
                # Match YOLO face box to person
                if person_bbox and yolo_face_boxes:
                    px1, py1, px2, py2 = person_bbox
                    best_box = None
                    
                    for (fx1, fy1, fx2, fy2, conf) in yolo_face_boxes:
                        cx = (fx1 + fx2) // 2
                        cy = (fy1 + fy2) // 2
                        if cx >= px1 and cx <= px2 and cy >= py1 and cy <= py2:
                            best_box = (fx1, fy1, fx2, fy2)
                            break
                        
                    if best_box is not None:
                        fx1, fy1, fx2, fy2 = best_box
                        fx1 = max(0, fx1)
                        fy1 = max(0, fy1)
                        fx2 = min(frame.shape[1] - 1, fx2)
                        fy2 = min(frame.shape[0] - 1, fy2)
                        face_crop = frame[fy1:fy2, fx1:fx2].copy()
    
                # Fallback: crop from keypoints
                if face_crop is None and p['keypoints'] is not None:
                    kp = p['keypoints']
                    try:
                        facial_points = []
                        facial_idx_map = {
                            'nose': 0, 'left_eye': 1, 'right_eye': 2,
                            'left_ear': 3, 'right_ear': 4
                        }
                        
                        for name, idx in facial_idx_map.items():
                            if idx < len(kp) and kp[idx][2] > 0.3:
                                facial_points.append((int(kp[idx][0]), int(kp[idx][1])))
                        
                        if facial_points:
                            xs = [pt[0] for pt in facial_points]
                            ys = [pt[1] for pt in facial_points]
                            
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            
                            width = max_x - min_x
                            height = max_y - min_y
                            
                            pad_x = max(int(width * 0.4), 30)
                            pad_y = max(int(height * 0.5), 40)
                            
                            x1 = max(0, min_x - pad_x)
                            y1 = max(0, min_y - pad_y)
                            x2 = min(frame.shape[1] - 1, max_x + pad_x)
                            y2 = min(frame.shape[0] - 1, max_y + pad_y)
                            
                            if (x2 - x1) >= 30 and (y2 - y1) >= 40:
                                face_crop = frame[y1:y2, x1:x2].copy()
                    except Exception as e:
                        print(f"[FightDetector] keypoint-based face crop error: {e}")
                        face_crop = None
    
                # Identify face
                if face_crop is not None:
                    try:
                        nm, dist = self.face_rec.identify(face_crop)
                        
                        # Temporal tracking
                        person_id = len(identified_names)
                        
                        if nm is not None and nm != "Unknown":
                            if not hasattr(self.face_rec, 'face_tracking'):
                                self.face_rec.face_tracking = {}
                            
                            if person_id not in self.face_rec.face_tracking:
                                self.face_rec.face_tracking[person_id] = {
                                    'name': nm,
                                    'conf': float(dist) if dist else 1.0,
                                    'frames': 1,
                                    'last_frame': frame_count
                                }
                            else:
                                tracked = self.face_rec.face_tracking[person_id]
                                if tracked['name'] == nm:
                                    tracked['frames'] += 1
                                    tracked['conf'] = 0.7 * tracked['conf'] + 0.3 * (float(dist) if dist else 1.0)
                                else:
                                    if float(dist) if dist else 1.0 > 0.35:
                                        nm = tracked['name']
                                    else:
                                        tracked['name'] = nm
                                tracked['last_frame'] = frame_count
                            
                            # Clean old tracking
                            to_remove = [pid for pid, track in self.face_rec.face_tracking.items()
                                        if frame_count - track['last_frame'] > 30]
                            for pid in to_remove:
                                del self.face_rec.face_tracking[pid]
                        
                        name_for_person = nm
                        face_boxes.append({'name': nm, 'dist': dist, 'crop_exists': True})
                        
                        # Telegram alert
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
                                    print("[FightDetector] telegram send error:", e)
                    except Exception as e:
                        print(f"[FightDetector] face identification error: {e}")
                        name_for_person = "Unknown"
                else:
                    name_for_person = "Unknown"
                    face_boxes.append({'name': name_for_person, 'dist': None, 'crop_exists': False})
    
                identified_names.append(name_for_person)
        else:
            # If recognition disabled, fill with placeholders
            identified_names = ["—"] * len(poses)
        
        # Use working_frame (blurred if enabled) for output
        out = working_frame

        # Detect fight
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

        # Store pose history
        self.pose_history.append(poses)

        # Draw skeletons on output
        for p in poses:
            color = (0, 0, 255) if fight_detected else (0, 255, 0)
            self.draw_skeleton(out, p['keypoints'], color)

        # Draw fight areas
        if fight_detected and fight_areas:
            for (x1, y1, x2, y2) in fight_areas:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(out, "FIGHT!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw identified names if available
        if identified_names:
            for i, p in enumerate(poses):
                if i < len(identified_names):
                    name = identified_names[i]
                    bbox = self.get_bbox(p['keypoints'])
                    if bbox and name and name != "—":
                        x1, y1, x2, y2 = bbox
                        label = f"{name}"
                        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Add metrics to return
        metrics['people_count'] = len(poses)
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
            if hasattr(detector, 'face_rec') and detector.face_rec:
                load_faces_from_folder(detector.face_rec, folder_path="faces/images")
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
    if detector is None or not hasattr(detector, 'face_rec') or detector.face_rec is None:
        return jsonify({'success': False, 'error': 'face_recognizer not initialized'})
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'success': False, 'error': 'provide file and name'})
    
    f = request.files['file']
    name = request.form['name']
    tmp = UPLOAD_DIR / f"tmp_{int(time.time())}_{secure_filename(f.filename)}"
    f.save(str(tmp))
    img = cv2.imread(str(tmp))
    ok, msg = detector.face_rec.register_face(name, img)
    try:
        tmp.unlink()
    except Exception:
        pass
    return jsonify({'success': ok, 'msg': msg})


@app.route('/reload_faces', methods=['POST'])
def reload_faces():
    global detector
    if detector is None or not hasattr(detector, 'face_rec'):
        return jsonify({'success': False, 'error': 'Detector not initialized'})

    try:
        load_faces_from_folder(detector.face_rec, folder_path="faces/images")
        return jsonify({
            'success': True,
            'database': list(detector.face_rec._mem_db.keys()),
            'count': len(detector.face_rec._mem_db)
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
