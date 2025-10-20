#!/usr/bin/env python3
"""
app.py — single-file fight detection server (YOLO-pose) with:
 - MJPEG /video_feed (annotated frames)
 - POST /start_stream (file upload or camera/URL)
 - POST /stop_stream
 - GET  /analytics  (snapshot, non-destructive)
 - GET  /stats_stream (SSE pushing latest stats each SSE_INTERVAL)
 - POST /settings (adjust thresholds)
 - job management for uploaded files -> uploads/job_<id>_result.json
"""
import os, time, uuid, json, threading, traceback
from pathlib import Path
from collections import deque
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import cv2, numpy as np

# Optional bot helpers (no-op if missing)
try:
    from bot import send_alert, send_photo
except Exception:
    def send_alert(*a, **k): pass
    def send_photo(*a, **k): pass

# ------------ Config ------------
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
ALERT_COOLDOWN = 8               # seconds between telegram alerts
ANALYTICS_SNAPSHOT_SIZE = 300
SKIP_FRAMES = 1                  # process every SKIP_FRAMES-th frame
RESIZE_WIDTH = 640               # set None to disable resize
SSE_INTERVAL = 0.6               # seconds between SSE pushes

# ------------- Detector (YOLO-pose) -------------
try:
    import torch
    from ultralytics import YOLO
except Exception:
    YOLO = None

class FightDetector:
    def __init__(self, model_path="yolov8n-pose.pt", device=None):
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
            raise RuntimeError("ultralytics YOLO not available; install ultralytics and torch")

        print(f"[FightDetector] loading pose model '{model_path}' on {self.device}")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # thresholds
        self.body_proximity_threshold = 120.0
        self.limb_proximity_threshold = 50.0
        self.fight_hold_duration = 60  # frames

        self.fight_detected = False
        self.fight_start_time = 0
        self.last_fight_detection = 0
        self.pose_history = deque(maxlen=512)

        self.analytics = {
            'total_detections': 0,
            'fight_duration_history': [],
            'people_count_history': [],          # {frame,count,timestamp}
            'detection_confidence_history': [],  # {frame,confidence,timestamp}
            'fight_events': []                   # {start_frame,start_time,confidence, duration?}
        }

        # COCO keypoint indices
        self.KP = {'nose':0,'left_eye':1,'right_eye':2,'left_ear':3,'right_ear':4,
                   'left_shoulder':5,'right_shoulder':6,'left_elbow':7,'right_elbow':8,
                   'left_wrist':9,'right_wrist':10,'left_hip':11,'right_hip':12,
                   'left_knee':13,'right_knee':14,'left_ankle':15,'right_ankle':16}

    def _dist(self,a,b):
        return float(np.linalg.norm(np.array(a)-np.array(b)))

    def get_person_center(self, kp):
        torso = [self.KP['left_shoulder'], self.KP['right_shoulder'], self.KP['left_hip'], self.KP['right_hip']]
        pts=[]
        for k in torso:
            if k < len(kp) and kp[k][2] > 0.5:
                pts.append(kp[k][:2])
        if pts:
            return np.mean(np.array(pts), axis=0)
        return None

    def get_bbox(self,kp):
        pts=[p[:2] for p in kp if p[2]>0.5]
        if not pts: return None
        arr=np.array(pts); x1,y1=arr.min(axis=0); x2,y2=arr.max(axis=0)
        return int(x1),int(y1),int(x2),int(y2)

    def _skeleton_lines(self,kp):
        conns = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        lines=[]
        for a,b in conns:
            if a < len(kp) and b < len(kp) and kp[a][2]>0.5 and kp[b][2]>0.5:
                lines.append(((int(kp[a][0]),int(kp[a][1])),(int(kp[b][0]),int(kp[b][1]))))
        return lines

    def _line_intersect(self,p1,p2,p3,p4):
        def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def check_limb_crossings(self,kp1,kp2):
        l1=self._skeleton_lines(kp1); l2=self._skeleton_lines(kp2)
        cnt=0
        for a1,b1 in l1:
            for a2,b2 in l2:
                if self._line_intersect(a1,b1,a2,b2): cnt+=1
        return cnt

    def check_close_limbs(self,kp1,kp2):
        limb_keys=[self.KP['left_wrist'],self.KP['right_wrist'],self.KP['left_elbow'],self.KP['right_elbow'],
                   self.KP['left_knee'],self.KP['right_knee'],self.KP['left_ankle'],self.KP['right_ankle']]
        contacts=0; min_d=float('inf')
        for a in limb_keys:
            if a < len(kp1) and kp1[a][2]>0.5:
                for b in limb_keys:
                    if b < len(kp2) and kp2[b][2]>0.5:
                        d=self._dist(kp1[a][:2], kp2[b][:2]); min_d=min(min_d,d)
                        if d < self.limb_proximity_threshold: contacts+=1
        return contacts, (min_d if min_d!=float('inf') else None)

    def detect_fight(self, poses, frame_count):
        if len(poses) < 2:
            return False, [], {'confidence':0}
        detected=False; areas=[]; metrics={'body_distances':[], 'limb_crossings':0, 'close_contacts':0, 'confidence':0}
        for i in range(len(poses)):
            for j in range(i+1,len(poses)):
                k1=poses[i]['keypoints']; k2=poses[j]['keypoints']
                c1=self.get_person_center(k1); c2=self.get_person_center(k2)
                body_close=False
                if c1 is not None and c2 is not None:
                    d=self._dist(c1,c2); metrics['body_distances'].append(d)
                    if d < self.body_proximity_threshold: body_close=True
                crosses=self.check_limb_crossings(k1,k2); metrics['limb_crossings']+=crosses
                close_contacts, min_limb = self.check_close_limbs(k1,k2); metrics['close_contacts']+=close_contacts
                if body_close or crosses>0 or close_contacts>0:
                    detected=True
                    b1=self.get_bbox(k1); b2=self.get_bbox(k2)
                    if b1 and b2:
                        x1=max(0,min(b1[0],b2[0])-20); y1=max(0,min(b1[1],b2[1])-20)
                        x2=max(b1[2],b2[2])+20; y2=max(b1[3],b2[3])+20
                        areas.append((x1,y1,x2,y2))
        conf=0.0
        if metrics['body_distances']:
            avg_d=float(np.mean(metrics['body_distances']))
            conf+=max(0.0,(self.body_proximity_threshold-avg_d)/self.body_proximity_threshold*40.0)
        conf+=min(metrics['limb_crossings']*20.0,30.0)
        conf+=min(metrics['close_contacts']*10.0,30.0)
        metrics['confidence']=min(conf,100.0)
        if detected:
            self.analytics['total_detections'] += 1
            self.analytics['detection_confidence_history'].append({'frame':frame_count,'confidence':metrics['confidence'],'timestamp':datetime.now().isoformat()})
        return detected, areas, metrics

    def draw_skeleton(self, frame, kps, color=(0,255,0)):
        conns = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        for kp in kps:
            if kp[2] > 0.5:
                cv2.circle(frame,(int(kp[0]),int(kp[1])),3,color,-1)
        for a,b in conns:
            if a < len(kps) and b < len(kps) and kps[a][2]>0.5 and kps[b][2]>0.5:
                p1=(int(kps[a][0]),int(kps[a][1])); p2=(int(kps[b][0]),int(kps[b][1]))
                cv2.line(frame,p1,p2,color,2)

    def process_frame(self, frame, frame_count):
        # optional resize for speed
        proc = frame
        if RESIZE_WIDTH:
            h,w = frame.shape[:2]
            if w != RESIZE_WIDTH:
                scale = RESIZE_WIDTH / float(w); new_h = int(h*scale)
                proc = cv2.resize(frame, (RESIZE_WIDTH, new_h))
        try:
            results = self.model(proc, verbose=False)
        except Exception as e:
            return frame.copy(), {}
        poses=[]
        for r in results:
            if getattr(r,"keypoints",None) is not None:
                for kp in r.keypoints.data:
                    arr=kp.cpu().numpy().copy()
                    if RESIZE_WIDTH:
                        sx = frame.shape[1] / float(proc.shape[1]); sy = frame.shape[0] / float(proc.shape[0])
                        arr[:,0]*=sx; arr[:,1]*=sy
                    poses.append({'keypoints': arr})
        self.pose_history.append(poses)
        self.analytics['people_count_history'].append({'frame':frame_count,'count':len(poses),'timestamp':datetime.now().isoformat()})
        fight, areas, metrics = self.detect_fight(poses, frame_count)
        if fight:
            if not self.fight_detected:
                self.fight_start_time = frame_count
                self.analytics['fight_events'].append({'start_frame':frame_count,'start_time':datetime.now().isoformat(),'confidence':metrics.get('confidence',0)})
            self.fight_detected = True; self.last_fight_detection = frame_count
        else:
            if self.fight_detected:
                frames_since = frame_count - self.fight_start_time
                frames_last = frame_count - self.last_fight_detection
                if frames_since >= self.fight_hold_duration and frames_last > 10:
                    dur_s = frames_since/30.0
                    self.analytics['fight_duration_history'].append(dur_s)
                    if self.analytics['fight_events']:
                        self.analytics['fight_events'][-1]['duration'] = dur_s
                    self.fight_detected = False
        out = frame.copy()
        for p in poses:
            color = (0,0,255) if self.fight_detected else (0,255,0)
            self.draw_skeleton(out,p['keypoints'], color=color)
        if self.fight_detected:
            dur=(frame_count-self.fight_start_time)/30.0
            cv2.putText(out, f"FIGHT DETECTED ({dur:.1f}s)", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),3,cv2.LINE_AA)
            for a in areas:
                cv2.rectangle(out,(a[0],a[1]),(a[2],a[3]),(0,0,255),3)
        else:
            cv2.putText(out, "Normal", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2,cv2.LINE_AA)
        cv2.putText(out, f"Persons: {len(poses)}", (30,out.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
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
latest_stats = {'people':0,'fights':0,'fps':0,'confidence':0.0,'timestamp':None}
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
        try: send_alert(text)
        except Exception: pass
        if frame_path:
            try: send_photo(frame_path, caption or "")
            except Exception: pass
    t = threading.Thread(target=job, daemon=True); t.start()

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
                time.sleep(0.02); continue
            frame_count += 1
            do_proc = (SKIP_FRAMES <= 1) or (frame_count % SKIP_FRAMES == 0)
            out_frame = None; metrics = {}
            if do_proc:
                try:
                    out_frame, metrics = detector.process_frame(frame, frame_count)
                    last_annot = out_frame.copy()
                except Exception as e:
                    out_frame = frame.copy(); metrics = {}
            else:
                out_frame = last_annot.copy() if last_annot is not None else frame.copy()

            # alerting
            if detector.fight_detected and (time.time() - last_alert_time > ALERT_COOLDOWN):
                snap = UPLOAD_DIR / f"alert_{int(time.time())}.jpg"
                try:
                    cv2.imwrite(str(snap), out_frame); snap_path=str(snap)
                except Exception:
                    snap_path = None
                txt = f"🚨 Fight detected! conf={metrics.get('confidence','N/A')} people={len(detector.pose_history[-1]) if detector.pose_history else 0}"
                _send_alert_nonblocking(txt, snap_path, caption=txt)
                last_alert_time = time.time()

            # publish current frame
            with frame_lock:
                current_frame = out_frame

            # update analytics structures
            try:
                people = len(detector.pose_history[-1]) if detector.pose_history else 0
                fight_flag = bool(detector.fight_detected)
                # fps estimate: running average each N frames
                processed += 1
                now = time.time()
                elapsed = now - t0
                if elapsed >= 0.5:
                    fps_est = int(round(processed / elapsed)) if elapsed > 0 else latest_stats.get('fps',0)
                    processed = 0
                    t0 = now
                else:
                    fps_est = latest_stats.get('fps',0)

                snap = {'frame':frame_count, 'fight':fight_flag, 'people':people, 'metrics':metrics, 'timestamp':datetime.now().isoformat()}
                analytics_buffer.append(snap)

                # update latest_stats atomically
                with latest_stats_lock:
                    latest_stats['people'] = people
                    latest_stats['fights'] = detector.analytics.get('total_detections',0)
                    latest_stats['fps'] = fps_est
                    # confidence: prefer metrics.confidence, fallback to last detection history
                    conf = float(metrics.get('confidence', 0.0))
                    if conf <= 0:
                        # fallback to last history if available
                        if detector.analytics.get('detection_confidence_history'):
                            conf = float(detector.analytics['detection_confidence_history'][-1].get('confidence',0.0))
                    latest_stats['confidence'] = conf
                    latest_stats['timestamp'] = snap['timestamp']
            except Exception:
                pass

    except Exception as e:
        print("[processing_loop] fatal:", e, traceback.format_exc())
    finally:
        stream_active = False
        try:
            if video_cap: video_cap.release()
        except Exception:
            pass
        # finalize job result if file
        if source_is_file and job_id:
            try:
                res = {'job_id': job_id, 'analytics': detector.analytics if detector else {}, 'ended_at': datetime.now().isoformat()}
                write_job_result(job_id, res)
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]['status']='finished'; JOBS[job_id]['result_path']=str(UPLOAD_DIR / f"job_{job_id}_result.json")
            except Exception as e:
                print("[warn] finalize job write failed:", e)
        print("[info] processing loop ended")

# ------------- MJPEG generator -------------
def frame_generator():
    global current_frame, stream_active
    while True:
        with frame_lock:
            frm = current_frame.copy() if current_frame is not None else None
        if frm is None:
            if not stream_active:
                time.sleep(0.1); continue
            time.sleep(0.03); continue
        try:
            _,buff = cv2.imencode(".jpg", frm)
            b = buff.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b + b'\r\n')
        except Exception:
            time.sleep(0.03); continue

# ------------- Flask routes -------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global detector, video_cap, proc_thread, stream_active
    if detector is None:
        try:
            detector = FightDetector()
        except Exception as e:
            return jsonify({'success':False,'error':f'Failed to init detector: {e}'})
    source_is_file=False; job_id=None
    if 'file' in request.files and request.files['file'].filename != '':
        f = request.files['file']; fn = secure_filename(f.filename)
        saved = UPLOAD_DIR / f"{int(time.time())}_{fn}"
        try:
            f.save(str(saved))
        except Exception as e:
            return jsonify({'success':False,'error':f'Failed save: {e}'})
        source = str(saved); source_is_file=True; job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {'status':'running','video':source,'started_at':time.time(),'result_path':None}
    else:
        data = request.get_json(silent=True) or request.form or request.values
        source = data.get('source','0')

    try:
        if stream_active:
            stream_active=False
            if proc_thread and proc_thread.is_alive(): proc_thread.join(timeout=1.0)
        # open capture
        if isinstance(source,str) and source.isdigit():
            idx = int(source); video_cap = cv2.VideoCapture(idx)
        else:
            video_cap = cv2.VideoCapture(source)
        if not video_cap.isOpened():
            return jsonify({'success':False,'error':'Could not open source'})
        try:
            video_cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        except Exception:
            pass
        stream_active=True
        proc_thread = threading.Thread(target=processing_loop, args=(source_is_file, job_id), daemon=True)
        proc_thread.start()
        return jsonify({'success':True,'streaming':True,'stream_url':'/video_feed','job_id':job_id})
    except Exception as e:
        return jsonify({'success':False,'error':str(e),'trace':traceback.format_exc()})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_active, proc_thread, video_cap
    stream_active=False
    if proc_thread and proc_thread.is_alive(): proc_thread.join(timeout=1.0)
    try:
        if video_cap: video_cap.release()
    except Exception:
        pass
    return jsonify({'success':True})

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics')
def analytics():
    recent = list(analytics_buffer)[-ANALYTICS_SNAPSHOT_SIZE:]
    with latest_stats_lock:
        ls = dict(latest_stats)
    analytics_copy = detector.analytics.copy() if detector else {}
    return jsonify({'success':True,'streaming':bool(stream_active),'recent_data':recent,'analytics':analytics_copy,'latest_stats':ls})

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
            try: return jsonify({'success':True,'status':'finished','analysis':json.loads(p.read_text(encoding='utf-8'))})
            except Exception: pass
        return jsonify({'success':False,'error':'Unknown job_id'}),404
    p = UPLOAD_DIR / f"job_{job_id}_result.json"
    res=None
    if p.exists():
        try: res=json.loads(p.read_text(encoding='utf-8'))
        except Exception: res=None
    return jsonify({'success':True,'status':meta.get('status'),'job':meta,'analysis':res})

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(str(UPLOAD_DIR.resolve()), filename, as_attachment=False)

@app.route('/settings', methods=['POST'])
def settings():
    global detector
    if not detector: return jsonify({'success':False,'error':'No detector running'})
    data = request.get_json(silent=True) or request.form or request.values
    try:
        if 'body_proximity_threshold' in data: detector.body_proximity_threshold = float(data['body_proximity_threshold'])
        if 'limb_proximity_threshold' in data: detector.limb_proximity_threshold = float(data['limb_proximity_threshold'])
        if 'fight_hold_duration' in data: detector.fight_hold_duration = int(data['fight_hold_duration'])
    except Exception:
        pass
    return jsonify({'success':True})

# ------------- main -------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
