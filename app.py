#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Super app.py — объединённый, улучшенный и более устойчивый сервер для ToBeLess AI
Включает:
- Advanced fight detection (pose-based, YOLO pose)
- Hazard detection (YOLO object + improved color/texture checks for fire/smoke)
- Heatmap генератор
- MJPEG stream, analytics, SSE stats, job handling, alerts via bot.py (если есть)
- Параметры и пороги скорректированы для уменьшения ложных срабатываний и повышения чувствительности к дракам
"""
import os, time, uuid, json, threading, traceback
from pathlib import Path
from collections import deque
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, send_file
from werkzeug.utils import secure_filename
import cv2, numpy as np


# === GLOBALS FOR LIVE STATS STREAM ===
SSE_INTERVAL = 0.8  # интервал в секундах
latest_stats = {}
latest_stats_lock = threading.Lock()

# optional telegram bot helpers (non-blocking)
try:
    from bot import send_alert, send_photo
except Exception:
    def send_alert(*a, **k): pass
    def send_photo(*a, **k): pass

# config
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
ALERT_COOLDOWN = 8.0
ANALYTICS_SNAPSHOT_SIZE = 400
SKIP_FRAMES = 1
RESIZE_WIDTH = 640
SSE_INTERVAL = 0.6
MJPEG_JPEG_QUALITY = 85

# try to import ultralytics YOLO
try:
    import torch
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ------------------ AdvancedFightDetector (YOLO-pose based) ------------------
class AdvancedFightDetector:
    def __init__(self, model_path="yolov8n-pose.pt", device=None):
        if YOLO is None:
            raise RuntimeError("ultralytics YOLO not available; install ultralytics")
        # choose device
        try:
            if device is None:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
        except Exception:
            device = "cpu"
        self.device = device
        print(f"[AdvancedFightDetector] Loading pose model '{model_path}' on {self.device}")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # COCO keypoints mapping
        self.KP = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

        # thresholds - tuned
        self.body_proximity_threshold = 120.0
        self.strike_velocity_threshold = 18.0   # reduced to detect quicker smaller movements
        self.aggressive_distance = 80.0
        self.ground_height_threshold = 0.7

        # temporal params
        self.min_fight_frames = 10   # fewer frames to confirm a fight (smoother buffer still applies)
        self.cooldown_frames = 25
        self.fight_confidence_buffer = deque(maxlen=20)
        self.pose_history = deque(maxlen=180)
        self.tension_score_history = deque(maxlen=40)
        self.person_tracks = {}
        self.analytics = {
            'total_detections': 0,
            'strike_count': 0,
            'fall_count': 0,
            'fight_events': [],
            'conflict_types': {'minor_scuffle':0,'active_fight':0,'group_conflict':0,'critical':0},
            'escalation_warnings': 0,
            'people_count_history': []
        }
        self.fight_detected = False
        self.escalation_warning = False
        self.consecutive_fight_frames = 0
        self.consecutive_normal_frames = 0

    # small helpers
    def _dist(self, a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _get_kp(self, kp, name, min_conf=0.45):
        idx = self.KP.get(name)
        if idx is None or idx >= len(kp): return None
        pt = kp[idx]
        if pt[2] < min_conf: return None
        return pt[:2]

    def _person_center(self, kp):
        pts = []
        for name in ('left_shoulder','right_shoulder','left_hip','right_hip'):
            p = self._get_kp(kp, name)
            if p is not None: pts.append(p)
        if len(pts) >= 2: return np.mean(np.array(pts),axis=0)
        return None

    def _bbox_from_kp(self, kp):
        pts = [p[:2] for p in kp if p[2] > 0.45]
        if not pts: return None
        arr = np.array(pts)
        x1,y1 = arr.min(axis=0); x2,y2 = arr.max(axis=0)
        return (int(x1),int(y1),int(x2),int(y2))

    def _calc_limb_velocity(self, kp_prev, kp_curr, limb_name):
        p_prev = self._get_kp(kp_prev, limb_name)
        p_curr = self._get_kp(kp_curr, limb_name)
        if p_prev is None or p_curr is None: return 0.0
        return self._dist(p_prev, p_curr)

    def _update_tracks(self, poses):
        # simple greedy matching by nearest center
        centers = []
        for pose in poses:
            c = self._person_center(pose['keypoints'])
            centers.append((pose, c))
        new_tracks = {}
        used = set()
        # match existing
        for tid, t in list(self.person_tracks.items()):
            last_center = t['centers'][-1] if t['centers'] else None
            if last_center is None: continue
            best = None; bestd = float('inf')
            for idx, (pose, c) in enumerate(centers):
                if idx in used or c is None: continue
                d = self._dist(last_center, c)
                if d < 160 and d < bestd:
                    bestd = d; best = (idx, pose, c)
            if best:
                idx, pose, c = best
                used.add(idx)
                new_tracks[tid] = {
                    'history': (t['history'] + [pose['keypoints']])[-120:],
                    'centers': (t['centers'] + [c])[-120:]
                }
        # add unmatched as new ids
        next_id = (max(self.person_tracks.keys())+1) if self.person_tracks else 1
        for idx, (pose, c) in enumerate(centers):
            if idx in used: continue
            new_tracks[next_id] = {'history':[pose['keypoints']],'centers':[c]}
            next_id += 1
        self.person_tracks = new_tracks

    def _calc_tension(self, poses):
        if len(poses) < 2: return 0.0
        t = 0.0
        for i in range(len(poses)):
            for j in range(i+1,len(poses)):
                k1 = poses[i]['keypoints']; k2 = poses[j]['keypoints']
                c1 = self._person_center(k1); c2 = self._person_center(k2)
                if c1 is None or c2 is None: continue
                d = self._dist(c1,c2)
                if d < 150: t += 10
                if d < 100: t += 15
                # aggressive poses add weight
                ag1 = self._aggressive_score(k1); ag2 = self._aggressive_score(k2)
                t += 0.25*(ag1+ag2)
        return min(t,100.0)

    def _aggressive_score(self, kp):
        score = 0
        # raised wrists
        lw = self._get_kp(kp,'left_wrist'); ls = self._get_kp(kp,'left_shoulder')
        rw = self._get_kp(kp,'right_wrist'); rs = self._get_kp(kp,'right_shoulder')
        if lw is not None and ls is not None and lw[1] < ls[1]-15: score += 12
        if rw is not None and rs is not None and rw[1] < rs[1]-15: score += 12
        # forward arm (straight)
        try:
            for side in [('left_shoulder','left_elbow','left_wrist'),('right_shoulder','right_elbow','right_wrist')]:
                a,b,c = side
                pa = self._get_kp(kp,a); pb = self._get_kp(kp,b); pc = self._get_kp(kp,c)
                if pa is not None and pb is not None and pc is not None:
                    v1 = np.array(pb)-np.array(pa); v2 = np.array(pc)-np.array(pb)
                    denom = (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
                    cosv = np.clip(np.dot(v1,v2)/denom, -1,1)
                    ang = np.degrees(np.arccos(cosv))
                    if ang > 150: score += 10
        except Exception:
            pass
        return score

    def _person_down(self, kp):
        nose = self._get_kp(kp,'nose'); lh = self._get_kp(kp,'left_hip'); rh = self._get_kp(kp,'right_hip')
        if nose is None or (lh is None and rh is None): return False
        hip = lh if lh is not None else rh
        return abs(nose[1]-hip[1]) < 60

    def process_frame(self, frame, frame_count):
        proc = frame
        if RESIZE_WIDTH:
            h,w = frame.shape[:2]
            if w != RESIZE_WIDTH:
                scale = RESIZE_WIDTH/float(w)
                new_h = int(h*scale)
                proc = cv2.resize(frame,(RESIZE_WIDTH,new_h))

        # inference
        try:
            results = self.model(proc, verbose=False)
        except Exception as e:
            # inference failed -> return no detection but don't crash
            return frame.copy(), {'detected':False,'areas':[],'confidence':0,'people_count':0,'conflict_type':'unknown','escalation_warning':{'active':False},'tension_score':0,'metrics':{}}

        poses = []
        for r in results:
            if getattr(r,"keypoints",None) is not None:
                for kp in r.keypoints.data:
                    arr = kp.cpu().numpy().copy()
                    if RESIZE_WIDTH:
                        sx = frame.shape[1]/float(proc.shape[1]); sy = frame.shape[0]/float(proc.shape[0])
                        arr[:,0] *= sx; arr[:,1] *= sy
                    poses.append({'keypoints':arr})

        # bookkeeping
        self.pose_history.append(poses)
        self.analytics['people_count_history'].append({'frame':frame_count,'count':len(poses),'timestamp':datetime.now().isoformat()})
        self._update_tracks(poses)
        tension = self._calc_tension(poses)
        self.tension_score_history.append(tension)

        # fight detection logic
        if len(poses) < 2:
            self.consecutive_normal_frames += 1
            self.consecutive_fight_frames = 0
            smoothed = np.mean(list(self.fight_confidence_buffer)) if self.fight_confidence_buffer else 0
            # draw skeletons only
            out = frame.copy()
            for p in poses: self._draw_skeleton(out,p['keypoints'],(0,255,0))
            return out, {'detected':False,'areas':[],'confidence':smoothed,'people_count':len(poses),'conflict_type':'unknown','escalation_warning':{'active':False},'tension_score':tension,'metrics':{}}

        fight_score = 0.0
        areas = []
        metrics = {'strikes':0,'defensive_poses':0,'aggressive_poses':0,'falls':0,'close_proximity':0}
        # pairwise analysis
        for i in range(len(poses)):
            for j in range(i+1,len(poses)):
                k1 = poses[i]['keypoints']; k2 = poses[j]['keypoints']
                c1 = self._person_center(k1); c2 = self._person_center(k2)
                if c1 is None or c2 is None: continue
                d = self._dist(c1,c2)
                if d < self.body_proximity_threshold:
                    fight_score += 22; metrics['close_proximity'] += 1
                # strikes: check velocities comparing with tracks history if available
                # check for both persons in tracks
                # person id mapping: naive mapping by index -> attempt to find matching track
                for pid, track in self.person_tracks.items():
                    # use last center similarity to map
                    if not track['centers']: continue
                    if self._dist(track['centers'][-1], c1) < 120:
                        # compute limb velocity using last pose in history
                        if len(track['history']) >= 2:
                            kp_prev = track['history'][-2]; kp_curr = k1
                            for limb in ('left_wrist','right_wrist','left_elbow','right_elbow'):
                                vel = self._calc_limb_velocity(kp_prev,kp_curr,limb)
                                if vel > self.strike_velocity_threshold:
                                    fight_score += 28; metrics['strikes'] += 1; self.analytics['strike_count'] += 1
                        break
                # defensive/aggressive poses
                if self._aggressive_score(k1) > 0: fight_score += 8; metrics['aggressive_poses'] += 1
                if self._aggressive_score(k2) > 0: fight_score += 8; metrics['aggressive_poses'] += 1
                # falls
                if self._person_down(k1): fight_score += 35; metrics['falls'] += 1; self.analytics['fall_count'] = self.analytics.get('fall_count',0)+1
                if self._person_down(k2): fight_score += 35; metrics['falls'] += 1; self.analytics['fall_count'] = self.analytics.get('fall_count',0)+1
                # area inclusion once score high
                if fight_score > 35:
                    b1 = self._bbox_from_kp(k1); b2 = self._bbox_from_kp(k2)
                    if b1 and b2:
                        x1 = min(b1[0],b2[0])-28; y1 = min(b1[1],b2[1])-28; x2 = max(b1[2],b2[2])+28; y2 = max(b1[3],b2[3])+28
                        areas.append((int(max(0,x1)),int(max(0,y1)),int(min(frame.shape[1],x2)),int(min(frame.shape[0],y2))))

        confidence = min(fight_score,100.0)
        self.fight_confidence_buffer.append(confidence)
        smoothed = float(np.mean(list(self.fight_confidence_buffer))) if self.fight_confidence_buffer else confidence

        # temporal smoothing decision
        detected = False
        if smoothed > 35.0:  # lowered threshold for sensitivity, but buffered smoothing prevents flicker
            self.consecutive_fight_frames += 1
            self.consecutive_normal_frames = 0
            if self.consecutive_fight_frames >= self.min_fight_frames:
                detected = True
                if not self.fight_detected:
                    self.fight_detected = True
                    self.analytics['total_detections'] += 1
                    self.analytics['fight_events'].append({'start_frame':frame_count,'start_time':datetime.now().isoformat(),'confidence':smoothed})
        else:
            self.consecutive_fight_frames = 0
            self.consecutive_normal_frames += 1
            if self.fight_detected and self.consecutive_normal_frames > self.cooldown_frames:
                self.fight_detected = False

        # classify conflict
        conflict_type = 'unknown'
        if metrics['falls'] > 0: conflict_type = 'critical'
        elif len(poses) > 2 and smoothed > 45: conflict_type = 'group_conflict'
        elif metrics['strikes'] > 0 or smoothed > 55: conflict_type = 'active_fight'
        elif smoothed > 40: conflict_type = 'minor_scuffle'

        # escalation check
        escalating = False
        if len(self.tension_score_history) >= 15:
            recent = np.mean(list(self.tension_score_history)[-5:])
            older = np.mean(list(self.tension_score_history)[:5]) if len(self.tension_score_history) > 10 else 0
            growth = recent - older
            if recent > 30 and growth > 8:
                escalating = True
                if not self.escalation_warning:
                    self.analytics['escalation_warnings'] += 1
                self.escalation_warning = True
            else:
                self.escalation_warning = False

        # draw annotated frame
        out = frame.copy()
        for p in poses:
            self._draw_skeleton(out,p['keypoints'],(0,0,255) if detected else (0,200,0))
        if detected:
            for a in areas:
                cv2.rectangle(out,(a[0],a[1]),(a[2],a[3]),(0,0,255),3)
        text = (f"{conflict_type.upper().replace('_',' ')} - {smoothed:.0f}%") if detected else ("NORMAL")
        color = (0,0,255) if detected else (0,200,0)
        cv2.putText(out,text,(18,46),cv2.FONT_HERSHEY_SIMPLEX,1.0,color,3,cv2.LINE_AA)
        # info panel
        info_y = out.shape[0]-100
        cv2.putText(out,f"People: {len(poses)}",(20,info_y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(out,f"Tension: {tension:.0f}",(20,info_y+28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(out,f"Strikes: {self.analytics.get('strike_count',0)}",(20,info_y+56),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        return out, {'detected':detected,'areas':areas,'confidence':smoothed,'people_count':len(poses),'conflict_type':conflict_type,'escalation_warning':{'active':escalating},'tension_score':tension,'metrics':metrics}

    def _draw_skeleton(self, frame, kp, color):
        conns = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        for p in kp:
            if p[2] > 0.45:
                cv2.circle(frame,(int(p[0]),int(p[1])),3,color,-1)
        for a,b in conns:
            if a < len(kp) and b < len(kp) and kp[a][2]>0.45 and kp[b][2]>0.45:
                cv2.line(frame,(int(kp[a][0]),int(kp[a][1])),(int(kp[b][0]),int(kp[b][1])),color,2)

    def get_analytics(self):
        return self.analytics

# ---------------- HazardDetector (YOLO + improved color) ----------------
class HazardDetector:
    def __init__(self, obj_model_path='yolov8n.pt'):
        self.enabled = False
        self.model = None
        self.statistics = {'fire_events':0,'smoke_events':0,'weapon_events':0}
        self.weapon_classes = set(['knife','scissors'])  # COCO names fallback
        if YOLO is not None:
            try:
                self.model = YOLO(obj_model_path)
                self.enabled = True
                print(f"[HazardDetector] Loaded object model '{obj_model_path}'")
            except Exception as e:
                print(f"[HazardDetector] Failed to load object model: {e}")
                self.enabled = False

    def detect(self, frame):
        results = {}
        # try model-based detection for weapons
        if self.enabled and self.model is not None:
            try:
                dets = self.model(frame, verbose=False)
                for res in dets:
                    if getattr(res, 'boxes', None) is None: continue
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        if conf < 0.35: continue
                        cls_id = int(box.cls[0])
                        name = res.names.get(cls_id, str(cls_id))
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1,y1,x2,y2 = map(int,xyxy)
                        if name in self.weapon_classes:
                            results.setdefault('weapon', []).append({'bbox':(x1,y1,x2,y2),'confidence':conf*100,'class':name})
                            self.statistics['weapon_events'] += 1
            except Exception as e:
                # model failed - fallback
                pass

        # advanced color/texture fire detection (tuned)
        fire_detections = self._detect_fire_advanced(frame)
        if fire_detections:
            results['fire'] = fire_detections
            self.statistics['fire_events'] += len(fire_detections)
        # advanced smoke detection (tuned stricter)
        smoke_detections = self._detect_smoke_strict(frame)
        if smoke_detections:
            results['smoke'] = smoke_detections
            self.statistics['smoke_events'] += len(smoke_detections)

        return results

    def _detect_fire_advanced(self, frame):
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # tuned ranges
        lower1 = np.array([0,150,160]); upper1 = np.array([18,255,255])
        lower2 = np.array([18,120,160]); upper2 = np.array([35,255,255])
        mask1 = cv2.inRange(hsv, lower1, upper1); mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h,w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 900: continue
            x,y,ww,hh = cv2.boundingRect(cnt)
            aspect = hh/float(ww) if ww>0 else 0
            if aspect < 0.4 or aspect > 6: continue
            roi = frame[y:y+hh, x:x+ww]
            if roi.size == 0: continue
            bright = float(np.mean(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)))
            if bright < 110: continue
            confidence = min(100.0, (area/1500.0)*(bright/2.0))
            if confidence > 40:
                detections.append({'bbox':(x,y,x+ww,y+hh),'confidence':confidence,'area':area})
        return detections if detections else None

    def _detect_smoke_strict(self, frame):
        # stronger smoke detection to reduce false positives
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # gray-ish low saturation areas with certain brightness
        lower = np.array([0,0,80]); upper = np.array([180,60,220])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections=[]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2500: continue   # stricter area
            x,y,ww,hh = cv2.boundingRect(cnt)
            roi = frame[y:y+hh, x:x+ww]
            if roi.size==0: continue
            # check texture using Laplacian variance (smoke has low texture)
            gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            var = cv2.Laplacian(gray,cv2.CV_64F).var()
            if var > 120: continue
            confidence = min(100.0, (area/3000.0)*(100.0/(var+1.0)))
            if confidence > 35:
                detections.append({'bbox':(x,y,x+ww,y+hh),'confidence':confidence,'area':area})
        return detections if detections else None

    def get_statistics(self): return self.statistics

# ------------------ HeatmapGenerator ------------------
class HeatmapGenerator:
    def __init__(self, grid_size=40):
        self.grid_size = grid_size
        self.heatmap_data = {}
        self.total_events = 0
        self.decay_rate = 0.985

    def add_event(self,x,y,weight=1.0):
        gx = int(x//self.grid_size); gy = int(y//self.grid_size)
        key = (gx,gy)
        self.heatmap_data[key] = self.heatmap_data.get(key,0.0)+weight
        self.total_events += 1

    def decay(self):
        for k in list(self.heatmap_data.keys()):
            self.heatmap_data[k] *= self.decay_rate
            if self.heatmap_data[k] < 0.05: del self.heatmap_data[k]

    def get_heatmap(self,width=640,height=480):
        heat = np.zeros((height,width),dtype=np.float32)
        if not self.heatmap_data:
            return cv2.applyColorMap(np.zeros((height,width),dtype=np.uint8),cv2.COLORMAP_JET)
        maxv = max(self.heatmap_data.values()) or 1.0
        for (gx,gy),val in self.heatmap_data.items():
            xs = gx*self.grid_size; ys = gy*self.grid_size
            xe = min(xs+self.grid_size,width); ye = min(ys+self.grid_size,height)
            if xs>=width or ys>=height: continue
            heat[ys:ye,xs:xe] = max(heat[ys:ye,xs:xe], float(val)/maxv)
        heat = cv2.GaussianBlur(heat,(51,51),0)
        cmap = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
        return cmap

    def get_hotspots(self,threshold=0.6):
        if not self.heatmap_data: return []
        maxv = max(self.heatmap_data.values()) or 1.0
        res=[]
        for (gx,gy),val in self.heatmap_data.items():
            if val/maxv >= threshold:
                x = gx*self.grid_size + self.grid_size//2; y = gy*self.grid_size + self.grid_size//2
                res.append({'x':int(x),'y':int(y),'intensity':float(val/maxv),'events':int(val)})
        return sorted(res,key=lambda r: r['intensity'],reverse=True)

# ------------------ MultiDetector ------------------
class MultiDetector:
    def __init__(self):
        print("[MultiDetector] initializing detectors...")
        # load fight detector; fail fast if model missing
        try:
            self.fight_detector = AdvancedFightDetector()
        except Exception as e:
            print(f"[MultiDetector] Failed to init fight detector: {e}")
            raise
        # hazard detector may be optional
        try:
            self.hazard_detector = HazardDetector()
        except Exception:
            self.hazard_detector = HazardDetector()  # will try to load in init
        self.heatmap_generator = HeatmapGenerator()
        print("[MultiDetector] ready")

    def process_frame(self, frame, frame_count):
        annotated, fight_res = self.fight_detector.process_frame(frame, frame_count)
        hazards = self.hazard_detector.detect(frame) if self.hazard_detector else {}
        res = {'fight':fight_res,'hazards':hazards}
        # update heatmap on fight events
        if fight_res.get('detected'):
            for area in fight_res.get('areas',[]):
                cx = (area[0]+area[2])//2; cy = (area[1]+area[3])//2
                self.heatmap_generator.add_event(cx,cy,weight=2.0)
        if frame_count % 30 == 0:
            self.heatmap_generator.decay()
        # draw hazards on annotated
        annotated = self._draw_hazards(annotated, hazards)
        return annotated, res

    def _draw_hazards(self, frame, hazards):
        for htype, dets in hazards.items():
            if not dets: continue
            for d in dets:
                x1,y1,x2,y2 = d['bbox']
                color = (0,69,255) if htype=='fire' else ((128,128,128) if htype=='smoke' else (0,0,200))
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                lbl = (('🔥 FIRE' if htype=='fire' else ('💨 SMOKE' if htype=='smoke' else ('🔪 WEAPON'))))
                txt = f"{lbl}: {int(d.get('confidence',0))}%"
                (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
                cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+8,y1),color,-1)
                cv2.putText(frame,txt,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
        return frame

    def get_heatmap(self,width=640,height=480):
        return self.heatmap_generator.get_heatmap(width,height)

    def get_analytics(self):
        return {
            'fight': self.fight_detector.get_analytics(),
            'hazards': self.hazard_detector.get_statistics() if self.hazard_detector else {},
            'heatmap_events': self.heatmap_generator.total_events
        }

# ------------------ Flask App & Globals ------------------
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
latest_stats = {'people':0,'fights':0,'fps':0,'confidence':0.0,'timestamp':None,'hazards':{},'escalation_warning':False,'conflict_type':'unknown','tension_score':0}
latest_stats_lock = threading.Lock()
last_alert_time = 0.0

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

# ------------------ Processing loop ------------------
def processing_loop(source_is_file=False, job_id=None):
    global detector, video_cap, current_frame, last_annot, stream_active, last_alert_time
    frame_count = 0; processed = 0; t0 = time.time()
    try:
        while stream_active and video_cap and video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                if source_is_file: break
                time.sleep(0.02); continue
            frame_count += 1
            do_proc = (SKIP_FRAMES <= 1) or (frame_count % SKIP_FRAMES == 0)
            out_frame = None; result = {}
            if do_proc:
                try:
                    out_frame, result = detector.process_frame(frame, frame_count)
                    last_annot = out_frame.copy()
                except Exception as e:
                    print(f"[ERROR] process_frame: {e}", traceback.format_exc())
                    out_frame = frame.copy(); result = {'fight':{'detected':False}, 'hazards':{}}
            else:
                out_frame = last_annot.copy() if last_annot is not None else frame.copy()
                result = {'fight':{'detected':False}, 'hazards':{}}

            # alerts: fight
            fight_data = result.get('fight',{})
            now = time.time()
            if fight_data.get('detected') and (now - last_alert_time > ALERT_COOLDOWN):
                snap = UPLOAD_DIR / f"alert_{int(now)}.jpg"
                snap_path = None
                try:
                    cv2.imwrite(str(snap), out_frame); snap_path = str(snap)
                except Exception:
                    snap_path = None
                conf = fight_data.get('confidence',0); people = fight_data.get('people_count',0)
                txt = f"🚨 {fight_data.get('conflict_type','unknown').upper()} | Conf={conf:.0f}% | People={people}"
                _send_alert_nonblocking(txt, snap_path, caption=txt)
                last_alert_time = now

            # alerts: hazards
            hazards = result.get('hazards',{})
            if hazards and (now - last_alert_time > ALERT_COOLDOWN):
                for htype, dets in hazards.items():
                    if dets:
                        snap = UPLOAD_DIR / f"hazard_{htype}_{int(now)}.jpg"; snap_path=None
                        try: cv2.imwrite(str(snap), out_frame); snap_path=str(snap)
                        except Exception: snap_path=None
                        txt = f"⚠️ {htype.upper()} detected! Count={len(dets)}"
                        _send_alert_nonblocking(txt, snap_path, caption=txt)
                        last_alert_time = now
                        break

            # publish current frame
            with frame_lock:
                current_frame = out_frame

            # update analytics
            try:
                people = fight_data.get('people_count',0)
                fight_flag = fight_data.get('detected',False)
                conflict_type = fight_data.get('conflict_type','unknown')
                escalation = fight_data.get('escalation_warning',{}).get('active',False)
                tension = fight_data.get('tension_score',0)
                processed += 1; now2 = time.time(); elapsed = now2 - t0
                if elapsed >= 0.5:
                    fps_est = int(round(processed/elapsed)) if elapsed>0 else latest_stats.get('fps',30)
                    processed = 0; t0 = now2
                else:
                    fps_est = latest_stats.get('fps',30)
                snap = {'frame':frame_count,'fight':fight_flag,'people':people,'conflict_type':conflict_type,'escalation_warning':escalation,'tension_score':tension,'hazards':hazards,'metrics':fight_data.get('metrics',{}),'timestamp':datetime.now().isoformat()}
                analytics_buffer.append(snap)

                with latest_stats_lock:
                    latest_stats['people'] = people
                    latest_stats['fights'] = detector.fight_detector.analytics.get('total_detections', 0)
                    latest_stats['fps'] = fps_est
                    latest_stats['confidence'] = float(fight_data.get('confidence', 0))  
                    latest_stats['timestamp'] = snap['timestamp']
                    latest_stats['hazards'] = hazards
                    latest_stats['escalation_warning'] = escalation
                    latest_stats['conflict_type'] = conflict_type
                    latest_stats['tension_score'] = tension
            except Exception as e:
                print(f"[ERROR] analytics update: {e}")

    except Exception as e:
        print(f"[FATAL] processing_loop: {e}", traceback.format_exc())
    finally:
        stream_active = False
        try:
            if video_cap: video_cap.release()
        except Exception: pass
        if source_is_file and job_id:
            try:
                res = {'job_id':job_id,'analytics':detector.get_analytics() if detector else {} ,'ended_at':datetime.now().isoformat()}
                write_job_result(job_id,res)
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]['status']='finished'; JOBS[job_id]['result_path']=str(UPLOAD_DIR / f"job_{job_id}_result.json")
            except Exception as e:
                print(f"[WARN] job finalize: {e}")
        print("[INFO] processing loop ended")

# ------------------ MJPEG generator ------------------
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
            _, buff = cv2.imencode(".jpg", frm, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_JPEG_QUALITY])
            b = buff.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b + b'\r\n')
        except Exception:
            time.sleep(0.03); continue

# ------------------ Flask routes ------------------
@app.route('/')
def index(): return render_template('index.html')
@app.route('/detection')
def detection(): return render_template('detection.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global detector, video_cap, proc_thread, stream_active
    if detector is None:
        try:
            detector = MultiDetector()
        except Exception as e:
            return jsonify({'success':False,'error':f'Failed init detector: {e}'})
    source_is_file=False; job_id=None
    if 'file' in request.files and request.files['file'].filename!='':
        f = request.files['file']; fn=secure_filename(f.filename)
        saved = UPLOAD_DIR / f"{int(time.time())}_{fn}"
        try: f.save(str(saved))
        except Exception as e: return jsonify({'success':False,'error':f'Failed save file: {e}'})
        source = str(saved); source_is_file=True; job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {'status':'running','video':source,'started_at':time.time(),'result_path':None}
    else:
        data = request.get_json(silent=True) or request.form or request.values
        source = data.get('source','0')
    try:
        if stream_active:
            stream_active = False
            if proc_thread and proc_thread.is_alive(): proc_thread.join(timeout=1.0)
        if isinstance(source,str) and source.isdigit(): video_cap = cv2.VideoCapture(int(source))
        else: video_cap = cv2.VideoCapture(source)
        if not video_cap.isOpened(): return jsonify({'success':False,'error':'Could not open source'})
        try: video_cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        except Exception: pass
        stream_active = True
        proc_thread = threading.Thread(target=processing_loop, args=(source_is_file, job_id), daemon=True); proc_thread.start()
        return jsonify({'success':True,'streaming':True,'stream_url':'/video_feed','job_id':job_id})
    except Exception as e:
        return jsonify({'success':False,'error':str(e),'trace':traceback.format_exc()})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_active, proc_thread, video_cap
    stream_active = False
    if proc_thread and proc_thread.is_alive(): proc_thread.join(timeout=1.0)
    try:
        if video_cap: video_cap.release()
    except Exception: pass
    return jsonify({'success':True})

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics')
def analytics():
    recent = list(analytics_buffer)[-ANALYTICS_SNAPSHOT_SIZE:]
    with latest_stats_lock: ls = dict(latest_stats)
    analytics_copy = detector.get_analytics() if detector else {}
    return jsonify({'success':True,'streaming':bool(stream_active),'recent_data':recent,'analytics':analytics_copy,'latest_stats':ls})

@app.route('/heatmap')
def heatmap():
    if not detector: return jsonify({'success':False,'error':'No detector'})
    try:
        heatmap_img = detector.get_heatmap(width=640,height=480)
        _, buffer = cv2.imencode('.png', heatmap_img)
        return Response(buffer.tobytes(), mimetype='image/png')
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

@app.route('/hotspots')
def hotspots():
    if not detector: return jsonify({'success':False,'error':'No detector'})
    try:
        spots = detector.heatmap_generator.get_hotspots(threshold=0.6)
        return jsonify({'success':True,'hotspots':spots,'total_events':detector.heatmap_generator.total_events})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)})

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
                return jsonify({'success':True,'status':'finished','analysis':json.loads(p.read_text(encoding='utf-8'))})
            except Exception: pass
        return jsonify({'success':False,'error':'Unknown job_id'}),404
    p = UPLOAD_DIR / f"job_{job_id}_result.json"
    res = None
    if p.exists():
        try: res = json.loads(p.read_text(encoding='utf-8'))
        except Exception: res = None
    return jsonify({'success':True,'status':meta.get('status'),'job':meta,'analysis':res})

@app.route('/uploads/<path:filename>')
def uploads(filename): return send_from_directory(str(UPLOAD_DIR.resolve()), filename, as_attachment=False)

@app.route('/settings', methods=['POST'])
def settings():
    global detector
    if not detector: return jsonify({'success':False,'error':'No detector running'})
    data = request.get_json(silent=True) or request.form or request.values
    try:
        fight_det = detector.fight_detector
        if 'body_proximity_threshold' in data:
            fight_det.body_proximity_threshold = float(data['body_proximity_threshold'])
        if 'strike_velocity_threshold' in data:
            fight_det.strike_velocity_threshold = float(data['strike_velocity_threshold'])
        if 'min_fight_frames' in data:
            fight_det.min_fight_frames = int(data['min_fight_frames'])
        if 'escalation_threshold' in data:
            # optional param for future toggles
            pass
    except Exception as e:
        print(f"[ERROR] settings: {e}")
    return jsonify({'success':True,'updated':True})

@app.route('/ping')
def ping(): return jsonify({'success':True,'msg':'pong','active':stream_active})

@app.route('/stats')
def stats():
    with latest_stats_lock: ls = dict(latest_stats)
    return jsonify({'success':True,'streaming':stream_active,'latest_stats':ls,'job_count':len(JOBS),'detector_ready':detector is not None})

# ------------------ main ------------------
if __name__ == "__main__":
    try: os.makedirs(UPLOAD_DIR, exist_ok=True)
    except Exception: pass
    print("[INFO] Starting ToBeLess AI server on 0.0.0.0:5000")
    # suggested run: python app.py  (ensure required models exist in working dir)
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
