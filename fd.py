# fd.py — улучшенный FightDetector (с трекингом, сглаживанием, направлением удара и устойчивой проверкой)
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch
from datetime import datetime
import time

class FightDetector:
    """
    Улучшенный детектор драк:
      - стабилизация keypoints (EMA)
      - простое трекинг-сопоставление между кадрами (greedy matching)
      - скорость конечностей нормализована по росту человека (bbox height)
      - проверка направления движения конечности в сторону уязвимой зоны
      - требование sustain_frames и повторяемости для подтверждения драки
    Совместимость:
      processed_frame, metrics = detector.process_frame(frame, frame_count)
      detector.fight_detected -> bool
      detector.analytics -> dict (total_detections, fight_events, ...)
      detector.pose_history -> deque (последние poses)
    """

    def __init__(self,
                 model_path="yolov8n-pose.pt",
                 device=None,
                 kp_conf_threshold=0.4,
                 ema_alpha=0.6,
                 pixel_speed_threshold=12.0,
                 rel_speed_threshold=0.035,
                 limb_proximity_threshold=60,
                 body_proximity_threshold=140,
                 sustain_frames=10,
                 gap_tolerance=4):
        # device selection
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        print(f"[FightDetector] device: {self.device}")

        # load YOLO pose model
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            # some backends ignore .to for cpu
            pass

        # config
        self.kp_conf_threshold = float(kp_conf_threshold)
        self.ema_alpha = float(ema_alpha)   # smoothing factor for keypoints
        self.pixel_speed_threshold = float(pixel_speed_threshold)
        self.rel_speed_threshold = float(rel_speed_threshold)
        self.limb_proximity_threshold = float(limb_proximity_threshold)
        self.body_proximity_threshold = float(body_proximity_threshold)
        self.sustain_frames = int(sustain_frames)
        self.gap_tolerance = int(gap_tolerance)

        # COCO keypoint indices (17)
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }

        # keypoints we watch for strikes
        self.fight_keypoints = [
            self.KEYPOINTS['left_wrist'], self.KEYPOINTS['right_wrist'],
            self.KEYPOINTS['left_elbow'], self.KEYPOINTS['right_elbow']
        ]

        # vulnerable zones (head / upper body)
        self.vulnerable_zones = [
            self.KEYPOINTS['nose'],
            self.KEYPOINTS['left_shoulder'], self.KEYPOINTS['right_shoulder'],
            self.KEYPOINTS['left_hip'], self.KEYPOINTS['right_hip']
        ]

        # tracking structures
        # tracks: tid -> {'kps': np(17,3), 'smoothed': np(17,3), 'last_seen': frame_count, 'bbox':(x1,y1,x2,y2)}
        self.tracks = {}
        self.track_centers = {}   # tid -> (x,y)
        self.next_track_id = 0

        # contact_records: (id_a, id_b) -> {'start_frame','last_frame','frames_in_contact','hits'}
        self.contact_records = {}

        # histories / analytics
        self.pose_history = deque(maxlen=50)
        self.vel_history = {}    # tid -> deque(last velocities)
        self.fight_detected = False
        self.fight_start_frame = None
        self.last_fight_detection = None

        self.analytics = {
            'total_detections': 0,
            'fight_duration_history': [],
            'people_count_history': [],
            'detection_confidence_history': [],
            'fight_events': []
        }

    # ----------------- helpers -----------------
    def calculate_distance(self, p1, p2):
        p1 = np.array(p1); p2 = np.array(p2)
        return float(np.linalg.norm(p1 - p2))

    def get_person_center(self, kps):
        torso_idx = [self.KEYPOINTS['left_shoulder'], self.KEYPOINTS['right_shoulder'],
                     self.KEYPOINTS['left_hip'], self.KEYPOINTS['right_hip']]
        pts = []
        for idx in torso_idx:
            if idx < len(kps) and kps[idx][2] > self.kp_conf_threshold:
                pts.append(kps[idx][:2])
        if pts:
            return np.mean(np.array(pts), axis=0)
        return None

    def get_bounding_box(self, kps):
        pts = []
        for kp in kps:
            if kp[2] > self.kp_conf_threshold:
                pts.append(kp[:2])
        if not pts:
            return None
        arr = np.array(pts)
        x_min, y_min = int(np.min(arr[:,0])), int(np.min(arr[:,1]))
        x_max, y_max = int(np.max(arr[:,0])), int(np.max(arr[:,1]))
        return (x_min, y_min, x_max, y_max)

    def ema_smooth(self, prev, curr):
        # prev, curr: numpy arrays shape (17,3) or None
        if prev is None:
            return curr.copy()
        sm = prev.copy()
        alpha = self.ema_alpha
        # Only smooth coordinates; keep confidence as curr confidence
        for i in range(min(len(curr), len(prev))):
            if curr[i][2] > 0:
                sm[i,0] = alpha * prev[i,0] + (1-alpha) * curr[i,0]
                sm[i,1] = alpha * prev[i,1] + (1-alpha) * curr[i,1]
                sm[i,2] = curr[i][2]   # use fresh confidence
            else:
                # if curr not confident, keep previous smoothed
                sm[i,2] = prev[i,2]
        return sm

    def get_skeleton_lines(self, keypoints):
        con = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16)
        ]
        lines = []
        for a,b in con:
            if a < len(keypoints) and b < len(keypoints) and keypoints[a][2] > self.kp_conf_threshold and keypoints[b][2] > self.kp_conf_threshold:
                p1 = (int(keypoints[a][0]), int(keypoints[a][1]))
                p2 = (int(keypoints[b][0]), int(keypoints[b][1]))
                lines.append((p1,p2,f"{a}-{b}"))
        return lines

    def check_close_limbs(self, kps1, kps2):
        limb_kps = [
            self.KEYPOINTS['left_wrist'], self.KEYPOINTS['right_wrist'],
            self.KEYPOINTS['left_elbow'], self.KEYPOINTS['right_elbow'],
            self.KEYPOINTS['left_knee'], self.KEYPOINTS['right_knee'],
            self.KEYPOINTS['left_ankle'], self.KEYPOINTS['right_ankle']
        ]
        close = 0; min_d = float('inf')
        for i in limb_kps:
            if i < len(kps1) and kps1[i][2] > self.kp_conf_threshold:
                p1 = kps1[i][:2]
                for j in limb_kps:
                    if j < len(kps2) and kps2[j][2] > self.kp_conf_threshold:
                        p2 = kps2[j][:2]
                        d = self.calculate_distance(p1,p2)
                        min_d = min(min_d, d)
                        if d < self.limb_proximity_threshold:
                            close += 1
        if min_d == float('inf'): min_d = 9999.0
        return close, min_d

    # ----------------- matching (greedy) -----------------
    def match_tracks(self, centers, max_dist=None):
        """
        centers: list of tuples or None
        returns mapping curr_idx -> track_id (int)
        """
        if max_dist is None:
            max_dist = self.body_proximity_threshold * 1.2
        mapping = {}
        used_tracks = set()
        prev_items = [(tid, self.track_centers.get(tid)) for tid in list(self.track_centers.keys())]

        # build candidate list (dist, curr_idx, tid)
        cand = []
        for curr_idx, c in enumerate(centers):
            if c is None:
                continue
            for (tid, pc) in prev_items:
                if pc is None:
                    continue
                d = self.calculate_distance(c, pc)
                cand.append((d, curr_idx, tid))
        cand.sort(key=lambda x: x[0])

        assigned_curr = set()
        for d, curr_idx, tid in cand:
            if curr_idx in assigned_curr or tid in used_tracks:
                continue
            if d <= max_dist:
                mapping[curr_idx] = tid
                assigned_curr.add(curr_idx)
                used_tracks.add(tid)

        # assign new ids for unmatched centers
        for idx, c in enumerate(centers):
            if idx in mapping:
                continue
            if c is None:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            mapping[idx] = tid

        return mapping

    # ----------------- main process -----------------
    def process_frame(self, frame, frame_count=0):
        """
        Input: BGR frame
        Returns: annotated_frame, metrics
        """
        # Run model
        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            print(f"[WARN] YOLO inference failed: {e}")
            return frame, {}

        # Extract poses
        poses = []
        for res in results:
            if getattr(res, "keypoints", None) is not None:
                for kps in res.keypoints.data:
                    poses.append(kps.cpu().numpy())

        # Analytics: people count
        people_count = len(poses)
        self.analytics['people_count_history'].append({
            'frame': frame_count, 'count': people_count, 'timestamp': datetime.now().isoformat()
        })

        # Build centers and bboxes
        centers = []
        bboxes = []
        for kps in poses:
            c = self.get_person_center(kps)
            centers.append(tuple(c) if c is not None else None)
            bboxes.append(self.get_bounding_box(kps))

        # match to existing tracks
        assignment = self.match_tracks(centers, max_dist=int(self.body_proximity_threshold*1.2))

        # metrics accumulator
        detection_metrics = {
            'body_distances': [],
            'limb_crossings': 0,
            'close_contacts': 0,
            'dangerous_contacts': 0,
            'confidence': 0.0
        }

        # compute pairwise interactions
        for i in range(len(poses)):
            for j in range(i+1, len(poses)):
                kps_i = poses[i]; kps_j = poses[j]

                # body distance
                ci = centers[i]; cj = centers[j]
                if ci is not None and cj is not None:
                    bd = self.calculate_distance(ci, cj)
                    detection_metrics['body_distances'].append(bd)
                else:
                    bd = None

                # limb crossings & close limbs
                crossings, _ = 0, []
                try:
                    crossings, _ = self.check_limb_crossings(kps_i, kps_j)
                except Exception:
                    crossings = 0
                detection_metrics['limb_crossings'] += crossings
                close_cnt, min_limb_dist = self.check_close_limbs(kps_i, kps_j)
                detection_metrics['close_contacts'] += close_cnt

                # check dangerous contacts (direction + speed + proximity)
                tid_i = assignment.get(i)
                tid_j = assignment.get(j)
                # ensure track entries exist (we will create them below if not)
                for (src_kps, src_idx, src_tid, tgt_kps, tgt_idx, tgt_tid) in [
                    (kps_i, i, tid_i, kps_j, j, tid_j),
                    (kps_j, j, tid_j, kps_i, i, tid_i)
                ]:
                    # for each monitored limb
                    for kp_idx in self.fight_keypoints:
                        if kp_idx >= len(src_kps) or kp_idx >= len(src_kps):
                            continue
                        kp = src_kps[kp_idx]
                        if kp[2] <= self.kp_conf_threshold:
                            continue
                        curr_pt = kp[:2]

                        # previous smoothed keypoints for this track
                        prev_sm = None
                        if src_tid is not None and src_tid in self.tracks and 'smoothed' in self.tracks[src_tid]:
                            prev_sm = self.tracks[src_tid]['smoothed']
                        if prev_sm is None:
                            continue
                        # previous point for this limb
                        if kp_idx >= len(prev_sm) or prev_sm[kp_idx][2] <= 0:
                            continue
                        prev_pt = prev_sm[kp_idx][:2]

                        # raw velocity (px per frame)
                        v_px = self.calculate_distance(curr_pt, prev_pt)

                        # normalize by scale (bbox height)
                        src_bbox = bboxes[src_idx]
                        scale = (src_bbox[3] - src_bbox[1]) if src_bbox else 120.0
                        rel_v = v_px / (scale + 1e-6)

                        # check if speed significant (either absolute or relative)
                        speed_sig = (v_px >= self.pixel_speed_threshold) or (rel_v >= self.rel_speed_threshold)
                        if not speed_sig:
                            continue

                        # check target vulnerable zones proximity and direction
                        for vz_idx in self.vulnerable_zones:
                            if vz_idx >= len(tgt_kps):
                                continue
                            vz = tgt_kps[vz_idx]
                            if vz[2] <= self.kp_conf_threshold:
                                continue
                            vz_pt = vz[:2]
                            dist = self.calculate_distance(curr_pt, vz_pt)
                            if dist > self.limb_proximity_threshold:
                                continue

                            # direction check: limb velocity vector should point toward vz_pt
                            v_vec = np.array(curr_pt) - np.array(prev_pt)
                            to_target = np.array(vz_pt) - np.array(curr_pt)
                            if np.linalg.norm(v_vec) < 1e-6 or np.linalg.norm(to_target) < 1e-6:
                                dir_ok = True
                            else:
                                cosang = np.dot(v_vec, to_target) / (np.linalg.norm(v_vec)*np.linalg.norm(to_target) + 1e-9)
                                dir_ok = cosang > 0.2  # moving broadly toward target
                            if not dir_ok:
                                continue

                            # It's a dangerous contact candidate (src -> tgt)
                            detection_metrics['dangerous_contacts'] += 1

                            # record contact between two track ids (create tid if None)
                            if src_tid is None:
                                src_tid = self._create_temp_track(src_idx, src_kps, centers[src_idx], bboxes[src_idx])
                                assignment[src_idx] = src_tid
                            if tgt_tid is None:
                                tgt_tid = self._create_temp_track(tgt_idx, tgt_kps, centers[tgt_idx], bboxes[tgt_idx])
                                assignment[tgt_idx] = tgt_tid

                            # unify key
                            a = int(min(src_tid, tgt_tid)); b = int(max(src_tid, tgt_tid))
                            key = (a,b)
                            rec = self.contact_records.get(key)
                            if rec is None:
                                self.contact_records[key] = {
                                    'start_frame': frame_count,
                                    'last_frame': frame_count,
                                    'frames_in_contact': 1,
                                    'hits': 1
                                }
                            else:
                                if frame_count - rec['last_frame'] <= self.gap_tolerance:
                                    rec['frames_in_contact'] += 1
                                    rec['hits'] += 1
                                else:
                                    rec['start_frame'] = frame_count
                                    rec['frames_in_contact'] = 1
                                    rec['hits'] = 1
                                rec['last_frame'] = frame_count

                            # break after first vulnerable zone matched for this limb
                            break

        # evaluate contact records to find confirmed pairs
        confirmed_pairs = []
        for k, rec in list(self.contact_records.items()):
            # if rec was stale (not updated recently) we can decay frames_in_contact (optional)
            if (frame_count - rec['last_frame']) > (self.gap_tolerance * 6):
                # drop stale record
                del self.contact_records[k]
                continue
            if rec['frames_in_contact'] >= self.sustain_frames and rec['hits'] >= 1:
                confirmed_pairs.append((k, rec))

        # compute confidence from metrics
        confidence = 0.0
        if detection_metrics['body_distances']:
            avg_bd = np.mean(detection_metrics['body_distances'])
            confidence += max(0, (self.body_proximity_threshold - avg_bd) / (self.body_proximity_threshold + 1e-6) * 25)
        confidence += min(detection_metrics['limb_crossings'] * 15, 25)
        confidence += min(detection_metrics['close_contacts'] * 6, 15)
        confidence += min(detection_metrics['dangerous_contacts'] * 20, 40)
        confidence = min(confidence, 100.0)
        detection_metrics['confidence'] = round(float(confidence), 2)

        # decide fight state
        fight_now = False
        if confirmed_pairs or detection_metrics['confidence'] >= 60.0:
            fight_now = True

        # state transitions + analytics like раньше
        if fight_now:
            if not self.fight_detected:
                self.fight_start_frame = frame_count
                self.analytics['fight_events'].append({
                    'start_frame': frame_count,
                    'start_time': datetime.now().isoformat(),
                    'confidence': detection_metrics['confidence'],
                    'pairs': [k for k,_ in confirmed_pairs]
                })
                self.analytics['total_detections'] += 1
            self.fight_detected = True
            self.last_fight_detection = frame_count
        else:
            if self.fight_detected:
                frames_since_start = frame_count - (self.fight_start_frame or frame_count)
                frames_since_last = frame_count - (self.last_fight_detection or frame_count)
                if frames_since_start >= self.sustain_frames and frames_since_last > 10:
                    duration = frames_since_start / 30.0
                    self.analytics['fight_duration_history'].append(duration)
                    if self.analytics['fight_events']:
                        self.analytics['fight_events'][-1]['duration'] = duration
                    self.fight_detected = False
                    self.fight_start_frame = None

        # store detection confidence history
        if detection_metrics['confidence'] > 0:
            self.analytics['detection_confidence_history'].append({
                'frame': frame_count,
                'confidence': detection_metrics['confidence'],
                'timestamp': datetime.now().isoformat()
            })

        # update track storage with smoothed keypoints
        # assignment: curr_idx -> tid exists for all centers (we ensured earlier)
        for idx, kps in enumerate(poses):
            tid = assignment.get(idx)
            if tid is None:
                tid = self.next_track_id
                self.next_track_id += 1
                assignment[idx] = tid
            prev_entry = self.tracks.get(tid)
            prev_sm = prev_entry['smoothed'] if (prev_entry and 'smoothed' in prev_entry) else None
            sm = self.ema_smooth(prev_sm, kps)
            self.tracks[tid] = {
                'kps': kps.copy(),
                'smoothed': sm,
                'last_seen': frame_count,
                'bbox': bboxes[idx]
            }
            if centers[idx] is not None:
                self.track_centers[tid] = centers[idx]
            # maintain vel_history container
            if tid not in self.vel_history:
                self.vel_history[tid] = deque(maxlen=10)

        # annotate frame cleanly using smoothed keypoints
        annotated = frame.copy()
        for idx, kps in enumerate(poses):
            tid = assignment.get(idx)
            entry = self.tracks.get(tid)
            sm_kps = entry['smoothed'] if entry is not None else kps
            color = (0,0,255) if self.fight_detected else (0,255,0)
            # draw keypoints
            for kp in sm_kps:
                if kp[2] > self.kp_conf_threshold:
                    cv2.circle(annotated, (int(kp[0]), int(kp[1])), 3, color, -1)
            # skeleton lines
            for (p1,p2,_) in self.get_skeleton_lines(sm_kps):
                cv2.line(annotated, p1, p2, color, 2)
            # bbox
            bbox = bboxes[idx]
            if bbox:
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # draw confirmed pairs with strong red boxes
        rev_map = {v:k for k,v in assignment.items()}  # tid -> idx
        for (pair_key, rec) in confirmed_pairs:
            a,b = pair_key
            idx_a = rev_map.get(a); idx_b = rev_map.get(b)
            if idx_a is not None:
                bb = bboxes[idx_a]
                if bb:
                    cv2.rectangle(annotated, (bb[0], bb[1]), (bb[2], bb[3]), (0,0,255), 4)
            if idx_b is not None:
                bb = bboxes[idx_b]
                if bb:
                    cv2.rectangle(annotated, (bb[0], bb[1]), (bb[2], bb[3]), (0,0,255), 4)

        # draw status text
        if self.fight_detected:
            duration = (frame_count - (self.fight_start_frame or frame_count)) / 30.0 if self.fight_start_frame else 0.0
            cv2.putText(annotated, f"FIGHT ({duration:.1f}s) - {detection_metrics['confidence']:.0f}%",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(annotated, f"Normal - {detection_metrics['confidence']:.0f}%",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # persons count
        try:
            cv2.putText(annotated, f"Persons: {people_count}", (20, annotated.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        except Exception:
            pass

        # store pose_history compatible with old app (list of {'keypoints': ...})
        pose_list = []
        for kps in poses:
            pose_list.append({'keypoints': kps})
        self.pose_history.append(pose_list)

        # metrics output (compatibility)
        metrics_out = {
            'confidence': detection_metrics['confidence'],
            'body_distances': detection_metrics['body_distances'],
            'limb_crossings': detection_metrics['limb_crossings'],
            'close_contacts': detection_metrics['close_contacts'],
            'dangerous_contacts': detection_metrics['dangerous_contacts']
        }

        return annotated, metrics_out

    # helper to create temporary track for unmatched persons (returns tid)
    def _create_temp_track(self, idx, kps, center, bbox):
        tid = self.next_track_id
        self.next_track_id += 1
        sm = kps.copy()
        self.tracks[tid] = {'kps': kps.copy(), 'smoothed': sm, 'last_seen': -1, 'bbox': bbox}
        if center is not None:
            self.track_centers[tid] = center
        if tid not in self.vel_history:
            self.vel_history[tid] = deque(maxlen=10)
        return tid
