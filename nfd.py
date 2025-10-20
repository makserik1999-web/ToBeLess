"""
nfd.py — гибридный детектор (YOLO detect + YOLO-pose) для распознавания драки.

Этот файл — обновлённая версия: оставлена совместимость с API в app.py,
но добавлены быстрые эвристики для повышения точности:
 - глобальный и локальный motion-score (frame diff)
 - motion boost к парному конфидэнсу
 - уменьшение веса при большом числе людей (толпа)
 - внутренняя временная устойчивость (sustain / per-frame threshold)
 - новые параметры: conf_threshold, max_conf_threshold, motion_threshold,
   frame_conf_threshold, crowd_size_limit, motion_boost_cap, sustain_fraction

Public API:
    analyze_video_segments(video_path, ...) -> dict

Dependencies:
    pip install ultralytics opencv-python-headless numpy
    (и torch если хочешь GPU)
"""
from pathlib import Path
import time, math, logging
import numpy as np
import cv2

logger = logging.getLogger("nfd")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Try to import ultralytics YOLO
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
    logger.info("ultralytics YOLO available.")
except Exception as e:
    _YOLO_AVAILABLE = False
    logger.info("ultralytics YOLO NOT available: %s", e)

# COCO keypoint indices & skeleton
KP_INDICES = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
SKELETON_CONNECTIONS = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]


def clamp_box(x1,y1,x2,y2,w,h):
    x1 = max(0, min(w-1, int(x1)))
    x2 = max(0, min(w-1, int(x2)))
    y1 = max(0, min(h-1, int(y1)))
    y2 = max(0, min(h-1, int(y2)))
    return x1,y1,x2,y2

def line_intersect(p1,p2,p3,p4):
    def ccw(A,B,C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return (ccw(p1,p3,p4) != ccw(p2,p3,p4)) and (ccw(p1,p2,p3) != ccw(p1,p2,p4))

def skeleton_lines_from_kp(kp, thr=0.25):
    lines=[]
    for a,b in SKELETON_CONNECTIONS:
        if a < len(kp) and b < len(kp) and kp[a][2]>thr and kp[b][2]>thr:
            p1=(float(kp[a][0]), float(kp[a][1])); p2=(float(kp[b][0]), float(kp[b][1]))
            lines.append((p1,p2,f"{a}-{b}"))
    return lines

def bbox_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0,(y1+y2)/2.0)


class YOLOPoseFight:
    """
    Person detect + pose per-person + improved heuristics.
    """
    def __init__(self, detect_model="yolov8n.pt", pose_model="yolov8n-pose.pt",
                 device=None, detect_conf=0.35, pose_conf=0.25,
                 body_threshold=120.0, limb_threshold=50.0):
        if not _YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package required")

        # device selection
        try:
            import torch
            if device is None:
                if torch.cuda.is_available():
                    device="cuda"
                elif getattr(torch.backends,"mps",None) and torch.backends.mps.is_available():
                    device="mps"
                else:
                    device="cpu"
        except Exception:
            device = device or "cpu"
        self.device = device

        logger.info("Loading detect model '%s' and pose model '%s' on %s", detect_model, pose_model, self.device)
        self.detector = YOLO(detect_model)
        try: self.detector.model.to(self.device)
        except Exception: pass

        self.pose_model = None
        try:
            self.pose_model = YOLO(pose_model)
            try: self.pose_model.model.to(self.device)
            except Exception: pass
        except Exception as e:
            logger.warning("Pose model not loaded: %s (fallback to detection-only)", e)
            self.pose_model = None

        self.detect_conf = float(detect_conf)
        self.pose_conf = float(pose_conf)
        self.body_threshold = float(body_threshold)
        self.limb_threshold = float(limb_threshold)

    def _detect_people(self, frame):
        res = self.detector(frame, verbose=False)
        boxes=[]
        for r in res:
            if getattr(r,"boxes",None) is None: continue
            for b in r.boxes:
                try:
                    cls = int(b.cls[0]) if hasattr(b.cls,"__len__") else int(b.cls)
                    conf = float(b.conf[0]) if hasattr(b.conf,"__len__") else float(b.conf)
                    xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy,"__len__") else (b.xyxy.tolist() if hasattr(b.xyxy,"tolist") else None)
                except Exception:
                    continue
                if cls==0 and conf>=self.detect_conf:
                    x1,y1,x2,y2 = map(int, xyxy[:4])
                    boxes.append((x1,y1,x2,y2,conf))
        return boxes

    def _pose_for_box(self, frame, box):
        if self.pose_model is None:
            return None
        x1,y1,x2,y2,_ = box
        h,w = frame.shape[:2]
        x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,w,h)
        if x2<=x1 or y2<=y1: return None
        crop = frame[y1:y2, x1:x2]
        try:
            res = self.pose_model(crop, verbose=False)
        except Exception as e:
            logger.debug("pose inference failed on crop: %s", e)
            return None
        for r in res:
            if getattr(r,"keypoints",None) is not None and len(r.keypoints.data)>0:
                kp = r.keypoints.data[0].cpu().numpy()
                kp_xyv = kp.copy()
                kp_xyv[:,0] += float(x1); kp_xyv[:,1] += float(y1)
                return kp_xyv
        return None

    def _skeleton_and_close(self,kp1,kp2,kp_vis_thr=0.25):
        if kp1 is None or kp2 is None: return 0,0,None
        lines1 = skeleton_lines_from_kp(kp1,kp_vis_thr)
        lines2 = skeleton_lines_from_kp(kp2,kp_vis_thr)
        crossings=0
        for p1,p2,_ in lines1:
            for p3,p4,_ in lines2:
                if line_intersect(p1,p2,p3,p4): crossings+=1
        limb_kps = [KP_INDICES['left_wrist'],KP_INDICES['right_wrist'],KP_INDICES['left_elbow'],KP_INDICES['right_elbow'],
                    KP_INDICES['left_knee'],KP_INDICES['right_knee'],KP_INDICES['left_ankle'],KP_INDICES['right_ankle']]
        close_contacts=0; min_d=float("inf")
        for i in limb_kps:
            if i < len(kp1) and kp1[i][2]>kp_vis_thr:
                p1 = kp1[i][:2]
                for j in limb_kps:
                    if j < len(kp2) and kp2[j][2]>kp_vis_thr:
                        p2 = kp2[j][:2]
                        d = float(np.linalg.norm(np.array(p1)-np.array(p2)))
                        min_d = min(min_d,d)
                        if d < self.limb_threshold: close_contacts+=1
        return crossings, close_contacts, (min_d if min_d!=float("inf") else None)

    def analyze_window(self, frames,
                       conf_threshold=30.0,
                       max_conf_threshold=50.0,
                       motion_threshold=0.015,
                       frame_conf_threshold=20.0,
                       motion_boost_cap=30.0,
                       crowd_size_limit=4,
                       sustain_fraction=0.6):
        """
        Analyze list of frames (short window). Returns:
           fight_flag (bool), avg_conf, max_conf, avg_people, details

        Additional heuristics:
         - compute global frame-diff motion_score
         - compute local (pair) motion in union-box area to boost pair conf
         - reduce confidence if crowd (many people)
         - require sustain across frames (sustain_fraction of frames above frame_conf_threshold)
        """
        per_frame_conf=[]; per_frame_people=[]
        details=[]
        h = frames[0].shape[0] if frames else 0
        w = frames[0].shape[1] if frames else 0

        # Precompute grayscale frames for motion
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        motion_scores = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            motion_score = float(np.sum(diff > 25)) / float(diff.size)
            motion_scores.append(motion_score)
        avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0

        # For each frame, detect people and (optionally) poses; compute per-pair conf
        people_boxes_per_frame = []
        kps_per_frame = []
        for frame in frames:
            people = self._detect_people(frame)
            people_boxes_per_frame.append(people)
            kps = []
            # compute poses only when pose_model available
            if self.pose_model is not None:
                for b in people:
                    k = self._pose_for_box(frame, b)
                    kps.append(k)
            else:
                kps = [None for _ in people]
            kps_per_frame.append(kps)

        for fi, frame in enumerate(frames):
            people = people_boxes_per_frame[fi]
            kps = kps_per_frame[fi]
            per_frame_people.append(len(people))
            frame_metrics={'pairs':[],'people':len(people),'motion':0.0}
            pair_confs=[]
            # compute body distances and pose crossings as before
            for a in range(len(people)):
                for b in range(a+1,len(people)):
                    box_a=people[a][:4]; box_b=people[b][:4]
                    center_a=bbox_center(box_a); center_b=bbox_center(box_b)
                    body_dist=float(np.linalg.norm(np.array(center_a)-np.array(center_b)))
                    crossings, close_contacts, min_limb = self._skeleton_and_close(kps[a] if a<len(kps) else None, kps[b] if b<len(kps) else None)
                    conf=0.0
                    # distance contribution
                    if body_dist < self.body_threshold:
                        conf += max(0.0, (self.body_threshold - body_dist)/self.body_threshold*50.0)
                    # skeleton crossing & limb contacts
                    conf += min(crossings*12.0, 30.0)
                    conf += min(close_contacts*10.0, 30.0)

                    # local motion: try to compute motion inside union bbox between this frame and next (if exists)
                    local_motion = 0.0
                    if fi < len(frames)-1:
                        # union bbox on current frame
                        x1 = int(min(box_a[0], box_b[0])); y1 = int(min(box_a[1], box_b[1]))
                        x2 = int(max(box_a[2], box_b[2])); y2 = int(max(box_a[3], box_b[3]))
                        x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2,w,h)
                        if x2>x1 and y2>y1:
                            crop0 = cv2.cvtColor(frames[fi][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            crop1 = cv2.cvtColor(frames[fi+1][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            if crop0.size>0 and crop1.size>0:
                                diffc = cv2.absdiff(crop1, crop0)
                                local_motion = float(np.sum(diffc > 20)) / float(diffc.size)
                    # boost conf by local motion (controlled, capped)
                    motion_boost = min(local_motion * 200.0, motion_boost_cap)  # scaling chosen empirically
                    conf += motion_boost

                    # if too many people, downweight conf (likely crowd / non-fight)
                    if len(people) > crowd_size_limit:
                        conf *= 0.6

                    conf = min(conf,100.0)
                    pair_confs.append(conf)
                    frame_metrics['pairs'].append({'pair':(a,b),'body_dist':body_dist,'crossings':crossings,'close_contacts':close_contacts,'min_limb':min_limb,'local_motion':local_motion,'conf':conf})
            frame_conf = float(max(pair_confs)) if pair_confs else 0.0
            per_frame_conf.append(frame_conf)
            frame_metrics['motion'] = float(avg_motion)  # global window-level motion
            details.append(frame_metrics)

        avg_conf = float(np.mean(per_frame_conf)) if per_frame_conf else 0.0
        max_conf = float(np.max(per_frame_conf)) if per_frame_conf else 0.0
        avg_people = int(round(np.mean(per_frame_people))) if per_frame_people else 0

        # Now improved heuristics combining confs and motion
        # Count frames above frame_conf_threshold
        high_frames = sum(1 for c in per_frame_conf if c >= frame_conf_threshold)
        required_high = max(1, int(math.ceil(len(frames) * float(sustain_fraction))))
        sustained = (high_frames >= required_high)

        fight_flag = False
        # 3 ways to trigger fight:
        # 1) average confidence high enough
        if avg_conf >= conf_threshold:
            fight_flag = True
        # 2) strong peak + motion present
        elif (max_conf >= max_conf_threshold) and (avg_motion >= motion_threshold):
            fight_flag = True
        # 3) sustained medium confidence across frames
        elif sustained and (avg_conf >= (conf_threshold * 0.6)):
            fight_flag = True

        return fight_flag, avg_conf, max_conf, avg_people, details


def sample_frame_indices(start,end,num):
    if num<=1:
        return [start]
    return list(np.linspace(start,end,num,dtype=int))


def analyze_video_segments(
    video_path: str,
    detect_model: str = "yolov8n.pt",
    pose_model: str = "yolov8n-pose.pt",
    window_sec: float = 1.0,
    step_sec: float = 1.0,
    detect_conf: float = 0.35,
    pose_conf: float = 0.25,
    body_threshold: float = 120.0,
    limb_threshold: float = 50.0,
    sample_per_window: int = 2,
    top_k: int = 5,
    verbose: bool = False,
    # new tuning params with sensible defaults
    conf_threshold: float = 30.0,
    max_conf_threshold: float = 50.0,
    motion_threshold: float = 0.015,
    frame_conf_threshold: float = 20.0,
    motion_boost_cap: float = 30.0,
    crowd_size_limit: int = 4,
    sustain_fraction: float = 0.6
):
    start = time.time()
    if verbose: logger.setLevel(logging.DEBUG)
    p = Path(video_path)
    if not p.exists(): return {"error":f"Video not found: {video_path}"}

    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return {"error":f"Cannot open video: {video_path}"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0 or math.isnan(fps): fps = 30.0
    duration = (total_frames / fps) if total_frames>0 else 0.0
    cap.release()

    logger.info("analyze_video_segments: %s fps=%.2f frames=%d", video_path, fps, total_frames)

    detector = YOLOPoseFight(detect_model=detect_model, pose_model=pose_model,
                             detect_conf=detect_conf, pose_conf=pose_conf,
                             body_threshold=body_threshold, limb_threshold=limb_threshold)

    window_frames = max(1, int(round(window_sec*fps)))
    step_frames = max(1, int(round(step_sec*fps)))
    starts = list(range(0, max(1, total_frames - window_frames + 1), step_frames))
    if not starts: starts=[0]

    window_results=[]
    for s in starts:
        e = min(total_frames-1, s+window_frames-1)
        idxs = sample_frame_indices(s, e, min(sample_per_window, max(1, e-s+1)))
        frames=[]
        cap = cv2.VideoCapture(str(video_path))
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frm = cap.read()
            if not ret: continue
            frames.append(frm)
        cap.release()
        if not frames:
            window_results.append({"start_frame":int(s),"end_frame":int(e),"start_sec":float(s)/fps,"end_sec":float(e+1)/fps,"preds":[],"fight":False,"max_conf":0.0,"people":0})
            continue

        # improved analyze_window with additional params
        fight_flag, avg_conf, max_conf, people_count, details = detector.analyze_window(
            frames,
            conf_threshold=conf_threshold,
            max_conf_threshold=max_conf_threshold,
            motion_threshold=motion_threshold,
            frame_conf_threshold=frame_conf_threshold,
            motion_boost_cap=motion_boost_cap,
            crowd_size_limit=crowd_size_limit,
            sustain_fraction=sustain_fraction
        )
        preds = [("fight", float(max_conf))] if fight_flag else [("no_fight", float(max_conf))]
        window_results.append({"start_frame":int(s),"end_frame":int(e),"start_sec":float(s)/fps,"end_sec":float(e+1)/fps,"preds":preds,"fight":bool(fight_flag),"max_conf":float(max_conf),"people":int(people_count),"details":details})

    # merge windows into segments
    segments=[]
    cur=None
    for w in window_results:
        if w.get("fight"):
            if cur is None:
                cur = {"start_frame":w["start_frame"],"end_frame":w["end_frame"],"start_sec":w["start_sec"],"end_sec":w["end_sec"],"max_conf":w.get("max_conf",0.0),"preds":list(w.get("preds",[]))}
            else:
                cur["end_frame"]=w["end_frame"]; cur["end_sec"]=w["end_sec"]
                if w.get("max_conf",0.0)>cur.get("max_conf",0.0): cur["max_conf"]=w.get("max_conf",0.0)
                cur["preds"].extend(w.get("preds",[]))
        else:
            if cur is not None:
                merged_preds={}
                for lab,conf in cur.get("preds",[]):
                    if lab not in merged_preds or conf>merged_preds[lab]: merged_preds[lab]=conf
                sorted_preds = sorted(merged_preds.items(), key=lambda x:x[1], reverse=True)[:top_k]
                cur["top_predictions"]=[(lab,float(conf)) for lab,conf in sorted_preds]
                cur["max_conf"]=float(cur.get("max_conf",0.0))
                segments.append(cur); cur=None
    if cur is not None:
        merged_preds={}
        for lab,conf in cur.get("preds",[]):
            if lab not in merged_preds or conf>merged_preds[lab]: merged_preds[lab]=conf
        sorted_preds=sorted(merged_preds.items(), key=lambda x:x[1], reverse=True)[:top_k]
        cur["top_predictions"]=[(lab,float(conf)) for lab,conf in sorted_preds]
        cur["max_conf"]=float(cur.get("max_conf",0.0))
        segments.append(cur)

    # compute overall top windows
    top_overall=[]
    try:
        sorted_w = sorted(window_results, key=lambda x:x.get("max_conf",0.0), reverse=True)[:top_k]
        for w in sorted_w:
            if w.get("max_conf",0.0)>0:
                top_overall.append(("fight" if w.get("fight") else "no_fight", float(w.get("max_conf",0.0))))
    except Exception:
        top_overall=[]

    elapsed = time.time() - start
    meta = {"mode":"yolo_pose_hybrid_v2","detect_model":str(detector.detector.model if hasattr(detector.detector,'model') else detector.detector),"pose_model":str(detector.pose_model.model if detector.pose_model is not None else None),"num_windows":len(starts),"num_processed_windows":len(window_results),"elapsed":elapsed}

    return {"video_path":str(video_path),"duration":duration,"fps":float(fps),"total_frames":int(total_frames),"segments":segments,"top_overall":top_overall,"windows":window_results,"meta":meta}
