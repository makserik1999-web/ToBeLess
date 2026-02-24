"""
Hybrid Fight Detector: YOLO-Pose + SlowFast

OPTIMIZED VERSION - Background threaded SlowFast for smooth FPS

Combines spatial pose detection with temporal action recognition
to reduce false positives in fight detection.

Approach:
1. YOLO-Pose detects WHERE people are and their spatial interactions
2. SlowFast analyzes WHAT action is happening temporally (background thread)
3. Only trigger fight alert when BOTH signals agree

This eliminates false positives like: hugs, crowds, dancing, sports
"""

import cv2
import time
import threading
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from video_buffer import TemporalFrameBuffer
from slowfast_detector import SlowFastDetector, ActionResult, create_slowfast_detector


class HybridFightDetector:
    """
    Hybrid fight detector combining pose-based and action recognition approaches.

    OPTIMIZED: SlowFast runs in a background daemon thread.
    - Main thread: YOLO-Pose + rendering (fast, never blocks)
    - Background thread: SlowFast inference (runs continuously on latest buffer)
    - Result: smooth FPS with accurate action classification
    """

    def __init__(
        self,
        pose_detector,  # Existing FightDetector instance
        slowfast_detector: Optional[SlowFastDetector] = None,
        buffer_size: int = 32,
        device: str = 'cuda',
        # Pose detection thresholds (inherited from pose_detector)
        body_proximity_threshold: float = 120.0,
        limb_proximity_threshold: float = 50.0,
        # SlowFast thresholds
        action_confidence_threshold: float = 0.4,
        violence_threshold: float = 0.6,
        # Hybrid fusion settings
        require_both: bool = True,
        action_weight: float = 0.7,
        inference_interval: int = 30,  # Ignored now (thread runs continuously)
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """
        Initialize hybrid detector with background SlowFast thread.

        Args:
            pose_detector: Existing FightDetector instance (YOLO-Pose)
            slowfast_detector: SlowFastDetector instance (or None to create)
            buffer_size: Frames to buffer for temporal analysis
            device: 'cuda' or 'cpu'
            body_proximity_threshold: Max distance for body proximity
            limb_proximity_threshold: Max distance for limb contacts
            action_confidence_threshold: Minimum action classification confidence
            violence_threshold: Minimum probability to classify action as violent
            require_both: If True, require BOTH pose and action signals
            action_weight: Weight for action signal in final decision (0-1)
            inference_interval: (legacy, ignored - thread runs continuously)
        """
        self.pose_detector = pose_detector
        self.buffer_size = buffer_size
        self.device = device

        # Initialize SlowFast detector (auto-detects RWF fine-tuned if available)
        if slowfast_detector is None:
            print("[HybridDetector] Initializing SlowFast detector...")
            try:
                # Use factory function for automatic model selection
                self.slowfast_detector = create_slowfast_detector(
                    model_type="auto",  # Auto-detect: prefer RWF if available
                    device=device,
                    confidence_threshold=action_confidence_threshold,
                    violence_threshold=violence_threshold
                )
            except Exception as e:
                print(f"[HybridDetector] Factory failed, falling back to Kinetics-400: {e}")
                self.slowfast_detector = SlowFastDetector(
                    labels_path="models/kinetics400_labels.json",
                    device=device,
                    confidence_threshold=action_confidence_threshold,
                    violence_threshold=violence_threshold
                )
        else:
            self.slowfast_detector = slowfast_detector

        # Frame buffer for temporal analysis (receives EVERY frame)
        self.frame_buffer = TemporalFrameBuffer(
            buffer_size=buffer_size,
            input_size=(224, 224)
        )

        # Thresholds
        self.body_proximity_threshold = body_proximity_threshold
        self.limb_proximity_threshold = limb_proximity_threshold
        self.action_confidence_threshold = action_confidence_threshold
        self.violence_threshold = violence_threshold

        # Fusion parameters
        self.require_both = require_both
        self.action_weight = action_weight
        self.pose_weight = 1.0 - action_weight
        self.inference_interval = inference_interval  # Keep for profiling display

        # State tracking
        self.frame_count = 0
        self.last_action_result = None  # Latest result from SlowFast thread
        self._fight_detected = False

        # Incident counting (cooldown-based, not per-frame)
        self._total_incidents = 0
        self.last_fight_event_time = 0.0  # Wall-clock time of last counted incident

        # Fight hold mechanism
        self.fight_hold_duration = 45
        self.last_fight_frame = -999

        # ---- SlowFast background thread ----
        self._slowfast_lock = threading.Lock()
        self._slowfast_result = None  # Protected by _slowfast_lock
        self._slowfast_result_time = 0.0  # Wall-clock time of last SlowFast result
        self._slowfast_running = True
        self._slowfast_thread = threading.Thread(
            target=self._slowfast_worker,
            daemon=True,
            name="SlowFast-Worker"
        )
        self._slowfast_inference_count = 0
        self._slowfast_last_time_ms = 0.0

        # Buffer-throttle counter: add frame to SlowFast buffer every 2nd frame
        # (halves preprocessing work while keeping enough temporal coverage)
        self._buf_frame_counter = 0

        # Statistics
        self.stats = {
            'total_frames': 0,
            'pose_detections': 0,
            'action_inferences': 0,
            'violence_detected': 0,
            'false_positives_avoided': 0,
            'detection_history': deque(maxlen=1000)
        }

        # Performance profiling (capped deques to prevent unbounded growth)
        self.profiling_stats = {
            'yolo_times': deque(maxlen=200),
            'slowfast_times': deque(maxlen=200),
            'fusion_times': deque(maxlen=200),
            'buffer_times': deque(maxlen=200),
            'slowfast_calls': 0,
            'slowfast_skips': 0,
            'yolo_calls': 0,
            'frames_processed': 0,
            'last_slowfast_time': 0.0,
        }

        print("")
        print("=" * 60)
        print("[HybridDetector] OPTIMIZED MODE (Background Thread)")
        print("=" * 60)
        print(f"  Mode: {'BOTH required' if require_both else 'Fusion weighted'}")
        print(f"  Weights: Pose={self.pose_weight:.2f}, Action={self.action_weight:.2f}")
        print(f"  Buffer size: {buffer_size} frames")
        print(f"  SlowFast: BACKGROUND THREAD (non-blocking)")
        print("=" * 60)
        print("")

        # Start the background thread
        self._slowfast_thread.start()
        print("[HybridDetector] SlowFast background thread started")

    def _slowfast_worker(self):
        """
        Background thread: runs SlowFast inference continuously.
        Picks up the latest buffer contents whenever the buffer is ready.
        Sleeps briefly between runs to avoid spinning.
        """
        print("[SlowFast-Thread] Worker started")
        while self._slowfast_running:
            try:
                if not self.frame_buffer.is_ready():
                    time.sleep(0.05)  # Wait for buffer to fill
                    continue

                t0 = time.perf_counter()
                action_result = self.slowfast_detector.detect(self.frame_buffer, top_k=3)
                t1 = time.perf_counter()

                elapsed_ms = (t1 - t0) * 1000
                self._slowfast_last_time_ms = elapsed_ms
                self._slowfast_inference_count += 1

                if action_result:
                    # Publish result (thread-safe)
                    with self._slowfast_lock:
                        self._slowfast_result = action_result
                        self._slowfast_result_time = time.time()

                    # Update profiling
                    self.profiling_stats['slowfast_times'].append(t1 - t0)
                    self.profiling_stats['slowfast_calls'] += 1
                    self.profiling_stats['last_slowfast_time'] = t1 - t0

                    if self._slowfast_inference_count % 5 == 1:
                        print(f"[SlowFast-Thread] #{self._slowfast_inference_count}: "
                              f"'{action_result.action}' ({action_result.confidence:.1%}) "
                              f"Violent={action_result.is_violent} | {elapsed_ms:.0f}ms")

                # Brief pause to avoid spinning (SlowFast is ~100-200ms anyway)
                time.sleep(0.01)

            except Exception as e:
                print(f"[SlowFast-Thread] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Back off on error

        print("[SlowFast-Thread] Worker stopped")

    def _get_latest_action_result(self) -> tuple:
        """Read latest SlowFast result (non-blocking, thread-safe).
        Returns (ActionResult|None, result_age_seconds).
        """
        with self._slowfast_lock:
            age = time.time() - self._slowfast_result_time if self._slowfast_result else float('inf')
            return self._slowfast_result, age

    def stop(self):
        """Stop the background SlowFast thread."""
        self._slowfast_running = False
        if self._slowfast_thread.is_alive():
            self._slowfast_thread.join(timeout=2.0)
        print("[HybridDetector] Stopped")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_count: int,
        run_yolo: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame through hybrid detection pipeline.

        IMPORTANT: This receives EVERY frame for the temporal buffer,
        but YOLO only runs when run_yolo=True.

        Args:
            frame: Input frame (BGR from OpenCV)
            frame_count: Current frame number
            run_yolo: Whether to run YOLO-Pose this frame

        Returns:
            (annotated_frame, detection_info) tuple
        """
        self.frame_count = frame_count
        self.stats['total_frames'] += 1
        self.profiling_stats['frames_processed'] += 1

        # ---- Add frame to temporal buffer every 2nd frame (halves preprocessing cost) ----
        _t_buf = time.perf_counter()
        self._buf_frame_counter += 1
        if self._buf_frame_counter % 2 == 0:
            self.frame_buffer.add_frame(frame)
        self.profiling_stats['buffer_times'].append(time.perf_counter() - _t_buf)

        # ---- Stage 1: YOLO-Pose (only when run_yolo=True) ----
        _t_yolo = time.perf_counter()
        annotated_frame, pose_info = self.pose_detector.process_frame(
            frame, frame_count, run_detection=run_yolo
        )
        _t_yolo_end = time.perf_counter()
        self.profiling_stats['yolo_times'].append(_t_yolo_end - _t_yolo)
        if run_yolo:
            self.profiling_stats['yolo_calls'] += 1

        pose_detected = pose_info.get('fight_detected', False)
        pose_confidence = pose_info.get('confidence', 0.0)

        if frame_count % 60 == 0:
            people_count = pose_info.get('people_count', 0)
            buf_len = len(self.frame_buffer)
            print(f"[Hybrid] Frame {frame_count}: People={people_count}, "
                  f"PoseFight={pose_detected}, Conf={pose_confidence:.1f}%, "
                  f"Buffer={buf_len}/{self.buffer_size} ({'ready' if self.frame_buffer.is_ready() else 'filling'}), "
                  f"SlowFast inferences={self._slowfast_inference_count}")

        if pose_detected:
            self.stats['pose_detections'] += 1

        # ---- Stage 2: Read latest SlowFast result (non-blocking) ----
        action_result, action_age = self._get_latest_action_result()
        action_detected = False
        action_confidence = 0.0

        if action_result:
            action_detected = action_result.is_violent
            action_confidence = action_result.confidence
            self.last_action_result = action_result
            self.last_action_result_age = action_age

        # ---- Hybrid decision fusion ----
        _t_fuse = time.perf_counter()
        fight_detected, hybrid_confidence, decision_reason = self._fuse_detections(
            pose_detected=pose_detected,
            pose_confidence=pose_confidence,
            action_detected=action_detected,
            action_confidence=action_confidence,
            people_count=pose_info.get('people_count', 0),
            action_age=action_age
        )
        self.profiling_stats['fusion_times'].append(time.perf_counter() - _t_fuse)

        # Track false positives avoided
        if pose_detected and not action_detected and action_result:
            self.stats['false_positives_avoided'] += 1

        # Apply fight hold mechanism + incident counting
        if fight_detected:
            # COUNT INCIDENTS: one per 5-second window (not per frame)
            current_time = time.time()
            if (current_time - self.last_fight_event_time) > 5.0:
                self._total_incidents += 1
                self.last_fight_event_time = current_time
                print(f"[Hybrid] NEW INCIDENT #{self._total_incidents} | {decision_reason}")

            self.last_fight_frame = frame_count
            self.stats['violence_detected'] += 1
        elif (frame_count - self.last_fight_frame) <= self.fight_hold_duration:
            fight_detected = True
            hybrid_confidence = max(hybrid_confidence, 40.0)
            decision_reason = f"Fight hold ({frame_count - self.last_fight_frame}/{self.fight_hold_duration})"

        self._fight_detected = fight_detected

        # ---- Build detection info ----
        detection_info = {
            'fight_detected': fight_detected,
            'confidence': hybrid_confidence,
            'pose_detected': pose_detected,
            'pose_confidence': pose_confidence,
            'action_detected': action_detected,
            'action_confidence': action_confidence,
            'action_class': action_result.action if action_result else None,
            'decision_reason': decision_reason,
            'people_count': pose_info.get('people_count', 0),
            'people_names': pose_info.get('people_names', []),
            'timestamp': datetime.now().isoformat()
        }

        self.stats['detection_history'].append({
            'frame': frame_count,
            'fight': fight_detected,
            'confidence': hybrid_confidence,
            'pose': pose_detected,
            'action': action_detected
        })

        # ---- Draw annotations based on hybrid decision ----
        poses = pose_info.get('poses', [])
        fight_areas = pose_info.get('fight_areas', [])

        if fight_detected:
            # If fight is confirmed but no boxes (e.g. action-only detection or
            # YOLO saw only one person), build fallback areas from available poses
            if not fight_areas:
                fight_areas = self._build_fallback_areas(poses, annotated_frame.shape)

            annotated_frame = self._draw_confirmed_fight(
                annotated_frame, poses, fight_areas, hybrid_confidence
            )
        elif pose_detected and action_result:
            annotated_frame = self._draw_filtered_info(annotated_frame, action_result)

        # Draw action overlay if SlowFast has produced results
        if action_result and self.frame_buffer.is_ready():
            annotated_frame = self._draw_action_overlay(annotated_frame, action_result)

        return annotated_frame, detection_info

    def _fuse_detections(
        self,
        pose_detected: bool,
        pose_confidence: float,
        action_detected: bool,
        action_confidence: float,
        people_count: int = 2,
        action_age: float = float('inf')
    ) -> Tuple[bool, float, str]:
        """Fuse pose and action detection signals."""
        pose_conf_norm = pose_confidence / 100.0
        action_conf_norm = action_confidence

        # DEBUG: Print raw values to understand model behavior
        if action_detected or pose_detected:
            print(f"[DEBUG] Fusion Input: Pose={pose_detected} ({pose_conf_norm:.2f}), "
                  f"Action={action_detected} ({action_conf_norm:.2f}), Age={action_age:.1f}s")

        # SMART FUSION LOGIC:
        # 1. SlowFast says "Fight" -> CONFIRM (always trusted)
        # 2. SlowFast says "NonFight" with HIGH confidence AND fresh result -> VETO
        # 3. SlowFast is unsure / stale / not ready -> Fallback to YOLO-Pose
        # 4. YOLO very confident (>80%) -> Can override weak NonFight veto

        VETO_CONF_THRESHOLD = 0.70   # SlowFast must be >70% confident to veto
        RESULT_MAX_AGE_SEC  = 3.0    # SlowFast result older than 3s is considered stale
        POSE_OVERRIDE_CONF  = 0.80   # YOLO can override NonFight if this confident

        if self.require_both:
            # Strict mode (Legacy)
            if pose_detected and action_detected:
                confidence = (pose_conf_norm * self.pose_weight +
                            action_conf_norm * self.action_weight)
                return True, confidence * 100, "Both pose and action detected violence"
            return False, 0.0, "Strict mode: requires both signals"
        else:
            # Smart Mode

            # 1. SlowFast confirms violence -> trust it (requires >=2 people)
            if action_detected:
                if people_count < 2 and action_conf_norm < 0.8:
                    return False, 0.0, f"Action ({action_conf_norm:.2f}) ignored: only {people_count} person"
                fused = action_conf_norm
                reason = f"Action detected violence: {action_conf_norm:.1%}"
                return True, fused * 100, reason

            # 2. Conditional VETO: only if SlowFast is fresh AND very confident about NonFight
            result_is_fresh = action_age < RESULT_MAX_AGE_SEC
            result_is_confident = action_conf_norm >= VETO_CONF_THRESHOLD
            pose_is_very_confident = pose_conf_norm >= POSE_OVERRIDE_CONF
            action_status = "pending"  # Reason string for fallback logging

            if self.last_action_result and not action_detected:
                if result_is_fresh and result_is_confident:
                    # Strong VETO — unless YOLO is extremely confident
                    if pose_is_very_confident and pose_detected:
                        # YOLO override: trust YOLO when it's very sure
                        fused = pose_conf_norm * 0.8
                        print(f"[Fusion] YOLO override: SlowFast says NonFight "
                              f"({action_conf_norm:.0%}) but YOLO very confident ({pose_conf_norm:.0%})")
                        return True, fused * 100, f"YOLO override (conf={pose_conf_norm:.0%})"
                    if pose_detected:
                        print(f"[Fusion] VETO: SlowFast NonFight ({action_conf_norm:.0%}, "
                              f"age={action_age:.1f}s) overrides YOLO ({pose_conf_norm:.0%})")
                    return False, 0.0, f"SlowFast veto (NonFight {action_conf_norm:.0%})"
                else:
                    # Stale or low-confidence NonFight -> don't veto, fallthrough to YOLO
                    action_status = "stale" if not result_is_fresh else f"nonfight ({action_conf_norm:.0%})"
                    if pose_detected:
                        print(f"[Fusion] Skipping weak VETO ({action_status}), using YOLO")

            # 3. Fallback: action model not ready / veto skipped -> use YOLO
            if pose_detected:
                fused = pose_conf_norm * 0.75
                if fused >= 0.55:  # Slightly lower threshold than before (was 0.60)
                    return True, fused * 100, f"Pose detected (conf={pose_conf_norm:.0%}), action={action_status}"

            return False, 0.0, "No signals detected"

    def _draw_action_overlay(
        self,
        frame: np.ndarray,
        action_result: ActionResult,
    ) -> np.ndarray:
        """Draw action recognition results directly on the frame (no allocation)."""
        h, w = frame.shape[:2]

        # Semi-transparent dark background over a small bottom-left region
        bx0, by0, bx1, by1 = 5, h - 78, min(w, 320), h - 5
        roi = frame[by0:by1, bx0:bx1]
        roi[:] = (roi * 0.35).astype(np.uint8)

        # Draw lines on the darkened ROI
        lines = [
            (f"Action: {action_result.action}", 0.55, (255, 255, 255), 1),
            (f"Conf: {action_result.confidence * 100:.1f}%", 0.48, (200, 200, 200), 1),
            ("VIOLENT" if action_result.is_violent else "Non-violent",
             0.55, (60, 60, 255) if action_result.is_violent else (50, 220, 50), 2),
            (f"SF: {self._slowfast_last_time_ms:.0f}ms #{self._slowfast_inference_count}",
             0.4, (140, 140, 140), 1),
        ]
        y = h - 68
        for text, scale, color, thick in lines:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thick, cv2.LINE_AA)
            y += 18

        return frame

    def _build_fallback_areas(
        self,
        poses: List[Dict],
        frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Build fallback bounding boxes when fight_areas is empty.

        Priority:
        1. Union box over all visible poses
        2. Full-frame banner if no pose keypoints are available
        """
        h, w = frame_shape[:2]

        if poses:
            all_xs, all_ys = [], []
            for p in poses:
                kps = p.get('keypoints', [])
                for kp in kps:
                    if len(kp) >= 3 and kp[2] > 0.3:
                        all_xs.append(int(kp[0]))
                        all_ys.append(int(kp[1]))

            if all_xs and all_ys:
                pad = 20
                x1 = max(0, min(all_xs) - pad)
                y1 = max(0, min(all_ys) - pad)
                x2 = min(w - 1, max(all_xs) + pad)
                y2 = min(h - 1, max(all_ys) + pad)
                return [(x1, y1, x2, y2)]

        # No poses at all → thin banner at top of frame
        return [(0, 0, w - 1, 50)]

    def _draw_confirmed_fight(
        self,
        frame: np.ndarray,
        poses: List[Dict],
        fight_areas: List[Tuple[int, int, int, int]],
        confidence: float
    ) -> np.ndarray:
        """
        Draw fight annotations when hybrid detector CONFIRMS a fight.

        Always draws:
        1. Red skeletons on all visible people
        2. Red bounding boxes around fight area(s)
        3. Prominent FIGHT banner at the top of the frame
        """
        h, w = frame.shape[:2]

        # 1. Red skeletons on all visible people
        for p in poses:
            kps = p.get('keypoints', [])
            self._draw_skeleton_colored(frame, kps, (0, 0, 255))

        # 2. Red bounding box + label for each fight area
        for (x1, y1, x2, y2) in fight_areas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"FIGHT! {confidence:.0f}%"
            label_y = y1 - 10 if y1 > 20 else y2 + 25
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 3. Always-visible alert banner at the top
        banner_h = 40
        # Semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        banner_text = f"!!! VIOLENCE DETECTED  {confidence:.0f}%  (Incident #{self._total_incidents}) !!!"
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(0, (w - text_size[0]) // 2)
        cv2.putText(frame, banner_text, (text_x, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def _draw_skeleton_colored(self, frame: np.ndarray, kps, color: Tuple[int, int, int]):
        """Draw skeleton with specified color."""
        conns = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                 (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

        for kp in kps:
            if len(kp) >= 3 and kp[2] > 0.5:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)

        for a, b in conns:
            if a < len(kps) and b < len(kps):
                if len(kps[a]) >= 3 and len(kps[b]) >= 3:
                    if kps[a][2] > 0.5 and kps[b][2] > 0.5:
                        p1 = (int(kps[a][0]), int(kps[a][1]))
                        p2 = (int(kps[b][0]), int(kps[b][1]))
                        cv2.line(frame, p1, p2, color, 2)

    def _draw_filtered_info(
        self,
        frame: np.ndarray,
        action_result: ActionResult
    ) -> np.ndarray:
        """Draw info when pose detected fight but action says non-violent."""
        cv2.putText(frame, f"Detected: {action_result.action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Non-violent interaction", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    # ---- Statistics & profiling ----

    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        stats = self.stats.copy()
        slowfast_stats = self.slowfast_detector.get_statistics()
        stats['slowfast'] = slowfast_stats

        if stats['total_frames'] > 0:
            stats['pose_detection_rate'] = stats['pose_detections'] / stats['total_frames']
            stats['violence_rate'] = stats['violence_detected'] / stats['total_frames']

        if stats['pose_detections'] > 0:
            stats['false_positive_reduction'] = (
                stats['false_positives_avoided'] / stats['pose_detections']
            )
        return stats

    def reset_statistics(self):
        """Reset all statistics and fight state."""
        self.stats = {
            'total_frames': 0,
            'pose_detections': 0,
            'action_inferences': 0,
            'violence_detected': 0,
            'false_positives_avoided': 0,
            'detection_history': deque(maxlen=1000)
        }
        self._total_incidents = 0
        self.last_fight_event_time = 0.0
        self._fight_detected = False
        self.last_fight_frame = -999
        self.last_action_result = None
        self.last_action_result_age = float('inf')
        with self._slowfast_lock:
            self._slowfast_result = None
            self._slowfast_result_time = 0.0
        self.reset_profiling_stats()

    def get_profiling_stats(self) -> Dict:
        """Get detailed performance profiling statistics."""
        ps = self.profiling_stats
        frames = ps['frames_processed'] or 1

        yolo_avg = (sum(ps['yolo_times']) / len(ps['yolo_times']) * 1000) if ps['yolo_times'] else 0
        slowfast_avg = (sum(ps['slowfast_times']) / len(ps['slowfast_times']) * 1000) if ps['slowfast_times'] else 0
        fusion_avg = (sum(ps['fusion_times']) / len(ps['fusion_times']) * 1000) if ps['fusion_times'] else 0
        buffer_avg = (sum(ps['buffer_times']) / len(ps['buffer_times']) * 1000) if ps['buffer_times'] else 0

        yolo_max = max(ps['yolo_times']) * 1000 if ps['yolo_times'] else 0
        slowfast_max = max(ps['slowfast_times']) * 1000 if ps['slowfast_times'] else 0

        total_sync = yolo_avg + fusion_avg + buffer_avg
        slowfast_freq = ps['slowfast_calls'] / frames if frames > 0 else 0

        total_all = yolo_avg + slowfast_avg + fusion_avg + buffer_avg
        if total_all > 0:
            breakdown = {
                'yolo_pct': (yolo_avg / total_all) * 100,
                'slowfast_pct': (slowfast_avg / total_all) * 100,
                'fusion_pct': (fusion_avg / total_all) * 100,
                'buffer_pct': (buffer_avg / total_all) * 100,
            }
        else:
            breakdown = {'yolo_pct': 0, 'slowfast_pct': 0, 'fusion_pct': 0, 'buffer_pct': 0}

        return {
            'yolo_pose': {
                'avg_ms': yolo_avg,
                'max_ms': yolo_max,
                'calls': ps['yolo_calls'],
            },
            'slowfast': {
                'avg_ms': slowfast_avg,
                'max_ms': slowfast_max,
                'calls': ps['slowfast_calls'],
                'skips': ps['slowfast_skips'],
                'last_time_ms': ps['last_slowfast_time'] * 1000,
                'run_frequency': slowfast_freq,
            },
            'fusion': {
                'avg_ms': fusion_avg,
                'calls': len(ps['fusion_times']),
            },
            'buffer': {
                'avg_ms': buffer_avg,
            },
            'summary': {
                'frames_processed': frames,
                'total_sync_ms': total_sync,
                'theoretical_fps': 1000 / total_sync if total_sync > 0 else 0,
                'bottleneck': 'YOLO-Pose' if yolo_avg >= slowfast_avg else 'SlowFast (bg)',
            },
            'breakdown_pct': breakdown,
        }

    def reset_profiling_stats(self):
        """Reset profiling statistics."""
        self.profiling_stats = {
            'yolo_times': deque(maxlen=200),
            'slowfast_times': deque(maxlen=200),
            'fusion_times': deque(maxlen=200),
            'buffer_times': deque(maxlen=200),
            'slowfast_calls': 0,
            'slowfast_skips': 0,
            'yolo_calls': 0,
            'frames_processed': 0,
            'last_slowfast_time': 0.0,
        }

    # ---- Compatibility properties for app.py ----

    @property
    def fight_detected(self):
        """Return HYBRID fight detection result (after SlowFast filtering)."""
        return self._fight_detected

    @property
    def pose_history(self):
        return self.pose_detector.pose_history

    @property
    def analytics(self):
        """Return filtered analytics using hybrid detection results.

        total_detections = number of distinct incidents (5-sec cooldown),
        NOT the raw frame count. This is what the UI 'Fights' counter reads.
        """
        base_analytics = self.pose_detector.analytics.copy()
        base_analytics['total_detections'] = self._total_incidents      # Incident count
        base_analytics['pose_detections_raw'] = self.stats['pose_detections']
        base_analytics['false_positives_avoided'] = self.stats['false_positives_avoided']
        base_analytics['action_inferences'] = self._slowfast_inference_count
        return base_analytics

    @property
    def body_proximity_threshold(self):
        return self.pose_detector.body_proximity_threshold

    @body_proximity_threshold.setter
    def body_proximity_threshold(self, value):
        self.pose_detector.body_proximity_threshold = value

    @property
    def limb_proximity_threshold(self):
        return self.pose_detector.limb_proximity_threshold

    @limb_proximity_threshold.setter
    def limb_proximity_threshold(self, value):
        self.pose_detector.limb_proximity_threshold = value

    @property
    def fight_hold_duration(self):
        return self.pose_detector.fight_hold_duration

    @fight_hold_duration.setter
    def fight_hold_duration(self, value):
        self.pose_detector.fight_hold_duration = value

    def update_thresholds(
        self,
        body_proximity_threshold: Optional[float] = None,
        limb_proximity_threshold: Optional[float] = None,
        action_confidence_threshold: Optional[float] = None,
        violence_threshold: Optional[float] = None,
        action_weight: Optional[float] = None
    ):
        """Update detection thresholds."""
        if body_proximity_threshold is not None:
            self.pose_detector.body_proximity_threshold = body_proximity_threshold
        if limb_proximity_threshold is not None:
            self.pose_detector.limb_proximity_threshold = limb_proximity_threshold
        if action_confidence_threshold is not None:
            self.action_confidence_threshold = action_confidence_threshold
            self.slowfast_detector.confidence_threshold = action_confidence_threshold
        if violence_threshold is not None:
            self.violence_threshold = violence_threshold
            self.slowfast_detector.violence_threshold = violence_threshold
        if action_weight is not None:
            self.action_weight = action_weight
            self.pose_weight = 1.0 - action_weight
        print(f"[HybridDetector] Updated thresholds")

    def __repr__(self) -> str:
        return (f"HybridFightDetector(frames={self.stats['total_frames']}, "
                f"violence={self.stats['violence_detected']}, "
                f"FP_avoided={self.stats['false_positives_avoided']})")
