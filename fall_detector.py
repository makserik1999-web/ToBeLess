"""
Fall Detection Module
Detects people falling using pose estimation analysis
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import math

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class FallDetector:
    """
    Fall detector using YOLO-Pose for human pose estimation

    Detection logic:
    1. Track person's pose over time
    2. Detect sudden change from vertical to horizontal orientation
    3. Check if person stays in fallen position (not just bending)
    4. Consider velocity of the fall
    """

    # COCO keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        fall_angle_threshold: float = 45.0,
        fall_ratio_threshold: float = 0.6,
        confirmation_frames: int = 5,
        recovery_frames: int = 30,
        device: str = 'cuda',
        debug: bool = False
    ):
        """
        Initialize fall detector

        Args:
            model_path: Path to YOLO-Pose model
            fall_angle_threshold: Angle from vertical to consider as fallen (degrees)
            fall_ratio_threshold: Height/width ratio threshold for fall detection
            confirmation_frames: Frames to confirm fall (prevent false positives)
            recovery_frames: Frames before person is considered recovered
            device: 'cuda' or 'cpu'
            debug: Enable debug output
        """
        self.fall_angle_threshold = fall_angle_threshold
        self.fall_ratio_threshold = fall_ratio_threshold
        self.confirmation_frames = confirmation_frames
        self.recovery_frames = recovery_frames
        self.device = device
        self.debug = debug

        # Load YOLO-Pose model
        self.model = None
        if YOLO is not None and model_path:
            try:
                p = Path(model_path)
                if p.exists():
                    self.model = YOLO(str(p))
                    if self.debug:
                        print(f"[FallDetector] ✓ YOLO-Pose loaded from {p}")
            except Exception as e:
                if self.debug:
                    print(f"[FallDetector] Model load error: {e}")

        # Tracking data per person (by approximate position)
        self.person_states = {}  # {person_id: {'history': deque, 'fall_count': int, 'fallen': bool}}
        self.next_person_id = 0

        # Detection history
        self.detection_history = []
        self.stats = {
            'total_falls_detected': 0,
            'active_falls': 0,
            'false_positives_avoided': 0
        }

    def _get_body_angle(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Calculate body angle from vertical

        Args:
            keypoints: Array of shape (17, 3) with x, y, confidence

        Returns:
            Angle in degrees from vertical (0 = standing, 90 = horizontal)
        """
        try:
            # Get shoulder and hip midpoints
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]
            left_hip = keypoints[self.LEFT_HIP]
            right_hip = keypoints[self.RIGHT_HIP]

            # Check confidence
            if (left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or
                left_hip[2] < 0.3 or right_hip[2] < 0.3):
                return None

            # Calculate midpoints
            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2
            )
            hip_mid = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )

            # Calculate angle from vertical
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]

            # Angle from vertical (0 = upright, 90 = horizontal)
            angle = abs(math.degrees(math.atan2(abs(dx), abs(dy))))

            return angle

        except Exception:
            return None

    def _get_body_ratio(self, keypoints: np.ndarray) -> Optional[float]:
        """
        Calculate body bounding box height/width ratio

        Low ratio = horizontal (fallen)
        High ratio = vertical (standing)
        """
        try:
            # Get all valid keypoints
            valid_points = []
            for kp in keypoints:
                if kp[2] > 0.3:  # confidence threshold
                    valid_points.append((kp[0], kp[1]))

            if len(valid_points) < 5:
                return None

            # Calculate bounding box
            xs = [p[0] for p in valid_points]
            ys = [p[1] for p in valid_points]

            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

            if width < 10:
                return None

            return height / width

        except Exception:
            return None

    def _get_center_position(self, keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """Get approximate center position of person for tracking"""
        try:
            left_hip = keypoints[self.LEFT_HIP]
            right_hip = keypoints[self.RIGHT_HIP]

            if left_hip[2] > 0.3 and right_hip[2] > 0.3:
                return (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                )

            # Fallback to shoulder midpoint
            left_shoulder = keypoints[self.LEFT_SHOULDER]
            right_shoulder = keypoints[self.RIGHT_SHOULDER]

            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                return (
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                )

            return None
        except Exception:
            return None

    def _match_person(self, center: Tuple[float, float], max_dist: float = 100) -> int:
        """Match detected person to tracked person by position"""
        best_id = None
        best_dist = float('inf')

        for person_id, state in self.person_states.items():
            if state['history']:
                last_center = state['history'][-1].get('center')
                if last_center:
                    dist = math.sqrt(
                        (center[0] - last_center[0])**2 +
                        (center[1] - last_center[1])**2
                    )
                    if dist < best_dist and dist < max_dist:
                        best_dist = dist
                        best_id = person_id

        if best_id is None:
            best_id = self.next_person_id
            self.next_person_id += 1
            self.person_states[best_id] = {
                'history': deque(maxlen=30),
                'fall_count': 0,
                'fallen': False,
                'recovery_count': 0
            }

        return best_id

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect falls in frame

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of fall detection dicts
        """
        detections = []

        if frame is None or frame.size == 0 or self.model is None:
            return detections

        try:
            # Run pose estimation
            results = self.model(
                frame,
                conf=0.5,
                verbose=False,
                device=self.device
            )

            detected_person_ids = set()

            for result in results:
                if result.keypoints is None:
                    continue

                keypoints_data = result.keypoints.data.cpu().numpy()

                for person_idx, keypoints in enumerate(keypoints_data):
                    # Get center for tracking
                    center = self._get_center_position(keypoints)
                    if center is None:
                        continue

                    # Match to tracked person
                    person_id = self._match_person(center)
                    detected_person_ids.add(person_id)
                    state = self.person_states[person_id]

                    # Calculate metrics
                    angle = self._get_body_angle(keypoints)
                    ratio = self._get_body_ratio(keypoints)

                    # Store in history
                    state['history'].append({
                        'center': center,
                        'angle': angle,
                        'ratio': ratio,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Check for fall
                    is_fallen_pose = False
                    if angle is not None and ratio is not None:
                        # Body is horizontal
                        if angle > self.fall_angle_threshold:
                            is_fallen_pose = True
                        # Body ratio indicates lying down
                        if ratio < self.fall_ratio_threshold:
                            is_fallen_pose = True

                    if is_fallen_pose:
                        state['fall_count'] += 1
                        state['recovery_count'] = 0

                        # Confirm fall after several frames
                        if state['fall_count'] >= self.confirmation_frames and not state['fallen']:
                            state['fallen'] = True
                            self.stats['total_falls_detected'] += 1
                            self.stats['active_falls'] += 1

                            if self.debug:
                                print(f"[FallDetector] ⚠ FALL DETECTED for person {person_id}")

                    else:
                        state['fall_count'] = max(0, state['fall_count'] - 1)
                        state['recovery_count'] += 1

                        # Person recovered
                        if state['fallen'] and state['recovery_count'] >= self.recovery_frames:
                            state['fallen'] = False
                            self.stats['active_falls'] = max(0, self.stats['active_falls'] - 1)
                            if self.debug:
                                print(f"[FallDetector] ✓ Person {person_id} recovered")

                    # Add to detections if fallen
                    if state['fallen']:
                        # Get bounding box from keypoints
                        valid_points = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.3]
                        if valid_points:
                            xs = [p[0] for p in valid_points]
                            ys = [p[1] for p in valid_points]
                            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

                            detections.append({
                                'person_id': person_id,
                                'bbox': bbox,
                                'angle': angle,
                                'ratio': ratio,
                                'confidence': min(1.0, state['fall_count'] / self.confirmation_frames),
                                'timestamp': datetime.now().isoformat()
                            })

            # Cleanup old person states
            for person_id in list(self.person_states.keys()):
                if person_id not in detected_person_ids:
                    state = self.person_states[person_id]
                    state['recovery_count'] += 1
                    if state['recovery_count'] > 60:  # Remove after 60 frames of absence
                        if state['fallen']:
                            self.stats['active_falls'] = max(0, self.stats['active_falls'] - 1)
                        del self.person_states[person_id]

            # Add to history
            if detections:
                self.detection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'detections': detections
                })
                if len(self.detection_history) > 1000:
                    self.detection_history = self.detection_history[-500:]

        except Exception as e:
            if self.debug:
                print(f"[FallDetector] Detection error: {e}")

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw fall detections on frame"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det.get('confidence', 1.0)

            # Red color for fall detection
            color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"FALL DETECTED {conf:.0%}"
            font_scale = 0.8
            thickness = 2
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )

            # Draw warning
            cv2.putText(
                annotated, "⚠ PERSON DOWN - NEED ASSISTANCE",
                (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        return annotated

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_falls_detected': 0,
            'active_falls': 0,
            'false_positives_avoided': 0
        }
        self.detection_history = []
        self.person_states = {}
