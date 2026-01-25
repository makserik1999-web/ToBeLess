"""
Hybrid Fight Detector: YOLO-Pose + SlowFast

Combines spatial pose detection with temporal action recognition
to reduce false positives in fight detection.

Approach:
1. YOLO-Pose detects WHERE people are and their spatial interactions
2. SlowFast analyzes WHAT action is happening temporally
3. Only trigger fight alert when BOTH signals agree

This eliminates false positives like: hugs, crowds, dancing, sports
"""

import cv2
import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from video_buffer import TemporalFrameBuffer
from slowfast_detector import SlowFastDetector, ActionResult


class HybridFightDetector:
    """
    Hybrid fight detector combining pose-based and action recognition approaches

    The detector uses a two-stage pipeline:
    Stage 1: YOLO-Pose spatial detection (fast, identifies potential interactions)
    Stage 2: SlowFast action classification (slower, verifies the action is violent)

    This dramatically reduces false positives while maintaining high accuracy.
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
        require_both: bool = True,  # Require both pose AND action signals
        action_weight: float = 0.85,  # How much to weight action vs pose (85% SlowFast, 15% Pose)
        inference_interval: int = 8,  # Run SlowFast every N frames
    ):
        """
        Initialize hybrid detector

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
            inference_interval: Run SlowFast every N frames (to save computation)
        """
        self.pose_detector = pose_detector
        self.buffer_size = buffer_size
        self.device = device

        # Initialize SlowFast detector
        if slowfast_detector is None:
            print("[HybridDetector] Initializing SlowFast detector...")
            self.slowfast_detector = SlowFastDetector(
                labels_path="models/kinetics400_labels.json",
                device=device,
                confidence_threshold=action_confidence_threshold,
                violence_threshold=violence_threshold
            )
        else:
            self.slowfast_detector = slowfast_detector

        # Frame buffer for temporal analysis
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
        self.inference_interval = inference_interval

        # State tracking
        self.frame_count = 0
        self.last_action_result = None
        self.last_pose_result = None
        self._fight_detected = False  # Hybrid fusion result (NOT pose-only)

        # Statistics
        self.stats = {
            'total_frames': 0,
            'pose_detections': 0,
            'action_inferences': 0,
            'violence_detected': 0,
            'false_positives_avoided': 0,  # Pose=fight, Action=non-fight
            'detection_history': deque(maxlen=1000)
        }

        print(f"[HybridDetector] Initialized")
        print(f"  Mode: {'BOTH required' if require_both else 'Fusion weighted'}")
        print(f"  Weights: Pose={self.pose_weight:.2f}, Action={self.action_weight:.2f}")
        print(f"  Inference interval: {inference_interval} frames")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_count: int
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame through hybrid detection pipeline

        Args:
            frame: Input frame (BGR from OpenCV)
            frame_count: Current frame number

        Returns:
            (annotated_frame, detection_info) tuple
        """
        self.frame_count = frame_count
        self.stats['total_frames'] += 1

        # Add frame to buffer
        self.frame_buffer.add_frame(frame)

        # Stage 1: YOLO-Pose spatial detection (always run, it's fast)
        annotated_frame, pose_info = self.pose_detector.process_frame(frame, frame_count)
        pose_detected = pose_info.get('fight_detected', False)
        pose_confidence = pose_info.get('confidence', 0.0)

        if pose_detected:
            self.stats['pose_detections'] += 1

        # Stage 2: SlowFast action recognition (run periodically)
        action_result = None
        action_detected = False
        action_confidence = 0.0

        # Run action recognition when buffer is ready and at inference interval
        if self.frame_buffer.is_ready() and frame_count % self.inference_interval == 0:
            action_result = self.slowfast_detector.detect(self.frame_buffer, top_k=3)

            if action_result:
                self.stats['action_inferences'] += 1
                self.last_action_result = action_result

                action_detected = action_result.is_violent
                action_confidence = action_result.confidence

        # Use last action result if we didn't just run inference
        elif self.last_action_result:
            action_result = self.last_action_result
            action_detected = action_result.is_violent
            action_confidence = action_result.confidence

        # Hybrid decision fusion
        fight_detected, hybrid_confidence, decision_reason = self._fuse_detections(
            pose_detected=pose_detected,
            pose_confidence=pose_confidence,
            action_detected=action_detected,
            action_confidence=action_confidence
        )

        # Track false positives avoided
        if pose_detected and not action_detected and action_result:
            # Pose said fight, but action said no violence
            self.stats['false_positives_avoided'] += 1

        if fight_detected:
            self.stats['violence_detected'] += 1

        # Update hybrid fight state (this is what app.py checks for alerts)
        self._fight_detected = fight_detected

        # Build detection info
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
            'timestamp': datetime.now().isoformat()
        }

        # Add to history
        self.stats['detection_history'].append({
            'frame': frame_count,
            'fight': fight_detected,
            'confidence': hybrid_confidence,
            'pose': pose_detected,
            'action': action_detected
        })

        # Draw action recognition overlay if available
        if action_result and self.frame_buffer.is_ready():
            annotated_frame = self._draw_action_overlay(
                annotated_frame,
                action_result,
                fight_detected
            )

        return annotated_frame, detection_info

    def _fuse_detections(
        self,
        pose_detected: bool,
        pose_confidence: float,
        action_detected: bool,
        action_confidence: float
    ) -> Tuple[bool, float, str]:
        """
        Fuse pose and action detection signals

        Args:
            pose_detected: Whether pose detector found fight
            pose_confidence: Pose detection confidence (0-100)
            action_detected: Whether action is violent
            action_confidence: Action classification confidence (0-1)

        Returns:
            (fight_detected, confidence, reason) tuple
        """
        # Normalize confidences to 0-1 range
        pose_conf_norm = pose_confidence / 100.0
        action_conf_norm = action_confidence

        if self.require_both:
            # Conservative: require BOTH signals
            if pose_detected and action_detected:
                # Both agree: fight
                confidence = (pose_conf_norm * self.pose_weight +
                            action_conf_norm * self.action_weight)
                return True, confidence * 100, "Both pose and action detected violence"

            elif pose_detected and not action_detected:
                # Pose says fight, action says no
                # This is likely a FALSE POSITIVE (hug, crowd, etc.)
                return False, pose_conf_norm * 100, "Pose detected, but action is non-violent"

            elif not pose_detected and action_detected:
                # Action says fight, but no physical proximity
                # Rare case, trust pose detector
                return False, action_conf_norm * 100, "Action detected, but no physical proximity"

            else:
                # Neither detected
                return False, 0.0, "No violence detected"

        else:
            # Fusion mode: weighted combination
            fused_confidence = (pose_conf_norm * self.pose_weight +
                              action_conf_norm * self.action_weight)

            # Require at least one signal to be positive
            if pose_detected or action_detected:
                # Apply threshold
                if fused_confidence >= 0.5:
                    reason = f"Fused detection (pose={pose_conf_norm:.2f}, action={action_conf_norm:.2f})"
                    return True, fused_confidence * 100, reason
                else:
                    return False, fused_confidence * 100, "Confidence below threshold"
            else:
                return False, 0.0, "No signals detected"

    def _draw_action_overlay(
        self,
        frame: np.ndarray,
        action_result: ActionResult,
        fight_detected: bool
    ) -> np.ndarray:
        """
        Draw action recognition results on frame

        Args:
            frame: Input frame
            action_result: ActionResult from SlowFast
            fight_detected: Whether fight was detected

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]

        # Background panel for action info
        panel_h = 120
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark background

        # Draw top action
        action_text = f"Action: {action_result.action}"
        conf_text = f"Confidence: {action_result.confidence * 100:.1f}%"

        cv2.putText(panel, action_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Violence indicator
        if action_result.is_violent:
            cv2.putText(panel, "VIOLENT ACTION", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(panel, "Non-violent", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Inference time
        time_text = f"{action_result.inference_time_ms:.1f}ms"
        cv2.putText(panel, time_text, (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Attach panel to bottom of frame
        result = np.vstack([frame, panel])

        return result

    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        stats = self.stats.copy()

        # Add SlowFast stats
        slowfast_stats = self.slowfast_detector.get_statistics()
        stats['slowfast'] = slowfast_stats

        # Calculate rates
        if stats['total_frames'] > 0:
            stats['pose_detection_rate'] = stats['pose_detections'] / stats['total_frames']
            stats['violence_rate'] = stats['violence_detected'] / stats['total_frames']

        if stats['pose_detections'] > 0:
            stats['false_positive_reduction'] = (
                stats['false_positives_avoided'] / stats['pose_detections']
            )

        return stats

    def reset_statistics(self):
        """Reset all statistics and fight state"""
        self.stats = {
            'total_frames': 0,
            'pose_detections': 0,
            'action_inferences': 0,
            'violence_detected': 0,
            'false_positives_avoided': 0,
            'detection_history': deque(maxlen=1000)
        }
        self._fight_detected = False
        self.last_action_result = None

    # Compatibility properties for app.py
    @property
    def fight_detected(self):
        """
        Return HYBRID fight detection result (after SlowFast filtering).

        IMPORTANT: This returns the filtered result, NOT the raw pose detection.
        This prevents false positives from hugging, crowds, dancing, etc.
        """
        return self._fight_detected

    @property
    def pose_history(self):
        """Forward pose_history from pose_detector for compatibility"""
        return self.pose_detector.pose_history

    @property
    def analytics(self):
        """
        Return filtered analytics using hybrid detection results.

        IMPORTANT: Uses violence_detected (filtered count) instead of
        pose_detector's total_detections (raw count) to avoid false positives
        from hugging, crowds, dancing etc.
        """
        # Start with pose detector analytics as base
        base_analytics = self.pose_detector.analytics.copy()

        # Override total_detections with FILTERED violence count
        # This is the key fix - we only count actual violence, not proximity
        base_analytics['total_detections'] = self.stats['violence_detected']

        # Add hybrid-specific stats
        base_analytics['pose_detections_raw'] = self.stats['pose_detections']
        base_analytics['false_positives_avoided'] = self.stats['false_positives_avoided']
        base_analytics['action_inferences'] = self.stats['action_inferences']

        return base_analytics

    @property
    def body_proximity_threshold(self):
        """Forward body_proximity_threshold from pose_detector"""
        return self.pose_detector.body_proximity_threshold

    @body_proximity_threshold.setter
    def body_proximity_threshold(self, value):
        """Set body_proximity_threshold on pose_detector"""
        self.pose_detector.body_proximity_threshold = value

    @property
    def limb_proximity_threshold(self):
        """Forward limb_proximity_threshold from pose_detector"""
        return self.pose_detector.limb_proximity_threshold

    @limb_proximity_threshold.setter
    def limb_proximity_threshold(self, value):
        """Set limb_proximity_threshold on pose_detector"""
        self.pose_detector.limb_proximity_threshold = value

    @property
    def fight_hold_duration(self):
        """Forward fight_hold_duration from pose_detector"""
        return self.pose_detector.fight_hold_duration

    @fight_hold_duration.setter
    def fight_hold_duration(self, value):
        """Set fight_hold_duration on pose_detector"""
        self.pose_detector.fight_hold_duration = value
        self.slowfast_detector.reset_statistics()

    def update_thresholds(
        self,
        body_proximity_threshold: Optional[float] = None,
        limb_proximity_threshold: Optional[float] = None,
        action_confidence_threshold: Optional[float] = None,
        violence_threshold: Optional[float] = None,
        action_weight: Optional[float] = None
    ):
        """Update detection thresholds"""
        if body_proximity_threshold is not None:
            self.body_proximity_threshold = body_proximity_threshold
            self.pose_detector.body_proximity_threshold = body_proximity_threshold

        if limb_proximity_threshold is not None:
            self.limb_proximity_threshold = limb_proximity_threshold
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
        """String representation"""
        return (f"HybridFightDetector(frames={self.stats['total_frames']}, "
                f"violence={self.stats['violence_detected']}, "
                f"FP_avoided={self.stats['false_positives_avoided']})")
