"""
Weapon Detection Module
Detects weapons (knives, guns, etc.) using YOLO object detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class WeaponDetector:
    """
    Weapon detector using YOLO model

    Can detect:
    - Knives (from COCO dataset)
    - Guns (requires custom model or specific weapon model)
    - Other weapons if custom model is provided
    """

    # COCO class IDs for potential weapons
    COCO_WEAPON_CLASSES = {
        43: 'knife',
        # Custom weapon model would have more classes
    }

    # Extended weapon classes for custom models
    WEAPON_CLASSES = {
        'knife': {'danger_level': 'high', 'color': (0, 0, 255)},
        'gun': {'danger_level': 'critical', 'color': (0, 0, 255)},
        'pistol': {'danger_level': 'critical', 'color': (0, 0, 255)},
        'rifle': {'danger_level': 'critical', 'color': (0, 0, 255)},
        'sword': {'danger_level': 'high', 'color': (0, 0, 255)},
        'bat': {'danger_level': 'medium', 'color': (0, 165, 255)},
        'hammer': {'danger_level': 'medium', 'color': (0, 165, 255)},
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        weapon_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = 'cuda',
        debug: bool = False
    ):
        """
        Initialize weapon detector

        Args:
            model_path: Path to general YOLO model (for knife detection)
            weapon_model_path: Path to specialized weapon model (optional)
            confidence_threshold: Minimum confidence for detection
            device: 'cuda' or 'cpu'
            debug: Enable debug output
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.debug = debug

        self.model = None
        self.weapon_model = None

        # Load general YOLO model (for COCO classes like knife)
        if YOLO is not None and model_path:
            try:
                p = Path(model_path)
                if p.exists():
                    self.model = YOLO(str(p))
                    print(f"[WeaponDetector] ✓ General YOLO loaded from {p}")
                else:
                    print(f"[WeaponDetector] General model not found: {p}")
            except Exception as e:
                print(f"[WeaponDetector] General YOLO load error: {e}")

        # Load specialized weapon model (pistol + knife)
        if YOLO is not None and weapon_model_path:
            try:
                p = Path(weapon_model_path)
                if p.exists():
                    self.weapon_model = YOLO(str(p))
                    classes = list(self.weapon_model.names.values())
                    print(f"[WeaponDetector] ✓ Weapon model loaded from {p} — classes: {classes}")
                else:
                    print(f"[WeaponDetector] Weapon model not found: {p} — run download_weapon_model.py")
            except Exception as e:
                print(f"[WeaponDetector] Weapon model load error: {e}")

        # Detection history for analytics
        self.detection_history = []
        self.stats = {
            'total_detections': 0,
            'knife_detections': 0,
            'gun_detections': 0,
            'other_weapon_detections': 0
        }

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect weapons in frame

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detection dicts with keys:
            - bbox: (x1, y1, x2, y2)
            - class_name: weapon type
            - confidence: detection confidence
            - danger_level: 'low', 'medium', 'high', 'critical'
        """
        detections = []

        if frame is None or frame.size == 0:
            return detections

        # Detect using general YOLO (COCO classes)
        if self.model is not None:
            try:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    verbose=False,
                    device=self.device
                )

                for result in results:
                    if result.boxes is None:
                        continue

                    for box in result.boxes:
                        cls_id = int(box.cls[0])

                        # Check if it's a weapon class
                        if cls_id in self.COCO_WEAPON_CLASSES:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            class_name = self.COCO_WEAPON_CLASSES[cls_id]

                            detection = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'class_name': class_name,
                                'confidence': conf,
                                'danger_level': self.WEAPON_CLASSES.get(
                                    class_name, {}
                                ).get('danger_level', 'medium'),
                                'timestamp': datetime.now().isoformat()
                            }
                            detections.append(detection)

            except Exception as e:
                if self.debug:
                    print(f"[WeaponDetector] Detection error: {e}")

        # Detect using specialized weapon model
        if self.weapon_model is not None:
            try:
                results = self.weapon_model(
                    frame,
                    conf=self.confidence_threshold,
                    verbose=False,
                    device=self.device
                )

                for result in results:
                    if result.boxes is None:
                        continue

                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        # Get class name from model
                        class_name = result.names.get(cls_id, 'weapon')

                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'class_name': class_name.lower(),
                            'confidence': conf,
                            'danger_level': self.WEAPON_CLASSES.get(
                                class_name.lower(), {}
                            ).get('danger_level', 'high'),
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)

            except Exception as e:
                if self.debug:
                    print(f"[WeaponDetector] Weapon model error: {e}")

        # Update stats
        if detections:
            self.stats['total_detections'] += len(detections)
            for det in detections:
                if 'knife' in det['class_name']:
                    self.stats['knife_detections'] += 1
                elif 'gun' in det['class_name'] or 'pistol' in det['class_name']:
                    self.stats['gun_detections'] += 1
                else:
                    self.stats['other_weapon_detections'] += 1

            # Add to history
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'detections': detections
            })

            # Keep history limited
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-500:]

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw weapon detections on frame

        Args:
            frame: BGR image
            detections: List of detection dicts

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            danger = det['danger_level']

            # Color based on danger level
            if danger == 'critical':
                color = (0, 0, 255)  # Red
            elif danger == 'high':
                color = (0, 69, 255)  # Orange-red
            elif danger == 'medium':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"WEAPON: {class_name.upper()} {conf:.0%}"
            font_scale = 0.7
            thickness = 2
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )

            # Draw danger warning
            if danger in ['critical', 'high']:
                warning = f"⚠ DANGER: {danger.upper()}"
                cv2.putText(
                    annotated, warning,
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        return annotated

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_detections': 0,
            'knife_detections': 0,
            'gun_detections': 0,
            'other_weapon_detections': 0
        }
        self.detection_history = []
