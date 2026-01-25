"""
SlowFast Action Recognition Detector

This module implements real-time action recognition using SlowFast R50 model.
It identifies violent actions to reduce false positives in fight detection.

Classes:
    SlowFastDetector: Main detector for action classification
    ActionResult: Data class for detection results
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from video_buffer import TemporalFrameBuffer


@dataclass
class ActionResult:
    """
    Result from action recognition

    Attributes:
        action: Predicted action class name
        confidence: Prediction confidence (0-1)
        is_violent: Whether action is classified as violent
        top_k_actions: List of (action, confidence) tuples for top predictions
        inference_time_ms: Time taken for inference in milliseconds
    """
    action: str
    confidence: float
    is_violent: bool
    top_k_actions: List[Tuple[str, float]]
    inference_time_ms: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'action': self.action,
            'confidence': self.confidence,
            'is_violent': self.is_violent,
            'top_k_actions': self.top_k_actions,
            'inference_time_ms': self.inference_time_ms
        }


class SlowFastDetector:
    """
    SlowFast-based action recognition detector

    Uses SlowFast R50 pretrained on Kinetics-400 to classify actions
    in video clips. Focuses on detecting violence-related actions.
    """

    # Violence-related action classes from Kinetics-400
    VIOLENCE_KEYWORDS = [
        'punch', 'punching', 'fight', 'fighting', 'slap', 'slapping',
        'hit', 'hitting', 'kick', 'kicking', 'wrestl', 'wrestling',
        'headbutt', 'headbutting', 'sword fight', 'boxing',
        'drop kick', 'side kick', 'high kick'
    ]

    # Non-violence physical actions (to distinguish from violence)
    NON_VIOLENCE_KEYWORDS = [
        'hug', 'hugging', 'embrac', 'dancing', 'exercis',
        'yoga', 'tai chi', 'stretching', 'clapping', 'waving',
        'shaking hands', 'high fiving'
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        labels_path: str = "models/kinetics400_labels.json",
        device: str = 'cuda',
        confidence_threshold: float = 0.3,
        violence_threshold: float = 0.4
    ):
        """
        Initialize SlowFast detector

        Args:
            model_path: Path to model weights (None = use pretrained)
            labels_path: Path to Kinetics-400 labels JSON
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for predictions
            violence_threshold: Minimum confidence to classify as violent
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.violence_threshold = violence_threshold

        # Load model
        print(f"[SlowFast] Loading model on {self.device}...")
        self.model = self._load_model(model_path)
        self.model.eval()

        # Load action labels
        self.labels = self._load_labels(labels_path)
        print(f"[SlowFast] Loaded {len(self.labels)} action classes")

        # Identify violence-related class indices
        self.violence_indices = self._identify_violence_classes()
        print(f"[SlowFast] Identified {len(self.violence_indices)} violence-related classes")

        # Statistics
        self.total_inferences = 0
        self.total_violence_detected = 0
        self.avg_inference_time = 0.0

    def _load_model(self, model_path: Optional[str] = None):
        """
        Load SlowFast R50 model

        Args:
            model_path: Path to saved weights or None for pretrained

        Returns:
            SlowFast model
        """
        try:
            from pytorchvideo.models.hub import slowfast_r50

            # Load pretrained model
            model = slowfast_r50(pretrained=True)

            # Load custom weights if provided
            if model_path and Path(model_path).exists():
                print(f"[SlowFast] Loading weights from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)

            model = model.to(self.device)
            return model

        except Exception as e:
            print(f"[ERROR] Failed to load SlowFast model: {e}")
            raise

    def _load_labels(self, labels_path: str) -> List[str]:
        """
        Load Kinetics-400 action labels

        Args:
            labels_path: Path to labels JSON file

        Returns:
            List of action class names
        """
        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            return labels
        except Exception as e:
            print(f"[ERROR] Failed to load labels: {e}")
            return []

    def _identify_violence_classes(self) -> List[int]:
        """
        Identify indices of violence-related action classes

        Returns:
            List of class indices for violent actions
        """
        violence_indices = []

        for idx, label in enumerate(self.labels):
            # Check if label contains violence keywords
            label_lower = label.lower()
            is_violent = any(kw in label_lower for kw in self.VIOLENCE_KEYWORDS)

            # Exclude if it contains non-violence keywords
            is_non_violent = any(kw in label_lower for kw in self.NON_VIOLENCE_KEYWORDS)

            if is_violent and not is_non_violent:
                violence_indices.append(idx)

        return violence_indices

    def detect(
        self,
        frame_buffer: TemporalFrameBuffer,
        top_k: int = 5
    ) -> Optional[ActionResult]:
        """
        Detect action in video clip from frame buffer

        Args:
            frame_buffer: TemporalFrameBuffer with 32 frames
            top_k: Number of top predictions to return

        Returns:
            ActionResult or None if buffer not ready
        """
        # Check if buffer is ready
        if not frame_buffer.is_ready():
            return None

        # Get clip from buffer
        clip = frame_buffer.get_clip(device=self.device)
        if clip is None:
            return None

        # Run inference
        import time
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(clip)

        inference_time_ms = (time.time() - start_time) * 1000

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

        # Build results
        top_k_actions = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            class_name = self.labels[class_idx] if class_idx < len(self.labels) else f"class_{class_idx}"
            top_k_actions.append((class_name, prob))

        # Determine if violent action
        top_action, top_confidence = top_k_actions[0]
        is_violent = self._is_violent_action(top_indices[0].cpu().numpy(), top_probs[0].cpu().numpy())

        # Update statistics
        self.total_inferences += 1
        if is_violent:
            self.total_violence_detected += 1

        # Update average inference time
        self.avg_inference_time = (
            (self.avg_inference_time * (self.total_inferences - 1) + inference_time_ms) /
            self.total_inferences
        )

        return ActionResult(
            action=top_action,
            confidence=top_confidence,
            is_violent=is_violent,
            top_k_actions=top_k_actions,
            inference_time_ms=inference_time_ms
        )

    def _is_violent_action(
        self,
        class_indices: np.ndarray,
        probabilities: np.ndarray
    ) -> bool:
        """
        Determine if detected action is violent

        Args:
            class_indices: Array of predicted class indices (top-k)
            probabilities: Array of prediction probabilities (top-k)

        Returns:
            True if violent action detected above threshold
        """
        # Check if any violence class is in top predictions with high confidence
        for idx, prob in zip(class_indices, probabilities):
            if idx in self.violence_indices and prob >= self.violence_threshold:
                return True

        # Also check weighted sum of violence class probabilities
        violence_prob_sum = 0.0
        for idx, prob in zip(class_indices, probabilities):
            if idx in self.violence_indices:
                violence_prob_sum += prob

        # If cumulative violence probability is high, classify as violent
        if violence_prob_sum >= self.violence_threshold:
            return True

        return False

    def get_statistics(self) -> Dict:
        """
        Get detector statistics

        Returns:
            Dictionary with detector stats
        """
        return {
            'total_inferences': self.total_inferences,
            'violence_detected': self.total_violence_detected,
            'violence_rate': (
                self.total_violence_detected / self.total_inferences
                if self.total_inferences > 0 else 0.0
            ),
            'avg_inference_time_ms': self.avg_inference_time,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'violence_threshold': self.violence_threshold
        }

    def reset_statistics(self):
        """Reset detector statistics"""
        self.total_inferences = 0
        self.total_violence_detected = 0
        self.avg_inference_time = 0.0

    def update_thresholds(
        self,
        confidence_threshold: Optional[float] = None,
        violence_threshold: Optional[float] = None
    ):
        """
        Update detection thresholds

        Args:
            confidence_threshold: Minimum confidence for predictions
            violence_threshold: Minimum confidence to classify as violent
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            print(f"[SlowFast] Updated confidence threshold: {confidence_threshold}")

        if violence_threshold is not None:
            self.violence_threshold = violence_threshold
            print(f"[SlowFast] Updated violence threshold: {violence_threshold}")

    def get_violence_classes(self) -> List[str]:
        """
        Get list of violence-related action classes

        Returns:
            List of violence class names
        """
        return [self.labels[idx] for idx in self.violence_indices]

    def __repr__(self) -> str:
        """String representation"""
        return (f"SlowFastDetector(device={self.device}, "
                f"inferences={self.total_inferences}, "
                f"avg_time={self.avg_inference_time:.1f}ms)")


class SlowFastBatchDetector(SlowFastDetector):
    """
    Batch version of SlowFast detector for processing multiple clips efficiently

    Useful when tracking multiple people simultaneously
    """

    def detect_batch(
        self,
        clips: List[List[torch.Tensor]],
        top_k: int = 5
    ) -> List[ActionResult]:
        """
        Detect actions in multiple clips simultaneously

        Args:
            clips: List of [slow_input, fast_input] clip pairs
            top_k: Number of top predictions per clip

        Returns:
            List of ActionResult objects
        """
        if not clips:
            return []

        import time
        start_time = time.time()

        # Stack clips into batches
        slow_batch = torch.cat([clip[0] for clip in clips], dim=0)
        fast_batch = torch.cat([clip[1] for clip in clips], dim=0)

        # Run inference
        with torch.no_grad():
            outputs = self.model([slow_batch, fast_batch])

        inference_time_ms = (time.time() - start_time) * 1000

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)

        # Process each result
        results = []
        for i in range(len(clips)):
            # Get top-k for this clip
            top_probs, top_indices = torch.topk(probs[i:i+1], k=top_k, dim=1)

            # Build top-k actions
            top_k_actions = []
            for j in range(top_k):
                class_idx = top_indices[0][j].item()
                prob = top_probs[0][j].item()
                class_name = self.labels[class_idx] if class_idx < len(self.labels) else f"class_{class_idx}"
                top_k_actions.append((class_name, prob))

            # Check if violent
            top_action, top_confidence = top_k_actions[0]
            is_violent = self._is_violent_action(
                top_indices[0].cpu().numpy(),
                top_probs[0].cpu().numpy()
            )

            results.append(ActionResult(
                action=top_action,
                confidence=top_confidence,
                is_violent=is_violent,
                top_k_actions=top_k_actions,
                inference_time_ms=inference_time_ms / len(clips)  # Per-clip time
            ))

            # Update statistics
            self.total_inferences += 1
            if is_violent:
                self.total_violence_detected += 1

        return results
