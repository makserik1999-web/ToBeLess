"""
Video Frame Buffer for Temporal Action Recognition

# This module provides efficient frame buffering for SlowFast temporal analysis.
# SlowFast can work with variable frame counts (default: 32 frames for standard R50 model).
# The sampling adapts automatically to the buffer size.

Classes:
    TemporalFrameBuffer: Thread-safe circular buffer for video frames
"""

import cv2
import numpy as np
import torch
from collections import deque
from threading import Lock
from typing import List, Optional, Tuple


class TemporalFrameBuffer:
    """
    Thread-safe circular buffer for storing video frames temporally

    SlowFast architecture requires:
    - Slow pathway: 8 frames uniformly sampled from buffer
    - Fast pathway: all frames in buffer

    This buffer maintains a sliding window for continuous action recognition.
    Sampling adapts automatically to buffer size (default: 16 frames).
    """

    def __init__(
        self,
        buffer_size: int = 32,
        input_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        """
        Initialize temporal frame buffer

        Args:
            buffer_size: Number of frames to buffer (default: 32 for standard R50 model)
            input_size: Target frame size (height, width) for model input
            normalize: Whether to normalize frames using ImageNet stats
        """
        self.buffer_size = buffer_size
        self.input_size = input_size
        self.normalize = normalize

        # Circular buffer for raw frames (BGR format from OpenCV)
        self.raw_frames = deque(maxlen=buffer_size)

        # Circular buffer for preprocessed frames (normalized tensors)
        self.preprocessed_frames = deque(maxlen=buffer_size)

        # Lock for thread-safe operations
        self.lock = Lock()

        # Frame counter
        self.frame_count = 0

        # ImageNet normalization parameters
        self.mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        self.std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a new frame to the buffer

        Args:
            frame: BGR image from OpenCV (H, W, 3)
        """
        with self.lock:
            # Store raw frame
            self.raw_frames.append(frame.copy())

            # Preprocess and store
            preprocessed = self._preprocess_frame(frame)
            self.preprocessed_frames.append(preprocessed)

            self.frame_count += 1

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame for SlowFast input

        Args:
            frame: BGR image from OpenCV (H, W, 3)

        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to input size
        frame_resized = cv2.resize(frame_rgb, self.input_size)

        # Convert to float32 and normalize to [0, 1]
        frame_float = frame_resized.astype(np.float32) / 255.0

        if self.normalize:
            # Apply ImageNet normalization
            frame_normalized = (frame_float - self.mean) / self.std
        else:
            frame_normalized = frame_float

        # Convert to tensor: (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)

        return frame_tensor

    def _is_ready_unlocked(self) -> bool:
        """Internal is_ready check without lock (call only when lock is held)"""
        return len(self.preprocessed_frames) >= self.buffer_size

    def is_ready(self) -> bool:
        """
        Check if buffer has enough frames for inference

        Returns:
            True if buffer is full (has buffer_size frames)
        """
        with self.lock:
            return self._is_ready_unlocked()

    def get_clip(self, device: str = 'cuda') -> Optional[List[torch.Tensor]]:
        """
        Get current clip as SlowFast dual-pathway input

        Returns:
            [slow_input, fast_input] tensors or None if buffer not ready
            - slow_input: (1, 3, 8, H, W)
            - fast_input: (1, 3, num_frames, H, W)
        """
        with self.lock:
            if not self._is_ready_unlocked():
                return None

            # Get all frames from buffer
            frames = list(self.preprocessed_frames)

            # Prepare SlowFast dual-pathway input
            slow_input, fast_input = self._prepare_slowfast_input(frames, device)

            return [slow_input, fast_input]

    def _prepare_slowfast_input(
        self,
        frames: List[torch.Tensor],
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare SlowFast dual-pathway input from frame list

        Args:
            frames: List of preprocessed frame tensors (C, H, W)
            device: 'cuda' or 'cpu'

        Returns:
            (slow_input, fast_input) tuple
        """
        # SlowFast temporal sampling (adaptive to buffer size):
        # - Slow pathway: 8 frames uniformly sampled
        # - Fast pathway: all frames in buffer

        num_frames = len(frames)

        # For slow pathway: sample 8 frames uniformly across the buffer
        # With 16 frames: indices 0, 2, 4, 6, 8, 10, 12, 14
        # With 32 frames: indices 0, 4, 8, 12, 16, 20, 24, 28
        slow_rate = max(1, num_frames // 8)
        slow_indices = [min(i * slow_rate, num_frames - 1) for i in range(8)]
        slow_frames = [frames[i] for i in slow_indices]

        # Use all frames for fast pathway
        fast_frames = frames[:num_frames]

        # Stack frames: (num_frames, C, H, W)
        slow_clip = torch.stack(slow_frames)
        fast_clip = torch.stack(fast_frames)

        # Rearrange to (C, num_frames, H, W) - SlowFast expected format
        slow_clip = slow_clip.permute(1, 0, 2, 3)
        fast_clip = fast_clip.permute(1, 0, 2, 3)

        # Add batch dimension: (1, C, num_frames, H, W)
        slow_input = slow_clip.unsqueeze(0).to(device)
        fast_input = fast_clip.unsqueeze(0).to(device)

        return slow_input, fast_input

    def get_raw_frames(self) -> List[np.ndarray]:
        """
        Get list of raw frames currently in buffer

        Returns:
            List of BGR frames (H, W, 3)
        """
        with self.lock:
            return list(self.raw_frames)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame from buffer

        Returns:
            Latest BGR frame or None if buffer empty
        """
        with self.lock:
            if len(self.raw_frames) > 0:
                return self.raw_frames[-1].copy()
            return None

    def clear(self) -> None:
        """Clear all frames from buffer"""
        with self.lock:
            self.raw_frames.clear()
            self.preprocessed_frames.clear()
            self.frame_count = 0

    def __len__(self) -> int:
        """Get current number of frames in buffer"""
        with self.lock:
            return len(self.preprocessed_frames)

    def __repr__(self) -> str:
        """String representation of buffer state"""
        return (f"TemporalFrameBuffer(size={len(self)}/{self.buffer_size}, "
                f"frames_processed={self.frame_count}, "
                f"ready={self.is_ready()})")


class MultiBufferManager:
    """
    Manage multiple frame buffers for tracking multiple people

    When YOLO-Pose detects multiple people, we need separate buffers
    to track each person's actions independently.
    """

    def __init__(
        self,
        buffer_size: int = 16,
        input_size: Tuple[int, int] = (224, 224),
        max_people: int = 10
    ):
        """
        Initialize multi-buffer manager

        Args:
            buffer_size: Frames per buffer (default: 16 for faster filling)
            input_size: Frame size for model
            max_people: Maximum number of people to track simultaneously
        """
        self.buffer_size = buffer_size
        self.input_size = input_size
        self.max_people = max_people

        # Dictionary: person_id -> TemporalFrameBuffer
        self.buffers = {}

        # Lock for thread-safe buffer management
        self.lock = Lock()

    def get_or_create_buffer(self, person_id: int) -> TemporalFrameBuffer:
        """
        Get existing buffer for person or create new one

        Args:
            person_id: Unique identifier for person

        Returns:
            TemporalFrameBuffer for this person
        """
        with self.lock:
            if person_id not in self.buffers:
                # Create new buffer if under limit
                if len(self.buffers) < self.max_people:
                    self.buffers[person_id] = TemporalFrameBuffer(
                        buffer_size=self.buffer_size,
                        input_size=self.input_size
                    )

            return self.buffers.get(person_id)

    def add_frame_for_person(
        self,
        person_id: int,
        frame: np.ndarray
    ) -> None:
        """
        Add frame to specific person's buffer

        Args:
            person_id: Person identifier
            frame: Cropped frame region for this person
        """
        buffer = self.get_or_create_buffer(person_id)
        if buffer is not None:
            buffer.add_frame(frame)

    def is_ready(self, person_id: int) -> bool:
        """Check if person's buffer is ready for inference"""
        with self.lock:
            if person_id in self.buffers:
                return self.buffers[person_id].is_ready()
            return False

    def get_clip(
        self,
        person_id: int,
        device: str = 'cuda'
    ) -> Optional[List[torch.Tensor]]:
        """Get clip for specific person"""
        with self.lock:
            if person_id in self.buffers:
                return self.buffers[person_id].get_clip(device)
            return None

    def cleanup_stale_buffers(self, active_ids: List[int]) -> None:
        """
        Remove buffers for people no longer in frame

        Args:
            active_ids: List of currently active person IDs
        """
        with self.lock:
            # Remove buffers not in active list
            stale_ids = set(self.buffers.keys()) - set(active_ids)
            for person_id in stale_ids:
                del self.buffers[person_id]

    def clear_all(self) -> None:
        """Clear all buffers"""
        with self.lock:
            for buffer in self.buffers.values():
                buffer.clear()
            self.buffers.clear()

    def get_buffer_count(self) -> int:
        """Get number of active buffers"""
        with self.lock:
            return len(self.buffers)

    def __repr__(self) -> str:
        """String representation"""
        return (f"MultiBufferManager(active_buffers={self.get_buffer_count()}, "
                f"max_people={self.max_people})")
