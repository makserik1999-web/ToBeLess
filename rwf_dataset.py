"""
RWF-2000 Dataset Loader for SlowFast Training

This module provides a PyTorchVideo-compatible dataset for the RWF-2000
violence detection dataset. It handles video loading, frame sampling,
and preprocessing for SlowFast dual-pathway input.

Classes:
    RWFDataset: Main dataset class for training/validation
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
from torch.utils.data import Dataset, DataLoader


class RWFDataset(Dataset):
    """
    RWF-2000 Dataset for SlowFast Violence Detection

    Loads videos and creates SlowFast dual-pathway inputs:
    - Slow pathway: 8 frames (every 4th frame from 32)
    - Fast pathway: 32 frames (all frames)

    Args:
        root_dir: Path to dataset root (e.g., datasets/RWF-2000/train)
        num_frames: Number of frames to sample (default: 32 for SlowFast)
        frame_size: Target frame size (height, width)
        transform: Optional additional transforms
        augment: Enable data augmentation for training
    """

    # Class mapping
    CLASS_NAMES = ['NonFight', 'Fight']
    CLASS_TO_IDX = {'NonFight': 0, 'Fight': 1}

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.augment = augment

        # ImageNet normalization
        self.mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        self.std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

        # Load video paths and labels
        self.samples = self._load_samples()
        print(f"[RWFDataset] Loaded {len(self.samples)} videos from {root_dir}")
        self._print_class_distribution()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all video paths and their labels

        Returns:
            List of (video_path, label) tuples
        """
        samples = []

        for class_name in self.CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[WARNING] Class directory not found: {class_dir}")
                continue

            label = self.CLASS_TO_IDX[class_name]

            # Find all video files
            for ext in ['*.avi', '*.mp4', '*.mkv', '*.mov']:
                for video_path in class_dir.glob(ext):
                    samples.append((str(video_path), label))

        return samples

    def _print_class_distribution(self):
        """Print class distribution"""
        counts = {name: 0 for name in self.CLASS_NAMES}
        for _, label in self.samples:
            class_name = self.CLASS_NAMES[label]
            counts[class_name] += 1

        print(f"[RWFDataset] Class distribution:")
        for name, count in counts.items():
            print(f"  {name}: {count} videos")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            ([slow_input, fast_input], label) tuple
            - slow_input: (C, 8, H, W) tensor
            - fast_input: (C, 32, H, W) tensor
            - label: 0 (NonFight) or 1 (Fight)
        """
        video_path, label = self.samples[idx]

        # Load and sample frames
        frames = self._load_video_frames(video_path)

        # Apply augmentation if enabled
        if self.augment:
            frames = self._augment_frames(frames)

        # Preprocess frames
        processed_frames = [self._preprocess_frame(f) for f in frames]

        # Create SlowFast dual-pathway input
        slow_input, fast_input = self._create_slowfast_input(processed_frames)

        return [slow_input, fast_input], label

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Load and sample frames from video

        Args:
            video_path: Path to video file

        Returns:
            List of BGR frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate sampling indices
        if total_frames >= self.num_frames:
            # Uniform sampling across video
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # If video is shorter, repeat frames
            indices = np.arange(total_frames)
            # Repeat to reach num_frames
            while len(indices) < self.num_frames:
                indices = np.concatenate([indices, indices])
            indices = indices[:self.num_frames]

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # Fallback: use last valid frame or black frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))

        cap.release()
        return frames

    def _augment_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply data augmentation to frames

        Args:
            frames: List of BGR frames

        Returns:
            Augmented frames
        """
        augmented = []

        # Decide augmentation parameters (same for all frames)
        do_flip = np.random.random() > 0.5
        brightness_factor = np.random.uniform(0.8, 1.2)
        contrast_factor = np.random.uniform(0.8, 1.2)

        for frame in frames:
            # Horizontal flip
            if do_flip:
                frame = cv2.flip(frame, 1)

            # Brightness/contrast adjustment
            frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=(brightness_factor - 1) * 50)

            augmented.append(frame)

        return augmented

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame for SlowFast

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 256 first (for cropping)
        frame_resized = cv2.resize(frame_rgb, (256, 256))

        # Center crop to target size
        h, w = frame_resized.shape[:2]
        th, tw = self.frame_size
        y = (h - th) // 2
        x = (w - tw) // 2
        frame_cropped = frame_resized[y:y+th, x:x+tw]

        # Normalize to [0, 1] and apply ImageNet normalization
        frame_float = frame_cropped.astype(np.float32) / 255.0
        frame_normalized = (frame_float - self.mean) / self.std

        # Convert to tensor (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)

        return frame_tensor

    def _create_slowfast_input(
        self,
        frames: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create SlowFast dual-pathway input from frames

        Args:
            frames: List of preprocessed frame tensors (C, H, W)

        Returns:
            (slow_input, fast_input) tuple
            - slow_input: (C, 8, H, W)
            - fast_input: (C, num_frames, H, W)
        """
        # Fast pathway: all frames
        fast_clip = torch.stack(frames)  # (T, C, H, W)
        fast_clip = fast_clip.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Slow pathway: sample every 4th frame (8 frames from 32)
        slow_rate = max(1, len(frames) // 8)
        slow_indices = [min(i * slow_rate, len(frames) - 1) for i in range(8)]
        slow_frames = [frames[i] for i in slow_indices]
        slow_clip = torch.stack(slow_frames)  # (8, C, H, W)
        slow_clip = slow_clip.permute(1, 0, 2, 3)  # (C, 8, H, W)

        return slow_clip, fast_clip


def collate_slowfast(batch):
    """
    Custom collate function for SlowFast batches

    Args:
        batch: List of ([slow, fast], label) tuples

    Returns:
        ([slow_batch, fast_batch], labels)
    """
    clips, labels = zip(*batch)

    # Stack slow and fast pathways separately
    slow_clips = torch.stack([c[0] for c in clips])  # (B, C, 8, H, W)
    fast_clips = torch.stack([c[1] for c in clips])  # (B, C, T, H, W)

    labels = torch.tensor(labels, dtype=torch.long)

    return [slow_clips, fast_clips], labels


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_frames: Frames per clip

    Returns:
        (train_loader, val_loader) tuple
    """
    train_dataset = RWFDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        augment=True
    )

    val_dataset = RWFDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_slowfast,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_slowfast,
        pin_memory=True
    )

    return train_loader, val_loader


# ============ Verification Script ============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RWF Dataset")
    parser.add_argument("--data_dir", default="datasets/RWF-2000/train", help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for test")
    args = parser.parse_args()

    print("=" * 60)
    print("RWF-2000 Dataset Verification")
    print("=" * 60)

    # Check if directory exists
    if not Path(args.data_dir).exists():
        print(f"\n[ERROR] Dataset directory not found: {args.data_dir}")
        print("\nPlease download RWF-2000 dataset first.")
        print("See datasets/README.md for instructions.")
        exit(1)

    # Create dataset
    dataset = RWFDataset(args.data_dir, augment=False)

    if len(dataset) == 0:
        print("\n[ERROR] No videos found in dataset!")
        exit(1)

    # Test loading a sample
    print(f"\nTesting sample loading...")
    try:
        clips, label = dataset[0]
        slow, fast = clips

        print(f"\n[SUCCESS] Sample loaded successfully!")
        print(f"  Slow pathway shape: {slow.shape}")  # Expected: (3, 8, 224, 224)
        print(f"  Fast pathway shape: {fast.shape}")  # Expected: (3, 32, 224, 224)
        print(f"  Label: {label} ({dataset.CLASS_NAMES[label]})")

        # Test dataloader
        print(f"\nTesting DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_slowfast
        )

        batch_clips, batch_labels = next(iter(loader))
        print(f"\n[SUCCESS] Batch loaded successfully!")
        print(f"  Slow batch shape: {batch_clips[0].shape}")  # (B, C, 8, H, W)
        print(f"  Fast batch shape: {batch_clips[1].shape}")  # (B, C, 32, H, W)
        print(f"  Labels: {batch_labels}")

    except Exception as e:
        print(f"\n[ERROR] Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "=" * 60)
    print("Dataset verification PASSED!")
    print("=" * 60)
