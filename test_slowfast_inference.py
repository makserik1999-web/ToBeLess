#!/usr/bin/env python3
"""
Test SlowFast model inference on real video clips

This script:
1. Loads the downloaded SlowFast R50 model
2. Processes a video file frame by frame
3. Extracts 32-frame temporal clips
4. Runs inference to classify actions
5. Displays top predictions with confidence scores
"""

import os
import sys
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import deque
import time

def load_kinetics_labels(labels_path):
    """Load Kinetics-400 action labels"""
    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        print(f"[ERROR] Failed to load labels: {e}")
        return None


def preprocess_frame(frame, input_size=(224, 224)):
    """
    Preprocess a single frame for SlowFast input

    Args:
        frame: BGR image from OpenCV
        input_size: Target size (height, width)

    Returns:
        Normalized tensor (C, H, W)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to input size
    frame_resized = cv2.resize(frame_rgb, input_size)

    # Convert to float32 and normalize to [0, 1]
    frame_float = frame_resized.astype(np.float32) / 255.0

    # Normalize using ImageNet stats
    mean = np.array([0.45, 0.45, 0.45])
    std = np.array([0.225, 0.225, 0.225])
    frame_normalized = (frame_float - mean) / std

    # Convert to tensor (H, W, C) -> (C, H, W)
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)

    return frame_tensor


def prepare_slowfast_input(frames, device='cuda'):
    """
    Prepare SlowFast dual-pathway input from frame buffer

    Args:
        frames: List of preprocessed frame tensors (C, H, W)
        device: 'cuda' or 'cpu'

    Returns:
        [slow_input, fast_input] tensors
    """
    # SlowFast uses 32 frames total
    # Slow pathway: 8 frames (sample every 4th frame)
    # Fast pathway: 32 frames (all frames)

    num_frames = len(frames)

    if num_frames < 32:
        # Pad with duplicate frames if not enough
        while len(frames) < 32:
            frames.append(frames[-1])

    # Sample 8 frames for slow pathway (every 4th frame)
    slow_indices = [i * 4 for i in range(8)]
    slow_frames = [frames[i] for i in slow_indices]

    # Use all 32 frames for fast pathway
    fast_frames = frames[:32]

    # Stack frames: (num_frames, C, H, W)
    slow_clip = torch.stack(slow_frames)
    fast_clip = torch.stack(fast_frames)

    # Rearrange to (C, num_frames, H, W)
    slow_clip = slow_clip.permute(1, 0, 2, 3)
    fast_clip = fast_clip.permute(1, 0, 2, 3)

    # Add batch dimension: (1, C, num_frames, H, W)
    slow_input = slow_clip.unsqueeze(0).to(device)
    fast_input = fast_clip.unsqueeze(0).to(device)

    return [slow_input, fast_input]


def run_inference(model, frames, labels, device='cuda', top_k=5):
    """
    Run SlowFast inference on a clip

    Args:
        model: SlowFast model
        frames: List of preprocessed frames
        labels: Action labels list
        device: 'cuda' or 'cpu'
        top_k: Number of top predictions to return

    Returns:
        List of (class_name, probability) tuples
    """
    # Prepare input
    inputs = prepare_slowfast_input(frames, device)

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)

    # Get probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    # Convert to list of (class, prob)
    results = []
    for i in range(top_k):
        class_idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        class_name = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"
        results.append((class_name, prob))

    return results


def process_video(video_path, model, labels, device='cuda', display=True):
    """
    Process entire video with SlowFast model

    Args:
        video_path: Path to video file or webcam index
        model: SlowFast model
        labels: Action labels
        device: 'cuda' or 'cpu'
        display: Whether to display video with predictions
    """
    # Open video
    if isinstance(video_path, int):
        cap = cv2.VideoCapture(video_path)
        print(f"[INFO] Opening webcam {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"[INFO] Opening video: {video_path}")

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print()

    # Frame buffer for temporal clip
    frame_buffer = deque(maxlen=32)

    # Violence-related keywords for highlighting
    violence_keywords = [
        'punch', 'fight', 'slap', 'hit', 'kick', 'wrestl',
        'headbutt', 'sword fight', 'boxing', 'drop kick'
    ]

    frame_count = 0
    inference_times = []

    print("[INFO] Processing video... (Press 'q' to quit)")
    print("=" * 70)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Preprocess frame
        preprocessed = preprocess_frame(frame)
        frame_buffer.append(preprocessed)

        # Run inference every 8 frames (overlap for smooth detection)
        if len(frame_buffer) == 32 and frame_count % 8 == 0:
            start_time = time.time()

            # Get predictions
            predictions = run_inference(model, list(frame_buffer), labels, device, top_k=5)

            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)

            # Display predictions
            print(f"\n[Frame {frame_count}/{total_frames}]")
            print(f"Inference time: {inference_time:.1f} ms")
            print("\nTop 5 predictions:")

            for i, (action, prob) in enumerate(predictions, 1):
                # Highlight violence-related actions
                is_violence = any(kw in action.lower() for kw in violence_keywords)
                marker = "[!!!]" if is_violence else "     "
                print(f"  {marker} {i}. {action:40s} {prob*100:5.2f}%")

            # Draw on frame if display enabled
            if display:
                # Add predictions overlay
                y_offset = 30
                for i, (action, prob) in enumerate(predictions[:3]):
                    text = f"{action}: {prob*100:.1f}%"
                    is_violence = any(kw in action.lower() for kw in violence_keywords)
                    color = (0, 0, 255) if is_violence else (0, 255, 0)

                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30

                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                           (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show frame
                cv2.imshow('SlowFast Inference Test', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Stopped by user")
                    break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # Print summary
    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print(f"Total frames processed: {frame_count}")
    print(f"Total inferences: {len(inference_times)}")

    if inference_times:
        avg_inference = np.mean(inference_times)
        max_inference = np.max(inference_times)
        min_inference = np.min(inference_times)
        avg_fps = 1000 / avg_inference if avg_inference > 0 else 0

        print(f"Average inference time: {avg_inference:.1f} ms")
        print(f"Min/Max inference time: {min_inference:.1f} / {max_inference:.1f} ms")
        print(f"Average FPS: {avg_fps:.1f} FPS")
    print("=" * 70)


def main():
    """Main test function"""
    print("=" * 70)
    print("SlowFast Inference Test")
    print("=" * 70)
    print()

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load labels
    labels_path = Path("models/kinetics400_labels.json")
    print(f"[INFO] Loading labels from: {labels_path}")
    labels = load_kinetics_labels(labels_path)

    if labels is None:
        print("[ERROR] Failed to load labels. Run download_slowfast_model.py first.")
        return

    print(f"[INFO] Loaded {len(labels)} action classes")
    print()

    # Load model
    print("[INFO] Loading SlowFast R50 model...")
    try:
        from pytorchvideo.models.hub import slowfast_r50

        model = slowfast_r50(pretrained=True)
        model.eval()
        model = model.to(device)

        print("[INFO] Model loaded successfully!")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Get video source
    if len(sys.argv) > 1:
        video_source = sys.argv[1]

        # Check if it's a webcam index
        if video_source.isdigit():
            video_source = int(video_source)
    else:
        # Default: use webcam 0
        print("[INFO] No video file specified. Using webcam 0")
        print("[INFO] Usage: python test_slowfast_inference.py <video_file or webcam_index>")
        print()
        video_source = 0

    # Process video
    process_video(video_source, model, labels, device, display=True)

    print("\n[INFO] Test complete!")


if __name__ == "__main__":
    main()
