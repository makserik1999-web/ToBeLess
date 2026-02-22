#!/usr/bin/env python3
"""
Test HybridFightDetector with webcam or video file

This script tests the hybrid detector (YOLO-Pose + SlowFast) without
modifying the main app.py. It's a standalone test to verify everything works.

Usage:
    python test_hybrid.py                    # Use webcam
    python test_hybrid.py video.mp4          # Use video file
    python test_hybrid.py 0                  # Use webcam 0
"""

import sys
import cv2
import time
from app import FightDetector
from hybrid_fight_detector import HybridFightDetector


def test_hybrid_detector(video_source=0):
    """
    Test hybrid detector with video source

    Args:
        video_source: Video file path, webcam index, or 0 for default webcam
    """
    print("=" * 70)
    print("HYBRID FIGHT DETECTOR TEST")
    print("=" * 70)
    print()

    # Initialize detectors
    print("[1/3] Initializing YOLO-Pose detector...")
    try:
        pose_detector = FightDetector(model_path="yolov8n-pose.pt", device='cuda')
        print("[OK] Pose detector initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pose detector: {e}")
        return

    print()
    print("[2/3] Initializing HybridFightDetector (this may take a moment)...")
    try:
        hybrid = HybridFightDetector(
            pose_detector=pose_detector,
            device='cuda',
            require_both=True,  # Conservative: both signals must agree
            action_weight=0.7,  # Trust action more (70% weight)
            inference_interval=8  # Run SlowFast every 8 frames
        )
        print("[OK] Hybrid detector initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize hybrid detector: {e}")
        print("Make sure you've run download_slowfast_model.py first!")
        return

    print()
    print("[3/3] Opening video source...")

    # Open video
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
        print(f"[OK] Opened webcam {video_source}")
    else:
        cap = cv2.VideoCapture(video_source)
        print(f"[OK] Opened video file: {video_source}")

    if not cap.isOpened():
        print("[ERROR] Failed to open video source")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: {width}x{height} @ {fps} FPS")
    print()
    print("=" * 70)
    print("PROCESSING VIDEO")
    print("=" * 70)
    print("Press 'q' to quit")
    print("Press 's' to show statistics")
    print("=" * 70)
    print()

    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_counter += 1

        # Calculate FPS
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            current_fps = fps_counter / (current_time - last_fps_time)
            fps_counter = 0
            last_fps_time = current_time

        # Process frame through hybrid detector
        annotated, info = hybrid.process_frame(frame, frame_count)

        # Display detection info
        if info['fight_detected']:
            print(f"\n[Frame {frame_count}] FIGHT DETECTED!")
            print(f"  Confidence: {info['confidence']:.1f}%")
            print(f"  Reason: {info['decision_reason']}")
            print(f"  Action: {info['action_class']}")
            print(f"  Pose detected: {info['pose_detected']} (conf: {info['pose_confidence']:.1f}%)")
            print(f"  Action detected: {info['action_detected']} (conf: {info['action_confidence']*100:.1f}%)")
            print()

        # Add FPS counter
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add frame counter
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Hybrid Detector Test (Press Q to quit, S for stats)', annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Stopped by user")
            break
        elif key == ord('s'):
            # Show statistics
            print("\n" + "=" * 70)
            print("CURRENT STATISTICS")
            print("=" * 70)
            stats = hybrid.get_statistics()
            print(f"Total frames: {stats['total_frames']}")
            print(f"Pose detections: {stats['pose_detections']}")
            print(f"Action inferences: {stats['action_inferences']}")
            print(f"Violence detected: {stats['violence_detected']}")
            print(f"False positives avoided: {stats['false_positives_avoided']}")

            if stats['pose_detections'] > 0:
                fp_reduction = (stats['false_positives_avoided'] / stats['pose_detections']) * 100
                print(f"False positive reduction: {fp_reduction:.1f}%")

            if 'slowfast' in stats:
                sf_stats = stats['slowfast']
                print(f"\nSlowFast statistics:")
                print(f"  Total inferences: {sf_stats['total_inferences']}")
                print(f"  Avg inference time: {sf_stats['avg_inference_time_ms']:.1f} ms")
                print(f"  Violence detected: {sf_stats['violence_detected']}")
            print("=" * 70)
            print()

    cap.release()
    cv2.destroyAllWindows()

    # Final statistics
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    stats = hybrid.get_statistics()

    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Average FPS: {stats['total_frames'] / elapsed_time:.1f}")
    print()

    print(f"Pose detections: {stats['pose_detections']}")
    print(f"Action inferences: {stats['action_inferences']}")
    print(f"Violence detected: {stats['violence_detected']}")
    print(f"False positives avoided: {stats['false_positives_avoided']}")

    if stats['pose_detections'] > 0:
        pose_rate = (stats['pose_detections'] / stats['total_frames']) * 100
        print(f"Pose detection rate: {pose_rate:.1f}%")

        fp_reduction = (stats['false_positives_avoided'] / stats['pose_detections']) * 100
        print(f"False positive reduction: {fp_reduction:.1f}%")

    if stats['total_frames'] > 0:
        violence_rate = (stats['violence_detected'] / stats['total_frames']) * 100
        print(f"Violence rate: {violence_rate:.2f}%")

    print()
    print("SlowFast Statistics:")
    if 'slowfast' in stats:
        sf_stats = stats['slowfast']
        print(f"  Total inferences: {sf_stats['total_inferences']}")
        print(f"  Average inference time: {sf_stats['avg_inference_time_ms']:.1f} ms")
        print(f"  Violence detected: {sf_stats['violence_detected']}")
        print(f"  Violence classes identified: {len(hybrid.slowfast_detector.violence_indices)}")

        # Show some violence classes
        violence_classes = hybrid.slowfast_detector.get_violence_classes()
        print(f"\n  Violence-related action classes:")
        for i, cls in enumerate(violence_classes[:10], 1):
            print(f"    {i}. {cls}")
        if len(violence_classes) > 10:
            print(f"    ... and {len(violence_classes) - 10} more")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def main():
    """Main function"""
    # Get video source from command line
    if len(sys.argv) > 1:
        source = sys.argv[1]

        # Check if it's a number (webcam index)
        if source.isdigit():
            source = int(source)

        print(f"Using video source: {source}")
    else:
        source = 0
        print("Using default webcam (index 0)")
        print("Usage: python test_hybrid.py <video_file or webcam_index>")

    print()

    # Run test
    test_hybrid_detector(source)


if __name__ == "__main__":
    main()
