"""
Test SlowFastDetector class

Verifies:
1. Model loads correctly
2. Can detect actions from frame buffer
3. Violence classification works
4. Statistics tracking works
"""

import cv2
import numpy as np
from video_buffer import TemporalFrameBuffer
from slowfast_detector import SlowFastDetector


def test_detector_initialization():
    """Test detector initialization"""
    print("=" * 70)
    print("Test 1: Detector Initialization")
    print("=" * 70)

    detector = SlowFastDetector(
        labels_path="models/kinetics400_labels.json",
        device='cuda',
        confidence_threshold=0.3,
        violence_threshold=0.4
    )

    print(f"Detector: {detector}")
    print(f"Device: {detector.device}")
    print(f"Total action classes: {len(detector.labels)}")
    print(f"Violence classes: {len(detector.violence_indices)}")

    # Print violence classes
    print("\nViolence-related classes:")
    violence_classes = detector.get_violence_classes()
    for i, cls in enumerate(violence_classes[:10], 1):
        print(f"  {i}. {cls}")
    if len(violence_classes) > 10:
        print(f"  ... and {len(violence_classes) - 10} more")

    print("[OK] Detector initialized successfully!")
    print()

    return detector


def test_action_detection(detector):
    """Test action detection with random frames"""
    print("=" * 70)
    print("Test 2: Action Detection")
    print("=" * 70)

    # Create frame buffer
    buffer = TemporalFrameBuffer(buffer_size=32, input_size=(224, 224))

    # Add 32 random frames
    print("Adding 32 random frames to buffer...")
    for i in range(32):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.add_frame(frame)

    print(f"Buffer ready: {buffer.is_ready()}")

    # Run detection
    print("\nRunning SlowFast inference...")
    result = detector.detect(buffer, top_k=5)

    if result:
        print(f"[OK] Detection successful!")
        print(f"\nTop action: {result.action}")
        print(f"Confidence: {result.confidence * 100:.2f}%")
        print(f"Is violent: {result.is_violent}")
        print(f"Inference time: {result.inference_time_ms:.1f} ms")

        print("\nTop 5 predictions:")
        for i, (action, conf) in enumerate(result.top_k_actions, 1):
            marker = "[!!!]" if any(kw in action.lower() for kw in detector.VIOLENCE_KEYWORDS) else "     "
            print(f"  {marker} {i}. {action:40s} {conf*100:5.2f}%")

        # Check statistics
        stats = detector.get_statistics()
        print(f"\nDetector statistics:")
        print(f"  Total inferences: {stats['total_inferences']}")
        print(f"  Violence detected: {stats['violence_detected']}")
        print(f"  Avg inference time: {stats['avg_inference_time_ms']:.1f} ms")
    else:
        print("[ERROR] Detection failed!")

    print()


def test_webcam_detection(detector):
    """Test detector with live webcam"""
    print("=" * 70)
    print("Test 3: Live Webcam Detection")
    print("=" * 70)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[SKIP] No webcam available")
        return

    print("[INFO] Opening webcam... (Press 'q' to quit)")

    buffer = TemporalFrameBuffer(buffer_size=32, input_size=(224, 224))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        buffer.add_frame(frame)

        # Run detection every 8 frames
        if buffer.is_ready() and frame_count % 8 == 0:
            result = detector.detect(buffer, top_k=3)

            if result:
                # Display results on frame
                y_offset = 30
                for i, (action, conf) in enumerate(result.top_k_actions):
                    is_violence = any(kw in action.lower() for kw in detector.VIOLENCE_KEYWORDS)
                    color = (0, 0, 255) if is_violence else (0, 255, 0)
                    text = f"{action}: {conf*100:.1f}%"

                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30

                # Display violence indicator
                if result.is_violent:
                    cv2.putText(frame, "VIOLENCE DETECTED!", (10, frame.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display frame info
        cv2.putText(frame, f"Frames: {len(buffer)}/32", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('SlowFast Detector Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final statistics
    stats = detector.get_statistics()
    print("\n[FINAL STATISTICS]")
    print(f"Total inferences: {stats['total_inferences']}")
    print(f"Violence detected: {stats['violence_detected']}")
    print(f"Violence rate: {stats['violence_rate'] * 100:.1f}%")
    print(f"Avg inference time: {stats['avg_inference_time_ms']:.1f} ms")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("*" * 70)
    print("SLOWFAST DETECTOR TEST SUITE")
    print("*" * 70)
    print()

    # Test 1: Initialization
    detector = test_detector_initialization()

    # Test 2: Basic detection
    test_action_detection(detector)

    # Test 3: Webcam (optional)
    response = input("Run webcam test? (y/n): ").strip().lower()
    if response == 'y':
        test_webcam_detection(detector)

    print("*" * 70)
    print("ALL TESTS COMPLETED!")
    print("*" * 70)
    print()


if __name__ == "__main__":
    main()
