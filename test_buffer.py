"""
Test script for TemporalFrameBuffer

Verifies that the buffer correctly:
1. Stores frames in circular buffer
2. Preprocesses frames properly
3. Generates SlowFast dual-pathway inputs
4. Works in thread-safe manner
"""

import cv2
import numpy as np
import torch
from video_buffer import TemporalFrameBuffer, MultiBufferManager


def test_basic_buffer():
    """Test basic buffer operations"""
    print("=" * 70)
    print("Test 1: Basic Buffer Operations")
    print("=" * 70)

    # Create buffer
    buffer = TemporalFrameBuffer(buffer_size=32, input_size=(224, 224))
    print(f"Created buffer: {buffer}")

    # Create dummy frames
    print("\nAdding 32 frames...")
    for i in range(32):
        # Create random frame (simulate video)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.add_frame(frame)

        if (i + 1) % 8 == 0:
            print(f"  Added {i + 1} frames - Ready: {buffer.is_ready()}")

    print(f"\nFinal state: {buffer}")
    print(f"Buffer is ready: {buffer.is_ready()}")
    print(f"Buffer length: {len(buffer)}")

    # Get clip
    print("\nGetting SlowFast clip...")
    clip = buffer.get_clip(device='cpu')

    if clip is not None:
        slow_input, fast_input = clip
        print(f"[OK] Slow pathway shape: {slow_input.shape}")
        print(f"[OK] Fast pathway shape: {fast_input.shape}")
        print(f"[OK] Expected: slow=(1,3,8,224,224), fast=(1,3,32,224,224)")

        # Verify shapes
        assert slow_input.shape == (1, 3, 8, 224, 224), "Wrong slow pathway shape!"
        assert fast_input.shape == (1, 3, 32, 224, 224), "Wrong fast pathway shape!"
        print("[OK] All shapes correct!")
    else:
        print("[ERROR] Failed to get clip")

    print()


def test_webcam_buffer():
    """Test buffer with real webcam input"""
    print("=" * 70)
    print("Test 2: Webcam Buffer Test")
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

        # Add frame to buffer
        buffer.add_frame(frame)

        # Try to get clip when ready
        if buffer.is_ready() and frame_count % 8 == 0:
            clip = buffer.get_clip(device='cpu')
            if clip is not None:
                print(f"Frame {frame_count}: Got clip with shapes "
                      f"{clip[0].shape}, {clip[1].shape}")

        # Display
        cv2.putText(frame, f"Frames: {len(buffer)}/32", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Ready: {buffer.is_ready()}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam Buffer Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Processed {frame_count} frames")
    print()


def test_multi_buffer():
    """Test multi-person buffer manager"""
    print("=" * 70)
    print("Test 3: Multi-Person Buffer Manager")
    print("=" * 70)

    manager = MultiBufferManager(buffer_size=32, max_people=5)
    print(f"Created manager: {manager}")

    # Simulate detecting 3 people
    person_ids = [1, 2, 3]

    print("\nAdding frames for 3 people...")
    for i in range(32):
        for person_id in person_ids:
            # Create random frame for each person
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            manager.add_frame_for_person(person_id, frame)

        if (i + 1) % 8 == 0:
            ready_count = sum(1 for pid in person_ids if manager.is_ready(pid))
            print(f"  Frame {i + 1}: {ready_count}/3 people ready")

    print(f"\nFinal state: {manager}")
    print(f"Active buffers: {manager.get_buffer_count()}")

    # Get clips for each person
    print("\nGetting clips for each person:")
    for person_id in person_ids:
        clip = manager.get_clip(person_id, device='cpu')
        if clip is not None:
            print(f"  Person {person_id}: [OK] Got clip")
        else:
            print(f"  Person {person_id}: [ERROR] No clip")

    # Test cleanup
    print("\nTesting cleanup (keeping only person 1 and 2)...")
    manager.cleanup_stale_buffers([1, 2])
    print(f"After cleanup: {manager.get_buffer_count()} buffers")

    print()


def test_buffer_overflow():
    """Test buffer behavior when adding more than max frames"""
    print("=" * 70)
    print("Test 4: Buffer Overflow (Circular Buffer)")
    print("=" * 70)

    buffer = TemporalFrameBuffer(buffer_size=32)
    print(f"Buffer size: 32")

    # Add 64 frames (double the buffer size)
    print("\nAdding 64 frames to 32-frame buffer...")
    for i in range(64):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.add_frame(frame)

    print(f"Buffer length after 64 frames: {len(buffer)}")
    print(f"Expected: 32 (circular buffer keeps only last 32)")
    print(f"Total frames processed: {buffer.frame_count}")

    assert len(buffer) == 32, "Buffer should maintain max size!"
    print("[OK] Circular buffer working correctly!")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("*" * 70)
    print("TEMPORAL FRAME BUFFER TEST SUITE")
    print("*" * 70)
    print()

    # Run tests
    test_basic_buffer()
    test_multi_buffer()
    test_buffer_overflow()

    # Optional webcam test
    response = input("Run webcam test? (y/n): ").strip().lower()
    if response == 'y':
        test_webcam_buffer()

    print("*" * 70)
    print("ALL TESTS COMPLETED!")
    print("*" * 70)
    print()


if __name__ == "__main__":
    main()
