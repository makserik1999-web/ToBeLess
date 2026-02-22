#!/usr/bin/env python3
"""
RWF-2000 Dataset Verification Script

Checks that the dataset is properly downloaded and structured.
"""

import os
from pathlib import Path


def verify_rwf_dataset(data_dir: str = "datasets/RWF-2000") -> bool:
    """
    Verify RWF-2000 dataset structure

    Expected structure:
    datasets/RWF-2000/
    ├── train/
    │   ├── Fight/
    │   └── NonFight/
    └── val/
        ├── Fight/
        └── NonFight/
    """
    data_path = Path(data_dir)

    print("=" * 60)
    print("RWF-2000 Dataset Verification")
    print("=" * 60)
    print(f"\nChecking: {data_path.absolute()}")

    errors = []
    warnings = []

    # Check root directory
    if not data_path.exists():
        print(f"\n[ERROR] Dataset directory not found: {data_path}")
        print("\nPlease download the RWF-2000 dataset and extract it to this location.")
        print("See datasets/README.md for instructions.")
        return False

    # Check required subdirectories
    required_dirs = [
        "train/Fight",
        "train/NonFight",
        "val/Fight",
        "val/NonFight"
    ]

    for subdir in required_dirs:
        subpath = data_path / subdir
        if not subpath.exists():
            errors.append(f"Missing directory: {subdir}")
        else:
            # Count videos
            video_count = 0
            for ext in ['*.avi', '*.mp4', '*.mkv', '*.mov']:
                video_count += len(list(subpath.glob(ext)))

            if video_count == 0:
                warnings.append(f"No videos found in: {subdir}")
            else:
                print(f"  ✓ {subdir}: {video_count} videos")

    # Report errors
    if errors:
        print("\n[ERRORS]")
        for err in errors:
            print(f"  ✗ {err}")
        return False

    # Report warnings
    if warnings:
        print("\n[WARNINGS]")
        for warn in warnings:
            print(f"  ⚠ {warn}")

    # Summary
    print("\n" + "-" * 60)

    # Count total videos
    total_train = 0
    total_val = 0

    for class_name in ["Fight", "NonFight"]:
        train_dir = data_path / "train" / class_name
        val_dir = data_path / "val" / class_name

        if train_dir.exists():
            for ext in ['*.avi', '*.mp4', '*.mkv', '*.mov']:
                total_train += len(list(train_dir.glob(ext)))

        if val_dir.exists():
            for ext in ['*.avi', '*.mp4', '*.mkv', '*.mov']:
                total_val += len(list(val_dir.glob(ext)))

    print(f"Total training videos: {total_train}")
    print(f"Total validation videos: {total_val}")
    print(f"Total videos: {total_train + total_val}")

    if total_train + total_val == 0:
        print("\n[ERROR] No videos found in dataset!")
        return False

    # Expected: ~2000 videos total
    if total_train + total_val < 1500:
        print(f"\n[WARNING] Expected ~2000 videos, found {total_train + total_val}")
        print("Dataset may be incomplete.")

    print("\n" + "=" * 60)
    print("Dataset verification PASSED!" if not warnings else "Dataset verification completed with warnings.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify RWF-2000 dataset")
    parser.add_argument("--data_dir", default="datasets/RWF-2000", help="Dataset directory")
    args = parser.parse_args()

    success = verify_rwf_dataset(args.data_dir)
    exit(0 if success else 1)
