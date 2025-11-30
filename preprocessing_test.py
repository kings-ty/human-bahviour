#!/usr/bin/env python3
"""
TEST VERSION - Process only 2 videos to verify the pipeline works
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict
import json
import argparse

# Import YOLOv8-Pose from ultralytics
from ultralytics import YOLO


# COCO 17 keypoint format used by YOLOv8-Pose
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Import functions from preprocessing.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    get_reference_point,
    normalize_keypoints,
    extract_pose_from_frame,
    process_video,
    pad_or_truncate_sequence,
    load_class_labels,
    get_video_label
)


def main():
    parser = argparse.ArgumentParser(
        description='TEST: Extract pose features from 2 videos only',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data_root', type=str, default='/home/ty/human-behaviour',
                       help='Root directory containing train_set and test_set')
    parser.add_argument('--output_dir', type=str, default='/home/ty/human-behaviour/pose_features_test',
                       help='Output directory for processed .npy files')
    parser.add_argument('--model_path', type=str, default='yolov8n-pose.pt',
                       help='Path to YOLOv8-Pose model weights')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum sequence length (frames)')
    parser.add_argument('--num_test_videos', type=int, default=2,
                       help='Number of videos to test')

    args = parser.parse_args()

    print("="*80)
    print("🧪 TEST MODE - YOLOv8-Pose Feature Extraction (2 videos only)")
    print("="*80)
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Max frames: {args.max_frames}")
    print(f"Number of test videos: {args.num_test_videos}")
    print("="*80)

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Load YOLOv8-Pose model
    print(f"Loading YOLOv8-Pose model: {args.model_path}")
    model = YOLO(args.model_path)
    print("✅ Model loaded successfully")
    print()

    # Load class labels
    annotations_path = os.path.join(args.data_root, 'annotations', 'classInd.txt')
    class_to_idx = load_class_labels(annotations_path)
    print(f"✅ Loaded {len(class_to_idx)} classes")
    print()

    # Get train videos
    dataset_dir = os.path.join(args.data_root, 'train_set')

    if not os.path.exists(dataset_dir):
        print(f"❌ Error: {dataset_dir} does not exist")
        return 1

    # Get all video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(dataset_dir).glob(f'*{ext}')))

    print(f"Found {len(video_files)} total videos in train_set")
    print(f"Processing first {args.num_test_videos} videos for testing...")
    print()

    # Take only first N videos for testing
    video_files = video_files[:args.num_test_videos]

    # Process each video
    sequences = []
    labels = []
    filenames = []

    for i, video_path in enumerate(video_files, 1):
        video_name = video_path.name
        print(f"[{i}/{args.num_test_videos}] Processing: {video_name}")

        # Extract pose sequence
        sequence, success = process_video(str(video_path), model, args.device, args.max_frames)

        if not success:
            print(f"  ❌ Failed to process {video_name}")
            continue

        # Get label
        label = get_video_label(video_name, class_to_idx)
        if label < 0:
            print(f"  ⚠️  Could not determine label for {video_name}")
            continue

        sequences.append(sequence)
        labels.append(label)
        filenames.append(video_name)

        print(f"  ✅ Success! Label: {label}, Sequence shape: {sequence.shape}")
        print()

    if len(sequences) == 0:
        print("❌ No videos processed successfully")
        return 1

    # Convert to numpy arrays
    sequences = np.array(sequences, dtype=np.float32)  # (N, max_frames, 17, 2)
    labels = np.array(labels, dtype=np.int64)  # (N,)

    print("="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    print(f"Successfully processed: {len(sequences)} videos")
    print(f"Sequence shape: {sequences.shape}")
    print(f"  - Videos: {sequences.shape[0]}")
    print(f"  - Frames per video: {sequences.shape[1]}")
    print(f"  - Keypoints: {sequences.shape[2]}")
    print(f"  - Coordinates (x,y): {sequences.shape[3]}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
    print()

    # Show sample data
    print("Sample keypoint data (first frame, first 3 keypoints):")
    print(sequences[0, 0, :3, :])
    print()

    # Save to disk
    output_path = os.path.join(args.output_dir, 'test_sequences.npy')
    labels_path = os.path.join(args.output_dir, 'test_labels.npy')
    filenames_path = os.path.join(args.output_dir, 'test_filenames.json')

    np.save(output_path, sequences)
    np.save(labels_path, labels)

    with open(filenames_path, 'w') as f:
        json.dump(filenames, f, indent=2)

    print("💾 Saved test results to:")
    print(f"  Sequences: {output_path}")
    print(f"  Labels: {labels_path}")
    print(f"  Filenames: {filenames_path}")
    print()

    print("="*80)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print()
    print("If this works, you can run the full preprocessing:")
    print("  python3 preprocessing.py --batch_mode both --device cuda")
    print()

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
