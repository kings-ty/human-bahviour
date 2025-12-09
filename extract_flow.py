#!/usr/bin/env python3
"""
Optical Flow Extraction Pipeline for Motion Stream.

This script generates "Visual Trajectory" data by extracting dense optical flow frames.
It samples 16 frames uniformly from each video, computes pixel-level motion vectors
using Farneback's algorithm, and saves them as images (x-flow and y-flow).

These flow images serve as the input for the ResNet-18 Motion Stream model.

Input: Raw Videos (e.g., .avi, .mp4)
Output: flow_data_16f/{split}/{video_name}/flow_x_{i}.jpg (and flow_y_{i}.jpg)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_flow(video_path, output_dir, num_frames=16):
    """
    Extracts dense optical flow from a video using uniform sampling.
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory where flow images will be saved.
        num_frames: Number of frames to sample per video (default 16 for ResNet).
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 1:
        cap.release()
        return False

    # Calculate indices for uniform sampling (avoiding the very last frame)
    # e.g., if total=100, num=16 -> [0, 6, 12, ..., 90]
    indices = np.linspace(0, total_frames - 2, num_frames, dtype=int) 
    
    video_name = Path(video_path).stem
    save_dir = output_dir / video_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    for i in indices:
        # Seek to the sampled frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame1 = cap.read()
        ret2, frame2 = cap.read() # Read the IMMEDIATE NEXT frame for flow calculation
        
        if not ret or not ret2: 
            break
        
        # Convert to Grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize to 224x224 (Standard Input Size for ResNet-18)
        # Optimizes storage and computation speed
        gray1 = cv2.resize(gray1, (224, 224))
        gray2 = cv2.resize(gray2, (224, 224))
        
        # Compute Dense Optical Flow (Farneback's Algorithm)
        # Parameters optimized for human motion capture
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Normalize Flow Vectors to 0-255 range for image storage
        # Maps -20..20 pixel displacement to 0..255 pixel intensity
        flow_x = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_y = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Save x and y components as separate grayscale images
        cv2.imwrite(str(save_dir / f"flow_x_{frame_count:02d}.jpg"), flow_x)
        cv2.imwrite(str(save_dir / f"flow_y_{frame_count:02d}.jpg"), flow_y)
        
        frame_count += 1
        
    cap.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract Optical Flow for Motion Stream")
    # Use Relative Paths by default
    parser.add_argument('--data_root', type=str, default='.', 
                       help='Root directory containing video folders (train_set, test_set)')
    parser.add_argument('--output_dir', type=str, default='flow_data_16f', 
                       help='Output directory for flow images')
    args = parser.parse_args()
    
    print(f"🚀 Starting 16-Frame Optical Flow Extraction...")
    print(f"   Input: {args.data_root}")
    print(f"   Output: {args.output_dir}")
    
    output_root = Path(args.output_dir)
    
    # Process both Train and Test sets
    for split in ['train_set', 'test_set']:
        video_dir = Path(args.data_root) / split
        
        if not video_dir.exists():
            print(f"⚠️  Skipping {split}: Directory not found.")
            continue
        
        # Gather all video files
        videos = list(video_dir.glob('*.avi')) + list(video_dir.glob('*.mp4'))
        print(f"Processing {split} ({len(videos)} videos)...")
        
        # Iterate and extract flow
        # Note: We run sequentially. Multiprocessing can be used but might overload edge CPUs.
        for vid in tqdm(videos, desc=f"Extracting {split}"):
            extract_flow(vid, output_root / split)
            
    print("✅ Flow Extraction Complete! Ready for ResNet Training.")

if __name__ == '__main__':
    main()