#!/usr/bin/env python3
"""
Test script for HRI30 pose extraction pipeline.
Tests the pose extractor on a single video file.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

from hri30_pose_extractor import HRI30PoseExtractor

def test_single_video():
    """Test pose extraction on a single video."""
    
    # Find a test video
    data_root = "/home/ty/human-bahviour"
    
    # Look for video files
    video_paths = []
    for split in ['train_set', 'test_set']:
        split_dir = os.path.join(data_root, split)
        if os.path.exists(split_dir):
            for ext in ['.avi', '.mp4', '.mov']:
                videos = list(Path(split_dir).glob(f"*{ext}"))
                video_paths.extend(videos)
                if len(video_paths) >= 1:  # Just need one for testing
                    break
        if len(video_paths) >= 1:
            break
    
    if not video_paths:
        print("❌ No video files found for testing!")
        print(f"Please ensure videos exist in:")
        print(f"  {data_root}/train_set/")
        print(f"  {data_root}/test_set/")
        return
    
    test_video = str(video_paths[0])
    print(f"🎯 Testing pose extraction on: {os.path.basename(test_video)}")
    
    try:
        # Initialize pose extractor
        print("📦 Initializing YOLOv8-pose model...")
        extractor = HRI30PoseExtractor()
        
        # Extract keypoints
        print("🔍 Extracting keypoints...")
        keypoints, metadata = extractor.extract_keypoints(test_video)
        
        # Print results
        print("\n" + "="*50)
        print("📊 EXTRACTION RESULTS")
        print("="*50)
        print(f"✅ Keypoints extracted successfully!")
        print(f"📐 Shape: {keypoints.shape}")
        print(f"📈 Frames processed: {len(keypoints)}")
        print(f"🎬 Original FPS: {metadata['original_fps']:.2f}")
        print(f"📏 Resolution: {metadata['original_resolution']} -> {metadata['target_resolution']}")
        
        # Analyze keypoints
        print(f"\n📊 Keypoint Analysis:")
        print(f"   • Total keypoints per frame: {keypoints.shape[1] // 3}")
        print(f"   • Features per frame: {keypoints.shape[1]} (x, y, conf for each keypoint)")
        
        # Check for valid detections
        valid_frames = np.sum(np.any(keypoints > 0, axis=1))
        print(f"   • Frames with valid detections: {valid_frames}/{len(keypoints)} ({valid_frames/len(keypoints)*100:.1f}%)")
        
        # Sample keypoint data (first frame)
        if len(keypoints) > 0:
            first_frame = keypoints[0].reshape(-1, 3)  # Reshape to (17, 3)
            valid_kpts = first_frame[first_frame[:, 2] > 0]  # Filter by confidence
            print(f"   • Valid keypoints in first frame: {len(valid_kpts)}/17")
        
        # Save test output
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(f"{output_dir}/test_keypoints.npy", keypoints)
        
        import json
        with open(f"{output_dir}/test_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Test output saved to: {output_dir}/")
        print(f"   • Keypoints: test_keypoints.npy")
        print(f"   • Metadata: test_metadata.json")
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_video()