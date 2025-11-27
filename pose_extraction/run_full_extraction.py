#!/usr/bin/env python3
"""
Run full HRI30 pose extraction pipeline.
Processes all videos in the dataset and creates the final training dataset.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from hri30_pose_extractor import process_hri30_dataset

def main():
    """Run the full pose extraction pipeline."""
    
    print("🎯 HRI30 Pose Extraction Pipeline")
    print("=" * 50)
    
    # Check and configure GPU/CPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"✅ Using GPU for 1.3x faster processing!")
        
        # Clear any CUDA_VISIBLE_DEVICES setting to enable GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    else:
        print(f"💻 No GPU detected, using CPU")
    
    # Configuration
    data_root = "/home/ty/human-bahviour"
    annotations_file = "/home/ty/human-bahviour/annotations/train_set_labels.csv"
    output_dir = "pose_dataset"
    
    print(f"📁 Data root: {data_root}")
    print(f"📋 Annotations: {annotations_file}")
    print(f"💾 Output directory: {output_dir}")
    
    # Check if data exists
    if not os.path.exists(data_root):
        print(f"❌ Data root directory not found: {data_root}")
        return
    
    train_dir = os.path.join(data_root, "train_set")
    test_dir = os.path.join(data_root, "test_set")
    
    if not os.path.exists(train_dir) and not os.path.exists(test_dir):
        print(f"❌ No train_set or test_set directories found!")
        print(f"   Expected: {train_dir} or {test_dir}")
        return
    
    # Count videos
    video_count = 0
    for split_dir in [train_dir, test_dir]:
        if os.path.exists(split_dir):
            for ext in ['.avi', '.mp4', '.mov']:
                video_count += len(list(Path(split_dir).glob(f"*{ext}")))
    
    print(f"🎬 Found {video_count} video files to process")
    
    if video_count == 0:
        print("❌ No video files found!")
        return
    
    # Estimate processing time based on device
    if torch.cuda.is_available():
        # GPU estimate: ~25 seconds per video (1.3x faster than CPU)
        estimated_time = video_count * 0.4  
        device_note = "(with GPU acceleration)"
    else:
        # CPU estimate: ~30 seconds per video
        estimated_time = video_count * 0.5
        device_note = "(CPU only)"
    
    print(f"⏱️  Estimated processing time: {estimated_time/60:.1f} minutes {device_note}")
    
    # Confirm before starting
    response = input("\n🤔 Continue with full extraction? [Y/n]: ").lower()
    if response and response != 'y' and response != 'yes':
        print("❌ Extraction cancelled.")
        return
    
    try:
        # Start processing
        start_time = time.time()
        print(f"\n🚀 Starting pose extraction at {time.strftime('%H:%M:%S')}")
        print("=" * 50)
        
        # Run the pipeline
        process_hri30_dataset(
            data_root=data_root,
            annotations_file=annotations_file if os.path.exists(annotations_file) else None,
            output_dir=output_dir
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 50)
        print("✅ EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"⏱️  Total processing time: {processing_time/60:.1f} minutes")
        print(f"📊 Average time per video: {processing_time/video_count:.1f} seconds")
        print(f"💾 Output saved to: {os.path.abspath(output_dir)}")
        
        # List output files
        if os.path.exists(output_dir):
            print(f"\n📋 Output files:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   • {file} ({size_mb:.1f} MB)")
        
        print(f"\n🎉 Dataset ready for LSTM/RNN training!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
    except Exception as e:
        print(f"\n❌ Extraction failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()