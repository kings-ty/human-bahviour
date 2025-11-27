#!/usr/bin/env python3
"""
Test GPU performance for pose extraction.
Compare CPU vs GPU speed and check memory usage.
"""

import os
import sys
import time
import torch
import psutil
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from hri30_pose_extractor import HRI30PoseExtractor

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def test_performance():
    """Test CPU vs GPU performance for pose extraction."""
    
    # Find a test video
    video_path = None
    for split in ['train_set', 'test_set']:
        split_dir = f"/home/ty/human-bahviour/{split}"
        if os.path.exists(split_dir):
            videos = [f for f in os.listdir(split_dir) if f.endswith('.avi')]
            if videos:
                video_path = os.path.join(split_dir, videos[0])
                break
    
    if not video_path:
        print("❌ No test video found!")
        return
    
    print("🎯 GPU vs CPU Performance Test")
    print("=" * 50)
    print(f"📹 Test video: {os.path.basename(video_path)}")
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("❌ No GPU available")
    
    print(f"💻 CPU cores: {psutil.cpu_count()}")
    print()
    
    results = {}
    
    # Test CPU performance
    print("🔄 Testing CPU performance...")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    
    start_time = time.time()
    try:
        extractor_cpu = HRI30PoseExtractor()
        keypoints_cpu, metadata_cpu = extractor_cpu.extract_keypoints(video_path)
        cpu_time = time.time() - start_time
        
        results['cpu'] = {
            'time': cpu_time,
            'frames': len(keypoints_cpu),
            'fps': len(keypoints_cpu) / cpu_time,
            'memory': psutil.Process().memory_info().rss / 1024**3
        }
        
        print(f"   ✅ CPU: {cpu_time:.2f}s ({results['cpu']['fps']:.1f} fps)")
        print(f"   📊 Memory: {results['cpu']['memory']:.2f} GB RAM")
        
    except Exception as e:
        print(f"   ❌ CPU test failed: {e}")
        results['cpu'] = None
    
    # Test GPU performance if available
    if gpu_available:
        print("\n🔄 Testing GPU performance...")
        
        # Clear environment to allow GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Check GPU memory before
        torch.cuda.empty_cache()
        gpu_mem_before = get_gpu_memory()
        
        start_time = time.time()
        try:
            extractor_gpu = HRI30PoseExtractor()
            keypoints_gpu, metadata_gpu = extractor_gpu.extract_keypoints(video_path)
            gpu_time = time.time() - start_time
            
            gpu_mem_after = get_gpu_memory()
            
            results['gpu'] = {
                'time': gpu_time,
                'frames': len(keypoints_gpu),
                'fps': len(keypoints_gpu) / gpu_time,
                'gpu_memory': gpu_mem_after - gpu_mem_before,
                'ram_memory': psutil.Process().memory_info().rss / 1024**3
            }
            
            print(f"   ✅ GPU: {gpu_time:.2f}s ({results['gpu']['fps']:.1f} fps)")
            print(f"   📊 GPU Memory: {results['gpu']['gpu_memory']:.2f} GB")
            print(f"   📊 RAM Memory: {results['gpu']['ram_memory']:.2f} GB")
            
        except Exception as e:
            print(f"   ❌ GPU test failed: {e}")
            print(f"   💡 Tip: Your GPU might have insufficient memory")
            results['gpu'] = None
    
    # Compare results
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    if results.get('cpu') and results.get('gpu'):
        speedup = results['gpu']['fps'] / results['cpu']['fps']
        time_reduction = (results['cpu']['time'] - results['gpu']['time']) / results['cpu']['time'] * 100
        
        print(f"🚀 GPU Speedup: {speedup:.1f}x faster")
        print(f"⏰ Time Reduction: {time_reduction:.1f}%")
        print(f"📈 CPU: {results['cpu']['fps']:.1f} fps")
        print(f"📈 GPU: {results['gpu']['fps']:.1f} fps")
        
        # Memory comparison
        print(f"\n💾 Memory Usage:")
        print(f"   • CPU RAM: {results['cpu']['memory']:.2f} GB")
        print(f"   • GPU VRAM: {results['gpu']['gpu_memory']:.2f} GB")
        print(f"   • GPU RAM: {results['gpu']['ram_memory']:.2f} GB")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if results['gpu']['gpu_memory'] < 1.0:  # Less than 1GB GPU memory
            print(f"   ✅ GPU is safe to use (only {results['gpu']['gpu_memory']:.2f} GB)")
            print(f"   🚀 Use GPU for {speedup:.1f}x faster processing!")
        else:
            print(f"   ⚠️  GPU uses {results['gpu']['gpu_memory']:.2f} GB")
            print(f"   🤔 Monitor GPU memory if processing large datasets")
        
        # Full dataset estimates
        total_videos = 2042 + 839  # train + test
        cpu_total_time = results['cpu']['time'] * total_videos / 3600  # hours
        gpu_total_time = results['gpu']['time'] * total_videos / 3600  # hours
        
        print(f"\n🎬 Full Dataset Estimates ({total_videos} videos):")
        print(f"   • CPU: {cpu_total_time:.1f} hours")
        print(f"   • GPU: {gpu_total_time:.1f} hours")
        print(f"   • Time saved: {cpu_total_time - gpu_total_time:.1f} hours")
        
    elif results.get('cpu'):
        print(f"📊 CPU Only Results:")
        print(f"   • Speed: {results['cpu']['fps']:.1f} fps")
        print(f"   • Memory: {results['cpu']['memory']:.2f} GB RAM")
        
        if not gpu_available:
            print(f"\n💡 No GPU detected. CPU performance is good for testing.")
        else:
            print(f"\n💡 GPU available but test failed. Try reducing video resolution.")
    
    return results

if __name__ == "__main__":
    test_performance()