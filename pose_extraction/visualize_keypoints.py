#!/usr/bin/env python3
"""
Visualize extracted keypoints to verify the extraction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_keypoints():
    """Analyze and visualize extracted keypoints."""
    
    # Load test keypoints
    keypoints_file = "test_output/test_keypoints.npy"
    if not os.path.exists(keypoints_file):
        print(f"❌ File not found: {keypoints_file}")
        print("Run test_single_video.py first!")
        return
    
    keypoints = np.load(keypoints_file)
    print(f"📊 Loaded keypoints: {keypoints.shape}")
    
    # Reshape to (frames, keypoints, 3)
    frames, features = keypoints.shape
    keypoints_reshaped = keypoints.reshape(frames, 17, 3)
    
    # Keypoint names
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Analysis
    print(f"\n📈 Keypoint Analysis:")
    print(f"   • Total frames: {frames}")
    print(f"   • Keypoints per frame: 17")
    print(f"   • Features per frame: {features} (x, y, confidence)")
    
    # Check confidence statistics
    confidences = keypoints_reshaped[:, :, 2]  # All confidence scores
    valid_confidences = confidences[confidences > 0]
    
    print(f"\n🎯 Detection Quality:")
    print(f"   • Average confidence: {np.mean(valid_confidences):.3f}")
    print(f"   • Min confidence: {np.min(valid_confidences):.3f}")
    print(f"   • Max confidence: {np.max(valid_confidences):.3f}")
    
    # Per-keypoint analysis
    print(f"\n📍 Per-Keypoint Detection Rate:")
    for i, name in enumerate(keypoint_names):
        detection_rate = np.mean(confidences[:, i] > 0) * 100
        avg_conf = np.mean(confidences[confidences[:, i] > 0, i]) if np.any(confidences[:, i] > 0) else 0
        print(f"   • {name:15}: {detection_rate:5.1f}% (avg conf: {avg_conf:.3f})")
    
    # Visualize first frame
    first_frame = keypoints_reshaped[0]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Keypoint positions
    valid_kpts = first_frame[first_frame[:, 2] > 0]
    if len(valid_kpts) > 0:
        ax1.scatter(valid_kpts[:, 0], valid_kpts[:, 1], 
                   c=valid_kpts[:, 2], cmap='viridis', s=100, alpha=0.7)
        ax1.set_title('First Frame Keypoints\n(Color = Confidence)')
        ax1.set_xlabel('X (Normalized)')
        ax1.set_ylabel('Y (Normalized)')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar.set_label('Confidence')
    
    # Plot 2: Confidence over time for key points
    time_axis = np.arange(frames)
    
    # Plot confidence for important keypoints
    important_kpts = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    for name in important_kpts:
        idx = keypoint_names.index(name)
        conf_series = confidences[:, idx]
        ax2.plot(time_axis, conf_series, label=name, alpha=0.7)
    
    ax2.set_title('Keypoint Confidence Over Time')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save visualization
    output_file = "test_output/keypoint_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {output_file}")
    
    # Show some sample data
    print(f"\n📋 Sample Data (First Frame):")
    print(f"Frame shape: {first_frame.shape}")
    print(f"Sample keypoints (nose, left_shoulder, right_shoulder):")
    for i, name in enumerate(['nose', 'left_shoulder', 'right_shoulder']):
        x, y, c = first_frame[keypoint_names.index(name)]
        print(f"   • {name:15}: ({x:6.3f}, {y:6.3f}, conf={c:.3f})")
    
    plt.show()

if __name__ == "__main__":
    analyze_keypoints()