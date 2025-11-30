#!/usr/bin/env python3
"""
Extract poses for CID29 and CID30 videos and append to existing test files
"""

import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from tqdm import tqdm

def extract_single_video_poses(video_path, model):
    """Extract poses from a single video"""
    cap = cv2.VideoCapture(video_path)
    poses = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints) > 0:
                # Take first person if multiple detected
                pose = keypoints[0].flatten()  # 17 keypoints * 2 = 34 values
            else:
                pose = np.zeros(34)  # No detection
        else:
            pose = np.zeros(34)  # No keypoints
            
        poses.append(pose)
    
    cap.release()
    return np.array(poses)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8n-pose.pt')

# Load existing data
print("Loading existing TRAIN data...")
train_sequences = np.load('pose_features/train_sequences.npy')
train_labels = np.load('pose_features/train_labels.npy')
with open('pose_features/train_filenames.json', 'r') as f:
    train_filenames = json.load(f)

print(f"Current train data shape: sequences={train_sequences.shape}, labels={train_labels.shape}")

# Process CID29 and CID30 videos
new_sequences = []
new_labels = []
new_filenames = []

video_files = [f for f in os.listdir('train_set') if f.endswith('.avi') and (f.startswith('CID29') or f.startswith('CID30'))]

for video_file in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join('train_set', video_file)
    
    # Extract poses
    poses = extract_single_video_poses(video_path, model)
    
    # Determine label
    if video_file.startswith('CID29'):
        label = 28  # WalkingWithDrill (29-1 for 0-based)
    else:  # CID30
        label = 29  # WalkingWithPolisher (30-1 for 0-based)
    
    new_sequences.append(poses)
    new_labels.append(label)
    new_filenames.append(video_file)
    
    print(f"Processed {video_file}: {poses.shape} poses, label={label}")

# Convert to numpy arrays - pad sequences to match existing format
max_frames = 60  # Same as existing train data
padded_sequences = []

for seq in new_sequences:
    if len(seq) > max_frames:
        # Truncate if too long
        padded_seq = seq[:max_frames]
    else:
        # Pad if too short
        padding = np.zeros((max_frames - len(seq), 34))
        padded_seq = np.vstack([seq, padding])
    
    # Reshape to (60, 17, 2) format
    padded_seq = padded_seq.reshape(max_frames, 17, 2)
    padded_sequences.append(padded_seq)

new_sequences = np.array(padded_sequences)  # (140, 60, 17, 2)
new_labels = np.array(new_labels)

print(f"New data: sequences={len(new_sequences)}, labels={new_labels.shape}")

# Append to existing data
updated_sequences = np.concatenate([train_sequences, new_sequences])
updated_labels = np.concatenate([train_labels, new_labels])
updated_filenames = train_filenames + new_filenames

print(f"Updated data: sequences={len(updated_sequences)}, labels={updated_labels.shape}")

# Save updated data
print("Saving updated data...")
np.save('pose_features/train_sequences_updated.npy', updated_sequences)
np.save('pose_features/train_labels_updated.npy', updated_labels)
with open('pose_features/train_filenames_updated.json', 'w') as f:
    json.dump(updated_filenames, f, indent=2)

print("✅ Successfully added CID29 and CID30 data to TRAIN set!")
print("New files created:")
print("- pose_features/train_sequences_updated.npy")
print("- pose_features/train_labels_updated.npy") 
print("- pose_features/train_filenames_updated.json")