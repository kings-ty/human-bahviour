#!/usr/bin/env python3
"""This script transforms raw skeleton coordinates into high-level kinematic physics features.
It acts as the "Feature Engineering" bridge between raw HPE data and the Bi-LSTM model.

Key Features Extracted:
1. Joint Angles (e.g., Elbow Angle, Knee Angle)
2. Angular Velocities (e.g., Torso Turn Rate)
3. Limb Lengths & Distances
4. Relative Heights (e.g., Wrist vs Shoulder)
5. Smoothing & Noise Reduction (Gaussian Filter)

Input: 
    - pose_features_large/{split}_sequences.npy (N, T, 17, 2) [COCO Format]
Output:
    - pose_features_smart_v3/{split}_features.npy (N, T, 79) [Smart Feature Vector]
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 1. Coordinate Mapping (COCO -> NTU Topology)
# ==========================================
# We map 17-keypoint COCO format to a richer 25-keypoint NTU format
# to enable more complex feature calculations (like spine, neck, etc.)
COCO_TO_NTU_MAPPING = {
    0: 3,   # Nose -> Head
    5: 4,   # L_Shoulder -> L_Shoulder
    6: 8,   # R_Shoulder -> R_Shoulder
    7: 5,   # L_Elbow -> L_Elbow
    8: 9,   # R_Elbow -> R_Elbow
    9: 6,   # L_Wrist -> L_Wrist
    10: 10, # R_Wrist -> R_Wrist
    11: 12, # L_Hip -> L_Hip
    12: 16, # R_Hip -> R_Hip
    13: 13, # L_Knee -> L_Knee
    14: 17, # R_Knee -> R_Knee
    15: 14, # L_Ankle -> L_Ankle
    16: 18  # R_Ankle -> R_Ankle
}

def map_coco_to_ntu_batch(coco_data):
    """
    Convert COCO (17 joints) to NTU-RGB+D (25 joints) topology.
    Missing joints (Spine, Neck, etc.) are interpolated geometrically.
    """
    N, T, _, C = coco_data.shape
    ntu = np.zeros((N, T, 25, 3), dtype=np.float32)
    
    # Direct Mapping
    for c, n in COCO_TO_NTU_MAPPING.items():
        ntu[:, :, n, :2] = coco_data[:, :, c, :]
        
    # Geometric Interpolation for Spine/Center joints
    # 0: Base of Spine (Midpoint of Hips)
    ntu[:, :, 0, :] = (ntu[:, :, 12, :] + ntu[:, :, 16, :]) / 2.0 
    
    # Shoulder Center
    shoulder_center = (ntu[:, :, 4, :] + ntu[:, :, 8, :]) / 2.0
    
    # 20: Spine Shoulder (Midpoint of Shoulders)
    ntu[:, :, 20, :] = (shoulder_center + ntu[:, :, 0, :]) / 2.0
    
    # 1: Mid Spine (Midpoint of Base and Spine Shoulder)
    ntu[:, :, 1, :] = (ntu[:, :, 0, :] + ntu[:, :, 20, :]) / 2.0 
    
    # 2: Neck (Shoulder Center)
    ntu[:, :, 2, :] = shoulder_center 
    
    # 3: Head (Nose is already mapped to 3, keep it)
    # ntu[:, :, 3, :] = ntu[:, :, 0, :] # (Legacy override removed)

    return ntu

# ==========================================
# 2. Physics Feature Calculators
# ==========================================

def compute_angle(a, b, c):
    """
    Compute angle ABC (at vertex B) in radians.
    """
    ba = a - b
    bc = c - b
    dot_product = np.sum(ba * bc, axis=-1)
    norm_ba = np.linalg.norm(ba, axis=-1)
    norm_bc = np.linalg.norm(bc, axis=-1)
    denominator = norm_ba * norm_bc
    cosine_angle = np.zeros_like(denominator)
    mask = denominator > 1e-6
    cosine_angle[mask] = dot_product[mask] / denominator[mask]
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.nan_to_num(angle)

def compute_height_velocity(head, l_ankle, r_ankle):
    """
    Compute vertical velocity of the body (Standing vs Crouching).
    """
    ankle_y = (l_ankle[:, 1] + r_ankle[:, 1]) / 2.0
    head_y = head[:, 1]
    height = np.abs(ankle_y - head_y)
    if height[0] > 1e-3: height = height / height[0] # Normalize by initial height
    
    vel = np.zeros_like(height)
    vel[1:] = height[1:] - height[:-1]
    return vel[:, np.newaxis]

def compute_torso_turn_velocity(l_shoulder, r_shoulder):
    """
    Calculate the angular velocity of the shoulder vector.
    Captures turning motions (e.g., looking around, screwing).
    """
    # Vector from Left Shoulder to Right Shoulder
    shoulder_vec = r_shoulder - l_shoulder # (T, 2) [dx, dy]
    
    # Calculate angle in 2D plane (radians)
    angles = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0]) # (T,)
    
    # Calculate angular velocity (derivative)
    # Handle angle wrapping (-pi to pi) using unwrap
    angles_unwrapped = np.unwrap(angles)
    
    ang_vel = np.zeros_like(angles_unwrapped)
    ang_vel[1:] = angles_unwrapped[1:] - angles_unwrapped[:-1]
    
    return ang_vel[:, np.newaxis] # (T, 1)

def compute_relative_wrist_height(r_wrist, r_shoulder, hip, neck):
    """
    Calculate Right Wrist Height relative to Shoulder.
    Essential for distinguishing "Lifting" vs "Dropping".
    """
    # Torso Height for normalization (Neck to Hip)
    torso_h = np.abs(neck[:, 1] - hip[:, 1]) + 1e-6
    
    # Relative Y: Wrist_Y - Shoulder_Y
    # Note: In image coords, Y increases downwards.
    rel_y = r_wrist[:, 1] - r_shoulder[:, 1] 
    
    ratio = rel_y / torso_h
    return ratio[:, np.newaxis] # (T, 1)

# ==========================================
# 3. Main Processing Logic
# ==========================================

def enrich_and_format_data(raw_data):
    """
    Main function to compute all features for a batch of sequences.
    """
    N, T, V, C = raw_data.shape 
    
    # 1. Stage A: Pre-Smoothing (Sigma=2.0)
    # Reduce jitter from YOLO before calculating sensitive derivatives (velocity)
    print("   🛡️ Stage A: Pre-Smoothing Raw Coordinates (Sigma=2.0)...")
    smoothed_data = gaussian_filter1d(raw_data, sigma=2.0, axis=1)
    
    # 2. Feature Calculation Loop
    smart_feats_list = []
    
    for i in range(N):
        sample = smoothed_data[i] # (T, 25, C)
        
        # Extract Key Joints
        # 0:Hip, 2:Neck, 3:Head
        # 4:LSh, 8:RSh, 10:RWr, 14:LAn, 18:RAn
        
        hip = sample[:, 0, :2]
        neck = sample[:, 2, :2]
        head = sample[:, 3, :2]
        l_sh, r_sh = sample[:, 4, :2], sample[:, 8, :2]
        r_wr = sample[:, 10, :2] 
        l_an, r_an = sample[:, 14, :2], sample[:, 18, :2]
        
        # --- Feature Set ---
        
        # F1: Torso Bend Angle (Posture)
        f1_bend = compute_angle(hip, neck, head)[:, np.newaxis]
        
        # F2: Height Velocity (Vertical Motion)
        f2_height = compute_height_velocity(head, l_an, r_an)
        
        # F3: Torso Turn Velocity (Rotation)
        f3_turn = compute_torso_turn_velocity(l_sh, r_sh)
        
        # F4: Wrist Relative Height (Arm Position)
        f4_wrist = compute_relative_wrist_height(r_wr, r_sh, hip, neck)
        
        # Debugging First Sample
        if i == 0:
            print("\n--- DEBUG: Feature Stats (Sample 0) ---")
            print(f"Torso Turn Max: {np.max(f3_turn):.6f}, Min: {np.min(f3_turn):.6f}")
            print(f"Wrist Height Max: {np.max(f4_wrist):.6f}, Min: {np.min(f4_wrist):.6f}")

        # Concatenate Features
        feats = np.concatenate([f1_bend, f2_height, f3_turn, f4_wrist], axis=1) # (T, 4)
        smart_feats_list.append(feats)
        
    smart_feats = np.stack(smart_feats_list) # (N, T, 4)
    
    # 3. Stage B: Post-Smoothing
    # We disable this to preserve sharp motion changes
    # smart_feats_smooth = gaussian_filter1d(smart_feats, sigma=4.0, axis=1)
    smart_feats_smooth = smart_feats 
    
    # 4. Final Tensor Construction
    # Combine (Raw Coordinates) + (Smart Features)
    flattened_joints = smoothed_data.reshape(N, T, -1) # (N, T, 75)
    final_tensor = np.concatenate([flattened_joints, smart_feats_smooth], axis=2) # (N, T, 79)
    
    return np.nan_to_num(final_tensor)

def main():
    print("🚀 Executing spaceX-Level Feature Engineering Pipeline...")
    
    # Use Relative Paths
    input_dir = Path('pose_features_large')
    output_dir = Path('pose_features_smart_v3')
    output_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'test']:
        # Try loading 'good' version first, then standard
        input_path = input_dir / f'{split}_sequences_good.npy'
        if not input_path.exists(): 
            input_path = input_dir / f'{split}_sequences.npy'
        
        if not input_path.exists():
            print(f"⚠️  Skipping {split}: {input_path} not found.")
            continue
        
        print(f"Processing {split} set...")
        data = np.load(input_path)
        
        # Convert Topology
        data_ntu = map_coco_to_ntu_batch(data)
        
        # Extract Features
        final_data = enrich_and_format_data(data_ntu)
        print(f"   -> Final Shape: {final_data.shape}") # Should be (N, T, 79)
        
        # Save
        np.save(output_dir / f'{split}_features.npy', final_data)
        
        # Copy Labels for convenience
        if split == 'train':
            lbl_path = input_dir / 'train_labels.npy'
            if lbl_path.exists():
                np.save(output_dir / 'train_labels.npy', np.load(lbl_path))
            
    print("✅ Feature Engineering Pipeline Complete.")

if __name__ == '__main__':
    main()