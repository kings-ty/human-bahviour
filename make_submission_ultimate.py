#!/usr/bin/env python3
"""
Ultimate Ensemble Submission Script.
Combines predictions from:
1. Motion Stream (ResNet-18 on Optical Flow)
2. Skeleton Stream (Bi-LSTM on Smart Features)
3. Object Detection Stream (Grounding DINO - optional veto power)

Produces the final accuracy report, including per-class performance analysis.
"""

import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
import os
import cv2
import random
from pathlib import Path
from torchvision import transforms 

# Import Models (Ensure these files are in the same directory)
from train_flow_resnet import FlowResNet
from train_lstm_smart_v2 import LSTMModel

# =========================================================
# Config & Paths
# =========================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

# Model Weights Paths
FLOW_MODEL_PATH = 'best_flow_resnet.pth'
LSTM_MODEL_PATH = 'data/best_lstm_v3_fold0.pth'

# DINO Object Detection Results (Optional)
DINO_RESULT_PATH = 'sap2/objects/object_predictions.json'
DINO_TXT_PATH = 'sap2/objects/validation_results.txt'

# Data Directories (Relative Paths)
DATA_DIR_FLOW = 'flow_data_16f/train_set' # Directory containing flow images folders
DATA_DIR_LSTM = 'pose_features_smart_v3'  # Directory containing .npy feature files
DATA_DIR_META = 'pose_features_large'     # Directory containing metadata (labels, filenames)

# Ensemble Weights (Tuned for HRI30)
# LSTM is the primary expert; Flow provides motion context.
W_FLOW = 0.1
W_LSTM = 0.9

# DINO Class Mapping (Object Name -> Action Class Indices)
# Used to penalize impossible actions (e.g., if "Drill" is detected, don't predict "Painting")
DRILL_INDICES = [3, 5, 7, 9, 11, 13, 15, 19, 22, 24, 26]
POLISHER_INDICES = [4, 6, 8, 10, 12, 14, 16, 18, 23, 25, 27]
OBJECT_INDICES = [21, 28]

CLASS_MAP = {
    'drill': DRILL_INDICES,
    'polisher': POLISHER_INDICES,
    'object': OBJECT_INDICES,
    'dumbbell': OBJECT_INDICES
}

# ImageNet Normalization (Required for Flow ResNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# =========================================================
# Dataset Class: Ultimate Fusion Dataset
# =========================================================
class UltimateDataset(Dataset):
    def __init__(self):
        """
        Loads and aligns data from both streams (Flow Images & Skeleton Features).
        Ensures strict alignment using filename mapping.
        """
        print("⚡ Initializing UltimateDataset with Sorted Directory Scan...")
        self.root_dir = DATA_DIR_FLOW 
        
        # 1. Scan Flow Directories (Ground Truth for available data)
        if not os.path.exists(self.root_dir):
             print(f"❌ Error: {self.root_dir} does not exist.")
             self.filenames = []
             return

        folder_names = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # 2. Load Metadata (Labels & Filenames)
        try:
            with open(f'{DATA_DIR_META}/train_filenames.json', 'r') as f:
                meta_filenames = json.load(f) # ["video1.avi", "video2.mp4", ...]
            meta_labels = np.load(f'{DATA_DIR_META}/train_labels.npy')
            
            # Load LSTM Features
            meta_features = np.load(f'{DATA_DIR_LSTM}/train_features.npy')
            
            # Load Normalization Stats
            mean = np.load(f'{DATA_DIR_LSTM}/feature_mean.npy')
            std = np.load(f'{DATA_DIR_LSTM}/feature_std.npy')
            meta_features = (meta_features - mean) / std # Normalize immediately
            
        except FileNotFoundError as e:
            print(f"❌ Error loading metadata or features: {e}")
            self.filenames = []
            return
        
        # 3. Create Mapping Table (Stem -> Index)
        # Allows matching "video1" (folder) to "video1.avi" (metadata)
        stem_to_idx = {Path(f).stem: i for i, f in enumerate(meta_filenames)}
        
        self.filenames = []
        self.labels = []
        self.physics_data = []
        
        missing_count = 0
        for folder in folder_names:
            key = Path(folder).stem 
            
            if key in stem_to_idx:
                idx = stem_to_idx[key]
                self.filenames.append(folder) # Keep folder name for loading images
                self.labels.append(meta_labels[idx])
                self.physics_data.append(meta_features[idx])
            else:
                missing_count += 1
                
        self.labels = np.array(self.labels)
        self.physics_data = np.array(self.physics_data)
        
        print(f"✅ Loaded {len(self.filenames)} samples (Aligned). Skipped: {missing_count}")

        # Initialize Transforms for Flow Data
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(), # HWC -> CHW, 0-255 -> 0-1
        ])

    def __len__(self):
        return len(self.filenames)

    def load_frames(self, vid_name):
        """Loads and stacks 16 flow frames for a given video."""
        vid_path = os.path.join(self.root_dir, vid_name)
        
        frames_np = [] 
        for i in range(16):
            p_x = os.path.join(vid_path, f"flow_x_{i:02d}.jpg")
            p_y = os.path.join(vid_path, f"flow_y_{i:02d}.jpg")
            
            # Fallback for non-padded filenames
            if not os.path.exists(p_x): p_x = os.path.join(vid_path, f"flow_x_{i}.jpg")
            if not os.path.exists(p_y): p_y = os.path.join(vid_path, f"flow_y_{i}.jpg")
            
            img = None
            if os.path.exists(p_x) and os.path.exists(p_y):
                try:
                    img_x = cv2.imread(p_x, cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(p_y, cv2.IMREAD_GRAYSCALE)
                    if img_x is not None and img_y is not None:
                        # Stack (X, Y, X) to simulate RGB for ResNet
                        img = np.stack([img_x, img_y, img_x], axis=2) 
                except Exception:
                    pass
            
            # Handle missing/corrupt frames with black image
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
            
            frames_np.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        processed_frames_tensors = [self.transform(frame) for frame in frames_np]
        
        # Stack into (T, C, H, W)
        return torch.stack(processed_frames_tensors)

    def __getitem__(self, idx):
        vid_name = self.filenames[idx]
        flow_tensor = self.load_frames(vid_name) 
        lstm_tensor = torch.tensor(self.physics_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return flow_tensor, lstm_tensor, label, vid_name

# =========================================================
# Helper Functions
# =========================================================
def load_dino_results(json_path, txt_path_fallback):
    """Loads object detection results from JSON or converts from TXT."""
    if os.path.exists(json_path):
        print(f"✅ Loading DINO context from JSON: {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
    
    if os.path.exists(txt_path_fallback):
        print(f"⚠️ JSON not found. Converting TXT ({txt_path_fallback}) to JSON...")
        dino_map = {}
        try:
            with open(txt_path_fallback, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(': ')
                    if len(parts) >= 2:
                        fname = parts[0].strip()
                        obj_name = parts[1].strip()
                        if obj_name != 'None': 
                            dino_map[fname] = obj_name
            
            with open(json_path, 'w') as f:
                json.dump(dino_map, f, indent=4)
            return dino_map
        except Exception:
            return {}

    print(f"❌ No DINO results found.")
    return {}

def apply_dino_correction(probs, vid_name, dino_map, match_counter=None):
    """
    Applies 'Negative Constraints' (Veto) based on detected objects.
    If 'Drill' is detected, heavily penalize non-drill classes.
    """
    detected_obj = dino_map.get(vid_name, None)
    if detected_obj is None:
        detected_obj = dino_map.get(vid_name + '.avi', None)
        
    if detected_obj is None or detected_obj == "None":
        return probs

    if match_counter is not None:
        match_counter['matched'] += 1

    if detected_obj not in CLASS_MAP:
        return probs

    target_indices = CLASS_MAP[detected_obj]
    
    # Create mask: True for allowed classes, False for disallowed
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[target_indices] = True
    
    # Penalize disallowed classes (Veto)
    probs[~mask] *= 0.01 
    
    # Re-normalize probability distribution
    probs = probs / probs.sum()
    
    return probs

# =========================================================
# Main Execution
# =========================================================
def main():
    print("🚀 Starting Ultimate Ensemble (Flow + LSTM + DINO)...")
    
    # 1. Initialize Dataset
    full_dataset = UltimateDataset() 
    if len(full_dataset) == 0:
        print("❌ Dataset is empty. Exiting.")
        return

    # --- VALIDATION MODE: Random Subset ---
    # random.seed(42) # Uncomment for reproducibility
    
    total_samples = len(full_dataset)
    subset_size = 420 # Validation set size
    
    if total_samples >= subset_size:
        indices = list(range(total_samples))
        random.shuffle(indices)
        val_indices = indices[:subset_size]
        
        dataset = Subset(full_dataset, val_indices)
        print(f"\n⚠️  [VALIDATION MODE ACTIVE]")
        print(f"   - Evaluating on {subset_size} random samples.")
    else:
        print(f"⚠️  Using full dataset ({total_samples} samples).")
        dataset = full_dataset

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 
    
    # 2. Load Models
    print(f"Loading Flow Model: {FLOW_MODEL_PATH}")
    try:
        model_flow = FlowResNet(num_classes=30).to(DEVICE)
        model_flow.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location=DEVICE))
        model_flow.eval()
    except FileNotFoundError:
        print(f"❌ Flow model not found at {FLOW_MODEL_PATH}")
        return
    
    print(f"Loading LSTM Model: {LSTM_MODEL_PATH}")
    try:
        model_lstm = LSTMModel(input_dim=79, hidden_size=128, num_classes=30, num_layers=2).to(DEVICE)
        model_lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
        model_lstm.eval()
    except FileNotFoundError:
        print(f"❌ LSTM model not found at {LSTM_MODEL_PATH}")
        return
    
    # 3. Load DINO Context
    dino_map = load_dino_results(DINO_RESULT_PATH, DINO_TXT_PATH)
    
    # 4. Inference Loop
    all_preds = []
    all_labels = []
    
    # Accuracy Counters
    correct_flow = 0
    correct_lstm = 0
    correct_ensemble = 0
    total = 0
    dino_stats = {'matched': 0}
    
    print("⚡ Running Inference...")
    
    with torch.no_grad():
        for flow, lstm, label, vid_names in tqdm(loader):
            flow, lstm = flow.to(DEVICE), lstm.to(DEVICE)
            label = label.to(DEVICE)
            
            # Model Forward Pass
            logits_flow = model_flow(flow)
            logits_lstm = model_lstm(lstm)
            
            prob_flow = F.softmax(logits_flow, dim=1)
            prob_lstm = F.softmax(logits_lstm, dim=1)
            
            # Individual Predictions (for Diagnosis)
            pred_flow = torch.argmax(prob_flow, dim=1)
            pred_lstm = torch.argmax(prob_lstm, dim=1)
            
            correct_flow += (pred_flow == label).sum().item()
            correct_lstm += (pred_lstm == label).sum().item()
            
            # Ensemble Fusion
            final_probs = (W_FLOW * prob_flow) + (W_LSTM * prob_lstm)
            
            # Apply DINO Correction
            for i in range(len(vid_names)):
                final_probs[i] = apply_dino_correction(final_probs[i], vid_names[i], dino_map, dino_stats)
            
            pred_ensemble = torch.argmax(final_probs, dim=1)
            correct_ensemble += (pred_ensemble == label).sum().item()
            
            total += label.size(0)
            
            all_preds.extend(pred_ensemble.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
    # 5. Reporting
    print("\n" + "="*50)
    print(f"📊 DIAGNOSIS REPORT")
    print(f" - Flow Stream Acc:   {correct_flow / total * 100:.2f}%")
    print(f" - LSTM Stream Acc:   {correct_lstm / total * 100:.2f}%")
    print(f" - Ensemble Acc:      {correct_ensemble / total * 100:.2f}%")
    print(f" - DINO Constraints:  Applied to {dino_stats['matched']} / {total} samples")
    print("="*50)

    print(f"\n📈 PER-CLASS ACCURACY REPORT:")
    class_correct = list(0. for i in range(30))
    class_total = list(0. for i in range(30))

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    
    for i in range(30):
        indices = np.where(all_labels_np == i)[0]
        class_total[i] = len(indices)
        if class_total[i] > 0:
            class_correct[i] = np.sum(all_preds_np[indices] == all_labels_np[indices])
            acc = 100 * class_correct[i] / class_total[i]
            print(f" - Class {i:02d}: {acc:6.2f}%  ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f" - Class {i:02d}:    N/A%  (0/0)")
            
    print("="*50)
    
    final_acc = accuracy_score(all_labels, all_preds)
    print(f"🏆 FINAL ENSEMBLE ACCURACY: {final_acc*100:.2f}%")

if __name__ == '__main__':
    main()