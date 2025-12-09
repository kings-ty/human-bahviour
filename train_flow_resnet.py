#!/usr/bin/env python3
"""
Training Script for Motion Stream (ResNet-18).

This script trains a ResNet-18 backbone on the sequence of 16 optical flow frames.
It treats the sequence as a 3D volume or averages the 2D predictions (TSN style).

Architecture:
    Input: (Batch, 16, 3, 224, 224) - 16 frames of Flow (X, Y, X stacked)
    Backbone: ResNet-18 (Pretrained on ImageNet)
    Consensus: Average Pooling over the temporal dimension (16 frames)
    Output: 30 Action Classes

Hyperparameters:
    - Optimizer: SGD with Momentum (0.9)
    - Learning Rate: 0.001 (Cosine Annealing)
    - Batch Size: 16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.model_selection import StratifiedKFold
import json

# =========================================================
# Dataset (Reads Flow Images)
# =========================================================
class FlowDataset(Dataset):
    """
    Custom Dataset to load optical flow image sequences.
    Reads 'flow_x_{i}.jpg' and 'flow_y_{i}.jpg' and stacks them into a 3-channel image.
    """
    def __init__(self, data_dir, video_list, labels, transform=None, num_segments=16):
        self.data_dir = Path(data_dir)
        self.video_list = video_list
        self.labels = labels
        self.transform = transform
        self.num_segments = num_segments

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        vid_name = self.video_list[idx]
        label = self.labels[idx]
        
        # Folder containing flow images for this video
        flow_dir = self.data_dir / Path(vid_name).stem
        
        frames = []
        # Load 16 frames (or less if missing)
        for i in range(self.num_segments):
            fx_path = flow_dir / f"flow_x_{i:02d}.jpg"
            fy_path = flow_dir / f"flow_y_{i:02d}.jpg"
            
            # Check for alternative naming if leading zeros missing
            if not fx_path.exists(): fx_path = flow_dir / f"flow_x_{i}.jpg"
            if not fy_path.exists(): fy_path = flow_dir / f"flow_y_{i}.jpg"
            
            if fx_path.exists() and fy_path.exists():
                try:
                    img_x = cv2.imread(str(fx_path), cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(str(fy_path), cv2.IMREAD_GRAYSCALE)
                    # Stack to make (H, W, 3) for ResNet compatibility (Expects 3 channels)
                    # We use (Flow_X, Flow_Y, Flow_X) as the 3 channels
                    img = np.stack([img_x, img_y, img_x], axis=2) 
                    frames.append(img)
                except Exception:
                    # corrupted file fallback
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                # If missing, pad with zeros
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
        # Convert List of Numpy Arrays to Tensor Batch
        # Output shape: (16, 3, 224, 224)
        processed_frames = []
        for frame in frames:
            # frame is (224, 224, 3) numpy
            # Convert to Tensor (C, H, W) and Scale 0-1
            tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            if self.transform:
                tensor_frame = self.transform(tensor_frame)
                
            processed_frames.append(tensor_frame)
            
        data = torch.stack(processed_frames) 
        return data, label

# =========================================================
# Model (ResNet-18 TSN)
# =========================================================
class FlowResNet(nn.Module):
    def __init__(self, num_classes=30):
        super(FlowResNet, self).__init__()
        # Use ResNet-18 backbone (Lightweight & Efficient)
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final Fully Connected layer
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Linear(num_ftrs, num_classes)
        
        # Fine-tuning: Unfreeze all layers to learn Motion features
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Input x: (Batch, Segments, Channels, H, W)
        b, s, c, h, w = x.size()
        
        # Merge Batch and Segments -> (B*S, C, H, W)
        # This allows processing all frames in parallel through the CNN
        x = x.view(b * s, c, h, w)
        
        # CNN Forward Pass
        out = self.base(x) # (B*S, Num_Classes)
        
        # Reshape back to separate Batch and Segments -> (B, S, Num_Classes)
        out = out.view(b, s, -1)
        
        # Temporal Consensus: Average Pooling across segments
        # "TSN" Strategy: Combine predictions from all snippets
        out = torch.mean(out, dim=1) # (B, Num_Classes)
        
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (lower for Xavier)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Optical Flow Training (ResNet-18) | Device: {device}")
    
    # 1. Load Data Metadata
    # We use the filenames/labels generated by the preprocessing step
    try:
        with open('pose_features_large/train_filenames.json', 'r') as f:
            filenames = json.load(f)
        labels = np.load('pose_features_large/train_labels.npy')
    except FileNotFoundError:
        print("❌ Error: Metadata files not found in 'pose_features_large/'. Run preprocessing first.")
        return

    # 2. Define Transforms
    # ResNet expects normalized inputs
    tf = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. K-Fold Cross Validation (Using Fold 0 only for this demo)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(filenames, labels)):
        if fold != 0: break # Only train one fold
        
        print(f"🔹 Starting FOLD {fold+1}/5")
        
        # Initialize Datasets
        train_ds = FlowDataset(
            'flow_data_16f/train_set', 
            [filenames[i] for i in train_idx], 
            labels[train_idx], 
            transform=tf
        )
        val_ds = FlowDataset(
            'flow_data_16f/train_set', 
            [filenames[i] for i in val_idx], 
            labels[val_idx], 
            transform=tf
        )
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Initialize Model, Loss, Optimizer
        model = FlowResNet(num_classes=30).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_acc = 0.0
        
        # Training Loop
        for epoch in range(args.epochs):
            model.train()
            t_loss, t_corr, t_tot = 0, 0, 0
            
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                t_loss += loss.item()
                t_corr += (out.argmax(1) == y).sum().item()
                t_tot += y.size(0)
                
            # Validation
            model.eval()
            v_corr, v_tot = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    v_corr += (out.argmax(1) == y).sum().item()
                    v_tot += y.size(0)
            
            t_acc = t_corr / t_tot * 100
            v_acc = v_corr / v_tot * 100
            scheduler.step()
            
            # Save Best Model
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), 'best_flow_resnet.pth')
                print(f"   🏆 New Best Model Saved! (Acc: {best_acc:.2f}%)")
                
            print(f"   Train Loss: {t_loss/len(train_loader):.4f} | Train Acc: {t_acc:.1f}% | Val Acc: {v_acc:.1f}%")

if __name__ == '__main__':
    main()