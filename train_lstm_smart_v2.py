#!/usr/bin/env python3
"""
Training Script for Skeleton Stream (Bi-LSTM).

This script trains a Bidirectional LSTM network on the 79-dimensional 
"Smart Kinematic Physics Features" extracted from the skeleton sequences.

Architecture:
    Input: (Batch, Sequence_Length, 79)
    Layer 1: Linear Projection + LayerNorm + ReLU
    Layer 2: Bi-Directional LSTM (2 Layers, Hidden=128)
    Layer 3: Self-Attention Mechanism (Weighted Sum of Temporal States)
    Output: 30 Action Classes

Key Components:
    - Bi-LSTM: Captures temporal dependencies in both forward/backward directions.
    - Attention: Allows the model to focus on critical frames (e.g., peak of a jump).
    - AdamW Optimizer: Weight decay for regularization.
    - ReduceLROnPlateau: Dynamic learning rate adjustment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import random
import os

# =============================================================================
# 🏛️ Neural Network Architecture
# =============================================================================

class Attention(nn.Module):
    """
    Self-Attention Mechanism.
    Computes a weighted sum of LSTM outputs to focus on important time steps.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        
        # Calculate attention scores for each time step
        weights = self.attention(lstm_output) # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1) # Normalize to sum to 1
        
        # Weighted sum (Context Vector)
        context_vector = torch.sum(weights * lstm_output, dim=1) # (batch, hidden_size)
        return context_vector, weights

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 1. Input Projection
        # Projects 79-dim raw features into a higher-dimensional latent space
        # LayerNorm stabilizes training for recurrent networks
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Bi-Directional LSTM
        # Processes sequence in both directions to capture context
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 3. Attention Mechanism
        # Input size is hidden_size * 2 because of Bi-Directional output
        self.attention = Attention(hidden_size * 2) 
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, 79)
        
        x = self.projection(x)
        
        # LSTM Forward Pass
        self.lstm.flatten_parameters() # Optimize memory layout for GPU
        lstm_out, _ = self.lstm(x) # Output: (batch, seq_len, hidden_size * 2)
        
        # Apply Attention to aggregate temporal information
        context, attn_weights = self.attention(lstm_out) # Output: (batch, hidden_size * 2)
        
        # Final Classification
        out = self.classifier(context)
        
        return out

# =============================================================================
# Dataset & Training Logic
# =============================================================================

class SmartFeatureDatasetV3(Dataset):
    def __init__(self, features, labels):
        # features: (N, 60, 79) numpy array
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Gradient Clipping (Prevents exploding gradients, crucial for LSTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    return total_loss / len(loader), 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Train Bi-LSTM on Smart Features")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Model Hyperparameters
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    parser.add_argument('--fold', type=int, default=0, help='Fold index for Cross-Validation (0-4)')
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Skeleton Stream Training (Bi-LSTM) | Device: {device}")
    
    # Data Paths
    data_dir = Path('pose_features_smart_v3')
    model_save_dir = Path('data')
    model_save_dir.mkdir(exist_ok=True) # Ensure 'data/' directory exists for saving weights

    try:
        features = np.load(data_dir / 'train_features.npy') # (N, 60, 79)
        labels = np.load(data_dir / 'train_labels.npy')
    except FileNotFoundError:
        print("❌ Error: 'train_features.npy' not found in 'pose_features_smart_v3/'.")
        print("   Please run 'python smart_features_v3.py' first.")
        return
    
    # Feature Normalization (Z-Score)
    # Critical for Neural Networks convergence
    print("   Normalizing features (Z-Score)...")
    mean = np.mean(features, axis=(0, 1))
    std = np.std(features, axis=(0, 1)) + 1e-6 # Avoid division by zero
    features = (features - mean) / std
    
    # Save normalization stats for inference time
    np.save(data_dir / 'feature_mean.npy', mean)
    np.save(data_dir / 'feature_std.npy', std)
    
    # K-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        if fold != args.fold: continue # Only run the specified fold
        
        print(f"🔹 Starting FOLD {fold+1}/5")
        
        train_ds = SmartFeatureDatasetV3(features[train_idx], labels[train_idx])
        val_ds = SmartFeatureDatasetV3(features[val_idx], labels[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Initialize Model
        model = LSTMModel(
            input_dim=79, 
            hidden_size=args.hidden_size, 
            num_classes=30, 
            num_layers=args.layers,
            dropout=args.dropout
        ).to(device)
        
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Optimizer: AdamW handles weight decay better than Adam
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        best_acc = 0.0
        
        # Training Loop
        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            
            # Step scheduler based on validation accuracy
            scheduler.step(v_acc)
            
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.state_dict(), model_save_dir / f'best_lstm_v3_fold{fold}.pth')
                
            if (epoch+1) % 5 == 0:
                print(f"   Ep {epoch+1:3d}: Train={t_acc:.1f}% | Val={v_acc:.1f}% (Best: {best_acc:.1f}%) | Loss: {t_loss:.3f}")
                
        print(f"✅ Fold {fold+1} Finished. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
