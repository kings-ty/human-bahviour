#!/usr/bin/env python3
"""
LSTM Training Script for Skeleton-based Action Recognition
Trains an LSTM model on pre-extracted pose features from YOLOv8-Pose

Model Architecture:
- Input: (Batch, Sequence_Length, Input_Dim)
  * Input_Dim = 34 (17 keypoints × 2 coordinates) or 51 (including confidence)
- 2-layer Bidirectional LSTM with Dropout
- Fully connected classifier for 30 action classes
- Optimized for Jetson Xavier AGX with CUDA support
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


# ========================================================================================
# Dataset Class
# ========================================================================================

class PoseSequenceDataset(Dataset):
    """
    Custom Dataset for loading pre-extracted pose sequences.

    Expected data format:
    - sequences: (N, T, 17, 2) where N=samples, T=sequence_length, 17=keypoints, 2=x,y
    - labels: (N,) integer class labels
    """

    def __init__(self, sequences_path: str, labels_path: str,
                 use_confidence: bool = False):
        """
        Args:
            sequences_path: Path to .npy file containing sequences
            labels_path: Path to .npy file containing labels
            use_confidence: If True, include confidence scores (not implemented yet)
        """
        self.sequences = np.load(sequences_path)  # (N, T, 17, 2)
        self.labels = np.load(labels_path)  # (N,)

        # Reshape sequences to (N, T, 34) - flatten keypoints
        N, T, num_keypoints, coords = self.sequences.shape
        self.sequences = self.sequences.reshape(N, T, num_keypoints * coords)

        print(f"Loaded dataset: {sequences_path}")
        print(f"  Sequences shape: {self.sequences.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Num classes: {len(np.unique(self.labels))}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence: (T, 34) tensor
            label: scalar tensor
        """
        sequence = torch.from_numpy(self.sequences[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return sequence, label


# ========================================================================================
# LSTM Model Architecture
# ========================================================================================

class ActionRecognitionLSTM(nn.Module):
    """
    Bidirectional LSTM for skeleton-based action recognition.

    Architecture:
    1. 2-layer Bidirectional LSTM (captures temporal dependencies)
    2. Dropout for regularization
    3. Fully connected layer for classification

    Why LSTM for pose sequences:
    - LSTMs can model temporal dependencies in sequential data
    - Bidirectional processing captures both past and future context
    - Suitable for variable-length sequences (with padding)
    - Efficient for skeleton-based data (low-dimensional input)
    """

    def __init__(self, input_dim: int = 34, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 30,
                 dropout: float = 0.5, bidirectional: bool = True):
        """
        Args:
            input_dim: Input feature dimension (34 for 17 keypoints × 2 coords)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            num_classes: Number of action classes (30 for HRI30)
            dropout: Dropout probability for regularization
            bidirectional: Use bidirectional LSTM
        """
        super(ActionRecognitionLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected classifier
        # If bidirectional, output dim is 2 * hidden_dim
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) input sequences

        Returns:
            (batch, num_classes) logits
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from both directions
        if self.bidirectional:
            # Concatenate last hidden states from forward and backward
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            h_forward = h_n[-2, :, :]  # Last layer forward
            h_backward = h_n[-1, :, :]  # Last layer backward
            h_last = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim * 2)
        else:
            h_last = h_n[-1, :, :]  # (batch, hidden_dim)

        # Apply dropout
        h_last = self.dropout(h_last)

        # Classification
        logits = self.fc(h_last)  # (batch, num_classes)

        return logits


# ========================================================================================
# Training Functions
# ========================================================================================

def train_one_epoch(model: nn.Module, dataloader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer,
                   device: str, epoch: int) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: LSTM model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for sequences, labels in pbar:
        # Move to device
        sequences = sequences.to(device)  # (batch, seq_len, input_dim)
        labels = labels.to(device)  # (batch,)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)  # (batch, num_classes)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model: nn.Module, dataloader: DataLoader,
            criterion: nn.Module, device: str, epoch: int) -> Tuple[float, float]:
    """
    Validate model.

    Args:
        model: LSTM model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

        for sequences, labels in pbar:
            # Move to device
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)

            # Compute loss
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ========================================================================================
# Main Training Loop
# ========================================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM for Skeleton-based Action Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/home/ty/human-behaviour/pose_features',
                       help='Directory containing processed .npy files')
    parser.add_argument('--train_sequences', type=str, default='train_sequences.npy',
                       help='Filename for training sequences')
    parser.add_argument('--train_labels', type=str, default='train_labels.npy',
                       help='Filename for training labels')
    parser.add_argument('--test_sequences', type=str, default='test_sequences.npy',
                       help='Filename for test sequences')
    parser.add_argument('--test_labels', type=str, default='test_labels.npy',
                       help='Filename for test labels')

    # Model arguments
    parser.add_argument('--input_dim', type=int, default=34,
                       help='Input dimension (34 for 17 keypoints × 2)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--num_classes', type=int, default=30,
                       help='Number of action classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler_step', type=int, default=20,
                       help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help='Learning rate decay factor')

    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default='hri30_lstm',
                       help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='/home/ty/human-behaviour/experiments_lstm',
                       help='Directory to save experiments')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval')

    args = parser.parse_args()

    print("="*80)
    print("LSTM Training for Skeleton-based Action Recognition")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*80)

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    device = torch.device(args.device)

    # Create experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.save_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # Save configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Setup TensorBoard
    writer = SummaryWriter(os.path.join(experiment_dir, 'tensorboard'))

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PoseSequenceDataset(
        sequences_path=os.path.join(args.data_dir, args.train_sequences),
        labels_path=os.path.join(args.data_dir, args.train_labels)
    )

    test_dataset = PoseSequenceDataset(
        sequences_path=os.path.join(args.data_dir, args.test_sequences),
        labels_path=os.path.join(args.data_dir, args.test_labels)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = ActionRecognitionLSTM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )

    # Move model to device
    model = model.to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma
    )

    # Training loop
    print("\nStarting training...")
    best_val_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch

            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'config': vars(args)
            }, best_model_path)

            print(f"  New best model saved! (Accuracy: {val_acc:.2f}%)")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

    # Training completed
    writer.close()

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"{'='*80}")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"Model saved to: {experiment_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
