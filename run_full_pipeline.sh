#!/bin/bash

# =============================================================================
# 🚀 FULL PIPELINE AUTOMATION SCRIPT
# Runs the complete Human Action Recognition pipeline from Raw Video to Final Submission.
#
# WARNING: This process may take several hours depending on your hardware.
# Recommended hardware: NVIDIA GPU with at least 8GB VRAM.
# =============================================================================

# Configuration
LOG_FILE="pipeline_execution.log"

# Function to log messages
log() {
    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Stop on error
set -e

log "========================================================"
log "   STARTING FULL PIPELINE EXECUTION"
log "========================================================"
log "Current Directory: $(pwd)"

# -----------------------------------------------------------------------------
# Step 1: Human Pose Estimation (HPE)
# -----------------------------------------------------------------------------
log "🦴 [Step 1/5] Extracting Poses using YOLOv8..."
# Assuming data_root is current directory. Adjust if videos are elsewhere.
python preprocessing.py \
    --data_root . \
    --output_dir pose_features_large \
    --max_frames 60 \
    --batch_mode both \
    --device cuda

log "✅ Step 1 Complete: Pose extraction finished."

# -----------------------------------------------------------------------------
# Step 2: Feature Engineering
# -----------------------------------------------------------------------------
log "🧠 [Step 2/5] Generating Smart Physics Features..."
python smart_features_v3.py 

log "✅ Step 2 Complete: Smart features generated."

# -----------------------------------------------------------------------------
# Step 3: Motion Stream (Optical Flow + ResNet)
# -----------------------------------------------------------------------------
log "🌊 [Step 3/5] Processing Motion Stream..."

# 3-1. Flow Extraction
log "   -> Extracting Dense Optical Flow (16 frames/video)..."
# This is time consuming.
python extract_flow.py \
    --data_root . \
    --output_dir flow_data_16f 

# 3-2. Train Flow Model
log "   -> Training Flow ResNet-18 (30 Epochs)..."
python train_flow_resnet.py \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001

log "✅ Step 3 Complete: Motion model trained."

# -----------------------------------------------------------------------------
# Step 4: Skeleton Stream (Bi-LSTM)
# -----------------------------------------------------------------------------
log "🤖 [Step 4/5] Training Skeleton Bi-LSTM (100 Epochs)..."
# Using 'Imperial' Config
python train_lstm_smart_v2.py \
    --epochs 100 \
    --batch_size 32 \
    --hidden_size 128 \
    --layers 2 \
    --dropout 0.3 \
    --fold 0

log "✅ Step 4 Complete: LSTM model trained."

# -----------------------------------------------------------------------------
# Step 5: Final Ensemble & Evaluation
# -----------------------------------------------------------------------------
log "🏆 [Step 5/5] Running Final Ensemble Evaluation..."
python make_submission_ultimate.py

log "========================================================"
log "🎉 PIPELINE EXECUTION SUCCESSFUL!"
log "   - Results saved in: pipeline_execution.log"
log "   - Final models: best_flow_resnet.pth, data/best_lstm_v3_fold0.pth"
log "========================================================"
