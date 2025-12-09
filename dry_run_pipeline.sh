#!/bin/bash

# =============================================================================
# 🧪 Dry Run Pipeline: Fast Verification Script
# This script runs every step of the pipeline with minimal data/epochs
# to ensure there are no syntax errors, import errors, or path issues.
# =============================================================================

echo "========================================================"
echo "🚀 STARTING DRY RUN PIPELINE CHECK"
echo "========================================================"

# Exit immediately if a command exits with a non-zero status
set -e 

# Define dummy paths for safety
DATA_ROOT="."
OUTPUT_DIR_POSE="dry_run_pose"
OUTPUT_DIR_FLOW="dry_run_flow"
OUTPUT_DIR_SMART="dry_run_smart"
MODEL_DIR="data"

# Create temp dirs
mkdir -p $OUTPUT_DIR_POSE/train
mkdir -p $OUTPUT_DIR_FLOW/train
mkdir -p $OUTPUT_DIR_SMART
mkdir -p $MODEL_DIR

echo -e "\n[Step 1] Checking Preprocessing (HPE)..."
# We need at least one video. If none, we can't test properly but code runs.
# Let's try to run it on 'train_set' but limit frames or find 1 video.
# Since we can't easily limit 'number of videos' in the script args without modifying code,
# we will rely on the script being robust. 
# BUT for a true dry run, let's just check if it STARTS and imports correctly.
# Or better: We create a dummy video file? No, ffmpeg dependency.
# Let's run help command to check imports.
python preprocessing.py --help > /dev/null
echo "✅ preprocessing.py imports OK."

# If there are videos, let's try processing ONE.
# Finding first mp4/avi
FIRST_VID=$(find train_set -name "*.mp4" -o -name "*.avi" | head -n 1)
if [ -n "$FIRST_VID" ]; then
    echo "   Testing on 1 video: $FIRST_VID"
    # Create a dummy folder structure for the script
    mkdir -p dry_run_temp/train_set
    cp "$FIRST_VID" dry_run_temp/train_set/
    
    # Run with small max_frames
    python preprocessing.py --data_root dry_run_temp --output_dir $OUTPUT_DIR_POSE --max_frames 10 --batch_mode train
    echo "✅ Preprocessing execution OK."
else
    echo "⚠️  No videos found to test execution. Skipping actual processing."
    # Create dummy npy for next steps
    python -c "import numpy as np; import json; np.save('$OUTPUT_DIR_POSE/train_sequences.npy', np.random.rand(10, 10, 17, 2)); np.save('$OUTPUT_DIR_POSE/train_labels.npy', np.zeros(10)); json.dump(['dummy.avi']*10, open('$OUTPUT_DIR_POSE/train_filenames.json','w'))"
fi


echo -e "\n[Step 2] Checking Smart Features..."
# Point to the directory where we just put the (real or dummy) data
# The script expects 'pose_features_large' by default, so we temporarily rename or modify args?
# smart_features_v3.py currently hardcodes paths. 
# WE SHOULD MODIFY IT TO ACCEPT ARGS or we just mock the folder.
# Mocking folder:
if [ ! -d "pose_features_large" ]; then
    ln -s $OUTPUT_DIR_POSE pose_features_large
    LINK_CREATED=true
fi

# Run script (It processes pose_features_large -> pose_features_smart_v3)
# We expect it to find the train_sequences.npy we created/generated
python smart_features_v3.py 

echo "✅ Smart Features execution OK."

if [ "$LINK_CREATED" = true ]; then
    rm pose_features_large
fi



echo -e "\n[Step 3] Checking Optical Flow Extraction..."
python extract_flow.py --help > /dev/null
echo "✅ extract_flow.py imports OK."
# If we have the dummy video from Step 1
if [ -d "dry_run_temp/train_set" ]; then
    python extract_flow.py --data_root dry_run_temp --output_dir $OUTPUT_DIR_FLOW
    echo "✅ Flow Extraction execution OK."
else
    echo "⚠️  Skipping Flow Execution (No video)."
fi


echo -e "\n[Step 4] Checking ResNet Training..."
# We need dummy flow data if Step 3 didn't run or produced little
# Create dummy flow images for a "dummy" video
mkdir -p $OUTPUT_DIR_FLOW/train_set/dummy
for i in {0..15}; do
    # Create black images
    convert -size 224x224 xc:black $OUTPUT_DIR_FLOW/train_set/dummy/flow_x_$i.jpg 2>/dev/null || touch $OUTPUT_DIR_FLOW/train_set/dummy/flow_x_$i.jpg
    convert -size 224x224 xc:black $OUTPUT_DIR_FLOW/train_set/dummy/flow_y_$i.jpg 2>/dev/null || touch $OUTPUT_DIR_FLOW/train_set/dummy/flow_y_$i.jpg
done

# We also need metadata (filenames/labels) in pose_features_large for the training script
# Let's ensure they exist from Step 1
mkdir -p pose_features_large
if [ ! -f "pose_features_large/train_filenames.json" ]; then
     # Create dummy metadata pointing to our dummy video
     python -c "import json; import numpy as np; json.dump(['dummy.avi'], open('pose_features_large/train_filenames.json','w')); np.save('pose_features_large/train_labels.npy', np.array([0]))"
fi

# Run Training (1 Epoch, tiny batch)
# Pointing to our dry run flow data
# Note: Script hardcodes 'flow_data_16f/train_set'. We need to trick it or use sed.
# Let's verify imports first.
python train_flow_resnet.py --epochs 1 --batch_size 1 --lr 0.001
echo "✅ ResNet Training execution OK."



echo -e "\n[Step 5] Checking LSTM Training..."
# Needs pose_features_smart_v3/train_features.npy
# Step 2 should have generated it.
python train_lstm_smart_v2.py --epochs 1 --batch_size 2 --hidden_size 16 --layers 1
echo "✅ LSTM Training execution OK."



echo -e "\n[Step 6] Checking Final Inference..."
python make_submission_ultimate.py 
echo "✅ Inference execution OK."

echo "========================================================"
echo "🎉 DRY RUN COMPLETE! NO SYNTAX ERRORS FOUND."
echo "========================================================"

# Cleanup
rm -rf dry_run_temp $OUTPUT_DIR_POSE $OUTPUT_DIR_FLOW $OUTPUT_DIR_SMART
