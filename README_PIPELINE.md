# 🚀 Project Pipeline: Human Action Recognition
**(Preprocessing -> HPE -> Smart Features -> LSTM & Flow -> Ensemble)**

This guide outlines the exact sequence of scripts used to run the full pipeline, from raw video data to final action recognition.

---

## ⚡ One-Click Automation (Recommended)
You can run the entire pipeline (Extraction -> Training -> Evaluation) automatically using the provided shell script. This is the easiest way to reproduce the results.

```bash
# Make the script executable
chmod +x run_full_pipeline.sh

# Run the full pipeline (May take hours)
./run_full_pipeline.sh

# Tip: Use 'nohup' to run in background
nohup ./run_full_pipeline.sh > full_run.out 2>&1 &
tail -f full_run.out
```
*Note: The script automatically handles directory creation and file linking between steps.*

---

## 🛠️ 1. Environment Setup
Make sure you have the necessary libraries installed.
```bash
pip install torch torchvision opencv-python ultralytics scikit-learn numpy pandas tqdm matplotlib
sudo apt-get install ffmpeg  # Required for video preprocessing
```

---

## 🎞️ 2. Preprocessing (Deinterlacing)
Before processing, we fix the video codec issues (DVSD interlacing) using ffmpeg.
*(Note: This is usually done via shell script or manually on raw data)*
```bash
# Example command (concept)
ffmpeg -i raw_video.avi -vf "yadif" -c:v libx264 -crf 18 -preset fast processed_video.mp4
```

---

## 🦴 3. Step 1: Human Pose Estimation (HPE)
Extract 2D Skeleton coordinates (17 joints) using **YOLOv8 Nano**.
*   **Input**: Raw Videos
*   **Output**: `pose_features_large/train_sequences.npy` (Coordinate Data)

```bash
# Run the preprocessing/extraction script
python preprocessing.py 
# (Or preprocessing_mit_logic.py if utilizing specific MIT logic)
```

---

## 🧠 4. Step 2: Feature Engineering (Smart Features)
Convert raw coordinates into **79-dim Kinematic Physics Features** (Velocity, Angles, Distances).
*   **Input**: `train_sequences.npy`
*   **Output**: `pose_features_smart_v3/train_features.npy`

```bash
python smart_features_v3.py
```

---

## 🌊 5. Step 3: Motion Stream (Optical Flow + ResNet)
### 3-1. Generate Optical Flow Images
Extract Dense Optical Flow (Farneback) from videos.
*   **Output**: `flow_data_16f/train_set/` (Saved images)

```bash
python extract_flow.py
```

### 3-2. Train Flow Model (ResNet-18)
Train the ResNet backbone on the generated flow images.
*   **Output**: `best_flow_resnet.pth`

```bash
python train_flow_resnet.py
```

---

## 🤖 6. Step 4: Skeleton Stream (Bi-LSTM)
Train the Bidirectional LSTM model on the Smart Features.
*   **Input**: `pose_features_smart_v3/train_features.npy`
*   **Output**: `data/best_lstm_v3_fold0.pth`

```bash
python train_lstm_smart_v2.py
```

---

## 🏆 7. Step 5: Final Ensemble & Evaluation
Combine predictions from **Flow ResNet** (10%) and **Bi-LSTM** (90%) for the final result.
*   **Mode**: Validation Mode (Random 420 samples) or Full Test
*   **Script**: `make_submission_ultimate.py`

```bash
# Run validation to see accuracy per class
python make_submission_ultimate.py
```

---

## ✅ 8. Smoke Testing (Quick Pipeline Verification)
Before running the full, time-consuming pipeline, it's recommended to perform a quick "Smoke Test" to ensure all scripts can run without fatal errors (e.g., syntax issues, missing imports, path errors). This script runs all pipeline steps with minimal data and epochs.

```bash
# Make the script executable
chmod +x dry_run_pipeline.sh

# Run the smoke test
./dry_run_pipeline.sh
```

---

### 📊 Visualization Tools
Generate figures for papers/reports.
```bash
# 1. Draw Architecture Diagram (Awesome Viz)
python generate_awesome_architecture.py

# 2. Draw Skeleton with Physics Annotations
python generate_paper_figure.py
```
