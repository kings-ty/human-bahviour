# HRI30 Pose Extraction Pipeline

A robust data preprocessing pipeline for the HRI30 industrial action recognition dataset that extracts human pose keypoints using YOLOv8-pose for LSTM/RNN training.

## Features

- 🎯 **YOLOv8-pose Integration**: Uses state-of-the-art pose detection
- 🔄 **Robust Processing**: Handles missing detections with interpolation
- 📐 **Smart Normalization**: Multiple normalization strategies for scale invariance
- 📊 **Batch Processing**: Efficient processing of entire video datasets
- 💾 **Multiple Output Formats**: CSV, NumPy, and JSON metadata
- 🏭 **HRI30 Optimized**: Specifically designed for industrial action recognition

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download YOLOv8-pose model** (happens automatically on first run):
```bash
# The script will automatically download yolov8n-pose.pt on first use
```

## Quick Start

### 1. Test with Single Video
```bash
cd pose_extraction
python test_single_video.py
```

### 2. Process Full Dataset
```bash
cd pose_extraction
python run_full_extraction.py
```

### 3. Custom Usage
```python
from hri30_pose_extractor import HRI30PoseExtractor

# Initialize extractor
extractor = HRI30PoseExtractor()

# Extract keypoints from single video
keypoints, metadata = extractor.extract_keypoints("path/to/video.avi")
print(f"Extracted shape: {keypoints.shape}")  # (n_frames, 51)

# Process multiple videos
video_paths = ["video1.avi", "video2.avi"]
labels = ["action1", "action2"]
dataset_path = extractor.process_video_batch(video_paths, labels)
```

## Pipeline Overview

### 1. **Keypoint Extraction**
- Detects human poses using YOLOv8-pose
- Extracts 17 COCO keypoints per frame
- Each keypoint has (x, y, confidence) values

### 2. **Robust Detection Handling**
- **Missing Person**: Uses previous frame or zero-padding
- **Low Confidence**: Filters unreliable detections
- **Multiple People**: Selects highest confidence detection

### 3. **Normalization Strategies**

#### Method 1: Image Normalization
```python
# Normalize to image dimensions [0, 1]
x_norm = x_coord / image_width
y_norm = y_coord / image_height
```

#### Method 2: Hip-Centered Normalization
```python
# Center coordinates around hip center
hip_center = (left_hip + right_hip) / 2
x_centered = (x_coord - hip_center_x) / image_width
y_centered = (y_coord - hip_center_y) / image_height
```

#### Method 3: Bounding Box Normalization
```python
# Scale relative to person's bounding box
bbox_center = (min_coord + max_coord) / 2
bbox_scale = max(bbox_width, bbox_height)
x_scaled = (x_coord - bbox_center_x) / bbox_scale
y_scaled = (y_coord - bbox_center_y) / bbox_scale
```

### 4. **Data Structure**

#### Keypoint Order (17 points):
1. nose, 2. left_eye, 3. right_eye, 4. left_ear, 5. right_ear
6. left_shoulder, 7. right_shoulder, 8. left_elbow, 9. right_elbow
10. left_wrist, 11. right_wrist, 12. left_hip, 13. right_hip
14. left_knee, 15. right_knee, 16. left_ankle, 17. right_ankle

#### Feature Vector per Frame:
```
[x1, y1, c1, x2, y2, c2, ..., x17, y17, c17]  # 51 dimensions
```

#### Output Formats:

**CSV Format:**
```
nose_x, nose_y, nose_conf, left_eye_x, ..., label, sequence_id
0.45,   0.32,   0.89,     0.42,       ..., Walking, 0
...
```

**NumPy Format:**
```python
data = np.load('hri30_pose_dataset.npz')
features = data['features']        # (total_frames, 51)
labels = data['labels']           # (total_frames,)
sequence_ids = data['sequence_ids'] # (total_frames,)
```

## Configuration

### Model Settings
```python
class HRI30PoseExtractor:
    def __init__(self):
        self.confidence_threshold = 0.5      # Person detection confidence
        self.min_keypoint_confidence = 0.3   # Keypoint confidence
        self.target_width = 720             # HRI30 target width
        self.target_height = 480            # HRI30 target height
```

### Customization
```python
# Use different YOLOv8 model
extractor = HRI30PoseExtractor("yolov8s-pose.pt")  # Better accuracy
extractor = HRI30PoseExtractor("yolov8x-pose.pt")  # Best accuracy

# Adjust confidence thresholds
extractor.confidence_threshold = 0.7
extractor.min_keypoint_confidence = 0.5
```

## File Structure

```
pose_extraction/
├── hri30_pose_extractor.py    # Main pipeline class
├── test_single_video.py       # Single video test script
├── run_full_extraction.py     # Full dataset processing
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── pose_dataset/              # Output directory
    ├── hri30_pose_dataset.csv     # CSV format dataset
    ├── hri30_pose_dataset.npz     # NumPy format dataset
    └── dataset_metadata.json     # Processing metadata
```

## Expected Input/Output

### Input
- **Videos**: `.avi`, `.mp4`, `.mov` files
- **Resolution**: Any (automatically resized to 720x480)
- **Format**: HRI30 dataset structure with `train_set/` and `test_set/` folders
- **Labels**: Optional CSV file with video_id, action_label, split

### Output
- **Features**: Shape `(total_frames, 51)` - 17 keypoints × 3 coordinates
- **Labels**: Action class for each frame
- **Metadata**: Video information, processing statistics

## Performance Notes

- **Speed**: ~30 seconds per video (depends on length and hardware)
- **Memory**: Processes videos sequentially to avoid memory issues
- **Storage**: ~1KB per frame in compressed format
- **Accuracy**: Depends on video quality and YOLOv8 model size

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

2. **No pose detected**:
- Check video quality
- Adjust `confidence_threshold`
- Verify person is visible in frame

3. **Import errors**:
```bash
pip install ultralytics opencv-python
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with LSTM/RNN

### TensorFlow/Keras Example
```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = np.load('pose_dataset/hri30_pose_dataset.npz')
X = data['features']  # (total_frames, 51)
y = data['labels']    # (total_frames,)

# Reshape for sequence modeling
# Group by sequence_ids to create sequences
sequences = []
labels = []

for seq_id in np.unique(data['sequence_ids']):
    mask = data['sequence_ids'] == seq_id
    sequences.append(X[mask])
    labels.append(y[mask][0])  # Assuming one label per sequence

# Pad sequences to same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_padded = pad_sequences(sequences, dtype='float32')

# Build model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(None, 51)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{hri30_pose_extraction,
    title={HRI30 Pose Extraction Pipeline for Industrial Action Recognition},
    author={AI Assistant},
    year={2024}
}
```