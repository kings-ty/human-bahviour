#!/usr/bin/env python3
"""
HRI30 Pose Extraction Pipeline
==============================

A robust data preprocessing pipeline for the HRI30 industrial action recognition dataset.
Extracts human pose keypoints using YOLOv8-pose and prepares data for LSTM/RNN training.

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from ultralytics import YOLO
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRI30PoseExtractor:
    """
    Human pose extraction pipeline for HRI30 dataset using YOLOv8-pose.
    
    Features:
    - Extracts 17 keypoints (x, y, confidence) per frame
    - Handles missing detections with interpolation/padding
    - Normalizes keypoints for scale invariance
    - Processes videos in batch mode
    """
    
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        """
        Initialize the pose extractor.
        
        Args:
            model_path: Path to YOLOv8-pose model weights
        """
        # COCO pose keypoints (17 points)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Load YOLOv8-pose model
        logger.info(f"Loading YOLOv8-pose model: {model_path}")
        self.model = YOLO(model_path)
        
        # Configuration
        self.confidence_threshold = 0.5
        self.min_keypoint_confidence = 0.3
        
        # Frame dimensions (HRI30 specific)
        self.target_width = 720
        self.target_height = 480
        
    def extract_keypoints(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract pose keypoints from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (keypoints_array, metadata)
            - keypoints_array: Shape (n_frames, 51) - 17 keypoints * 3 (x, y, conf)
            - metadata: Dict with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {os.path.basename(video_path)}")
        logger.info(f"Properties: {frame_count} frames, {fps:.2f} FPS, {width}x{height}")
        
        keypoints_sequence = []
        previous_keypoints = None
        
        try:
            for frame_idx in tqdm(range(frame_count), desc="Extracting keypoints"):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame to target resolution if needed
                if (width, height) != (self.target_width, self.target_height):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                # Extract keypoints for current frame
                keypoints = self._extract_frame_keypoints(frame)
                
                # Handle missing detections
                if keypoints is None:
                    if previous_keypoints is not None:
                        # Use previous frame's keypoints
                        keypoints = previous_keypoints.copy()
                        logger.debug(f"Frame {frame_idx}: Using previous keypoints")
                    else:
                        # Pad with zeros if no previous keypoints available
                        keypoints = np.zeros(51)  # 17 keypoints * 3
                        logger.debug(f"Frame {frame_idx}: Padding with zeros")
                else:
                    previous_keypoints = keypoints
                
                keypoints_sequence.append(keypoints)
                
        finally:
            cap.release()
        
        # Convert to numpy array
        keypoints_array = np.array(keypoints_sequence)
        
        # Metadata
        metadata = {
            'video_path': video_path,
            'original_fps': fps,
            'frame_count': len(keypoints_sequence),
            'original_resolution': (width, height),
            'target_resolution': (self.target_width, self.target_height)
        }
        
        return keypoints_array, metadata
    
    def _extract_frame_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract keypoints from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Flattened keypoints array [x1,y1,c1, x2,y2,c2, ...] or None if no detection
        """
        # Run YOLOv8-pose inference
        results = self.model(frame, verbose=False)
        
        if not results or len(results) == 0:
            return None
            
        # Get the first result
        result = results[0]
        
        # Check if keypoints are detected
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None
            
        # Get bounding boxes and keypoints
        boxes = result.boxes
        keypoints = result.keypoints.data
        
        if len(keypoints) == 0:
            return None
        
        # Filter by confidence if boxes are available
        if boxes is not None and len(boxes.conf) > 0:
            # Get the detection with highest confidence
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            if boxes.conf[best_idx] < self.confidence_threshold:
                return None
            best_keypoints = keypoints[best_idx]
        else:
            # Use first detection if no confidence scores
            best_keypoints = keypoints[0]
        
        # Convert to numpy and flatten
        kpts = best_keypoints.cpu().numpy()  # Shape: (17, 3)
        
        # Normalize keypoints
        normalized_kpts = self._normalize_keypoints(kpts, frame.shape)
        
        # Flatten: [x1, y1, c1, x2, y2, c2, ...]
        return normalized_kpts.flatten()
    
    def _normalize_keypoints(self, keypoints: np.ndarray, frame_shape: Tuple) -> np.ndarray:
        """
        Normalize keypoints for scale and position invariance.
        
        Args:
            keypoints: Raw keypoints array (17, 3)
            frame_shape: (height, width, channels)
            
        Returns:
            Normalized keypoints array (17, 3)
        """
        height, width = frame_shape[:2]
        normalized = keypoints.copy()
        
        # Extract coordinates and confidence
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        confidences = keypoints[:, 2]
        
        # Only normalize points with sufficient confidence
        valid_mask = confidences > self.min_keypoint_confidence
        
        if not np.any(valid_mask):
            # No valid keypoints, return zeros
            return np.zeros_like(keypoints)
        
        # Method 1: Normalize to image dimensions
        normalized[:, 0] = x_coords / width   # X coordinates to [0, 1]
        normalized[:, 1] = y_coords / height  # Y coordinates to [0, 1]
        
        # Method 2: Center normalization using hip center
        # Get hip center (average of left_hip and right_hip)
        left_hip_idx, right_hip_idx = 11, 12
        
        if (confidences[left_hip_idx] > self.min_keypoint_confidence and 
            confidences[right_hip_idx] > self.min_keypoint_confidence):
            
            hip_center_x = (x_coords[left_hip_idx] + x_coords[right_hip_idx]) / 2
            hip_center_y = (y_coords[left_hip_idx] + y_coords[right_hip_idx]) / 2
            
            # Shift coordinates to make hip center the origin
            normalized[valid_mask, 0] = (x_coords[valid_mask] - hip_center_x) / width
            normalized[valid_mask, 1] = (y_coords[valid_mask] - hip_center_y) / height
        
        # Method 3: Scale normalization using bounding box
        if np.any(valid_mask):
            valid_x = x_coords[valid_mask]
            valid_y = y_coords[valid_mask]
            
            if len(valid_x) > 0:
                # Calculate bounding box of valid keypoints
                bbox_width = np.max(valid_x) - np.min(valid_x)
                bbox_height = np.max(valid_y) - np.min(valid_y)
                
                # Avoid division by zero
                if bbox_width > 0 and bbox_height > 0:
                    bbox_scale = max(bbox_width, bbox_height)
                    
                    # Scale relative to bounding box size
                    normalized[valid_mask, 0] = (x_coords[valid_mask] - np.mean(valid_x)) / bbox_scale
                    normalized[valid_mask, 1] = (y_coords[valid_mask] - np.mean(valid_y)) / bbox_scale
        
        # Keep confidence scores unchanged
        normalized[:, 2] = confidences
        
        # Set invalid keypoints to zero
        normalized[~valid_mask] = 0.0
        
        return normalized
    
    def process_video_batch(self, 
                          video_paths: List[str], 
                          labels: Optional[List[str]] = None,
                          output_dir: str = "output") -> str:
        """
        Process multiple videos and save the dataset.
        
        Args:
            video_paths: List of paths to video files
            labels: Optional list of action labels for each video
            output_dir: Directory to save output files
            
        Returns:
            Path to the saved dataset file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_sequences = []
        all_labels = []
        all_metadata = []
        
        logger.info(f"Processing {len(video_paths)} videos...")
        
        for i, video_path in enumerate(tqdm(video_paths, desc="Processing videos")):
            try:
                # Extract keypoints
                keypoints, metadata = self.extract_keypoints(video_path)
                
                # Store data
                all_sequences.append(keypoints)
                all_metadata.append(metadata)
                
                # Add label if provided
                if labels and i < len(labels):
                    all_labels.extend([labels[i]] * len(keypoints))
                else:
                    # Extract label from filename if possible
                    video_name = os.path.basename(video_path)
                    label = self._extract_label_from_filename(video_name)
                    all_labels.extend([label] * len(keypoints))
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                continue
        
        # Create dataset
        return self._save_dataset(all_sequences, all_labels, all_metadata, output_dir)
    
    def _extract_label_from_filename(self, filename: str) -> str:
        """
        Extract action label from HRI30 filename format.
        
        Args:
            filename: Video filename
            
        Returns:
            Extracted action label
        """
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        
        # HRI30 format: CID##_SID##_VID##
        # For now, return the full base name as label
        # This can be customized based on your label mapping
        return base_name
    
    def _save_dataset(self, 
                     sequences: List[np.ndarray], 
                     labels: List[str], 
                     metadata: List[Dict],
                     output_dir: str) -> str:
        """
        Save the processed dataset to files.
        
        Args:
            sequences: List of keypoint sequences
            labels: List of labels
            metadata: List of metadata dicts
            output_dir: Output directory
            
        Returns:
            Path to saved dataset file
        """
        # Concatenate all sequences
        all_frames = np.vstack(sequences)
        
        # Create feature names
        feature_names = []
        for kpt_name in self.keypoint_names:
            feature_names.extend([f"{kpt_name}_x", f"{kpt_name}_y", f"{kpt_name}_conf"])
        
        # Create DataFrame
        df = pd.DataFrame(all_frames, columns=feature_names)
        df['label'] = labels
        
        # Add sequence information
        sequence_ids = []
        for i, seq in enumerate(sequences):
            sequence_ids.extend([i] * len(seq))
        df['sequence_id'] = sequence_ids
        
        # Save CSV
        csv_path = os.path.join(output_dir, "hri30_pose_dataset.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV dataset: {csv_path}")
        
        # Save NumPy arrays
        np_path = os.path.join(output_dir, "hri30_pose_dataset.npz")
        np.savez_compressed(
            np_path,
            features=all_frames,
            labels=np.array(labels),
            sequence_ids=np.array(sequence_ids),
            feature_names=np.array(feature_names)
        )
        logger.info(f"Saved NumPy dataset: {np_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'total_frames': len(all_frames),
                'total_sequences': len(sequences),
                'feature_dim': all_frames.shape[1],
                'keypoint_names': self.keypoint_names,
                'video_metadata': metadata
            }, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Print dataset summary
        logger.info("="*50)
        logger.info("DATASET SUMMARY")
        logger.info("="*50)
        logger.info(f"Total frames: {len(all_frames):,}")
        logger.info(f"Total sequences: {len(sequences)}")
        logger.info(f"Feature dimensions: {all_frames.shape[1]}")
        logger.info(f"Unique labels: {len(set(labels))}")
        logger.info(f"Average frames per sequence: {len(all_frames)/len(sequences):.1f}")
        
        return csv_path


def process_hri30_dataset(data_root: str, 
                         annotations_file: Optional[str] = None,
                         output_dir: str = "pose_dataset"):
    """
    Process the entire HRI30 dataset.
    
    Args:
        data_root: Root directory containing train_set and test_set folders
        annotations_file: Optional CSV file with labels
        output_dir: Output directory for processed dataset
    """
    extractor = HRI30PoseExtractor()
    
    # Find all video files
    video_extensions = ['.avi', '.mp4', '.mov']
    video_paths = []
    
    for split in ['train_set', 'test_set']:
        split_dir = os.path.join(data_root, split)
        if os.path.exists(split_dir):
            for ext in video_extensions:
                pattern = f"*{ext}"
                videos = list(Path(split_dir).glob(pattern))
                video_paths.extend([str(v) for v in videos])
    
    logger.info(f"Found {len(video_paths)} video files")
    
    # Load labels if annotations file provided
    labels = None
    if annotations_file and os.path.exists(annotations_file):
        df_labels = pd.read_csv(annotations_file, header=None, names=['video_id', 'action', 'split'])
        label_dict = {row['video_id']: row['action'] for _, row in df_labels.iterrows()}
        
        labels = []
        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            labels.append(label_dict.get(video_id, 'unknown'))
    
    # Process videos
    extractor.process_video_batch(video_paths, labels, output_dir)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="HRI30 Pose Extraction Pipeline")
    parser.add_argument("--data_root", type=str, default="/home/ty/human-bahviour",
                       help="Root directory containing HRI30 dataset")
    parser.add_argument("--annotations", type=str, 
                       default="/home/ty/human-bahviour/annotations/train_set_labels.csv",
                       help="Path to annotations CSV file")
    parser.add_argument("--output_dir", type=str, default="pose_dataset",
                       help="Output directory for processed dataset")
    parser.add_argument("--single_video", type=str, default=None,
                       help="Process single video file for testing")
    
    args = parser.parse_args()
    
    if args.single_video:
        # Test with single video
        extractor = HRI30PoseExtractor()
        keypoints, metadata = extractor.extract_keypoints(args.single_video)
        print(f"Extracted keypoints shape: {keypoints.shape}")
        print(f"Metadata: {metadata}")
        
        # Save test output
        os.makedirs("test_output", exist_ok=True)
        np.save("test_output/test_keypoints.npy", keypoints)
        with open("test_output/test_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Test output saved to test_output/")
        
    else:
        # Process full dataset
        process_hri30_dataset(
            data_root=args.data_root,
            annotations_file=args.annotations,
            output_dir=args.output_dir
        )