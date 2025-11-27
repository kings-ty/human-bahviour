#!/usr/bin/env python3
"""
GPU-optimized version of HRI30 pose extractor.
Processes multiple frames in batches for better GPU utilization.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GPUOptimizedPoseExtractor:
    """
    GPU-optimized pose extractor with batch processing.
    """
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", batch_size: int = 8):
        """
        Initialize GPU-optimized extractor.
        
        Args:
            model_path: Path to YOLOv8-pose model
            batch_size: Number of frames to process simultaneously
        """
        # Force GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Load model
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to(self.device)
        
        # Configuration
        self.confidence_threshold = 0.5
        self.min_keypoint_confidence = 0.3
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Batch size: {batch_size}")
    
    def extract_keypoints_batch(self, video_path: str) -> Tuple[np.ndarray, dict]:
        """
        Extract keypoints using GPU batch processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (keypoints_array, metadata)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        logger.info(f"Loaded {len(frames)} frames, processing in batches of {self.batch_size}")
        
        # Process frames in batches
        all_keypoints = []
        
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            
            # Process batch on GPU
            batch_keypoints = self._process_frame_batch(batch_frames)
            all_keypoints.extend(batch_keypoints)
        
        # Convert to array
        keypoints_array = np.array(all_keypoints)
        
        metadata = {
            'video_path': video_path,
            'original_fps': fps,
            'frame_count': len(frames),
            'original_resolution': (width, height),
            'batch_size': self.batch_size,
            'device': str(self.device)
        }
        
        return keypoints_array, metadata
    
    def _process_frame_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of frames on GPU.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of keypoint arrays
        """
        batch_keypoints = []
        
        # Convert frames to tensor batch
        if torch.cuda.is_available():
            # Process batch on GPU
            results = self.model(frames, verbose=False, device=self.device)
        else:
            # Fallback to CPU
            results = self.model(frames, verbose=False)
        
        # Extract keypoints from results
        for i, result in enumerate(results):
            frame_keypoints = self._extract_single_frame_keypoints(result, frames[i].shape)
            
            if frame_keypoints is None:
                # Use previous frame or zeros
                if batch_keypoints:
                    frame_keypoints = batch_keypoints[-1].copy()
                else:
                    frame_keypoints = np.zeros(51)
            
            batch_keypoints.append(frame_keypoints)
        
        return batch_keypoints
    
    def _extract_single_frame_keypoints(self, result, frame_shape: Tuple) -> Optional[np.ndarray]:
        """
        Extract keypoints from single frame result.
        """
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return None
        
        # Get best detection
        keypoints = result.keypoints.data
        boxes = result.boxes
        
        if boxes is not None and len(boxes.conf) > 0:
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            if boxes.conf[best_idx] < self.confidence_threshold:
                return None
            best_keypoints = keypoints[best_idx]
        else:
            best_keypoints = keypoints[0]
        
        # Normalize and flatten
        kpts = best_keypoints.cpu().numpy()  # Shape: (17, 3)
        normalized_kpts = self._normalize_keypoints(kpts, frame_shape)
        
        return normalized_kpts.flatten()
    
    def _normalize_keypoints(self, keypoints: np.ndarray, frame_shape: Tuple) -> np.ndarray:
        """Normalize keypoints for scale invariance."""
        height, width = frame_shape[:2]
        normalized = keypoints.copy()
        
        # Extract coordinates
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        confidences = keypoints[:, 2]
        
        # Only normalize points with sufficient confidence
        valid_mask = confidences > self.min_keypoint_confidence
        
        if not np.any(valid_mask):
            return np.zeros_like(keypoints)
        
        # Simple normalization to [0, 1]
        normalized[:, 0] = x_coords / width
        normalized[:, 1] = y_coords / height
        normalized[:, 2] = confidences
        
        # Set invalid keypoints to zero
        normalized[~valid_mask] = 0.0
        
        return normalized


def test_gpu_optimization():
    """Test GPU-optimized vs standard extractor."""
    import time
    import sys
    import os
    
    # Find test video
    video_path = None
    for split in ['train_set', 'test_set']:
        split_dir = f"/home/ty/human-bahviour/{split}"
        if os.path.exists(split_dir):
            videos = [f for f in os.listdir(split_dir) if f.endswith('.avi')]
            if videos:
                video_path = os.path.join(split_dir, videos[0])
                break
    
    if not video_path:
        print("❌ No test video found!")
        return
    
    print("🚀 Testing GPU Batch Optimization")
    print("=" * 50)
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        print(f"\n🔄 Testing batch size: {batch_size}")
        
        try:
            start_time = time.time()
            
            # GPU optimized extractor
            extractor = GPUOptimizedPoseExtractor(batch_size=batch_size)
            keypoints, metadata = extractor.extract_keypoints_batch(video_path)
            
            processing_time = time.time() - start_time
            fps = len(keypoints) / processing_time
            
            print(f"   ✅ Time: {processing_time:.2f}s ({fps:.1f} fps)")
            print(f"   📊 Frames: {len(keypoints)}")
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   💾 GPU Memory: {gpu_memory:.3f} GB")
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n✅ Batch optimization test completed!")


if __name__ == "__main__":
    test_gpu_optimization()