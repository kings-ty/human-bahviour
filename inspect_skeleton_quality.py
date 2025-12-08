import numpy as np
from pathlib import Path

def inspect_data():
    path = Path('/home/ty/human-bahviour/pose_features_large/test_sequences.npy')
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"🔍 Inspecting: {path}")
    data = np.load(path) # (N, T, V, C) or (N, T, V, C, M)
    
    # Shape check
    print(f"   Shape: {data.shape}")
    
    # Assume shape is (N, T, V, C) where C=2 (x,y) or C=3 (x,y,score)
    # If C=2, we check (0,0) count.
    
    # 1. Missing Joints (0,0) Analysis
    # Checking for absolute zero (0.0) which usually indicates missing detection
    zero_count = np.sum(np.all(np.abs(data[..., :2]) < 1e-5, axis=-1))
    total_points = data.shape[0] * data.shape[1] * data.shape[2]
    missing_ratio = (zero_count / total_points) * 100
    
    print(f"\n📊 Quality Report")
    print(f"   - Total Keypoints: {total_points:,}")
    print(f"   - Missing (0,0): {zero_count:,} ({missing_ratio:.2f}%)")
    
    if missing_ratio < 5.0:
        print("   ✅ Quality: EXCELLENT (Less than 5% missing)")
    elif missing_ratio < 15.0:
        print("   ⚠️ Quality: MODERATE (5-15% missing, might need interpolation)")
    else:
        print("   ❌ Quality: POOR (High missing rate, YOLO struggled)")

    # 2. Coordinate Range Check
    # NTU data is usually normalized, or raw pixels?
    # HRI30 is likely raw pixels.
    x_max = np.max(data[..., 0])
    y_max = np.max(data[..., 1])
    print(f"\n📏 Coordinate Range")
    print(f"   - X max: {x_max:.2f}")
    print(f"   - Y max: {y_max:.2f}")
    
    if x_max > 2.0:
        print("   ℹ️  Data seems to be RAW PIXELS (not normalized).")
        print("      -> That's why we use /3.0 scaling. Correct.")
    else:
        print("   ℹ️  Data seems to be NORMALIZED (-1~1).")
        print("      -> If so, /3.0 scaling might be making it TOO SMALL.")

if __name__ == "__main__":
    inspect_data()
