
import numpy as np
import sys
from pathlib import Path

def main():
    print("Checking doc_embeddings.npy stats...")
    path = Path("data/msmarco_index/doc_embeddings.npy")
    if not path.exists():
        print(f"File not found: {path}")
        return

    try:
        # Load a small slice to avoid memory crash if it's huge, but we need stats
        # Use mmap_mode to check without full load
        arr = np.load(str(path), mmap_mode='r')
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        
        # Check first 100
        slice_100 = arr[:100]
        print(f"First 100 stats:")
        print(f"  Min: {np.min(slice_100)}")
        print(f"  Max: {np.max(slice_100)}")
        print(f"  Mean: {np.mean(slice_100)}")
        print(f"  Norms: {np.linalg.norm(slice_100, axis=1)}")
        
        # Check for Infs/NaNs in first chunk
        if not np.isfinite(slice_100).all():
            print("WARNING: Non-finite values found in first 100!")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    main()
