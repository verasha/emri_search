print("Starting minimal test...")
import numpy as np
from numba import cuda
import cupy as cp

print("CUDA available:", cuda.is_available())
print("GPU count:", cuda.device_count())

try:
    print("Testing CuPy...")
    x = cp.array([1, 2, 3])
    print("CuPy works:", x)
    
    print("Testing basic few import...")
    import few
    print("Few imported successfully")
    
    print("All basic tests passed!")
    
except Exception as e:
    print(f"Error: {e}")