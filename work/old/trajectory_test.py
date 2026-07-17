print("Testing trajectory generator backends...")
import numpy as np
from numba import cuda
import cupy as cp
import few
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux

print("Test 1: CPU backend...")
try:
    traj_cpu = EMRIInspiral(func=KerrEccEqFlux, force_backend="cpu", use_gpu=False)
    print("✓ CPU trajectory generator OK")
except Exception as e:
    print(f"✗ CPU trajectory generator failed: {e}")

print("Test 2: CUDA12x backend...")
try:
    traj_gpu = EMRIInspiral(func=KerrEccEqFlux, force_backend="cuda12x", use_gpu=True)
    print("✓ GPU trajectory generator OK")
except Exception as e:
    print(f"✗ GPU trajectory generator failed: {e}")

print("All trajectory tests completed!")