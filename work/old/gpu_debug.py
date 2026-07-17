#!/usr/bin/env python3

print("=== GPU Debug Test ===")

import sys
import os

print("1. Testing basic CUDA availability...")
try:
    from numba import cuda
    print(f"✓ CUDA available: {cuda.is_available()}")
    if cuda.is_available():
        print(f"✓ CUDA devices: {len(cuda.gpus)}")
        for i, gpu in enumerate(cuda.gpus):
            print(f"  GPU {i}: {gpu.name}")
except Exception as e:
    print(f"✗ CUDA test failed: {e}")
    sys.exit(1)

print("\n2. Testing CuPy...")
try:
    import cupy as cp
    print("✓ CuPy imported successfully")
    
    # Test basic GPU memory allocation
    print("Testing GPU memory allocation...")
    x = cp.array([1, 2, 3])
    print(f"✓ Basic CuPy array created: {x}")
    
    # Check GPU memory
    mempool = cp.get_default_memory_pool()
    print(f"✓ GPU memory used: {mempool.used_bytes()} bytes")
    print(f"✓ GPU memory total: {mempool.total_bytes()} bytes")
    
except Exception as e:
    print(f"✗ CuPy test failed: {e}")
    sys.exit(1)

print("\n3. Testing FEW trajectory only (no waveform)...")
try:
    import few
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode import KerrEccEqFlux
    
    print("✓ FEW trajectory imports successful")
    
    # Test trajectory creation without full waveform
    print("Creating trajectory generator...")
    traj = EMRIInspiral(func="KerrEccEqFlux", use_gpu=True)
    print("✓ Trajectory generator created successfully")
    
except Exception as e:
    print(f"✗ FEW trajectory test failed: {e}")
    sys.exit(1)

print("\n4. Testing individual FEW components...")
try:
    from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
    from few.summation.interpolatedmodesum import InterpolatedModeSum
    from few.utils.ylm import GetYlms
    
    print("✓ All FEW component imports successful")
    
    # Test amplitude interpolator
    print("Creating amplitude interpolator...")
    amp_gen = AmpInterpKerrEccEq(use_gpu=True)
    print("✓ Amplitude generator created")
    
    # Test Ylm generator
    print("Creating Ylm generator...")
    ylm_gen = GetYlms(use_gpu=True)
    print("✓ Ylm generator created")
    
    # Test summation 
    print("Creating summation generator...")
    sum_gen = InterpolatedModeSum(use_gpu=True)
    print("✓ Summation generator created")
    
except Exception as e:
    print(f"✗ FEW components test failed: {e}")
    print(f"Error details: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All tests passed! ===")
print("The issue is likely in the FastKerrEccentricEquatorialFlux initialization")
print("Try running with CPU only: use_gpu=False")