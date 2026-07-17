#!/usr/bin/env python3

print("=== Step-by-step Debug ===")
import sys
import time

def test_step(step_name, func):
    print(f"\n--- Testing: {step_name} ---")
    start_time = time.time()
    try:
        result = func()
        elapsed = time.time() - start_time
        print(f"✓ {step_name} SUCCESS ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ {step_name} FAILED ({elapsed:.2f}s): {e}")
        sys.exit(1)

# Step 1: Basic imports
def step1():
    import numpy as np
    from numba import cuda
    import cupy as cp
    return cuda.is_available()

cuda_available = test_step("Basic imports + CUDA check", step1)
print(f"CUDA available: {cuda_available}")

# Step 2: Few import
def step2():
    import few
    return few

few_module = test_step("Few import", step2)

# Step 3: Waveform import
def step3():
    from few.waveform import FastKerrEccentricEquatorialFlux
    return FastKerrEccentricEquatorialFlux

WaveformClass = test_step("FastKerrEccentricEquatorialFlux import", step3)

# Step 4: Try CPU-only initialization
def step4():
    inspiral_kwargs = {
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0,
        "include_minus_m": False, 
        "use_gpu": False  # CPU only first
    }
    
    waveform_gen = WaveformClass(
        inspiral_kwargs=inspiral_kwargs,
        use_gpu=False,
    )
    return waveform_gen

cpu_waveform = test_step("CPU-only waveform initialization", step4)

# Step 5: Try GPU initialization (if CUDA available)
if cuda_available:
    def step5():
        inspiral_kwargs = {
            "func": 'KerrEccEqFlux',
            "DENSE_STEPPING": 0,
            "include_minus_m": False, 
            "use_gpu": True
        }
        
        waveform_gen = WaveformClass(
            inspiral_kwargs=inspiral_kwargs,
            use_gpu=True,
        )
        return waveform_gen
    
    gpu_waveform = test_step("GPU waveform initialization (no force_backend)", step5)
    
    # Step 6: Try with force_backend
    def step6():
        inspiral_kwargs = {
            "func": 'KerrEccEqFlux',
            "DENSE_STEPPING": 0,
            "include_minus_m": False, 
            "use_gpu": True,
            "force_backend": "cuda12x"
        }
        
        amplitude_kwargs = {"force_backend": "cuda12x"}
        Ylm_kwargs = {"force_backend": "cuda12x"}
        sum_kwargs = {"force_backend": "cuda12x", "pad_output": True}
        
        waveform_gen = WaveformClass(
            inspiral_kwargs=inspiral_kwargs,
            amplitude_kwargs=amplitude_kwargs,
            Ylm_kwargs=Ylm_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=True,
        )
        return waveform_gen
    
    forced_gpu_waveform = test_step("GPU waveform with force_backend=cuda12x", step6)

print("\n=== All tests completed successfully! ===")