#!/usr/bin/env python3

print("Starting CPU-only debug test...")
import numpy as np

print("Importing few...")
import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum 
from few.utils.ylm import GetYlms
from few import get_file_manager
from few.waveform import FastKerrEccentricEquatorialFlux

print("Configuring few...")
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# Force CPU usage
use_gpu = False
dist = 1.0  
dt = 10     
T = 1.0     

print("Setting up CPU-only waveform generator...")

# CPU-only configuration
inspiral_kwargs={
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False, 
    "use_gpu": False,  # Force CPU
}

amplitude_kwargs = {
    "use_gpu": False,  # Force CPU
}

Ylm_kwargs = {
    "use_gpu": False,  # Force CPU
}

sum_kwargs = {
    "use_gpu": False,  # Force CPU
    "pad_output": True,
}

print("Creating FastKerrEccentricEquatorialFlux with CPU...")
try:
    waveform_gen = FastKerrEccentricEquatorialFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
    )
    print("✓ CPU waveform generator created successfully!")
    
    # Quick test
    print("Testing basic waveform generation...")
    M = 1e6  # Solar masses
    a = 0.1  # Dimensionless spin
    p0 = 12.0  # Initial separation
    e0 = 0.35  # Initial eccentricity
    x0 = 1.0  # Inclination cosine
    qK = 1e-4  # Mass ratio
    qS = 5.0  # Spin of smaller BH
    
    # Very short waveform test
    wave = waveform_gen(M, a, p0, e0, x0, qK, qS, dist, T=0.1, dt=10.0)
    print(f"✓ Generated waveform shape: {wave.shape}")
    print("✓ CPU version works! The issue is GPU-specific.")
    
except Exception as e:
    print(f"✗ Even CPU version failed: {e}")
    import traceback
    traceback.print_exc()