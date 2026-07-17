print("Testing FEW components step by step...")
import numpy as np
from numba import cuda
import cupy as cp
import few

print("Step 1: Basic imports...")
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
print("✓ Trajectory imports OK")

from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
print("✓ Amplitude imports OK")

from few.summation.interpolatedmodesum import InterpolatedModeSum
print("✓ Summation imports OK")

from few.utils.ylm import GetYlms
print("✓ YLM imports OK")

print("Step 2: Testing individual component creation...")

try:
    print("Creating trajectory generator...")
    traj = EMRIInspiral(func=KerrEccEqFlux, force_backend="cuda12x", use_gpu=True)
    print("✓ Trajectory generator OK")
except Exception as e:
    print(f"✗ Trajectory generator failed: {e}")

try:
    print("Creating amplitude interpolator...")
    amp = AmpInterpKerrEccEq(force_backend="cuda12x")
    print("✓ Amplitude interpolator OK")
except Exception as e:
    print(f"✗ Amplitude interpolator failed: {e}")

try:
    print("Creating mode sum...")
    interpolate_mode_sum = InterpolatedModeSum(force_backend="cuda12x")
    print("✓ Mode sum OK")
except Exception as e:
    print(f"✗ Mode sum failed: {e}")

try:
    print("Creating YLM generator...")
    ylm_gen = GetYlms(include_minus_m=False, force_backend="cuda12x")
    print("✓ YLM generator OK")
except Exception as e:
    print(f"✗ YLM generator failed: {e}")

print("All individual components tested!")