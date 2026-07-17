print("Starting imports...")
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

print("Importing few...")
import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum 


from few.utils.ylm import GetYlms

from few import get_file_manager

from few.waveform import FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

import os
import sys

print("Importing GWfuncs...")
import localgit.FEWNEW.work.GWfuncs_backup as GWfuncs_backup
print("Importing cupy...")
import cupy as cp
import gc

print("Configuring few...")
# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

print("Importing dynesty...")
import dynesty

# GPU configuration and missing variables
use_gpu = True
dist = 1.0  # Distance in Gpc => TODO: do make this a parameter
dt = 10     # Time step
T = 1.0     # Total time

# Check GPU availability
if not cuda.is_available():
    print("Warning: CUDA not available, falling back to CPU")
    use_gpu = False

print("Setting up waveform generator...")
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
        "use_gpu" : use_gpu,
        "force_backend": "cuda12x"  # Force GPU
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": "cuda12x" # Force GPU
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    "pad_output": True,
}

print("Creating FastKerrEccentricEquatorialFlux...")

# Clear GPU memory first
if use_gpu:
    print("Clearing GPU memory...")
    try:
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    except:
        pass

# Add timeout protection
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Waveform generator initialization timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout

try:
    # Kerr eccentric flux
    waveform_gen = FastKerrEccentricEquatorialFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
    )
    signal.alarm(0)  # Cancel timeout
except TimeoutError:
    print("ERROR: Waveform generator timed out - try CPU version")
    print("Setting use_gpu=False and retrying...")
    use_gpu = False
    inspiral_kwargs["use_gpu"] = False
    waveform_gen = FastKerrEccentricEquatorialFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
    )

print("Waveform generator created successfully!")
print("Memory test passed - stopping here to check memory usage.")
print("If you see this message, the issue is NOT in the waveform generator initialization.")