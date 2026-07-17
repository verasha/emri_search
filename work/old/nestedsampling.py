print('Importing libraries...')
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum 


from few.utils.ylm import GetYlms

from few import get_file_manager

from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform

from few.utils.geodesic import get_fundamental_frequencies

from few.utils.constants import YRSID_SI

import os
import sys

# Change to the desired directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

# Add it to Python path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglike
import modeselector
import dynesty
# import gc
# import pickle
import cupy as cp

print('Done importing libraries.')
print('Tuning FEW configuration...')
# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration 
use_gpu = True
force_backend = "cuda12x"  
dt = 10     # Time step
T = 0.25     # Total time


print('Checking available backends...')
for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]: 
    print(f" - Backend '{backend}': {"available" if few.has_backend(backend) else "unavailable"}")  


print('Initializing waveform generator...')
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": "cuda12x" # Force GPU
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    "pad_output": True,
}

print("Creating GenerateEMRIWaveform class...")
# Kerr eccentric flux
waveform_gen = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, 
    frame='detector',
    inspiral_kwargs=inspiral_kwargs, 
    amplitude_kwargs=amplitude_kwargs, 
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu
)

print('Done initializing waveform generator.')
print('Setting up GW functions and parameters...')
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

#Generating data (true)

m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4
xI0 = 1.0 #NOTE: fixed, equatorial
dist = 3 
qS = 0.5 
phiS = 1
qK = 1 #NOTE: fixed, degenerate
phiK = 1.5 #NOTE: fixed, degenerate
# Phases
Phi_phi0 = 0.4
Phi_theta0 = 0.0 # NOTE: fixed, equatorial
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

loglike_obj = loglike.LogLike(params_star, waveform_gen, gwf, M_init=30, verbose=False, distance_threshold=1.5)
print('Done setting up GW functions and parameters.')
print('Starting nested sampling...')
## PRIOR: masses log-uniform
def prior_transform(utheta):
      um1, um2, ua, up0, ue0, uxI0, udist, uqS, uphiS, uqK, uphiK, uPhi_phi0, uPhi_theta0, uPhi_r0 = utheta

      # 3sigma ranges (exc e0)
      m1_range = (9.9999999987e+05, 1.0000000001e+06)
      m2_range = (2.9999921753e+01, 3.0000078247e+01)
      a_range = (6.9989929349e-01, 7.0010070651e-01)
      p0_range = (7.4994459434e+00, 7.5005540566e+00)
      e0_range = (3.9999126130e-01, 4.0000873870e-01)
      dist_range = (2.9566301838e+00, 3.0433698162e+00)
      qS_range = (4.7685409100e-01, 5.2314590900e-01)
      phiS_range = (9.7607316244e-01, 1.0239268376e+00)
      Phi_phi0_range = (3.6664072881e-01, 4.3335927119e-01)
      Phi_r0_range = (4.8936765646e-01, 5.1063234354e-01)

      # Log-uniform for masses
      m1 = 10**(np.log10(m1_range[0]) + um1 * (np.log10(m1_range[1]) - np.log10(m1_range[0])))
      m2 = 10**(np.log10(m2_range[0]) + um2 * (np.log10(m2_range[1]) - np.log10(m2_range[0])))

      # Uniform for other parameters
      a = (a_range[1] - a_range[0]) * ua + a_range[0]
      p0 = (p0_range[1] - p0_range[0]) * up0 + p0_range[0]
      e0 = (e0_range[1] - e0_range[0]) * ue0 + e0_range[0]
      xI0 = 1.0  # Fixed value
      dist = (dist_range[1] - dist_range[0]) * udist + dist_range[0]
      qS = (qS_range[1] - qS_range[0]) * uqS + qS_range[0]
      phiS = (phiS_range[1] - phiS_range[0]) * uphiS + phiS_range[0]
      qK = 1.0  # Fixed value
      phiK = 1.5  # Fixed value
      Phi_phi0 = (Phi_phi0_range[1] - Phi_phi0_range[0]) * uPhi_phi0 + Phi_phi0_range[0]
      Phi_theta0 = 0.0  # Fixed value
      Phi_r0 = (Phi_r0_range[1] - Phi_r0_range[0]) * uPhi_r0 + Phi_r0_range[0]

      return m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0

rstate = np.random.default_rng(7)
with dynesty.pool.Pool(16, loglike_obj, prior_transform) as pool:
    dsampler = dynesty.DynamicNestedSampler(
        loglike_obj,  
        prior_transform,
        ndim=14,
        bound='multi',
        sample='rwalk',
        rstate=rstate
    )
    dsampler.run_nested()
    
#TODO: fix error when trying to have checkpoint file => cannot pickle 'module' object

# NOTE: Save end results in pickle as a workaround
results = dsampler.results

import pickle
# Save results using pickle
with open('nestedsampling_results.pkl', 'wb') as f:
    pickle.dump(results, f)