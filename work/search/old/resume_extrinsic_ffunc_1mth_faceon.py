"""
Resume a saved PARIS sampler for the extrinsic 1-month face-on search.

Usage:
  python resume_extrinsic_ffunc_1mth_faceon.py
"""

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

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

from few.utils.constants import YRSID_SI
from smt.sampling_methods import LHS


import os
import sys

# Changing directory to FEWNEW/work
# to import stuffs
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglike_timemax  # TIME-MAXIMIZED VERSION
import parismc
import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration
use_gpu = True
force_backend = "cuda12x"
dt = 10     # Time step
T = 1/12     # Total time
print(f"Using dt = {dt} seconds, T = {T} years")

print('Initializing waveform generator...')
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}

amplitude_kwargs = {
    "force_backend": force_backend
}

Ylm_kwargs = {
    "force_backend": force_backend,
}

sum_kwargs_comb = {
    "force_backend": force_backend,
    "pad_output": True,
}

sum_kwargs_sep = {
    "force_backend": force_backend,
    "pad_output": True,
    "separate_modes": True,
}

print("Creating GenerateEMRIWaveform class...")
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_comb,
    use_gpu=use_gpu
)

waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_sep,
    use_gpu=use_gpu
)

print('Done initializing waveform generator.')

print("Creating GravWaveAnalysis class...")
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")

# Source parameters — must match original search
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 11.7
e0 = 0.4
xI0 = 1.0
dist = 0.9  # Gpc
qS = np.pi/2
phiS = 0.
qK = np.pi - qS
phiK = np.pi + phiS
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

# n-indexed mode selection parameters
n_vals = np.arange(-1, 6)  # n from -1 to 5
ell = 2  # quadrupole only

loglike_obj = loglike_timemax.LogLikeTimeMax(
    params_star,
    waveform_gen_comb,
    gwf,
    verbose=False,
    waveform_gen_sep=waveform_gen_sep,
    ell=ell,
    n_vals=n_vals,
    M_mode=None
)

print('Done initializing loglike class.')
print('Calculating SNR...')
data = loglike_obj.signal
data_snr = gwf.rhostat(data)
print('SNR calculated:', data_snr)
print("Setting up log_density and prior functions...")


def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)

    for i in range(n_samples):
        logm1, logm2, a, p0, e0, dist, qS, phiS = params[i]
        m1 = 10**logm1
        m2 = 10**logm2
        qK = np.pi - qS
        phiK = np.pi + phiS

        try:
            loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
            log_likes[i] = 10*loglike
        except (AssertionError, Exception):
            log_likes[i] = -np.inf

    return log_likes


def prior_transform(u):
    logm1lim = [5.9, 6.2]
    logm2lim = [1.3, 1.6]
    alim = [0.4, 0.95]
    p0lim = [9.0, 14.5]
    e0lim = [0.3, 0.5]
    distlim = [0.5, 1.3]
    qSlim = [1.0, 2.0]
    phiSlim = [-0.5, 0.5]

    transformed = np.zeros_like(u)

    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    transformed[:, 5] = (distlim[1] - distlim[0]) * u[:, 5] + distlim[0]
    transformed[:, 6] = (qSlim[1] - qSlim[0]) * u[:, 6] + qSlim[0]
    transformed[:, 7] = (phiSlim[1] - phiSlim[0]) * u[:, 7] + phiSlim[0]

    return transformed


print('Done setting up log-likelihood and prior.')

# Change to the search directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

# Load saved sampler
state_path = './extrinsic_ffunc_1mth_faceon_box1/sampler_state.pkl'
print(f'Loading sampler state from: {state_path}')

if not os.path.isfile(state_path):
    print(f"Sampler state not found at: {state_path}")
    print("Please run extrinsic_ffunc_1mth_faceon.py first.")
    exit(1)

sampler = parismc.Sampler.load_state(state_path)

# Rebind the functions
try:
    sampler.log_density_func_original = log_density
    if hasattr(sampler, "prior_transform") and sampler.prior_transform is not None:
        sampler.prior_transform = prior_transform
    if getattr(sampler, "prior_transform", None) is not None:
        sampler.log_density_func = sampler.transformed_log_density_func
    else:
        sampler.log_density_func = sampler.log_density_func_original
except Exception as e:
    print(f"Warning: Could not rebind functions: {e}")

print('Done loading sampler.')
print(f"Sampler ndim: {sampler.ndim}")
print(f"Sampler n_seed: {sampler.n_seed}")
print(f"Sampler current_iter: {getattr(sampler, 'current_iter', None)}")

# Continue sampling
print('Resuming sampling...')
more_iters = int(1e5)
out_dir = './extrinsic_ffunc_1mth_faceon_box1_resumed/'

sampler.run_sampling(
    num_iterations=more_iters,
    savepath=out_dir,
    print_iter=100,
    stop_dlogZ=0.01
)
print('Done resuming sampling.')
