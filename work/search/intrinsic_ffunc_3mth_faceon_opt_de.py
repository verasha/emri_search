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

import GWfuncs_pure
import loglike_pure
# import modeselectoralt
import parismc
# import gc
import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration 
use_gpu = True
force_backend = "cuda12x"  
dt = 10     # Time step
T = 3/12     # Total time




print(f"Using dt = {dt} seconds, T = {T} years")



print('Initializing waveform generator...')
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": force_backend # Force GPU
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": force_backend,  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs_comb = {
    "force_backend": force_backend,  # Force GPU
    "pad_output": True,
}

sum_kwargs_sep = {
    "force_backend": force_backend,  # Force GPU
    "pad_output": True,
    "separate_modes": True,
}

print("Creating GenerateEMRIWaveform class...")
# Kerr eccentric flux
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, 
    frame='detector',
    inspiral_kwargs=inspiral_kwargs, 
    amplitude_kwargs=amplitude_kwargs, 
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_comb,
    use_gpu=use_gpu
)

# Kerr eccentric flux
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
gwf = GWfuncs_pure.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")


# Source parameters
m1 = 1e6
m2 = 1e1
a = 0.7
p0 = 9
e0 = 0.4
xI0 = 1.0
dist = 1.8  # Gpc
qS = np.pi
phiS = 0.
qK =  0.
phiK = 0.
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

# n-indexed mode selection parameters
n_vals = np.arange(-1,6)  # n from -1 to 5
ell = 2  # quadrupole only

# NOTE: change verbose argument for debugging
# Using n-indexed mode selection
loglike_obj = loglike_pure.LogLikePure(
    params_star,
    waveform_gen_comb,
    gwf,
    verbose=False,
    waveform_gen_sep=waveform_gen_sep,
    ell=ell,
    n_vals=n_vals,
    M_mode=None  # No SNR filtering, use all n-groups
)

print('Done initializing loglike class.')
print('Calculating SNR...')
data = loglike_obj.signal
data_snr = gwf.rhostat(data)
print('SNR calculated:', data_snr)
print("Setting up log_density and prior functions...")
# make this from log_density([param_true])
_TARGET_LOGLIKE = 5.90716261


def log_density(params):
    global _PARIS_EARLY_STOP_HIT
    params = np.asarray(params)

    def eval_one(x):
        global _PARIS_EARLY_STOP_HIT
        if _PARIS_EARLY_STOP_HIT:
            return float('-inf')
        try:
            logm1, logm2, a, p0, e0 = x
            m1 = 10**logm1
            m2 = 10**logm2

            fstat = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))            
        except Exception:
            return float('-inf')
        
        if fstat >= _TARGET_LOGLIKE:
            _PARIS_EARLY_STOP_HIT = True
            try:
                print(f"[EARLY-STOP] SNR {fstat:.6f} >= {_TARGET_LOGLIKE}; future calls => -inf")
            except Exception:
                pass

        return fstat
        
    if params.ndim == 1:
        return eval_one(params)
    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = eval_one(params[i])
    return out


_PARIS_EARLY_STOP_HIT = False

# ─────────────────────────────────────────────
# Load anneal12 → compute DE bounds from posterior
# (same approach as paris2_Sdel_new.ipynb + ellipse_prior_study.py)
# ─────────────────────────────────────────────
_p2_lo = np.array([5.6, 0.8, 0.3,  8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])
S_ANNEAL = 30   # anneal12 final annealing factor

def prior_transform(u):
    out = np.zeros_like(u)
    widths = _p2_hi - _p2_lo
    for d in range(5):
        out[:, d] = widths[d] * u[:, d] + _p2_lo[d]
    return out

print('Loading anneal12 sampler...')
_sampler = parismc.Sampler.load_state(
    './search/intrinsic_ffunc_3mth_snr32_anneal12/sampler_state.pkl'
)

# Best-fit (maxld) point in physical space
_pts_u   = _sampler.searched_points_list[0]       # unit-cube
_logdens = _sampler.searched_log_densities_list[0]
mu_center = prior_transform(_pts_u[np.argmax(_logdens)].reshape(1, -1))[0]
print(f'anneal12 maxld/S = {np.max(_logdens)/S_ANNEAL:.4f}')
print(f'mu_center: {mu_center}')

# Importance-weighted covariance (same as ellipse_prior_study.py)
_samples, _weights = _sampler.get_samples_with_weights(flatten=True)
_weights = _weights / _weights.sum()
_rng = np.random.default_rng(0)
_idx = _rng.choice(len(_samples), size=50_000, replace=True, p=_weights)
_post = _samples[_idx]
cov_posterior = np.cov(_post.T)   # annealed (no de-anneal)
sigma_diag    = np.sqrt(np.diag(cov_posterior))
print(f'Posterior 1-sigma (annealed): {sigma_diag}')
del _sampler, _samples, _weights, _idx, _post

# DE bounds: mu_center ± N_SIGMA * sigma_diag, clipped to flat prior
N_SIGMA_DE = 2.0
half   = N_SIGMA_DE * sigma_diag
lo     = np.clip(mu_center - half, _p2_lo, _p2_hi)
hi     = np.clip(mu_center + half, _p2_lo, _p2_hi)
bounds = [(lo[d], hi[d]) for d in range(5)]

labels = ['logm1', 'logm2', 'a', 'p0', 'e0']
print(f'\nDE bounds (N_sigma={N_SIGMA_DE}):')
for d, lab in enumerate(labels):
    print(f'  {lab}: [{lo[d]:.5f}, {hi[d]:.5f}]  (mu={mu_center[d]:.5f}, true={param_true[d]:.5f})')

# ─────────────────────────────────────────────
# Differential evolution
# ─────────────────────────────────────────────

def neg_logden(x):
    val = log_density(np.array([x]))[0]
    return -val if np.isfinite(val) else 1e10

def _stop_callback(xk, convergence):
    return _PARIS_EARLY_STOP_HIT

from scipy.optimize import differential_evolution

result_new = differential_evolution(
    neg_logden, bounds=bounds,
    maxiter=2000, tol=1e-8, seed=42, disp=True,
    popsize=20, mutation=(0.5, 1.0), recombination=0.7,
    callback=_stop_callback, polish=False,
)

print(f'\n=== DE Result ===')
print(f'Peak   : {result_new.x}')
print(f'loglike: {-result_new.fun:.8f}')
print(f'Target : {_TARGET_LOGLIKE:.8f}')
print(f'True   : {param_true}')
print(f'Diff   : {result_new.x - np.array(param_true)}')