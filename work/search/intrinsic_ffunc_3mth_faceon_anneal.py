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

import GWfuncs
import loglike_timemax  # TIME-MAXIMIZED VERSION
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
gwf = GWfuncs.GravWaveAnalysis(T, dt)

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
loglike_obj = loglike_timemax.LogLikeTimeMax(
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

sampler_instance = None  # injected before run_sampling

anneal_state = {
    'S': 1.0,
    'phase': 'normal',           # 'normal' | 'waiting' | 'annealing'
    'phase_start_iter': None,    # iter when current phase began
    'merge_max_ld': None,        # max_ld at the moment n_proc=1 (LHS baseline)
    'ref_max_ld': None,          # raw max_ld at start of current interval
    'ref_iter': None,            # iter when ref_max_ld was recorded
    'stuck_count': 0,            # consecutive intervals with no max_ld improvement
}


def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0] 
    log_likes = np.zeros(n_samples)


    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2

        try:
            loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])) * anneal_state['S']
        except Exception:
            loglike = -np.inf
        log_likes[i] = loglike

    return log_likes

def prior_transform(u):
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8,1.3]
    alim = [0.3, 0.99]
    p0lim = [8.0, 11.0]
    e0lim = [0.2, 0.5]

    transformed = np.zeros_like(u)

    # Uniform in log for masses

    # m1
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]

    # m2
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]

    # Linear in others

    # a
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]

    # p0
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]

    # e0
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]

    return transformed

def inverse_prior_transform(params):
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99]
    p0lim = [8.0, 11.0]
    e0lim = [0.2, 0.5]

    params = np.asarray(params)
    u = np.zeros_like(params)

    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])

    return u



print('Done setting up log-likelihood and prior.')
print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
    alpha=int(100),                    # Use recent samples for weighting. 
    trail_size=int(1e3),          # Maximum trials per iteration
    boundary_limiting=True,        # Enable boundary constraints
    use_beta=False,                # Use beta correction for boundaries
    gamma=500,                    # Covariance update frequency NOTE: changed from 100
    exclude_scale_z=np.inf,       # No exclusion based on weights
    use_pool=False,               # Set to True for multiprocessing
    keep_dead_processes=True
)

# Reverse annealing parameters
anneal_wait_iter = 1000  # iters to wait at each stage
S_max = 10               # max scaling factor
anneal_alpha = 1000      # alpha when annealing starts
anneal_stuck_max = 5     # force S++ after this many consecutive stuck intervals

print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')

# Change to the search directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')


ndim = 5
n_seed = 10

sigma = 1e-5
init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]

print('Done setting up initial covariance matrix.')

print('Initializing sampler...')
sampler = parismc.Sampler(
    ndim=ndim, 
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Done initializing sampler.')

print('Loading external LHS samples from pkl...')
import pickle as _pkl
with open('/scratch/e1498138/lhs/lhs_snr32_checkpoints/lhs_snr32_final.pkl', 'rb') as _f:
    _phys_pts, _logden = _pkl.load(_f)
external_lhs_points          = inverse_prior_transform(_phys_pts)
external_lhs_log_densities   = _logden
print(f'Loaded {len(_logden)} LHS samples.')

print('Running sampling...')

def anneal_callback(sampler, i):
    global anneal_state
    state = anneal_state
    S = state['S']
    n_proc = sampler.n_proc

    if state['phase'] == 'normal':
        if n_proc == 1:
            state['merge_max_ld'] = sampler.max_logden_list[0]
            print(f"[Anneal] n_proc=1 reached at iter {i}. Baseline max_ld={state['merge_max_ld']:.5f}. Waiting {anneal_wait_iter} iters before annealing.", flush=True)
            state['phase'] = 'waiting'
            state['phase_start_iter'] = i
            sampler.config.alpha = anneal_alpha

    elif state['phase'] == 'waiting':
        # Fixed wait: always wait anneal_wait_iter iters after merge before starting
        if i - state['phase_start_iter'] >= anneal_wait_iter:
            state['S'] = 5.0
            state['phase'] = 'annealing'
            state['ref_max_ld'] = sampler.max_logden_list[0]
            state['ref_iter'] = i
            print(f"[Anneal] Starting annealing at iter {i}: S=5.0 (max_ld={state['ref_max_ld']:.5f})", flush=True)

    elif state['phase'] == 'annealing':
        if i - state['ref_iter'] >= anneal_wait_iter:
            current_max_ld = sampler.max_logden_list[0]
            ref = state['ref_max_ld']
            # Increment S if sampler found something better, or force if stuck too long
            if current_max_ld > ref and S < S_max:
                new_S = S + 1.0
                state['S'] = new_S
                state['stuck_count'] = 0
                print(f"[Anneal] max_ld improved ({ref:.5f} -> {current_max_ld:.5f}). S: {S:.0f} -> {new_S:.0f} at iter {i}", flush=True)
            elif state['stuck_count'] >= anneal_stuck_max and S < S_max:
                new_S = S + 1.0
                state['S'] = new_S
                state['stuck_count'] = 0
                print(f"[Anneal] Stuck for {anneal_stuck_max} intervals ({ref:.5f} -> {current_max_ld:.5f}). Forcing S: {S:.0f} -> {new_S:.0f} at iter {i}", flush=True)
            else:
                state['stuck_count'] += 1
                print(f"[Anneal] max_ld not improved ({ref:.5f} -> {current_max_ld:.5f}). Staying S={S:.0f} (stuck={state['stuck_count']}/{anneal_stuck_max})", flush=True)

            # Update reference for next interval
            state['ref_max_ld'] = sampler.max_logden_list[0]
            state['ref_iter'] = i


def combined_callback(sampler, i):
    anneal_callback(sampler, i)
    if i % 1000 == 0 and i > 0:
        sampler.save_state()


sampler_instance = sampler

sampler.run_sampling(
    num_iterations=int(1e5),
    savepath='./intrinsic_ffunc_3mth_snr32_anneal5',
    print_iter=100,
    callback=combined_callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done running sampling.')
