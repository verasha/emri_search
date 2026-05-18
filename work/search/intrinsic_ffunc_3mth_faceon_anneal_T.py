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

# Changing directory to work dir
# to import my modules
dir_work ='/nfs/home/svu/e1498138/localgit/FEWNEW/work/'

os.chdir(dir_work)
sys.path.insert(0, dir_work)

import GWfuncs
import loglike_timemax_T  # TIME-MAXIMIZED VERSION, with T as param
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
T = 2     # Total time
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
loglike_obj = loglike_timemax_T.LogLike(
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

# annealing dict
anneal_state = {
    'S':          3.0,
    'stage':      0,
    'ref_max_ld': None,
    'ref_iter':   0,
}

S_schedule  = [3.0, 10.0, 30.0, 100.0]
stuck_iters = 10000

def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)


    for i in range(n_samples):
        logm1, logm2, a, p0, e0, T_prop = params[i]
        m1 = 10**logm1
        m2 = 10**logm2

        try:
            loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0,
                                            dist, qS, phiS, qK, phiK,
                                            Phi_phi0, Phi_theta0, Phi_r0,
                                            T_prop])) * anneal_state['S']
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
    Tlim = [3/12,2]

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

    # T
    transformed[:, 5] = (Tlim[1] - Tlim[0]) * u[:, 5] + Tlim[0]


    return transformed

def inverse_prior_transform(params):
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99]
    p0lim = [8.0, 11.0]
    e0lim = [0.2, 0.5]
    Tlim = [3/12,2]

    params = np.asarray(params)
    u = np.zeros_like(params)

    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5] = (params[:, 5] - Tlim[0]) / (Tlim[1] - Tlim[0])

    return u



print('Done setting up log-likelihood and prior.')

def custom_terminate_condition(sampler, proc_idx):
    """
    Terminate a process if:
    1. Its max_logden has been stable for > 1000 iterations, AND
    2. Its best log-density lags more than 'gap' below the global best
       (meaning it is stuck in a low-likelihood region).
    """
    stable_count = sampler._proc_max_ld_stable_count[proc_idx]

    if stable_count > 1000:
        global_best = max(sampler.max_logden_list)
        proc_best   = sampler.max_logden_list[proc_idx]
        gap = 50.0  # tune: SNR^2/2 units; 50 ≈ large mismatch

        if proc_best < global_best - gap:
            print(
                f"\n[Terminator] Process {proc_idx} terminated: "
                f"stable {stable_count} iters, "
                f"proc_best={proc_best:.2f} vs global_best={global_best:.2f} "
                f"(gap={global_best - proc_best:.2f})",
                flush=True,
            )
            return True

    return False

print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
    alpha=int(1e3),                    # Use recent samples for weighting.  # NOTE: changed so can forget cov from prev stage
    trail_size=int(1e3),          # Maximum trials per iteration
    boundary_limiting=True,        # Enable boundary constraints
    use_beta=True,                # Use beta correction for boundaries
    integral_num=int(1e5),        # MC samples for beta estimation
    gamma=500,                    # Covariance update frequency NOTE: changed from 100
    exclude_scale_z=np.inf,       # No exclusion based on weights
    use_pool=False,               # Set to True for multiprocessing
    keep_dead_processes=True,
    terminate_proc_condition=custom_terminate_condition,
    seed = 323
)

print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')

# Change to the search directory
dir_search =  os.path.join(dir_work, 'search')
os.chdir(dir_search)
sys.path.insert(0, dir_search)

ndim = 6
n_seed = 10

dir_scratch='/scratch/e1498138/'

print('Loading top processes from saved sampler state...')
_state_pkl = dir_scratch + 'paris1/int_3mth_snr32_8/sampler_state.pkl'
with open(_state_pkl, 'rb') as _f:
    _prev_sampler = pickle.load(_f)

_sorted_idx = np.argsort(-np.array(_prev_sampler.max_logden_list))[:n_seed]
print(f'Top {n_seed} process max_logden: {[_prev_sampler.max_logden_list[i] for i in _sorted_idx]}')

external_lhs_points        = np.array([_prev_sampler.now_means[i] for i in _sorted_idx])
external_lhs_log_densities = np.array([_prev_sampler.max_logden_list[i] for i in _sorted_idx])
init_cov_list              = [_prev_sampler.now_covariances[i].copy() for i in _sorted_idx]
print(f'Loaded {n_seed} seed points from saved state.')

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




_stop_flag = [False]

def anneal_callback(sampler, i):
    global anneal_state
    state = anneal_state

    if state['ref_max_ld'] is None:
        state['ref_max_ld'] = sampler.max_logden_list[0]
        state['ref_iter']   = i
        return

    current_max = sampler.max_logden_list[0]
    stage       = state['stage']
    S           = S_schedule[stage]

    if current_max > state['ref_max_ld']:
        state['ref_max_ld'] = current_max
        state['ref_iter']   = i
        print(f"S={S:.0f} improved -> {current_max:.5f} at iter {i}", flush=True)
        return

    if i - state['ref_iter'] >= stuck_iters:
        if stage < len(S_schedule) - 1:
            new_S = S_schedule[stage + 1]
            state['stage']      = stage + 1
            state['S']          = new_S
            state['ref_max_ld'] = sampler.max_logden_list[0]
            state['ref_iter']   = i
            print(f"Stuck {stuck_iters} iters at S={S:.0f}. Jumping -> S={new_S:.0f} at iter {i}", flush=True)
        else:
            if not _stop_flag[0]:
                print(f"Stuck {stuck_iters} iters at S={S:.0f} (final stage). Stopping at iter {i}.", flush=True)
                _stop_flag[0] = True


def combined_callback(sampler, i):
    anneal_callback(sampler, i)
    if _stop_flag[0]:
        sampler.stop_sampling = True
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

print('Running sampling...')
savepath = dir_scratch+'paris2/int_3mth_snr32_dur_anneal2'
sampler.run_sampling(
    num_iterations=int(1e4),
    savepath=savepath,
    print_iter=100,
    callback=combined_callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
    stop_max_ld_stable_iters=int(1e4),
)
print('Done running sampling.')
