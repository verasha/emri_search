"""
Resume a saved paris2_T sampler and continue sampling with T-annealing.

Usage:
  python resume_intrinsic_ffunc_3mth_faceon_paris2_T.py
"""

import numpy as np
import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.constants import YRSID_SI

import os
import sys

dir_work = '/nfs/home/svu/e1498138/localgit/FEWNEW/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

import GWfuncs
import loglike_timemax_T
import parismc
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 2     # Full data duration
print(f"Using dt = {dt} seconds, T = {T} years")

print('Initializing waveform generator...')
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs       = {"force_backend": force_backend}
sum_kwargs_comb  = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep   = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

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
gwf = GWfuncs.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")

m1 = 1e6
m2 = 1e1
a  = 0.7
p0 = 9
e0 = 0.4
xI0 = 1.0
dist = 1.8
qS = np.pi
phiS = 0.
qK  = 0.
phiK = 0.
Phi_phi0   = 0.4
Phi_theta0 = 0.0
Phi_r0     = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

n_vals = np.arange(-1, 6)
ell = 2

loglike_obj = loglike_timemax_T.LogLike(
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
data_snr = gwf.rhostat(loglike_obj.signal)
print('SNR:', data_snr)

print("Setting up log_density and prior functions...")

# NOTE: Set stage/T to match where the run was when it was saved.
# stage 0 -> T=3/12, stage 1 -> T=6/12, stage 2 -> T=1.0, stage 3 -> T=1.5, stage 4 -> T=2.0
anneal_state = {
    'T':          3/12,
    'stage':      0,
    'ref_max_ld': None,
    'ref_iter':   0,
}

T_schedule  = [3/12, 6/12, 1.0, 1.5, 2.0]
stuck_iters = int(1e5)

def log_density(params):
    params = np.asarray(params)
    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)

    T_current = anneal_state['T']

    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2
        try:
            loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0,
                                            dist, qS, phiS, qK, phiK,
                                            Phi_phi0, Phi_theta0, Phi_r0,
                                            T_current]))
        except Exception:
            loglike = -np.inf
        log_likes[i] = loglike

    return log_likes

def prior_transform(u):
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8, 1.3]
    alim     = [0.3, 0.99]
    p0lim    = [8.0, 11.0]
    e0lim    = [0.2, 0.5]

    transformed = np.zeros_like(u)
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    transformed[:, 2] = (alim[1]     - alim[0])     * u[:, 2] + alim[0]
    transformed[:, 3] = (p0lim[1]    - p0lim[0])    * u[:, 3] + p0lim[0]
    transformed[:, 4] = (e0lim[1]    - e0lim[0])    * u[:, 4] + e0lim[0]
    return transformed

print('Done setting up log-likelihood and prior.')

# Load saved sampler
dir_search = os.path.join(dir_work, 'search')
os.chdir(dir_search)
sys.path.insert(0, dir_search)

dir_scratch = '/scratch/e1498138/'
state_path  = dir_scratch + 'paris2/int_3mth_snr32_dur_anneal/sampler_state.pkl'

print(f'Loading sampler state from: {state_path}')
if not os.path.isfile(state_path):
    print(f"Sampler state not found at: {state_path}")
    exit(1)

sampler = parismc.Sampler.load_state(state_path)

try:
    sampler.log_density_func_original = log_density
    if getattr(sampler, "prior_transform", None) is not None:
        sampler.prior_transform = prior_transform
        sampler.log_density_func = sampler.transformed_log_density_func
    else:
        sampler.log_density_func = log_density
except Exception as e:
    print(f"Warning: could not rebind functions: {e}")

print('Done loading sampler.')
print(f"Sampler ndim: {sampler.ndim}")
print(f"Sampler current_iter: {getattr(sampler, 'current_iter', None)}")

def anneal_callback(sampler, i):
    global anneal_state
    state = anneal_state

    if state['ref_max_ld'] is None:
        state['ref_max_ld'] = sampler.max_logden_list[0]
        state['ref_iter']   = i
        return

    current_max = sampler.max_logden_list[0]
    stage       = state['stage']
    T_cur       = T_schedule[stage]

    if current_max > state['ref_max_ld']:
        state['ref_max_ld'] = current_max
        state['ref_iter']   = i
        print(f"T={T_cur:.2f}yr improved -> {current_max:.5f} at iter {i}", flush=True)
        return

    if i - state['ref_iter'] >= stuck_iters:
        if stage < len(T_schedule) - 1:
            new_T = T_schedule[stage + 1]
            state['stage']      = stage + 1
            state['T']          = new_T
            state['ref_max_ld'] = sampler.max_logden_list[0]
            state['ref_iter']   = i
            print(f"Stuck {stuck_iters} iters at T={T_cur:.2f}yr. Jumping -> T={new_T:.2f}yr at iter {i}", flush=True)
        else:
            print(f"T={T_cur:.2f}yr (final stage), still running at iter {i}.", flush=True)


def combined_callback(sampler, i):
    anneal_callback(sampler, i)
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

print('Resuming sampling...')
savepath = dir_scratch + 'paris2/int_3mth_snr32_dur_anneal_resumed'
sampler.run_sampling(
    num_iterations=int(5e5),
    savepath=savepath,
    print_iter=100,
    callback=combined_callback,
)
print('Done resuming sampling.')
