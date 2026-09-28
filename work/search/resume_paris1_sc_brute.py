"""
Resume a saved paris1_sc_brute sampler and continue sampling.

Usage:
  python resume_paris1_sc_brute.py
"""

import numpy as np
import few
import os
import sys
import pickle

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_timemax_noise import LogLike

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12/12 #NOTE: changed!
N_SEGS = 12  # must match --n-segs used to build the brute-force LHS seed
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_SEGS}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Mojito light EMRI_G
m1 = 6.72e5
m2 = 9.84e1
a = 0.5
p0 = 15.7117
e0 = 0.7440
xI0 = 1.0
dist = 4.755
qS = 0.5906
phiS = 3.5808
qK = 1.1707
phiK = 3.8495
Phi_phi0 = 3.1940
Phi_theta0 = 3.3780     # fixed: not searched
Phi_r0 = 0.6038

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, dist, np.cos(qS), phiS, np.cos(qK), phiK, Phi_phi0, Phi_r0]

n_vals = np.arange(-1, 6)
ell = 2

print('Initializing LogLike...')
loglike_obj = LogLike(
    params=params_star,
    waveform_response=waveform_response,
    gwf=gwf,
    add_noise=True,
    seed=42,
    verbose=False,
    ell=ell,
    n_vals=n_vals,
    M_mode=None,
)
print('LogLike initialized.')


def log_density(params):
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        (logm1, logm2, a_i, p0_i, e0_i, dist_i,
         cos_qS_i, phiS_i, cos_qK_i, phiK_i, Phi_phi0_i, Phi_r0_i) = params[i]
        try:
            qS_i = np.arccos(cos_qS_i)
            qK_i = np.arccos(cos_qK_i)
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist_i, qS_i, phiS_i, qK_i, phiK_i,
                Phi_phi0_i, Phi_theta0, Phi_r0_i,
                T=T, dt=dt,
            ))
            out[i] = float(gwf.SNR_semicoherent(loglike_obj.signal, h_temp, N_seg=N_SEGS, phase_max=True))
        except Exception:
            pass
    return out


# Broad priors — same bounds as run_lhs_noise_sc_brute.py. xI0 (only 1.0 is
# valid for the waveform model) and Phi_theta0 are held fixed (not
# searched); sky/spin polar angles sampled uniform in cosine.
LOGM1_LIM = [5.6,  6.4]
LOGM2_LIM = [1.7,  2.3]
A_LIM     = [0.3,  0.999]
P0_LIM    = [13.5,  16.5]
E0_LIM    = [0.6,  0.9]
DIST_LIM  = [3,  5]
COSQS_LIM = [-1.0, 1.0]
PHIS_LIM  = [0.0,  2 * np.pi]
COSQK_LIM = [-1.0, 1.0]
PHIK_LIM  = [0.0,  2 * np.pi]
PHIPHI0_LIM = [0.0, 2 * np.pi]
PHIR0_LIM   = [0.0, 2 * np.pi]


def prior_transform(u):
    t = np.zeros_like(u)
    t[:, 0]  = (LOGM1_LIM[1] - LOGM1_LIM[0]) * u[:, 0] + LOGM1_LIM[0]
    t[:, 1]  = (LOGM2_LIM[1] - LOGM2_LIM[0]) * u[:, 1] + LOGM2_LIM[0]
    t[:, 2]  = (A_LIM[1] - A_LIM[0]) * u[:, 2] + A_LIM[0]
    t[:, 3]  = (P0_LIM[1] - P0_LIM[0]) * u[:, 3] + P0_LIM[0]
    t[:, 4]  = (E0_LIM[1] - E0_LIM[0]) * u[:, 4] + E0_LIM[0]
    t[:, 5]  = (DIST_LIM[1] - DIST_LIM[0]) * u[:, 5] + DIST_LIM[0]
    t[:, 6]  = (COSQS_LIM[1] - COSQS_LIM[0]) * u[:, 6] + COSQS_LIM[0]
    t[:, 7]  = (PHIS_LIM[1] - PHIS_LIM[0]) * u[:, 7] + PHIS_LIM[0]
    t[:, 8]  = (COSQK_LIM[1] - COSQK_LIM[0]) * u[:, 8] + COSQK_LIM[0]
    t[:, 9]  = (PHIK_LIM[1] - PHIK_LIM[0]) * u[:, 9] + PHIK_LIM[0]
    t[:, 10] = (PHIPHI0_LIM[1] - PHIPHI0_LIM[0]) * u[:, 10] + PHIPHI0_LIM[0]
    t[:, 11] = (PHIR0_LIM[1] - PHIR0_LIM[0]) * u[:, 11] + PHIR0_LIM[0]
    return t


print('Done setting up log-likelihood and prior.')

# Load saved sampler state
dir_scratch = '/scratch/e1498138'
savepath = dir_scratch + '/paper/ext/emri_g/stage1_merge_brute_smol/'
state_path = savepath + 'sampler_state.pkl'
print(f'Loading sampler state from: {state_path}')

# Write resumed run's state/samples to a separate directory so the
# original stage1_merge_brute_smol state is left untouched.
resume_savepath = dir_scratch + '/paper/ext/emri_g/stage1_merge_brute_smol_resume/'
os.makedirs(resume_savepath, exist_ok=True)

if not os.path.isfile(state_path):
    print(f"Sampler state not found at: {state_path}")
    print("Please run paris1_sc_brute.py first.")
    exit(1)

sampler = parismc.Sampler.load_state(state_path)

# Rebind functions
try:
    sampler.log_density_func_original = log_density
    if hasattr(sampler, 'prior_transform') and sampler.prior_transform is not None:
        sampler.prior_transform = prior_transform
    if getattr(sampler, 'prior_transform', None) is not None:
        sampler.log_density_func = sampler.transformed_log_density_func
    else:
        sampler.log_density_func = sampler.log_density_func_original
except Exception as e:
    print(f"Warning: Could not rebind functions: {e}")

print('Done loading sampler.')
print(f"Sampler ndim: {sampler.ndim}")
print(f"Sampler n_seed: {sampler.n_seed}")
print(f"Sampler current_iter: {getattr(sampler, 'current_iter', None)}")


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()


# Continue sampling
print('Resuming paris1_sc_brute sampling...')
out_dir = resume_savepath

sampler.run_sampling(
    num_iterations=int(5e4),
    savepath=out_dir,
    print_iter=10,
    callback=callback,
)
print('Done.')
print('Savepath:', out_dir)
