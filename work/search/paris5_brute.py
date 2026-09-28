"""
ParisMC merge for the paris5 "brute" pipeline, scoring with the coherent
(pure) f-statistic (loglike_pure_noise.LogLike), same EMRI_G source, 12-dim
(logm1, logm2, a, p0, e0, dist, cosqS, phiS, cosqK, phiK, Phi_phi0, Phi_r0)
parameter space, and box prior bounds as paris5_lhs_box_brute.py
(run via run_lhs_noise.pbs).

xI0 and Phi_theta0 remain fixed (not searched), same as
paris5_lhs_box_brute.py.

Loads the LHS grid precomputed by paris5_lhs_box_brute.py
(paper/ext/emri_g/f_1e5_brute/lhs_final.pkl) -- points and coherent-f
log_densities -- and uses it to seed a ParisMC sampler that merges/refines
within that same box.
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
from loglike_pure_noise import LogLike

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12 / 12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source (Mojito light EMRI_G, matches paris5_lhs_box_brute.py)
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
Phi_theta0 = 3.3780
Phi_r0 = 0.6038

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, dist,
              np.cos(qS), phiS, np.cos(qK), phiK, Phi_phi0, Phi_r0]

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
print(f'SNR (rhostat): {float(gwf.SNR(gwf.freq_wave(loglike_obj.signal))):.4f}')


def log_density(params):
    """
    Coherent (pure) f-statistic (loglike_pure_noise.LogLike), row-by-row,
    over the 12-dim brute parameter space
    [logm1, logm2, a, p0, e0, dist, cosqS, phiS, cosqK, phiK, Phi_phi0, Phi_r0].
    xI0 and Phi_theta0 are held fixed at the injected values.
    """
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        (logm1, logm2, a_i, p0_i, e0_i, dist_i,
         cos_qS_i, phiS_i, cos_qK_i, phiK_i, Phi_phi0_i, Phi_r0_i) = params[i]
        try:
            qS_i = np.arccos(cos_qS_i)
            qK_i = np.arccos(cos_qK_i)
            theta = [
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist_i, qS_i, phiS_i, qK_i, phiK_i,
                Phi_phi0_i, Phi_theta0, Phi_r0_i,
            ]
            out[i] = float(loglike_obj(theta))
        except Exception:
            pass
    return out


def prior_transform(u):
    # matches paris5_lhs_box_brute.py box (nsigma=2 for all params except
    # e0 and phir0, which get 6sigma)
    logm1lim   = [5.82615, 5.83032]
    logm2lim   = [1.99273, 1.99370]
    alim       = [0.49650, 0.50850]
    p0lim      = [15.64820, 15.72680]
    e0lim      = [0.74301, 0.74413]
    distlim    = [3.94257, 4.75567]
    cosqSlim   = [0.62966, 1.0]
    phiSlim    = [3.15906, 4.06029]
    cosqKlim   = [0.31455, 1.0]
    phiKlim    = [1.66135, 4.53748]
    Phiphi0lim = [1.48468, 5.32255]
    Phir0lim   = [0.00000, 2 * np.pi]

    t = np.zeros_like(u)
    t[:, 0]  = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    t[:, 1]  = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    t[:, 2]  = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    t[:, 3]  = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]
    t[:, 4]  = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    t[:, 5]  = (distlim[1] - distlim[0]) * u[:, 5] + distlim[0]
    t[:, 6]  = (cosqSlim[1] - cosqSlim[0]) * u[:, 6] + cosqSlim[0]
    t[:, 7]  = (phiSlim[1] - phiSlim[0]) * u[:, 7] + phiSlim[0]
    t[:, 8]  = (cosqKlim[1] - cosqKlim[0]) * u[:, 8] + cosqKlim[0]
    t[:, 9]  = (phiKlim[1] - phiKlim[0]) * u[:, 9] + phiKlim[0]
    t[:, 10] = (Phiphi0lim[1] - Phiphi0lim[0]) * u[:, 10] + Phiphi0lim[0]
    t[:, 11] = (Phir0lim[1] - Phir0lim[0]) * u[:, 11] + Phir0lim[0]
    return t


def inverse_prior_transform(params):
    logm1lim   = [5.82615, 5.83032]
    logm2lim   = [1.99273, 1.99370]
    alim       = [0.49650, 0.50850]
    p0lim      = [15.64820, 15.72680]
    e0lim      = [0.74301, 0.74413]
    distlim    = [3.94257, 4.75567]
    cosqSlim   = [0.62966, 1.0]
    phiSlim    = [3.15906, 4.06029]
    cosqKlim   = [0.31455, 1.0]
    phiKlim    = [1.66135, 4.53748]
    Phiphi0lim = [1.48468, 5.32255]
    Phir0lim   = [0.00000, 2 * np.pi]

    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0]  = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1]  = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2]  = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3]  = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4]  = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5]  = (params[:, 5] - distlim[0]) / (distlim[1] - distlim[0])
    u[:, 6]  = (params[:, 6] - cosqSlim[0]) / (cosqSlim[1] - cosqSlim[0])
    u[:, 7]  = (params[:, 7] - phiSlim[0]) / (phiSlim[1] - phiSlim[0])
    u[:, 8]  = (params[:, 8] - cosqKlim[0]) / (cosqKlim[1] - cosqKlim[0])
    u[:, 9]  = (params[:, 9] - phiKlim[0]) / (phiKlim[1] - phiKlim[0])
    u[:, 10] = (params[:, 10] - Phiphi0lim[0]) / (Phiphi0lim[1] - Phiphi0lim[0])
    u[:, 11] = (params[:, 11] - Phir0lim[0]) / (Phir0lim[1] - Phir0lim[0])
    return u


print('Setting up ParisMC...')
config = parismc.SamplerConfig(
    merge_type='distance',
    alpha=int(1e3),
    trail_size=int(1e3),
    boundary_limiting=True,
    use_beta=True,
    integral_num=int(1e5),
    gamma=500,
    exclude_scale_z=np.inf,
    use_pool=False,
    keep_dead_processes=True,
)

ndim = 12
n_seed = 10
sigma = 1e-2
init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config,
)

print('Getting LHS points...')
dir_scratch = '/scratch/e1498138'

lhs_path = dir_scratch + '/paper/ext/emri_g/f_1e5_brute/lhs_final.pkl'

with open(lhs_path, 'rb') as f:
    lhs_data = pickle.load(f)
phys_pts = np.asarray(lhs_data['lhs_phys'])
log_densities = np.asarray(lhs_data['log_densities'])

valid = np.isfinite(log_densities)
external_lhs_points = inverse_prior_transform(phys_pts[valid])
external_lhs_log_densities = log_densities[valid]
print(f'Loaded {valid.sum()} / {len(log_densities)} finite LHS evaluations.')
savepath = dir_scratch + '/paper/ext/emri_g/f_1e5_brute_merge'


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()


print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(5e4),
    savepath=savepath,
    print_iter=10,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
print('Savepath:', savepath)
