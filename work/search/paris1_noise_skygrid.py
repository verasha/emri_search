"""
Seed 25 independent PARIS processes on a 5x5 grid in (qS, phiS) around the
stage-1 MAP point, to map out sky-localization secondary modes directly
instead of relying on chains to stumble onto them.

Mass/spin/p0/e0 are held fixed at the stage-1 MAP values; qS and phiS are
each swept over 5 points spanning +/- pi/2 around the MAP sky position
(qS clipped to [0, pi], phiS wrapped mod 2*pi). Prior box is unchanged.
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
T = 3/12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

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
Phi_theta0 = 3.3780
Phi_r0 = 0.6038

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, qS, phiS]

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
    log_likes = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i, qS_i, phiS_i = params[i]
        try:
            log_likes[i] = loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS_i, phiS_i, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            pass
    return log_likes


def prior_transform(u):
    logm1lim = [5.6,  6.4]
    logm2lim = [1.7, 2.3]
    alim = [0.3, 0.99]
    p0lim = [13.5, 16.5]
    e0lim = [0.6, 0.9]
    qSlim = [0.0, np.pi]
    phiSlim = [0.0, 2 * np.pi]
    t = np.zeros_like(u)
    t[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    t[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    t[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    t[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]
    t[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    t[:, 5] = (qSlim[1] - qSlim[0]) * u[:, 5] + qSlim[0]
    t[:, 6] = (phiSlim[1] - phiSlim[0]) * u[:, 6] + phiSlim[0]
    return t


def inverse_prior_transform(params):
    logm1lim = [5.6,  6.4]
    logm2lim = [1.7, 2.3]
    alim = [0.3, 0.99]
    p0lim = [13.5, 16.5]
    e0lim = [0.6, 0.9]
    qSlim = [0.0, np.pi]
    phiSlim = [0.0, 2 * np.pi]
    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5] = (params[:, 5] - qSlim[0]) / (qSlim[1] - qSlim[0])
    u[:, 6] = (params[:, 6] - phiSlim[0]) / (phiSlim[1] - phiSlim[0])
    return u


print('Setting up ParisMC...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,
    alpha=int(1e5),
    trail_size=int(1e3),
    boundary_limiting=True,
    use_beta=True,
    integral_num=int(1e5),
    gamma=500,
    exclude_scale_z=np.inf,
    use_pool=False,
    keep_dead_processes=True
)

ndim = 7

# --- Stage-1 MAP point (emri_g_stage1_f.ipynb, maxld_pt1) ---
# [logm1, logm2, a, p0, e0, qS, phiS]
map1 = np.array([5.83098287, 1.99866379, 0.50439036, 15.70515014, 0.7438042, 1.73721063, 1.32380263])
qS0, phiS0 = map1[5], map1[6]

qSlim = (0.0, np.pi)
phiSlim = (0.0, 2 * np.pi)

n_grid = 5
offsets = np.linspace(-np.pi / 2, np.pi / 2, n_grid)

qS_grid = np.clip(qS0 + offsets, *qSlim)
phiS_grid = (phiS0 + offsets) % (2 * np.pi)

qS_mesh, phiS_mesh = np.meshgrid(qS_grid, phiS_grid, indexing='ij')
qS_flat = qS_mesh.ravel()
phiS_flat = phiS_mesh.ravel()

n_seed = n_grid * n_grid  # 25
print(f'Building {n_seed} grid seed points around MAP1 sky position (qS0={qS0:.4f}, phiS0={phiS0:.4f})...')
print(f'qS grid: {qS_grid}')
print(f'phiS grid: {phiS_grid}')

grid_points_phys = np.tile(map1, (n_seed, 1))
grid_points_phys[:, 5] = qS_flat
grid_points_phys[:, 6] = phiS_flat

print('Evaluating log-density at grid seed points...')
grid_log_densities = log_density(grid_points_phys)
n_bad = np.sum(~np.isfinite(grid_log_densities))
if n_bad > 0:
    print(f'WARNING: {n_bad} / {n_seed} grid points have non-finite log-density '
          f'(likely qS clipped to boundary). These seeds will start from a weak point.')
for i in range(n_seed):
    print(f'  seed {i:2d}: qS={grid_points_phys[i,5]:.4f}, phiS={grid_points_phys[i,6]:.4f}, '
          f'logL={grid_log_densities[i]:.4f}')

external_lhs_points = inverse_prior_transform(grid_points_phys)
external_lhs_log_densities = grid_log_densities

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

savepath = '/scratch/e1498138/paper/ext/emri_g/stage1_skygrid/'


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()


print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(1e5),
    savepath=savepath,
    print_iter=10,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
print('Savepath:', savepath)
