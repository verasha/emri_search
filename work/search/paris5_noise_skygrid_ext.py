"""
Seed 25 independent PARIS processes on a 5x5 grid in (qS, phiS) around the
stage3 (paris5_lhs_ext_stage2.py) max-log-density point, to map out
sky-localization secondary modes directly instead of relying on chains to
stumble onto them -- same idea as paris1_noise_skygrid.py, but scored with
the coherent (pure) f-statistic (loglike_pure_noise.LogLike) and using the
paris5 ext source and stage3 ellipse prior box.

Mass/spin/p0/e0 are held fixed at the stage3 max-log-density values; qS and
phiS are each swept over 5 points spanning +/- pi/2 around that sky position
(qS clipped to [0, pi], phiS wrapped mod 2*pi). Prior box is the stage3
ellipse box (paper/ext/stage3/lhs_f.pkl's ellipse_lo/ellipse_hi), unchanged.
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

# Source (matches paper/ext/stage2_anneal, paris4_sc_ext.py, and
# paris5_lhs_ext_stage2.py)
m1, m2, a, p0, e0, xI0 = 1e6, 1e1, 0.7, 9.0, 0.4, 1.0
dist, qS, phiS, qK, phiK = 4.5, np.pi, 0., 0., 0.
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5
params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]
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
print(f'SNR (rhostat): {float(gwf.SNR(gwf.freq_wave(loglike_obj.signal))):.4f}')


def log_density(params):
    params = np.asarray(params)
    log_likes = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i, qS_i, phiS_i = params[i]
        theta = [
            10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS_i, phiS_i, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
        ]
        try:
            log_likes[i] = float(loglike_obj(theta))
        except Exception:
            pass
    return log_likes


# ── Prior box: stage3 ellipse box (paris5_lhs_ext_stage2.py) ──────────────────

dir_scratch = '/scratch/e1498138'
lhs_path = dir_scratch + '/paper/ext/stage3/lhs_f.pkl'

print(f'Loading stage3 LHS grid from {lhs_path}...')
with open(lhs_path, 'rb') as f:
    lhs_data = pickle.load(f)

ellipse_lo = lhs_data['ellipse_lo']
ellipse_hi = lhs_data['ellipse_hi']


def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u


def inverse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)


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
    keep_dead_processes=True,
)

ndim = 7

# --- Stage3 max-log-density point (paris5_lhs_ext_stage2.py's lhs_f.pkl) ---
# [logm1, logm2, a, p0, e0, qS, phiS]
lhs_log_densities = lhs_data['log_densities']
maxld_idx = np.argmax(lhs_log_densities)
map1 = lhs_data['lhs_phys'][maxld_idx]
print(f'stage3 maxld: {lhs_log_densities[maxld_idx]:.4f}')
print(f'stage3 maxld point: {map1}')
qS0, phiS0 = map1[5], map1[6]

qSlim = (ellipse_lo[5], ellipse_hi[5])
phiSlim = (ellipse_lo[6], ellipse_hi[6])

n_grid = 5
offsets = np.linspace(-np.pi / 2, np.pi / 2, n_grid)

qS_grid = np.clip(qS0 + offsets, *qSlim)
phiS_grid = (phiS0 + offsets) % (2 * np.pi)

qS_mesh, phiS_mesh = np.meshgrid(qS_grid, phiS_grid, indexing='ij')
qS_flat = qS_mesh.ravel()
phiS_flat = phiS_mesh.ravel()

n_seed = n_grid * n_grid  # 25
print(f'Building {n_seed} grid seed points around stage3 MAP sky position (qS0={qS0:.4f}, phiS0={phiS0:.4f})...')
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

savepath = dir_scratch + '/paper/ext/stage3_skygrid/'


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
