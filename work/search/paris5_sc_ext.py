"""
ParisMC merge for the paris5 (ext) pipeline, scoring with the coherent
(pure) f-statistic (loglike_pure_noise.LogLike), same EMRI_C source,
7-dim (logm1, logm2, a, p0, e0, cosqS, phiS) parameter space, and box
prior bounds as paris5_lhs_box.py.

Loads the LHS grid precomputed by paris5_lhs_box.py
(paper/ext/emri_c/stage2_f_1e4/lhs_final.pkl) -- points, coherent-f
log_densities, and the box_lo/box_hi bounds it was drawn from -- and uses
it to seed a ParisMC sampler that merges/refines within that same box.
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
dt = 10
T = 23 / 12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)


def icrs_to_ecliptic(qK_icrs, phiK_icrs, eps_deg=23.43929111):
    """
    Convert a sky direction (colatitude, longitude) from the ICRS/equatorial
    frame (qK = pi/2 - dec, phiK = ra) to the ecliptic frame (SSB frame used
    by the LISA response, is_ecliptic_latitude=False convention) via the
    standard equatorial<->ecliptic rotation with J2000 mean obliquity eps.
    """
    dec = np.pi / 2 - qK_icrs
    ra = phiK_icrs
    eps = np.deg2rad(eps_deg)

    sin_beta = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(ra) * np.sin(eps)
    beta = np.arcsin(sin_beta)

    y = np.cos(dec) * np.sin(ra) * np.cos(eps) + np.sin(dec) * np.sin(eps)
    x = np.cos(dec) * np.cos(ra)
    lam = np.arctan2(y, x) % (2 * np.pi)

    qK_ecl = np.pi / 2 - beta
    phiK_ecl = lam
    return qK_ecl, phiK_ecl


# Mojito light EMRI_C (matches paris5_lhs_box.py)
m1 = 3.54e6
m2 = 8.01e1
a = 0.950
p0 = 7.3890
e0 = 0.3160
xI0 = 1.0000
dist = 4.858
qS = 2.0420
phiS = 5.7873
qK, phiK = icrs_to_ecliptic(1.6633, 1.7431)  # catalog qK/phiK are in ICRS, convert to ecliptic
Phi_phi0 = 2.4044
Phi_theta0 = 0.2850
Phi_r0 = 0.9743

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, np.cos(qS), phiS]

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
    """Coherent (pure) f-statistic, batched."""
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = params[i]
        qS_i = np.arccos(cos_qS_i)
        theta = [
            10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS_i, phiS_i, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
        ]
        try:
            out[i] = float(loglike_obj(theta))
        except Exception:
            pass
    return out


# ── Load paris5_lhs_box.py's LHS grid: box prior bounds + coherent-f evals ────

dir_scratch = '/scratch/e1498138'
lhs_path = dir_scratch + '/paper/ext/emri_c/stage2_f_1e4/lhs_final.pkl'

print(f'Loading LHS grid from {lhs_path}...')
with open(lhs_path, 'rb') as f:
    lhs_data = pickle.load(f)

ellipse_lo = lhs_data['box_lo']
ellipse_hi = lhs_data['box_hi']
lhs_phys = lhs_data['lhs_phys']
lhs_log_densities = lhs_data['log_densities']

param_names = ['logm1', 'logm2', 'a', 'p0', 'e0', 'cosqS', 'phiS']
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]')


def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u


def inverse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)


print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,
    alpha=int(1e3),
    trail_size=int(1e3),
    boundary_limiting=True,
    use_beta=True,
    integral_num=int(1e5),
    gamma=500,
    exclude_scale_z=np.inf,
    use_pool=False,
    keep_dead_processes=True,
    merge_type='distance',
)

ndim = 7
n_seed = 10
init_cov = np.eye(ndim) * 1e-4
init_cov_list = [init_cov] * n_seed

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config,
)

print('Preparing external LHS points from box LHS grid...')
valid = np.isfinite(lhs_log_densities)
external_lhs_points = np.clip(
    inverse_prior_transform(lhs_phys[valid]), 0.0, 1.0
)
external_lhs_log_densities = lhs_log_densities[valid]
print(f'Loaded {valid.sum()} / {len(lhs_log_densities)} finite LHS evaluations.')


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()


savepath = dir_scratch + '/paper/ext/emri_c/stage2_f_1e4_merge'

print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(1e4),
    savepath=savepath,
    print_iter=100,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done running sampling.')
print('Savepath:', savepath)
