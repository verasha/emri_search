"""
"ext" (7-dim, qS/phiS-extended) counterpart of paris3_sc.py.

Same source, LogLike, and S6 semi-coherent scoring as paris3_lhs_s6_ext.py
(which precomputed the LHS seed at /scratch/e1498138/paper/ext/paris3_lhs_s6/lhs_s6.pkl).
Runs a ParisMC search over the 7-dim (logm1, logm2, a, p0, e0, qS, phiS) box,
using that LHS grid (and its ellipsoid prior bounds) as the external seed.
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
# sys.path.insert(0, "/nfs/home/svu/e1498138/parismc_dev")

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 10
T = 23/12 #NOTE: changed!
N_SEGS = 6
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_SEGS}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# # Source (matches paper/ext/stage1_anneal.ipynb / paris3_lhs_s6_ext.py)
# m1, m2, a, p0, e0, xI0 = 1e6, 1e1, 0.7, 9.0, 0.4, 1.0
# dist, qS, phiS, qK, phiK = 4.5, np.pi, 0., 0., 0.
# Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5

# Source (Mojito light EMRI_G, matches paris2_sc_ext.py)

# m1 = 6.72e5
# m2 = 9.84e1
# a = 0.5
# p0 = 15.7117
# e0 = 0.7440
# xI0 = 1.0
# dist = 4.755
# qS = 0.5906
# phiS = 3.5808
# qK = 1.1707
# phiK = 3.8495
# Phi_phi0 = 3.1940
# Phi_theta0 = 3.3780
# Phi_r0 = 0.6038


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

# Mojito light EMRI_C
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

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
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


def log_density(params):
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = params[i]
        try:
            qS_i = np.arccos(cos_qS_i)
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS_i, phiS_i, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                T=T, dt=dt,
            ))
            out[i] = float(gwf.SNR_semicoherent(loglike_obj.signal, h_temp, N_seg=N_SEGS, phase_max=True))
        except Exception:
            pass
    return out



def prior_transform(u):
    # broad EMRI C (pre-anneal)
    # logm1lim = [6.15,  6.95]
    # logm2lim = [1.60,  2.20]
    # alim = [0.70,  0.999]
    # p0lim = [5.89,  8.89]
    # e0lim = [0.17,  0.47]
    # cosqSlim = [-1.0,  1.0]
    # phiSlim = [0.0,  2*np.pi]

    # EMRI C, matches lhs_sc_s6/final.pkl box (run_lhs_noise_sc_ext.py, "# EMRI_C")
    logm1lim = [6.54808, 6.54911]
    logm2lim = [1.90229, 1.90370]
    alim = [0.94932, 0.95003]
    p0lim = [7.38774, 7.39225]
    e0lim = [0.31562, 0.31965]
    cosqSlim = [-1.0, 1.0]
    phiSlim = [0.0, 2 * np.pi]

    t = np.zeros_like(u)
    t[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    t[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    t[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    t[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]
    t[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    t[:, 5] = (cosqSlim[1] - cosqSlim[0]) * u[:, 5] + cosqSlim[0]
    t[:, 6] = (phiSlim[1] - phiSlim[0]) * u[:, 6] + phiSlim[0]
    return t


def inverse_prior_transform(params):
    # EMRI C, matches lhs_sc_s6/final.pkl box (run_lhs_noise_sc_ext.py, "# EMRI_C")
    logm1lim = [6.54808, 6.54911]
    logm2lim = [1.90229, 1.90370]
    alim = [0.94932, 0.95003]
    p0lim = [7.38774, 7.39225]
    e0lim = [0.31562, 0.31965]
    cosqSlim = [-1.0, 1.0]
    phiSlim = [0.0, 2 * np.pi]

    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5] = (params[:, 5] - cosqSlim[0]) / (cosqSlim[1] - cosqSlim[0])
    u[:, 6] = (params[:, 6] - phiSlim[0]) / (phiSlim[1] - phiSlim[0])
    return u


print('Setting up ParisMC...')
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
    seed=6342
)

ndim = 7
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

lhs_path = dir_scratch + '/paper/ext/emri_c/lhs_sc_s6/final.pkl'

with open(lhs_path, 'rb') as f:
    phys_pts, det_snr = pickle.load(f)

valid = np.isfinite(det_snr)
external_lhs_points = inverse_prior_transform(phys_pts[valid])
external_lhs_log_densities = det_snr[valid]
print(f'Loaded {valid.sum()} / {len(det_snr)} finite LHS evaluations.')
savepath = dir_scratch + f'/paper/ext/emri_c/stage2_merge_s6/'


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
