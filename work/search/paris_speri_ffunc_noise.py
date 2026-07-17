"""
PARIS refinement of the coherent one-stop function f on the Speri near-circular
source, seeded by the Speri SBI posterior samples.

Modeled on paris5_noise.py. Differences (both intentional):
  1. Source params = the Speri injection (best_results.h5), not the eccentric demo.
  2. Sampled coordinates are (log10 m1, m2, a, Tpl, ef) -- the SBI parameter
     space -- NOT (p0, e0). (Tpl, ef) is converted to (p0, e0) by backward
     trajectory integration inside log_density (emri_utils recipe). This keeps
     the prior compact and lets the SBI samples be used directly as seeds.
  3. PARIS is seeded with the SBI posterior samples (re-evaluated under f on THIS
     noise realization) instead of a blind ellipse-LHS grid.

No T-annealing: fixed T, single coherent f objective throughout.

f is coherent, no maximization: loglike_pure_noise uses Re(<d|h>)/rho (no abs,
no time/phase shift).
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

# trajectory generator for the (Tpl, ef) -> (p0, e0) conversion
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12/12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

# --- (Tpl, ef) -> (p0, e0) converter (same recipe as emri_utils.py) -----------
_traj = EMRIInspiral(func=KerrEccEqFlux)
_BUF = _traj.inspiral_generator.func.separatrix_buffer_dist


def tplef_to_p0e0(m1, m2, a, Tpl, ef, x0=1.0):
    """Place orbit at separatrix with eccentricity ef, integrate back Tpl yr;
    p0, e0 at t=0 are the max p and e along the (inward-decaying) trajectory."""
    from few.utils.geodesic import get_separatrix
    p_pl = _BUF + get_separatrix(a, ef, x0) + 1e-6
    out = _traj(m1, m2, a, p_pl, ef, x0, T=float(Tpl), integrate_backwards=True)
    return float(np.max(out[1])), float(np.max(out[2]))


print('Building ResponseWrapper...')
# pad_output=True: source plunges at Tpl=0.902yr < T=1yr (paper Fig.2), so the
# waveform must be zero-padded out to the 1yr grid (data = noise after plunge).
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen,
                                            pad_output=True)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# --- Source parameters: Speri near-circular injection (best_results.h5) --------
m1 = 1337921.2715760823
m2 = 27.09194489452272
a = 0.8635975383155665
p0 = 7.855665849709651
e0 = 0.017295005700448653
xI0 = 1.0
dist = 3.191602792552005
qS = 1.461597549736258
phiS = 2.7031389624243194
qK = 0.7101677235968459
phiK = 0.5046062323196301
Phi_phi0 = 1.5329007115882451
Phi_theta0 = 3.494293502033031
Phi_r0 = 3.3878102006676967

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]

n_vals = np.arange(-1, 6)
ell = 2

print('Initializing LogLike (injects signal + noise, seed=42)...')
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

# --- Prior box (SBI support). Sampled coords: log10 m1, m2, a, Tpl, ef ---------
BOUNDS = {
    'log10_m1': [5.70, 6.40],
    'm2':       [19.0, 50.0],
    'a':        [0.30, 0.93],
    'Tpl':      [0.85, 0.93],
    'ef':       [0.00, 0.10],
}
_lo = np.array([BOUNDS[k][0] for k in ['log10_m1', 'm2', 'a', 'Tpl', 'ef']])
_hi = np.array([BOUNDS[k][1] for k in ['log10_m1', 'm2', 'a', 'Tpl', 'ef']])
ndim = 5


def prior_transform(u):
    u = np.asarray(u)
    return _lo + (_hi - _lo) * u


def inverse_prior_transform(params):
    params = np.asarray(params)
    return (params - _lo) / (_hi - _lo)


def log_density(params):
    params = np.asarray(params)
    log_likes = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        logm1, m2_i, a_i, Tpl_i, ef_i = params[i]
        try:
            p0_i, e0_i = tplef_to_p0e0(10**logm1, m2_i, a_i, Tpl_i, ef_i)
            f_val = loglike_obj(np.array([
                10**logm1, m2_i, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            f_val = -np.inf
        log_likes[i] = f_val
    return log_likes


print('Done setting up log-likelihood and prior.')

# --- Seeds: Speri SBI posterior samples, re-evaluated under f ------------------
SBI_NPZ = '/nfs/home/svu/e1498138/EMRI-Search/paper_figures/sample_lls_n1000000_b100_p8.npz'
N_SEED_LHS = 2000   # PARIS initialization seeds

print(f'Loading SBI samples from {SBI_NPZ} ...')
_d = np.load(SBI_NPZ)
_s, _ll = _d['samples'], _d['sample_lls']   # (N,5) = log10m1, m2, a, Tpl, ef

# keep only samples inside our box
_in = np.all((_s >= _lo) & (_s <= _hi), axis=1)
_s_in, _ll_in = _s[_in], _ll[_in]
print(f'  {_in.sum()} / {len(_s)} SBI samples fall inside the box')

# pick the high-search-statistic region + a random spread for coverage
_order = np.argsort(_ll_in)
_top = _order[-int(0.75 * N_SEED_LHS):]
_rng = np.random.default_rng(0)
_rest = _rng.choice(len(_s_in), N_SEED_LHS - len(_top), replace=False)
_pick = np.unique(np.concatenate([_top, _rest]))
seed_params = _s_in[_pick]                       # physical, our coords
external_lhs_points = inverse_prior_transform(seed_params)   # u-space

print(f'Re-evaluating f on {len(seed_params)} SBI seeds (this realization)...')
external_lhs_log_densities = log_density(seed_params)
_finite = np.isfinite(external_lhs_log_densities)
external_lhs_points = external_lhs_points[_finite]
external_lhs_log_densities = external_lhs_log_densities[_finite]
print(f'  {_finite.sum()} finite-f seeds retained; '
      f'max f = {external_lhs_log_densities.max():.4f}')

# --- ParisMC sampler (config identical to paris5_noise.py; no annealing) -------
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
    merge_type='distance'
)

n_seed = 10
init_cov = np.eye(ndim) * 1e-5      # u-space seed covariance (as in paris5_noise)
init_cov_list = [init_cov] * n_seed

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Sampler initialized.')

dir_scratch = '/scratch/e1498138'
os.makedirs(f'{dir_scratch}/paris_speri_ffunc_noise', exist_ok=True)
filepath = f'{dir_scratch}/paris_speri_ffunc_noise/int_1yr'


def callback(sampler, i):
    if i % 1000 == 0 and i > 0:
        sampler.save_state()


print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(5e4),
    savepath=filepath,
    print_iter=100,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
