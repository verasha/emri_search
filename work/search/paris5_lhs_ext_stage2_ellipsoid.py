"""
Stage3 counterpart of paris5_lhs_ext_stage2.py that uses an ellipsoid
"throwaway" LHS instead of a uniform axis-aligned box: points are drawn via
LHS in Cholesky-whitened space over the intrinsic dims (logm1, logm2, a, p0,
e0) and any point landing outside the N-sigma unit sphere is discarded
(box1_lhs_ellipsoid.py's approach), rather than filling the full bounding box.

The source, LogLike (coherent f-statistic via loglike_pure_noise), and 7-dim
(logm1, logm2, a, p0, e0, qS, phiS) parameter space match
paris5_lhs_ext_stage2.py, and the ellipsoid center/covariance are loaded the
same way from paper/ext/stage2_anneal (paris4_sc_ext.py)'s sampler state.
qS, phiS are not part of the ellipsoid -- they are drawn uniformly over the
full stage2 prior range ([0, pi] and [0, 2*pi]) for each surviving intrinsic
point, exactly as in paris5_lhs_ext_stage2.py.

Saves unit-cube points, physical points, coherent-f log_densities, and prior
bounds.
"""
import numpy as np
import pickle
import time
import os
import sys

import few
from smt.sampling_methods import LHS

os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/search/')

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_pure_noise import LogLike
import parismc

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("WARNING")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12 / 12
print(f"dt={dt}s  T={T}yr  TDI gen={tdi_gen}")

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source (matches paper/ext/stage2_anneal / paris4_sc_ext.py)
m1, m2, a, p0, e0, xI0 = 1e6, 1e1, 0.7, 9.0, 0.4, 1.0
dist, qS, phiS, qK, phiK = 4.5, np.pi, 0., 0., 0.
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5
params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, qS, phiS]

n_vals = np.arange(-1, 6)
ell = 2

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
print(f'SNR (rhostat): {float(gwf.SNR(gwf.freq_wave(loglike_obj.signal))):.4f}')


def f_score(phys_point):
    """Coherent (pure) f-statistic at a physical point [logm1, logm2, a, p0, e0, qS, phiS]."""
    logm1, logm2, a_i, p0_i, e0_i, qS_i, phiS_i = phys_point
    theta = [
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        xI0, dist, qS_i, phiS_i, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
    ]
    return float(loglike_obj(theta))


# ── Load paper/ext/stage2_anneal for ellipse center and covariance ────────────
# Prior box matches paris4_sc_ext.py's prior_transform (stage2_anneal box).

_p2_lo = np.array([5.92138, 0.89200, 0.69030, 8.00000, 0.39955, 0.0, 0.0])
_p2_hi = np.array([6.11220, 1.01955, 0.77933, 9.45620, 0.44961, np.pi, 2 * np.pi])

def _stub_prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

def log_density(params):
    raise RuntimeError("stub")

def prior_transform(u):
    return _stub_prior_transform(u)

import __main__
__main__.log_density    = log_density
__main__.prior_transform = prior_transform

stage2_anneal_path = '/scratch/e1498138/paper/ext/stage2_anneal/sampler_state.pkl'
print(f'Loading stage2_anneal sampler from {stage2_anneal_path}...')
sampler_2 = parismc.Sampler.load_state(stage2_anneal_path)

all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = _stub_prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'stage2_anneal maxld: {all_logden[maxld_idx]:.4f}')
print(f'stage2_anneal maxld point: {mu_center}')

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx_rs].T)
print('stage2_anneal posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_2, samples_p2, weights_p2, idx_rs

# ── Ellipsoid (Cholesky-truncated, throwaway) over the intrinsic dims ─────────
# qS, phiS have no useful localization restriction here, so they always take
# the FULL stage2 prior range and are drawn independently and uniformly --
# they are not part of the ellipsoid.

param_names   = ['logm1', 'logm2', 'a', 'p0', 'e0', 'qS', 'phiS']
N_SIGMA = 7.0
n_intrinsic = 5

mu_intrinsic  = mu_center[:n_intrinsic]
cov_intrinsic = cov_posterior[:n_intrinsic, :n_intrinsic]
sigma_diag    = np.sqrt(np.diag(cov_posterior))

ellipse_lo = np.clip(mu_center - N_SIGMA * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi = np.clip(mu_center + N_SIGMA * sigma_diag, _p2_lo, _p2_hi)
ellipse_lo[5:7] = _p2_lo[5:7]
ellipse_hi[5:7] = _p2_hi[5:7]

print(f'Ellipse prior bounds (per-dim sigma, N_sigma={N_SIGMA}):')
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  '
          f'mu={mu_center[i]:.5f}  sigma={sigma_diag[i]:.5f}')

def ellipse_prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_ellipse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ── Generate LHS in Cholesky space, filter (throwaway) by unit sphere ─────────
# Intrinsic dims: draw in the N_sigma Cholesky-whitened ball and discard
# anything landing outside the unit sphere -- an ellipsoid, not a box.
# qS, phiS: full prior range, drawn uniformly (LHS) for each surviving point.

ndim  = 7
N_LHS = int(5e5)  # oversample: ~16% of a 5D LHS cube lands inside the unit ball

savepath = f'/scratch/e1498138/paper/ext/stage3_ellipsoid/lhs_f.pkl'
os.makedirs(os.path.dirname(savepath), exist_ok=True)

print(f'Generating {N_LHS} LHS points in {n_intrinsic}D Cholesky space...')
_L     = np.linalg.cholesky(cov_intrinsic)
_scale = N_SIGMA  # uniform per-dim N_sigma applied after Cholesky

_lhs_sampler = LHS(xlimits=np.column_stack([-np.ones(n_intrinsic), np.ones(n_intrinsic)]))
lhs_z_raw    = _lhs_sampler(N_LHS)
sphere_mask  = np.sum(lhs_z_raw ** 2, axis=1) <= 1.0
lhs_z_inside = lhs_z_raw[sphere_mask]
n_inside     = int(sphere_mask.sum())
print(f'LHS points inside unit sphere: {n_inside} / {N_LHS} ({100*sphere_mask.mean():.1f}%)')

lhs_intrinsic = mu_intrinsic + _scale * (_L @ lhs_z_inside.T).T
lhs_intrinsic = np.clip(lhs_intrinsic, _p2_lo[:n_intrinsic], _p2_hi[:n_intrinsic])

print(f'Drawing qS, phiS uniformly (LHS) for {n_inside} surviving points...')
_sky_sampler = LHS(xlimits=np.column_stack([ellipse_lo[5:7], ellipse_hi[5:7]]))
lhs_sky      = _sky_sampler(n_inside)

lhs_phys = np.concatenate([lhs_intrinsic, lhs_sky], axis=1)
lhs_u    = np.clip(
    np.array([inverse_ellipse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)
log_densities = np.full(n_inside, -np.inf)

# ── Evaluate coherent f-statistic ──────────────────────────────────────────────

print(f'Evaluating coherent f-statistic on {n_inside} points...')
t0 = time.time()

for i in range(n_inside):
    if i % 500 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (n_inside - i) / rate
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    try:
        ld = f_score(lhs_phys[i])
        log_densities[i] = ld if np.isfinite(ld) else -np.inf
    except Exception:
        log_densities[i] = -np.inf

elapsed = time.time() - t0
print(f'Done. {n_inside} evals in {elapsed:.0f}s ({n_inside/elapsed:.1f}/s)')

n_finite = np.isfinite(log_densities).sum()
print(f'Finite log_densities: {n_finite} / {n_inside}')
if n_finite > 0:
    print(f'Max log_density: {np.max(log_densities[np.isfinite(log_densities)]):.6f}')
    print(f'Mean log_density (finite): {np.mean(log_densities[np.isfinite(log_densities)]):.6f}')

# ── Save ──────────────────────────────────────────────────────────────────────

save_data = {
    'lhs_u':          lhs_u,
    'lhs_phys':       lhs_phys,
    'log_densities':  log_densities,
    'ellipse_lo':     ellipse_lo,
    'ellipse_hi':     ellipse_hi,
    'mu_center':      mu_center,
    'cov_posterior':  cov_posterior,
    'sigma_diag':     sigma_diag,
    'N_SIGMA':        N_SIGMA,
    'N_LHS':          N_LHS,
    'n_inside':       n_inside,
    'statistic':      'coherent_f (loglike_pure_noise)',
    'ellipsoid':      True,
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
