"""
Precompute LHS grid inside the paris1 (stage1_anneal) posterior ellipsoid for the
paris5 noise search, scoring each point with the coherent (pure) f-statistic
(loglike_obj(theta), from loglike_pure_noise.LogLike) instead of a semi-coherent
statistic — same scoring approach as paris5_lhs.py.

This is the "ext" (7-dim, qS/phiS-extended) counterpart of paris5_lhs.py: the
source, LogLike, and 7-dim (logm1, logm2, a, p0, e0, qS, phiS) parameter space
match paper/ext/stage1_anneal.ipynb, and the ellipsoid center/covariance are
loaded from that stage's sampler state (paper/ext/stage1_anneal/sampler_state.pkl),
same as paris3_lhs_s6_ext.py.

For logm1, logm2, a, p0, e0 the LHS box is the N-sigma bounding box around the
stage1_anneal posterior, same as paris3_lhs_s6_ext.py. For qS and phiS there is
no useful sky-localization information yet at this stage, so those two
dimensions are sampled over their FULL prior range ([0, pi] and [0, 2*pi])
rather than an ellipsoid-restricted box.

Loads paper/ext/stage1_anneal to get ellipse center and covariance.
Saves unit-cube points, physical points, coherent-f log_densities, and prior bounds.
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

# Source (matches paper/ext/stage1_anneal.ipynb)
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


# ── Load paper/ext/stage1_anneal for ellipse center and covariance ────────────

_p1_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2, 0.0, 0.0])
_p1_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5, np.pi, 2 * np.pi])

def _stub_prior_transform(u):
    return _p1_lo + (_p1_hi - _p1_lo) * u

def log_density(params):
    raise RuntimeError("stub")

def prior_transform(u):
    return _stub_prior_transform(u)

import __main__
__main__.log_density    = log_density
__main__.prior_transform = prior_transform

stage1_anneal_path = '/scratch/e1498138/paper/ext/stage1_anneal/sampler_state.pkl'
print(f'Loading stage1_anneal sampler from {stage1_anneal_path}...')
sampler_1 = parismc.Sampler.load_state(stage1_anneal_path)

all_pts_u  = sampler_1.searched_points_list[0]
all_logden = sampler_1.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = _stub_prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'stage1_anneal maxld: {all_logden[maxld_idx]:.4f}')
print(f'stage1_anneal maxld point: {mu_center}')

samples_p1, weights_p1 = sampler_1.get_samples_with_weights(flatten=True)
weights_p1 = weights_p1 / weights_p1.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p1), size=50_000, replace=True, p=weights_p1)
cov_posterior = np.cov(samples_p1[idx_rs].T)
print('stage1_anneal posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_1, samples_p1, weights_p1, idx_rs

# ── Ellipsoid bounding box ─────────────────────────────────────────────────────
# e0 has high Fisher information → tight posterior at the (possibly wrong) mode.
# Use N_SIGMA >> 1 so true e0=0.4 is safely inside the LHS volume.
# qS, phiS have no useful localization at this stage, so they are handled
# separately below and always take the FULL prior range.

param_names   = ['logm1', 'logm2', 'a', 'p0', 'e0', 'qS', 'phiS']
N_SIGMA = 25.0
N_SIGMA_PER_DIM = np.array([N_SIGMA, N_SIGMA, N_SIGMA,
                             N_SIGMA, N_SIGMA, 0.0, 0.0])

sigma_diag = np.sqrt(np.diag(cov_posterior))
ellipse_lo = np.clip(mu_center - N_SIGMA_PER_DIM * sigma_diag, _p1_lo, _p1_hi)
ellipse_hi = np.clip(mu_center + N_SIGMA_PER_DIM * sigma_diag, _p1_lo, _p1_hi)

# qS, phiS: full prior range, no ellipsoid restriction.
ellipse_lo[5:7] = _p1_lo[5:7]
ellipse_hi[5:7] = _p1_hi[5:7]

print(f'Ellipse prior bounds (per-dim sigma):')
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  '
          f'mu={mu_center[i]:.5f}  sigma={sigma_diag[i]:.5f}  N_sigma={N_SIGMA_PER_DIM[i]:.0f}')

def ellipse_prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_ellipse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ── Generate LHS uniformly in the N-sigma box (full range for qS, phiS) ───────
# No ellipsoid truncation: points fill the axis-aligned box [ellipse_lo,
# ellipse_hi] (mu ± N_sigma * sigma_diag per dim, full range for qS/phiS), not
# the correlated ellipsoid.

ndim  = 7
N_LHS = int(5e5)

print(f'Generating {N_LHS} LHS points in {ndim}D box...')
_lhs_sampler = LHS(xlimits=np.column_stack([ellipse_lo, ellipse_hi]))
lhs_phys     = _lhs_sampler(N_LHS)
lhs_phys     = np.clip(lhs_phys, _p1_lo, _p1_hi)
n_inside     = N_LHS
lhs_u    = np.clip(
    np.array([inverse_ellipse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)

# ── Evaluate coherent f-statistic ──────────────────────────────────────────────

print(f'Evaluating coherent f-statistic on {n_inside} points...')
log_densities = np.full(n_inside, -np.inf)
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

savepath = f'/scratch/e1498138/paper/ext/paris5_lhs/lhs_f.pkl'
os.makedirs(os.path.dirname(savepath), exist_ok=True)
save_data = {
    'lhs_u':          lhs_u,
    'lhs_phys':       lhs_phys,
    'log_densities':  log_densities,
    'ellipse_lo':     ellipse_lo,
    'ellipse_hi':     ellipse_hi,
    'mu_center':      mu_center,
    'cov_posterior':     cov_posterior,
    'sigma_diag':        sigma_diag,
    'N_SIGMA_PER_DIM':   N_SIGMA_PER_DIM,
    'N_SIGMA':       N_SIGMA,
    'N_LHS':             N_LHS,
    'n_inside':       n_inside,
    'statistic':      'coherent_f (loglike_pure_noise)',
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
