"""
Precompute LHS grid inside the paris4 (stage2_anneal) posterior ellipsoid for the
paris5 noise search, scoring each point with the coherent (pure) f-statistic
(loglike_obj(theta), from loglike_pure_noise.LogLike) instead of a semi-coherent
statistic — same scoring approach as paris5_lhs.py / paris5_lhs_ext.py.

This is the stage2 counterpart of paris5_lhs_ext.py: the source, LogLike, and
7-dim (logm1, logm2, a, p0, e0, qS, phiS) parameter space match
paper/ext/stage2_anneal (paris4_sc_ext.py), and the ellipsoid center/covariance
are loaded from that stage's sampler state
(paper/ext/stage2_anneal/sampler_state.pkl) instead of stage1_anneal. The prior
box is also narrowed to the stage2_anneal box (paris4_sc_ext.py's
prior_transform bounds) rather than the wide stage1 box.

For logm1, logm2, a, p0, e0 the LHS box is the N-sigma bounding box around the
stage2_anneal posterior, clipped to the stage2 prior box. For qS and phiS the
box still takes the FULL prior range ([0, pi] and [0, 2*pi]) since sky
localization is not being restricted here.

Loads paper/ext/stage2_anneal to get ellipse center and covariance.
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

# ── Ellipsoid bounding box ─────────────────────────────────────────────────────
# e0 has high Fisher information → tight posterior at the (possibly wrong) mode.
# Use N_SIGMA >> 1 so the box stays wide; it is clipped to the stage2 prior box
# below regardless, so a large N_SIGMA just means "fill the whole stage2 box".
# qS, phiS have no useful localization restriction here, so they always take
# the FULL stage2 prior range.

param_names   = ['logm1', 'logm2', 'a', 'p0', 'e0', 'qS', 'phiS']
N_SIGMA = 3.0
N_SIGMA_PER_DIM = np.array([N_SIGMA, N_SIGMA, N_SIGMA,
                             N_SIGMA, N_SIGMA, 0.0, 0.0])

sigma_diag = np.sqrt(np.diag(cov_posterior))
ellipse_lo = np.clip(mu_center - N_SIGMA_PER_DIM * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi = np.clip(mu_center + N_SIGMA_PER_DIM * sigma_diag, _p2_lo, _p2_hi)

# qS, phiS: full stage2 prior range, no ellipsoid restriction.
ellipse_lo[5:7] = _p2_lo[5:7]
ellipse_hi[5:7] = _p2_hi[5:7]

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
# ellipse_hi] (mu ± N_sigma * sigma_diag per dim, clipped to the stage2 prior
# box, full range for qS/phiS), not the correlated ellipsoid.

ndim  = 7
N_LHS = int(1e5)

savepath = f'/scratch/e1498138/paper/ext/stage3/lhs_f.pkl'
ckptpath = f'/scratch/e1498138/paper/ext/stage3/lhs_f_checkpoint.pkl'
os.makedirs(os.path.dirname(savepath), exist_ok=True)

CKPT_EVERY = 2000  # points between checkpoint writes

if os.path.exists(ckptpath):
    print(f'Resuming from checkpoint {ckptpath}...')
    with open(ckptpath, 'rb') as f:
        ckpt = pickle.load(f)
    lhs_phys      = ckpt['lhs_phys']
    lhs_u         = ckpt['lhs_u']
    log_densities = ckpt['log_densities']
    n_inside      = ckpt['n_inside']
    i_start       = int(ckpt['next_idx'])
    print(f'  {i_start}/{n_inside} points already evaluated; continuing.')
else:
    print(f'Generating {N_LHS} LHS points in {ndim}D box...')
    _lhs_sampler = LHS(xlimits=np.column_stack([ellipse_lo, ellipse_hi]))
    lhs_phys     = _lhs_sampler(N_LHS)
    lhs_phys     = np.clip(lhs_phys, _p2_lo, _p2_hi)
    n_inside     = N_LHS
    lhs_u    = np.clip(
        np.array([inverse_ellipse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
    )
    log_densities = np.full(n_inside, -np.inf)
    i_start = 0
    # Persist the generated points immediately so a walltime kill mid-eval
    # can resume against the *same* LHS draw instead of regenerating it.
    with open(ckptpath, 'wb') as f:
        pickle.dump({
            'lhs_phys': lhs_phys, 'lhs_u': lhs_u,
            'log_densities': log_densities,
            'n_inside': n_inside, 'next_idx': i_start,
        }, f)

# ── Evaluate coherent f-statistic ──────────────────────────────────────────────

print(f'Evaluating coherent f-statistic on {n_inside} points (starting at {i_start})...')
t0 = time.time()

for i in range(i_start, n_inside):
    if i % 500 == 0 and i > i_start:
        elapsed = time.time() - t0
        rate = (i - i_start) / elapsed
        eta  = (n_inside - i) / rate
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    try:
        ld = f_score(lhs_phys[i])
        log_densities[i] = ld if np.isfinite(ld) else -np.inf
    except Exception:
        log_densities[i] = -np.inf

    if (i + 1) % CKPT_EVERY == 0:
        with open(ckptpath, 'wb') as f:
            pickle.dump({
                'lhs_phys': lhs_phys, 'lhs_u': lhs_u,
                'log_densities': log_densities,
                'n_inside': n_inside, 'next_idx': i + 1,
            }, f)

elapsed = time.time() - t0
print(f'Done. {n_inside - i_start} evals in {elapsed:.0f}s ({(n_inside - i_start)/elapsed:.1f}/s)')

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

if os.path.exists(ckptpath):
    os.remove(ckptpath)
