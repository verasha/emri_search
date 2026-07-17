"""
Precompute LHS grid inside a UNIFORM BOX for paris3 noise search.

Unlike paris3_lhs.py (which LHS-samples a cube, rejects to the unit sphere, then
rotates by the posterior Cholesky into an ellipsoid), this version samples
uniformly and directly inside the per-dim box  mu +/- N_sigma * sigma.

Motivation: the paris2 posterior mode is strongly biased from the true injection
(true logm2 is ~13.5 sigma from the mode, true a is ~10 sigma).  A per-dim box
needs N_sigma >= ~13.6 to contain the truth; the correlated ellipsoid needs an
even larger radius.  A box also uses ALL N_LHS points (no ~83% sphere rejection)
and preserves the LHS space-filling stratification in every dimension.

Saves unit-cube points, physical points, log_densities, and prior bounds.
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
T = 3 / 12
print(f"dt={dt}s  T={T}yr  TDI gen={tdi_gen}")

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

m1, m2, a, p0, e0, xI0 = 1e6, 1e1, 0.7, 9.0, 0.4, 1.0
dist, qS, phiS, qK, phiK = 4.5, np.pi, 0., 0., 0.
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5
params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]

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

# ── Load paris2_noise/int_1yr_s12 for box center and covariance ────────────────

_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

def _stub_prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

def log_density(params):
    raise RuntimeError("stub")

def prior_transform(u):
    return _stub_prior_transform(u)

import __main__
__main__.log_density    = log_density
__main__.prior_transform = prior_transform

paris2_noise_path = '/scratch/e1498138/paris2_sc/int_1yr_s12/sampler_state.pkl'
print(f'Loading paris2_noise sampler from {paris2_noise_path}...')
sampler_2 = parismc.Sampler.load_state(paris2_noise_path)

all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = _stub_prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
paris2_maxld = float(all_logden[maxld_idx])
print(f'paris2_noise maxld: {paris2_maxld:.4f}')
print(f'paris2_noise maxld point: {mu_center}')

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx_rs].T)
sigma_diag = np.sqrt(np.diag(cov_posterior))
print('paris2_noise posterior 1-sigma (diag):', sigma_diag)

del sampler_2, samples_p2, weights_p2, idx_rs

# ── Uniform box:  mu +/- N_sigma * sigma  (per dimension) ──────────────────────
# The paris2 mode is biased along the mass-ratio / spin degeneracy (a secondary
# likelihood mode): true logm2 is ~13.55 sigma out and true a is ~10.07 sigma out,
# while logm1/p0/e0 are only 1-2 sigma off.  So widen N_sigma ONLY in the two
# bias-prone dims (logm2, a) and keep the well-behaved dims tight -- otherwise a
# global large N_sigma blows a and p0 out to most of the physical range.
#                            logm1  logm2   a     p0    e0
param_names     = ['logm1', 'logm2', 'a', 'p0', 'e0']
N_SIGMA_PER_DIM = np.array([   6.0,   15.0, 15.0,  6.0,  6.0])

box_lo = np.clip(mu_center - N_SIGMA_PER_DIM * sigma_diag, _p2_lo, _p2_hi)
box_hi = np.clip(mu_center + N_SIGMA_PER_DIM * sigma_diag, _p2_lo, _p2_hi)

print('Uniform box prior bounds (per-dim sigma):')
for i, name in enumerate(param_names):
    print(f'  {name}: [{box_lo[i]:.5f}, {box_hi[i]:.5f}]  '
          f'mu={mu_center[i]:.5f}  sigma={sigma_diag[i]:.5f}  N_sigma={N_SIGMA_PER_DIM[i]:.0f}')

def box_prior_transform(u):
    return box_lo + (box_hi - box_lo) * u

def inverse_box_prior_transform(params):
    return (np.asarray(params) - box_lo) / (box_hi - box_lo)

# ── Generate LHS directly on the uniform box (no sphere rejection) ─────────────

ndim  = 5
N_LHS = int(5e5)

print(f'Generating {N_LHS} LHS points in {ndim}D uniform box...')
_lhs_sampler = LHS(xlimits=np.column_stack([np.zeros(ndim), np.ones(ndim)]))
lhs_u    = _lhs_sampler(N_LHS)                 # [0,1]^5, all points used
lhs_phys = box_prior_transform(lhs_u)
lhs_phys = np.clip(lhs_phys, _p2_lo, _p2_hi)
n_points = N_LHS
print(f'LHS points on box: {n_points} (all used, no rejection)')

# ── Evaluate log_density ───────────────────────────────────────────────────────

print(f'Evaluating log_density on {n_points} points...')
log_densities = np.full(n_points, -np.inf)
t0 = time.time()

for i in range(n_points):
    if i % 500 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (n_points - i) / rate
        print(f'  [{i}/{n_points}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    logm1, logm2, a_i, p0_i, e0_i = lhs_phys[i]
    try:
        ld = loglike_obj(np.array([
            10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
        ]))
        log_densities[i] = ld
    except Exception:
        log_densities[i] = -np.inf

elapsed = time.time() - t0
print(f'Done. {n_points} evals in {elapsed:.0f}s ({n_points/elapsed:.1f}/s)')

n_finite = np.isfinite(log_densities).sum()
print(f'Finite log_densities: {n_finite} / {n_points}')
if n_finite > 0:
    ld_max = np.max(log_densities[np.isfinite(log_densities)])
    print(f'Max log_density: {ld_max:.6f}')
    print(f'Mean log_density (finite): {np.mean(log_densities[np.isfinite(log_densities)]):.6f}')

    # ── Did the wide box find anything beating the (biased) paris2 mode? ────────
    print(f'\nparis2 mode log_density: {paris2_maxld:.4f}')
    print(f'best LHS log_density:    {ld_max:.4f}   '
          f'(delta = {ld_max - paris2_maxld:+.4f})')
    if ld_max > paris2_maxld:
        print('  -> LHS BEATS the paris2 mode: paris2 mode is a SECONDARY mode.')
    else:
        print('  -> nothing beats the paris2 mode within this box.')

    true_phys = np.array([6.0, 1.0, 0.7, 9.0, 0.4])   # [logm1, logm2, a, p0, e0]
    topk = 10
    order = np.argsort(log_densities)[::-1][:topk]
    print(f'\nTop {topk} points (logm1, logm2, a, p0, e0)  [true = {true_phys}]:')
    for rank, idx in enumerate(order):
        p = lhs_phys[idx]
        print(f'  #{rank+1:2d}  ld={log_densities[idx]:10.4f}  '
              f'[{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.4f} {p[4]:.4f}]')

# ── Save ──────────────────────────────────────────────────────────────────────

savepath = f'/scratch/e1498138/paris3_noise/lhs_box.pkl'
os.makedirs(os.path.dirname(savepath), exist_ok=True)
save_data = {
    'lhs_u':          lhs_u,
    'lhs_phys':       lhs_phys,
    'log_densities':  log_densities,
    'box_lo':         box_lo,
    'box_hi':         box_hi,
    # aliases for downstream code that expects the ellipse_* keys
    'ellipse_lo':     box_lo,
    'ellipse_hi':     box_hi,
    'mu_center':      mu_center,
    'cov_posterior':  cov_posterior,
    'sigma_diag':     sigma_diag,
    'N_SIGMA_PER_DIM': N_SIGMA_PER_DIM,
    'paris2_maxld':   paris2_maxld,
    'N_LHS':          N_LHS,
    'n_points':       n_points,
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
