"""
Precompute LHS grid inside the paris2 posterior ellipsoid for the paris3 noise
search, scoring each point with the S6 semi-coherent statistic
(gwf.SNR_semicoherent(signal, h, N_seg=6)) instead of the coherent "pure" f-stat.

The coherent statistic peak is far narrower than the LHS grid spacing, so a
fully-coherent scan misses the signal. The semi-coherent statistic (6 segments)
broadens the peak in the intrinsic parameters, trading some SNR for a wider
basin of attraction that a grid/sampler can actually resolve.

Loads paris2_sc/int_1yr_s12 to get ellipse center and covariance.
Saves unit-cube points, physical points, S6 log_densities, and prior bounds.
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

# ── Semi-coherent statistic configuration ─────────────────────────────────────
N_SEG = 6  # S6: number of segments for the semi-coherent statistic

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

m1, m2, a, p0, e0, xI0 = 1e6, 1e1, 0.7, 9.0, 0.4, 1.0
dist, qS, phiS, qK, phiK = 4.5, np.pi, 0., 0., 0.
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5
params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]

n_vals = np.arange(-1, 6)
ell = 2

# LogLike is used only to build the injected data (signal + noise) in .signal.
# The S6 score below reads loglike_obj.signal directly and does not call
# loglike_obj(...), so mode selection here only affects the injection, not scoring.
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

# Injected data time series (signal + colored noise), time-domain (n_chan, N).
data_signal = loglike_obj.signal


def s6_score(phys_point):
    """S6 semi-coherent statistic at a physical point [logm1, logm2, a, p0, e0]."""
    logm1, logm2, a_i, p0_i, e0_i = phys_point
    h = gwf.xp.array(waveform_response(
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        xI0, dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt,
    ))
    return float(gwf.SNR_semicoherent(data_signal, h, N_seg=N_SEG))


# ── Load paris2_sc/int_1yr_s12 for ellipse center and covariance ──────────────

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
print(f'paris2_noise maxld: {all_logden[maxld_idx]:.4f}')
print(f'paris2_noise maxld point: {mu_center}')

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx_rs].T)
print('paris2_noise posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_2, samples_p2, weights_p2, idx_rs

# ── Ellipsoid bounding box ─────────────────────────────────────────────────────
# e0 has high Fisher information → tight posterior at the (possibly wrong) mode.
# Use N_SIGMA_E0 >> N_SIGMA_OTHER so true e0=0.4 is safely inside the LHS volume.

param_names   = ['logm1', 'logm2', 'a', 'p0', 'e0']
N_SIGMA_1 = 15.0
N_SIGMA_2    = 5.0
N_SIGMA_PER_DIM = np.array([N_SIGMA_2, N_SIGMA_1, N_SIGMA_1,
                             N_SIGMA_2, N_SIGMA_2])

sigma_diag = np.sqrt(np.diag(cov_posterior))
ellipse_lo = np.clip(mu_center - N_SIGMA_PER_DIM * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi = np.clip(mu_center + N_SIGMA_PER_DIM * sigma_diag, _p2_lo, _p2_hi)

print(f'Ellipse prior bounds (per-dim sigma):')
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  '
          f'mu={mu_center[i]:.5f}  sigma={sigma_diag[i]:.5f}  N_sigma={N_SIGMA_PER_DIM[i]:.0f}')

def ellipse_prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_ellipse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ── Generate LHS uniformly in the N-sigma box ─────────────────────────────────
# No ellipsoid truncation: points fill the axis-aligned box [ellipse_lo,
# ellipse_hi] (mu ± N_sigma * sigma_diag per dim), not the correlated ellipsoid.

ndim  = 5
N_LHS = int(5e5)

print(f'Generating {N_LHS} LHS points in {ndim}D N-sigma box...')
_lhs_sampler = LHS(xlimits=np.column_stack([ellipse_lo, ellipse_hi]))
lhs_phys     = _lhs_sampler(N_LHS)
lhs_phys     = np.clip(lhs_phys, _p2_lo, _p2_hi)
n_inside     = N_LHS
lhs_u    = np.clip(
    np.array([inverse_ellipse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)

# ── Evaluate S6 semi-coherent statistic ────────────────────────────────────────

print(f'Evaluating S{N_SEG} semi-coherent statistic on {n_inside} points...')
log_densities = np.full(n_inside, -np.inf)
t0 = time.time()

for i in range(n_inside):
    if i % 500 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (n_inside - i) / rate
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    try:
        ld = s6_score(lhs_phys[i])
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

savepath = f'/scratch/e1498138/paris3_noise/lhs_s{N_SEG}.pkl'
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
    'N_SIGMA_1':     N_SIGMA_1,
    'N_SIGMA_2':        N_SIGMA_2,
    'N_LHS':             N_LHS,
    'n_inside':       n_inside,
    'N_SEG':          N_SEG,
    'statistic':      f'SNR_semicoherent(N_seg={N_SEG})',
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
