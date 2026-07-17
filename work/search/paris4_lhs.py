"""
Precompute LHS grid inside 3-sigma ellipsoid for paris3 noise search.
Loads paris3_noise/int_3mth_noise_6 to get ellipse center and covariance.
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
T = 12 / 12
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


_p2_lo = np.array([5.94889, 0.99336, 0.62652, 8.74104, 0.37427])
_p2_hi = np.array([6.03054, 1.13038, 0.99000, 9.68094, 0.41300])

def _stub_prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

def log_density(params):
    raise RuntimeError("stub")

def prior_transform(u):
    return _stub_prior_transform(u)

import __main__
__main__.log_density    = log_density
__main__.prior_transform = prior_transform

paris3_noise_path = '/scratch/e1498138/paris3_sc/int_1yr_s6/sampler_state.pkl'
print(f'Loading paris3_noise sampler from {paris3_noise_path}...')
sampler_2 = parismc.Sampler.load_state(paris3_noise_path)

all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = _stub_prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'paris3_noise maxld: {all_logden[maxld_idx]:.4f}')
print(f'paris3_noise maxld point: {mu_center}')

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx_rs].T)
print('paris3_noise posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_2, samples_p2, weights_p2, idx_rs

# ── Ellipsoid bounding box ─────────────────────────────────────────────────────
# e0 has high Fisher information → tight posterior at the (possibly wrong) mode.
# Use N_SIGMA_E0 >> N_SIGMA_OTHER so true e0=0.4 is safely inside the LHS volume.

param_names   = ['logm1', 'logm2', 'a', 'p0', 'e0']
N_SIGMA_1 = 3.0
N_SIGMA_2    = 3.0
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

# ── Generate LHS in Cholesky space, filter by unit sphere ─────────────────────

ndim  = 5
N_LHS = int(5e5)
_L     = np.linalg.cholesky(cov_posterior)
_scale = np.diag(N_SIGMA_PER_DIM)  # per-dim N_sigma applied after Cholesky

print(f'Generating {N_LHS} LHS points in {ndim}D Cholesky space...')
_lhs_sampler = LHS(xlimits=np.column_stack([-np.ones(ndim), np.ones(ndim)]))
lhs_z_raw    = _lhs_sampler(N_LHS)
sphere_mask  = np.sum(lhs_z_raw ** 2, axis=1) <= 1.0
lhs_z_inside = lhs_z_raw[sphere_mask]
n_inside     = sphere_mask.sum()
print(f'LHS points inside unit sphere: {n_inside} / {N_LHS} ({100*sphere_mask.mean():.1f}%)')

lhs_phys = mu_center + (_scale @ _L @ lhs_z_inside.T).T
lhs_phys = np.clip(lhs_phys, _p2_lo, _p2_hi)
lhs_u    = np.clip(
    np.array([inverse_ellipse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)

# ── Evaluate log_density ───────────────────────────────────────────────────────

print(f'Evaluating log_density on {n_inside} points...')
log_densities = np.full(n_inside, -np.inf)
t0 = time.time()

for i in range(n_inside):
    if i % 500 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (n_inside - i) / rate
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

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
print(f'Done. {n_inside} evals in {elapsed:.0f}s ({n_inside/elapsed:.1f}/s)')

n_finite = np.isfinite(log_densities).sum()
print(f'Finite log_densities: {n_finite} / {n_inside}')
if n_finite > 0:
    print(f'Max log_density: {np.max(log_densities[np.isfinite(log_densities)]):.6f}')
    print(f'Mean log_density (finite): {np.mean(log_densities[np.isfinite(log_densities)]):.6f}')

# ── Save ──────────────────────────────────────────────────────────────────────

savepath = f'/scratch/e1498138/paris4_noise/lhs_f.pkl'
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
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
