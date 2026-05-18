"""
Precompute LHS grid inside 3-sigma ellipsoid for paris3.
Saves unit-cube points, physical points, log_densities, and prior bounds
so paris3 can load them without re-evaluating.
"""
import numpy as np
import pickle
import time
import os
import sys

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from smt.sampling_methods import LHS

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_pure
import parismc

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration
use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 12/12   # 1 year

print(f"Using dt = {dt} seconds, T = {T} years")

inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs_comb = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

print("Creating GenerateEMRIWaveform class...")
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_comb,
    use_gpu=use_gpu
)
waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_sep,
    use_gpu=use_gpu
)
print('Done initializing waveform generator.')

gwf = GWfuncs.GravWaveAnalysis(T, dt)

# Source parameters
m1 = 1e6; m2 = 1e1; a = 0.7; p0 = 9; e0 = 0.4
xI0 = 1.0; dist = 1.8
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

n_vals = np.arange(-1, 6)
ell = 2

loglike_obj = loglike_pure.LogLikePure(
    params_star, waveform_gen_comb, gwf, verbose=False,
    waveform_gen_sep=waveform_gen_sep, ell=ell, n_vals=n_vals, M_mode=None
)
print('Done initializing loglike class.')

data = loglike_obj.signal
data_snr = float(gwf.rhostat(data))
print('SNR:', data_snr)

# ── Load paris2 to get ellipse center and covariance ─────────────────────────
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

def _stub_log_density(params):
    raise RuntimeError("stub")

def _stub_prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

import __main__
__main__.log_density = _stub_log_density
__main__.prior_transform = _stub_prior_transform

print('Loading paris2 sampler...')
sampler_2 = parismc.Sampler.load_state(
    '/scratch/e1498138/paris2/int_3mth_snr32_1/sampler_state.pkl'
)

all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = _stub_prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'stage 2 maxld: {all_logden[maxld_idx]:.4f}')
print(f'stage 2 maxld point: {mu_center}')

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx_rs].T)
print('stage 2 posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_2, samples_p2, weights_p2, idx_rs

# ── Ellipsoid bounding box (hypercube prior) ─────────────────────────────────
N_SIGMA_PRIOR = 3.0
sigma_diag = np.sqrt(np.diag(cov_posterior))
ellipse_lo = np.clip(mu_center - N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi = np.clip(mu_center + N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)

print(f'Ellipse prior ({N_SIGMA_PRIOR:.0f}σ) bounds:')
param_names = ['logm1', 'logm2', 'a', 'p0', 'e0']
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  (mu={mu_center[i]:.5f})')

def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ── Generate LHS in Cholesky space, filter by sphere ─────────────────────────
ndim = 5
N_LHS = int(5e5)
_L = np.linalg.cholesky(cov_posterior)

print(f'Generating {N_LHS} LHS points in Cholesky space...')
_lhs_sampler = LHS(xlimits=np.column_stack([-np.ones(ndim), np.ones(ndim)]))
lhs_z_raw = _lhs_sampler(N_LHS)
sphere_mask = np.sum(lhs_z_raw ** 2, axis=1) <= 1.0
lhs_z_inside = lhs_z_raw[sphere_mask]
n_inside = sphere_mask.sum()
print(f'LHS points inside ellipse: {n_inside} / {N_LHS} ({100*sphere_mask.mean():.1f}%)')

# Transform to physical space
lhs_phys = mu_center + N_SIGMA_PRIOR * (_L @ lhs_z_inside.T).T
# Transform to unit-cube space
lhs_u = np.clip(
    np.array([inverse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)

# ── Evaluate log_density ─────────────────────────────────────────────────────
print(f'Evaluating log_density on {n_inside} points...')

log_densities = np.full(n_inside, -np.inf)
t0 = time.time()
for i in range(n_inside):
    if i % 500 == 0:
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (n_inside - i) / rate if rate > 0 else 0
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    logm1, logm2, a_i, p0_i, e0_i = lhs_phys[i]
    try:
        ld = loglike_obj(np.array([
            10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
        ]))
        log_densities[i] = ld
    except Exception as e:
        log_densities[i] = -np.inf

elapsed = time.time() - t0
print(f'Done. {n_inside} evals in {elapsed:.0f}s ({n_inside/elapsed:.1f}/s)')

n_finite = np.isfinite(log_densities).sum()
print(f'Finite log_densities: {n_finite} / {n_inside}')
if n_finite > 0:
    print(f'Max log_density: {np.max(log_densities[np.isfinite(log_densities)]):.6f}')
    print(f'Mean log_density (finite): {np.mean(log_densities[np.isfinite(log_densities)]):.6f}')

# ── Save ─────────────────────────────────────────────────────────────────────
savepath = f'./precomputed_lhs_paris3_1yr_new_{N_LHS:.0e}.pkl'
save_data = {
    'lhs_u': lhs_u,                    # unit-cube points (for external_lhs_points)
    'lhs_phys': lhs_phys,              # physical-space points
    'log_densities': log_densities,     # for external_lhs_log_densities
    'ellipse_lo': ellipse_lo,
    'ellipse_hi': ellipse_hi,
    'mu_center': mu_center,
    'cov_posterior': cov_posterior,
    'N_SIGMA_PRIOR': N_SIGMA_PRIOR,
    'N_LHS': N_LHS,
    'n_inside': n_inside,
    'T': T,
    'dt': dt,
    'SNR': data_snr,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
