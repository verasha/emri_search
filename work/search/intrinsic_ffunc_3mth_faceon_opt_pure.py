import numpy as np
import os
import sys
from scipy.optimize import minimize

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

import GWfuncs
import loglike_pure   # NON-timemax, NON-phasemax
import parismc

# ─────────────────────────────────────────────
# Waveform / loglike setup  (same as paris2)
# ─────────────────────────────────────────────

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu       = True
force_backend = "cuda12x"
dt = 10
T  = 3/12
print(f"Using dt={dt}s, T={T} years")

inspiral_kwargs  = {"func": 'KerrEccEqFlux', "DENSE_STEPPING": 0, "include_minus_m": False}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs       = {"force_backend": force_backend}
sum_kwargs_comb  = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep   = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

print('Initializing waveform generators...')
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, frame='detector',
    inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_comb, use_gpu=use_gpu
)
waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, frame='detector',
    inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_sep, use_gpu=use_gpu
)

gwf = GWfuncs.GravWaveAnalysis(T, dt)

m1 = 1e6;  m2 = 1e1;  a = 0.7;  p0 = 9;  e0 = 0.4
xI0 = 1.0; dist = 1.8
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true  = [np.log10(m1), np.log10(m2), a, p0, e0]

n_vals = np.arange(-1, 6)
ell    = 2

print('Initializing loglike_pure (non-timemax, non-phasemax)...')
loglike_obj = loglike_pure.LogLikePure(
    params_star, waveform_gen_comb, gwf,
    verbose=False, waveform_gen_sep=waveform_gen_sep,
    ell=ell, n_vals=n_vals, M_mode=None
)

data     = loglike_obj.signal
data_snr = float(gwf.rhostat(data))
print(f'SNR: {data_snr}')

# ─────────────────────────────────────────────
# Step 1: Load paris2_1 to reconstruct ellipse bounds
# (needed to unpickle paris3_2 whose prior_transform uses ellipse_lo/hi)
# ─────────────────────────────────────────────

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

print('Loading paris2_1 to reconstruct ellipse prior bounds...')

_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

def log_density(params):   # stub for unpickling paris2_1
    raise RuntimeError("stub")

def prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

sampler_p2_1 = parismc.Sampler.load_state(
    './intrinsic_ffunc_3mth_snr32_paris2/sampler_state.pkl'
)
all_pts_u_p2  = sampler_p2_1.searched_points_list[0]
all_logden_p2 = sampler_p2_1.searched_log_densities_list[0]
mu_center     = prior_transform(all_pts_u_p2[np.argmax(all_logden_p2)].reshape(1, -1))[0]
print(f'Paris2_1 maxld point (phys): {mu_center}')

# Posterior covariance from paris2_1 importance-weight resampling
samples_p2_1, weights_p2_1 = sampler_p2_1.get_samples_with_weights(flatten=True)
weights_p2_1 = weights_p2_1 / weights_p2_1.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2_1), size=50_000, replace=True, p=weights_p2_1)
cov_posterior = np.cov(samples_p2_1[idx_rs].T)
del sampler_p2_1, samples_p2_1, weights_p2_1, idx_rs

# Reconstruct paris3 ellipse bounds (same as paris3.py)
N_SIGMA_PRIOR = 3.0
sigma_diag  = np.sqrt(np.diag(cov_posterior))
ellipse_lo  = np.clip(mu_center - N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi  = np.clip(mu_center + N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
print(f'Ellipse lo: {ellipse_lo}')
print(f'Ellipse hi: {ellipse_hi}')

# Redefine prior_transform for paris3 tight box (needed to unpickle paris3_2)
def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

# ─────────────────────────────────────────────
# Step 2: Load paris3_2 → extract best (maxld) point
# ─────────────────────────────────────────────

# Starting point: anneal12 best point (closer to true params than paris3_2)
best_fit = np.array([6.00556816, 1.00404448, 0.71049625, 8.94018508, 0.39701817])
print(f'Starting point (anneal12 best): {best_fit}')

# ─────────────────────────────────────────────
# log_density with early-stop at target SNR
# ─────────────────────────────────────────────

_TARGET_LOGLIKE = 5.907  # loglike_pure at true params (< data_snr due to multi-mode beta*chi_sq correction)
_EARLY_STOP_HIT = False

def log_density(params):
    global _EARLY_STOP_HIT
    params = np.asarray(params)

    def eval_one(x):
        global _EARLY_STOP_HIT
        if _EARLY_STOP_HIT:
            return float('-inf')
        try:
            logm1, logm2, a_i, p0_i, e0_i = x
            fstat = loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            return float('-inf')

        if fstat >= _TARGET_LOGLIKE:
            _EARLY_STOP_HIT = True
            print(f'[EARLY-STOP] loglike {fstat:.6f} >= target {_TARGET_LOGLIKE:.6f}; stopping.', flush=True)

        return fstat

    if params.ndim == 1:
        return eval_one(params)
    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = eval_one(params[i])
    return out

# ─────────────────────────────────────────────
# Nelder-Mead optimization
# ─────────────────────────────────────────────

print(f'\nStarting point (paris3_2 best): {best_fit}')
print(f'loglike at start: {log_density(best_fit):.6f}')
print(f'Target loglike  : {_TARGET_LOGLIKE:.6f}')
print(f'True point      : {param_true}')

_best_seen   = [best_fit.copy()]
_best_logden = [log_density(best_fit)]

def neg_logden(x):
    global _EARLY_STOP_HIT
    # enforce ellipse bounds — prevent wandering into oscillatory dead zones
    if np.any(x < ellipse_lo) or np.any(x > ellipse_hi):
        return 1e10
    val = log_density(x)
    if np.isfinite(val) and val > _best_logden[0]:
        _best_logden[0] = val
        _best_seen[0]   = x.copy()
        print(f'  [opt] new best: {val:.8f}  params: {x}', flush=True)
    return -val if np.isfinite(val) else 1e10

# Small initial simplex — steps must be < one phase oscillation period
step = np.array([1e-3, 1e-3, 1e-3, 5e-3, 1e-3])
initial_simplex = np.vstack([best_fit, best_fit + np.diag(step)])

print('\nRunning Nelder-Mead...')
result = minimize(
    neg_logden, x0=best_fit, method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000,
             'adaptive': True, 'initial_simplex': initial_simplex}
)

best_fit_new = _best_seen[0]
logden_new   = _best_logden[0]

print(f'\n=== Result ===')
print(f'Peak : {best_fit_new}')
print(f'loglike: {logden_new:.8f}')
print(f'Target : {_TARGET_LOGLIKE:.8f}')
print(f'True   : {param_true}')
print(f'Diff from true: {best_fit_new - np.array(param_true)}')
