import numpy as np
import os
import sys

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_pure
import parismc

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

from misc import compute_fisher_parallelotope
from lisatools.sensitivity import CornishLISASens

# ─────────────────────────────────────────────
# Waveform / loglike setup  (same as paris3)
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

print('Initializing loglike_pure...')
loglike_obj = loglike_pure.LogLikePure(
    params_star, waveform_gen_comb, gwf,
    verbose=False, waveform_gen_sep=waveform_gen_sep,
    ell=ell, n_vals=n_vals, M_mode=None
)

data     = loglike_obj.signal
data_snr = float(gwf.rhostat(data))
print(f'SNR: {data_snr}')

_TARGET_LOGLIKE = 5.907  # loglike_pure at true params (< data_snr due to multi-mode beta*chi_sq correction)
_EARLY_STOP_HIT = False

# ─────────────────────────────────────────────
# Fisher parallelotope prior
# compute_fisher_parallelotope uses 'channels' and 'noise_kwargs' from
# enclosing scope — define them here before calling
# ─────────────────────────────────────────────

channels     = ["A"]
noise_kwargs = {'sens_fn': CornishLISASens, 'return_type': 'PSD'}

N_SIGMA_PRIOR = 3.0

_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

ctx = {'T': T, 'dt': dt}

# anneal12 best point (log-mass space) → convert to linear for Fisher
_a12_best = np.array([6.00556816, 1.00404448, 0.71049625, 8.94018508, 0.39701817])
fisher_params_a12 = list(params_star)
fisher_params_a12[0] = 10**_a12_best[0]   # m1
fisher_params_a12[1] = 10**_a12_best[1]   # m2
fisher_params_a12[2] = _a12_best[2]       # a
fisher_params_a12[3] = _a12_best[3]       # p0
fisher_params_a12[4] = _a12_best[4]       # e0

print('Computing Fisher matrix via StableEMRIFisher (centred on anneal12 best)...')
Q, b, meta = compute_fisher_parallelotope(
    ctx             = ctx,
    fisher_params   = fisher_params_a12,          # full 14-param, anneal12 intrinsics
    params_to_infer = ['m1', 'm2', 'a', 'p0', 'e0'],
    additional_kwargs = {},
    use_gpu         = True,
    _TARGET_SNR     = data_snr,
    prior_sigma_range = N_SIGMA_PRIOR,
    using_evec      = False,                      # axis-aligned box, no SNR scaling needed
)
print(f'Fisher meta: {meta}')

# b[i] = N_SIGMA_PRIOR * sigma_i  in (m1, m2, a, p0, e0) linear space
# Convert mass parameters to log10 space via Jacobian: sigma(log10 m) = sigma(m) / (m * ln10)
b_log = b.copy()
b_log[0] = b[0] / (m1 * np.log(10))
b_log[1] = b[1] / (m2 * np.log(10))

sigma_fisher = b_log / N_SIGMA_PRIOR
print(f'Fisher 1-sigma (log-mass space): {sigma_fisher}')

# Prior centered on anneal12 best point (same as Fisher evaluation point)
mu_center = _a12_best.copy()

ellipse_lo = np.clip(mu_center - b_log, _p2_lo, _p2_hi)
ellipse_hi = np.clip(mu_center + b_log, _p2_lo, _p2_hi)

# Diagonal covariance from Fisher (no cross-terms for using_evec=False)
cov_posterior = np.diag(sigma_fisher**2)
cov_inv       = np.diag(1.0 / sigma_fisher**2)

# Sanity check
param_true_phys = np.array(param_true, dtype=float)
diff_true = param_true_phys - mu_center   # should be zero (centered on true)
maha_true = np.sqrt(diff_true @ cov_inv @ diff_true)
print(f'Mahalanobis distance (center -> true point): {maha_true:.3f}σ  (should be 0)')

print(f'Ellipse prior ({N_SIGMA_PRIOR:.0f}σ) bounds:')
pnames = ['logm1', 'logm2', 'a', 'p0', 'e0']
for i, name in enumerate(pnames):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  sigma={sigma_fisher[i]:.5f}')

# ─────────────────────────────────────────────
# prior_transform for tight box
# ─────────────────────────────────────────────

def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ─────────────────────────────────────────────
# log_density: NON-phasemax + ellipse guard
# ─────────────────────────────────────────────

def log_density(params):
    global _EARLY_STOP_HIT
    params = np.asarray(params)
    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)

    for i in range(n_samples):
        if _EARLY_STOP_HIT:
            log_likes[i] = -np.inf
            continue

        logm1, logm2, a_i, p0_i, e0_i = params[i]

        diff  = params[i] - mu_center
        maha2 = diff @ cov_inv @ diff
        if maha2 > N_SIGMA_PRIOR ** 2:
            log_likes[i] = -np.inf
            continue

        try:
            loglike = loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            loglike = -np.inf

        if np.isfinite(loglike) and loglike >= _TARGET_LOGLIKE:
            _EARLY_STOP_HIT = True
            print(f'[EARLY-STOP] loglike {loglike:.6f} >= target {_TARGET_LOGLIKE:.6f}', flush=True)

        log_likes[i] = loglike

    return log_likes

print('Testing log_density at true params...')
print('  logden at true:', log_density(mu_center.reshape(1, -1)))

# ─────────────────────────────────────────────
# PARIS sampler setup
# ─────────────────────────────────────────────

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
    keep_dead_processes=True
)

ndim   = 5
n_seed = 1

new_J     = np.diag(ellipse_hi - ellipse_lo)
new_J_inv = np.diag(1.0 / (ellipse_hi - ellipse_lo))
init_cov  = new_J_inv @ cov_posterior @ new_J_inv.T
print('Init cov (unit-cube) diagonal:', np.diag(init_cov))

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=[init_cov],
    prior_transform=prior_transform,
    config=config
)

start_u    = inverse_prior_transform(mu_center)
start_logd = log_density(mu_center.reshape(1, -1))
print('Starting point (phys):', mu_center)
print('Starting log_density: ', start_logd)

def callback(sampler, i):
    global _EARLY_STOP_HIT
    if _EARLY_STOP_HIT:
        sampler.stop_sampling = True
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

print('Running paris3_try sampling...')
sampler.run_sampling(
    num_iterations=int(2e5),
    savepath='./intrinsic_ffunc_3mth_snr32_paris3_try',
    print_iter=100,
    callback=callback,
    external_lhs_points=start_u.reshape(1, -1),
    external_lhs_log_densities=start_logd,
    stop_max_ld_stable_iters=int(1e4)
)
print('Done.')
