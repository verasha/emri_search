import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum


from few.utils.ylm import GetYlms

from few import get_file_manager

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

from few.utils.constants import YRSID_SI
from smt.sampling_methods import LHS


import os
import sys

# Changing directory to FEWNEW/work
# to import stuffs
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_pure   # NON-timemax, NON-phasemax
import parismc
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")


# GPU configuration
use_gpu = True
force_backend = "cuda12x"
dt = 10     # Time step
T = 3/12     # Total time

print(f"Using dt = {dt} seconds, T = {T} years")

inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0,
        "include_minus_m": False,
}

amplitude_kwargs = {
    "force_backend": force_backend
}

Ylm_kwargs = {
    "force_backend": force_backend,
}

sum_kwargs_comb = {
    "force_backend": force_backend,
    "pad_output": True,
}

sum_kwargs_sep = {
    "force_backend": force_backend,
    "pad_output": True,
    "separate_modes": True,
}

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

print("Creating GravWaveAnalysis class...")
gwf = GWfuncs.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")


# Source parameters
m1 = 1e6
m2 = 1e1
a = 0.7
p0 = 9
e0 = 0.4
xI0 = 1.0
dist = 1.8  # Gpc
qS = np.pi
phiS = 0.
qK =  0.
phiK = 0.
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

n_vals = np.arange(-1,6)
ell = 2

loglike_obj = loglike_pure.LogLikePure(
    params_star,
    waveform_gen_comb,
    gwf,
    verbose=False,
    waveform_gen_sep=waveform_gen_sep,
    ell=ell,
    n_vals=n_vals,
    M_mode=None
)

print('Done initializing loglike class.')

data     = loglike_obj.signal
data_snr = float(gwf.rhostat(data))
print('SNR:', data_snr)

_TARGET_LOGLIKE = data_snr # only stop once hit target
_EARLY_STOP_HIT = False

# LOAD PARIS STAGE 2

print('Loading paris2_1 sampler to get ellipse center and covariance...')

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

# prior bounds
_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

def log_density(params):      
    raise RuntimeError("stub")

def prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

sampler_2 = parismc.Sampler.load_state(
    './intrinsic_ffunc_3mth_snr32_anneal12/sampler_state.pkl'
)

S_ANNEAL = 30

# Max logden point from paris2_1 (in physical space)
all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'stage 2 maxld: {all_logden[maxld_idx]:.4f}')
print(f'stage 2 maxld point:    {mu_center}')

# Posterior covariance from paris2_1: importance-weight resample then np.cov
samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = S_ANNEAL* np.cov(samples_p2[idx_rs].T)  # physical-space posterior covariance
print('stage 2 posterior 1-sigma (diag, using S):', np.sqrt(np.diag(cov_posterior)))

del sampler_2, samples_p2, weights_p2, idx_rs  # free memory

# ─────────────────────────────────────────────
# Ellipsoidal prior definition
# ─────────────────────────────────────────────

N_SIGMA_PRIOR = 1.0   # ellipse radius in units of posterior sigma

# Tight bounding box: mu_center ± N_SIGMA_PRIOR * sigma, clipped to original prior
sigma_diag  = np.sqrt(np.diag(cov_posterior))
ellipse_lo  = np.clip(mu_center - N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi  = np.clip(mu_center + N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
cov_inv     = np.linalg.inv(cov_posterior)

# Sanity check: is the true point inside the ellipse?
param_true_phys = np.array([np.log10(m1), np.log10(m2), a, p0, e0])
diff_true = param_true_phys - mu_center
maha_true = np.sqrt(diff_true @ cov_inv @ diff_true)
print(f'Mahalanobis distance (maxld_pt -> true point): {maha_true:.3f}σ')
if maha_true > N_SIGMA_PRIOR:
    print(f'WARNING: true point is OUTSIDE the {N_SIGMA_PRIOR:.0f}σ ellipse!')
else:
    print(f'OK: true point is inside the {N_SIGMA_PRIOR:.0f}σ ellipse.')

print(f'Ellipse prior ({N_SIGMA_PRIOR:.0f}σ) bounds:')
param_names = ['logm1', 'logm2', 'a', 'p0', 'e0']
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  (mu={mu_center[i]:.5f})')

# Redefine prior_transform and its inverse for the tight box
def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_prior_transform(params):
    params = np.asarray(params)
    return (params - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ─────────────────────────────────────────────
# log_density: NON-phasemax  + ellipse guard
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

        # Ellipse guard: reject points outside N_SIGMA_PRIOR ellipse
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
            print(f'[EARLY-STOP] loglike {loglike:.6f} >= target {_TARGET_LOGLIKE:.6f}; future calls => -inf', flush=True)

        log_likes[i] = loglike

    return log_likes

print('Testing log_density at maxld center...')
print('  logden at mu_center:', log_density(mu_center.reshape(1, -1)))

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

# Initial proposal covariance: posterior covariance transformed to new unit-cube space
# new_J = diag(ellipse_hi - ellipse_lo); cov_unit = J^{-1} @ cov_phys @ J^{-T}
new_J       = np.diag(ellipse_hi - ellipse_lo)
new_J_inv   = np.diag(1.0 / (ellipse_hi - ellipse_lo))
init_cov    = new_J_inv @ cov_posterior @ new_J_inv.T
init_cov_list = [init_cov]

print('Init cov (unit-cube space) diagonal:', np.diag(init_cov))

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)

# Start from paris2 maxld point
start_u    = inverse_prior_transform(mu_center)
start_logd = log_density(mu_center.reshape(1, -1))
print('Starting point (phys):', mu_center)
print('Starting point (unit):', start_u)
print('Starting log_density: ', start_logd)

external_lhs_points        = start_u.reshape(1, -1)
external_lhs_log_densities = start_logd

def paris3_callback(sampler, i):
    global _EARLY_STOP_HIT
    if _EARLY_STOP_HIT:
        sampler.stop_sampling = True
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

print('Running paris3 sampling...')
sampler.run_sampling(
    num_iterations=int(2e5),
    savepath='./intrinsic_ffunc_3mth_snr32_paris3_4',
    print_iter=100,
    callback=paris3_callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
    stop_max_ld_stable_iters=int(1e4)
)
print('Done.')
