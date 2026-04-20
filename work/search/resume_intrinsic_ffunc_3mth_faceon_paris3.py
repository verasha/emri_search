"""
Resume paris3 sampler from saved state.
State: ./intrinsic_ffunc_3mth_snr32_paris3_1yr/sampler_state.pkl
"""
import numpy as np
import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
import os
import sys

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_pure
import parismc

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 12/12

inspiral_kwargs = {"func": 'KerrEccEqFlux', "DENSE_STEPPING": 0, "include_minus_m": False}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs_comb = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep  = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

print("Initializing waveform generators...")
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

m1 = 1e6; m2 = 1e1; a = 0.7; p0 = 9; e0 = 0.4
xI0 = 1.0; dist = 1.8
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5
params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

n_vals = np.arange(-1, 6)
ell = 2

print("Initializing loglike_pure...")
loglike_obj = loglike_pure.LogLikePure(
    params_star, waveform_gen_comb, gwf, verbose=False,
    waveform_gen_sep=waveform_gen_sep, ell=ell, n_vals=n_vals, M_mode=None
)
print(f"SNR: {float(gwf.rhostat(loglike_obj.signal)):.4f}")

def log_density(params):
    params = np.asarray(params)
    log_likes = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i = params[i]
        try:
            log_likes[i] = loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            log_likes[i] = -np.inf
    return log_likes

def prior_transform(u):
    logm1lim = [5.98699, 6.04277]
    logm2lim = [0.98935, 1.02067]
    alim     = [0.67449, 0.78962]
    p0lim    = [8.48613, 9.14781]
    e0lim    = [0.38373, 0.40714]
    transformed = np.zeros_like(u)
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    transformed[:, 2] = (alim[1]     - alim[0])     * u[:, 2] + alim[0]
    transformed[:, 3] = (p0lim[1]    - p0lim[0])    * u[:, 3] + p0lim[0]
    transformed[:, 4] = (e0lim[1]    - e0lim[0])    * u[:, 4] + e0lim[0]
    return transformed

# ── Load saved state ──────────────────────────────────────────────────────────
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

state_path = './intrinsic_ffunc_3mth_snr32_paris3_1yr/sampler_state.pkl'
print(f'Loading sampler state from {state_path}...')
if not os.path.isfile(state_path):
    print(f"State not found. Exiting.")
    exit(1)

sampler = parismc.Sampler.load_state(state_path)

sampler.log_density_func_original = log_density
if getattr(sampler, 'prior_transform', None) is not None:
    sampler.prior_transform = prior_transform
    sampler.log_density_func = sampler.transformed_log_density_func
else:
    sampler.log_density_func = sampler.log_density_func_original

print(f"Loaded. current_iter={getattr(sampler, 'current_iter', '?')}")
print(f"max_logden_list: {sampler.max_logden_list}")

def paris3_callback(sampler, i):
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

print('Resuming for 1e5 more iterations...')
sampler.run_sampling(
    num_iterations=int(1e5),
    savepath='./intrinsic_ffunc_3mth_snr32_paris3_1yr_resume',
    print_iter=100,
    callback=paris3_callback,
    stop_max_ld_stable_iters=int(1e4)
)
print('Done.')
