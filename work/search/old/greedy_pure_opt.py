"""
Greedy hill-climber using loglike_pure

Proposal: N(p_max, cov_prop) where cov_prop is loaded from previous result.
"""
import numpy as np
import pickle
import time
import os
import sys

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

os.chdir('/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
# import loglike_pure_hopper as loglike_pure
import loglike_pure 

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("warning")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = (12+6)/12

print(f"dt={dt}s, T={T}yr")

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

# Source parameters
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

data_snr = float(gwf.rhostat(loglike_obj.signal))
print(f"SNR: {data_snr:.4f}")

# ── Load precomputed LHS for prior bounds ─────────────────────────────────────
pkl_path = '/home/svu/e1498138/localgit/FEWNEW/work/search/precomputed_lhs_paris3_1yr_1e+05.pkl'
print(f"Loading precomputed LHS from {pkl_path}...")
with open(pkl_path, 'rb') as f:
    lhs_data = pickle.load(f)

ellipse_lo = lhs_data['ellipse_lo']
ellipse_hi = lhs_data['ellipse_hi']

# ── Load result as starting point and proposal covariance ─────────────
# seed_path = '/home/svu/e1498138/localgit/FEWNEW/work/search/greedy_timeonly_results/seed08.pkl'
seed_path = '/home/svu/e1498138/localgit/FEWNEW/work/search/greedy_pure_paris3_results_1_25yr.pkl'
print(f"Loading prev results from {seed_path}...")
with open(seed_path, 'rb') as f:
    seed_data = pickle.load(f)

p_max      = seed_data['p_max_final'].copy()
cov_prop   = seed_data['cov_prop']
print(f"cov_prop (diag): {np.sqrt(np.diag(cov_prop))}")

# ── Loglike wrapper ───────────────────────────────────────────────────────────
def eval_loglike(phys_params):
    logm1, logm2, a_i, p0_i, e0_i = phys_params
    return loglike_obj(np.array([
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
    ]))

def in_prior(p):
    return np.all(p >= ellipse_lo) and np.all(p <= ellipse_hi)

logden_max = float(eval_loglike(p_max))
print(f"Point start: {p_max}")
print(f"Recomputed starting logden (pure): {logden_max:.4f}")

# ── Greedy hill-climb ─────────────────────────────────────────────────────────
N_ITER      = 10_000
print_every = 100
rng         = np.random.default_rng(42)

history_logden = [logden_max]
history_params = [p_max.copy()]
n_accept = 0

print(f"\nRunning greedy hill-climb for {N_ITER} iterations...")
t0 = time.time()

for i in range(1, N_ITER + 1):
    # Gaussian proposal centred on current best
    p_prop = rng.multivariate_normal(p_max, cov_prop)

    if not in_prior(p_prop):
        continue

    try:
        logden_prop = eval_loglike(p_prop)
    except Exception:
        continue

    # Greedy: accept only improvements
    if logden_prop > logden_max:
        logden_max = logden_prop
        p_max = p_prop.copy()
        n_accept += 1
        history_logden.append(logden_max)
        history_params.append(p_max.copy())

    if i % print_every == 0:
        elapsed = time.time() - t0
        print(f"  iter={i:6d}  max_logden={logden_max:.6f}  accepts={n_accept}"
              f"  p_max={p_max}  elapsed={elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\nDone. {N_ITER} iters in {elapsed:.0f}s ({N_ITER/elapsed:.1f}/s)")
print(f"Total accepts: {n_accept}")
print(f"Final max logden: {logden_max:.6f}")
print(f"Final p_max: {p_max}")

# ── Save results ──────────────────────────────────────────────────────────────
savepath = '/home/svu/e1498138/localgit/FEWNEW/work/search/greedy_pure_paris3_results_1_5yr.pkl'
results = {
    'history_logden': np.array(history_logden),
    'history_params': np.array(history_params),
    'p_max_final': p_max,
    'logden_max_final': logden_max,
    'cov_prop': cov_prop,
    'N_ITER': N_ITER,
    'n_accept': n_accept,
    'T': T,
    'dt': dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved to {savepath}")
