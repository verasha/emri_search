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
import parismc

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("warning")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 12/12

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

def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0] 
    log_likes = np.zeros(n_samples)


    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2

        try:
            loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
        except Exception:
            loglike = -np.inf
        log_likes[i] = loglike
        print(loglike)

    return log_likes

def prior_transform(u):
    u = np.asarray(u)
    out = np.zeros_like(u, dtype=float)

    logm1lim = [5.98699, 6.04277]
    logm2lim = [0.98935, 1.02067]
    alim     = [0.67449, 0.78962]
    p0lim    = [8.48613, 9.14781]
    e0lim    = [0.38373, 0.40714]

    out[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    out[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    out[:, 2] = (alim[1]     - alim[0])     * u[:, 2] + alim[0]
    out[:, 3] = (p0lim[1]    - p0lim[0])    * u[:, 3] + p0lim[0]
    out[:, 4] = (e0lim[1]    - e0lim[0])    * u[:, 4] + e0lim[0]
    return out


sampler_path = "/scratch/e1498138/localgit/FEWNEW/work/intrinsic_ffunc_3mth_snr32_paris3_1yr/sampler_state.pkl"

sampler = parismc.Sampler.load_state(sampler_path)

# find process with largest max log-density
best_proc = int(np.argmax(sampler.max_logden_list))

# best point in that process
lds = np.asarray(sampler.searched_log_densities_list[best_proc])
pts = np.asarray(sampler.searched_points_list[best_proc])
best_idx = int(np.argmax(lds))
p_max = prior_transform(pts[best_idx:best_idx+1])[0]

# fixed proposal covariance from parismc sampler (read once, not updated)
scales = np.array([
    6.04277 - 5.98699,
    1.02067 - 0.98935,
    0.78962 - 0.67449,
    9.14781 - 8.48613,
    0.40714 - 0.38373,
])
S = np.diag(scales)
SCALE = 1e-6
cov_prop = SCALE * S @ np.asarray(sampler.now_covariances[best_proc]) @ S

# seed_path = '/home/svu/e1498138/localgit/FEWNEW/work/search/greedy_timeonly_results/seed08.pkl'
# seed_path = '/home/svu/e1498138/localgit/FEWNEW/work/search/greedy_pure_results.pkl'
# print(f"Loading prev results from {seed_path}...")
# with open(seed_path, 'rb') as f:
#     seed_data = pickle.load(f)

# p_max      = seed_data['p_max_final'].copy()
# cov_prop   = seed_data['cov_prop']
print(f"cov_prop = {SCALE} * S@cov_parismc@S  (diag sigma: {np.sqrt(np.diag(cov_prop))})")

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
savepath = '/home/svu/e1498138/localgit/FEWNEW/work/search/greedy_pure_paris3_results_1yr.pkl'
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
