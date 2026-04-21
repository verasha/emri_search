"""
Greedy hill-climber using loglike_pure (no phase-max, no time-max).

Reads seed index from --seed-idx (or PBS_ARRAY_INDEX env var).
Starts from the N-th best precomputed LHS point.
cov_prop = 0.01 * cov_prior (paris2 posterior covariance).
"""
import numpy as np
import pickle
import time
import os
import sys
import argparse

# ── Seed index from PBS array or CLI ─────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed-idx', type=int, default=None)
args, _ = parser.parse_known_args()

if args.seed_idx is not None:
    SEED_IDX = args.seed_idx
elif 'PBS_ARRAY_INDEX' in os.environ:
    SEED_IDX = int(os.environ['PBS_ARRAY_INDEX'])
else:
    SEED_IDX = 0

print(f"[seed_idx={SEED_IDX}] Starting")

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

os.chdir('/nfs/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/emri_search/work/')

import GWfuncs
import loglike_pure_hopper as loglike_pure

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("warning")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 12/12

inspiral_kwargs = {"func": 'KerrEccEqFlux', "DENSE_STEPPING": 0, "include_minus_m": False}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs_comb = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep  = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

print(f"[{SEED_IDX}] Initializing waveform generators...")
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

print(f"[{SEED_IDX}] Initializing loglike_pure...")
loglike_obj = loglike_pure.LogLikePure(
    params_star, waveform_gen_comb, gwf, verbose=False,
    waveform_gen_sep=waveform_gen_sep, ell=ell, n_vals=n_vals, M_mode=None
)
print(f"[{SEED_IDX}] SNR: {float(gwf.rhostat(loglike_obj.signal)):.4f}")

# ── Load precomputed LHS ──────────────────────────────────────────────────────
pkl_path = '/nfs/home/svu/e1498138/emri_search/work/search/precomputed_lhs_paris3_1yr_1e+05.pkl'
with open(pkl_path, 'rb') as f:
    lhs_data = pickle.load(f)

lhs_phys      = lhs_data['lhs_phys']
log_densities = lhs_data['log_densities']
cov_prior     = lhs_data['cov_posterior']
ellipse_lo    = lhs_data['ellipse_lo']
ellipse_hi    = lhs_data['ellipse_hi']

# Top-10 seeds sorted by logden descending
top10_idx = np.argsort(log_densities)[::-1][:10]
start_idx = top10_idx[SEED_IDX]
p_max     = lhs_phys[start_idx].copy()

print(f"[{SEED_IDX}] Starting from LHS rank {SEED_IDX}: {p_max}  logden={log_densities[start_idx]:.4f}")

# ── Proposal covariance ───────────────────────────────────────────────────────
SCALE    = 0.01
cov_prop = SCALE * cov_prior
print(f"[{SEED_IDX}] cov_prop diag sigma: {np.sqrt(np.diag(cov_prop))}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def eval_loglike(p):
    logm1, logm2, a_i, p0_i, e0_i = p
    return loglike_obj(np.array([
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
    ]))

def in_prior(p):
    return np.all(p >= ellipse_lo) and np.all(p <= ellipse_hi)

# ── Evaluate starting point ───────────────────────────────────────────────────
print(f"[{SEED_IDX}] Evaluating start with pure loglike...")
logden_max = eval_loglike(p_max)
print(f"[{SEED_IDX}] Starting logden (pure): {logden_max:.4f}")

# ── Greedy hill-climb ─────────────────────────────────────────────────────────
N_ITER      = 10_000
print_every = 500
rng         = np.random.default_rng(SEED_IDX)

history_logden = [logden_max]
history_params = [p_max.copy()]
n_accept = 0

print(f"[{SEED_IDX}] Running {N_ITER} iterations...")
t0 = time.time()

for i in range(1, N_ITER + 1):
    p_prop = rng.multivariate_normal(p_max, cov_prop)

    if not in_prior(p_prop):
        continue

    try:
        logden_prop = eval_loglike(p_prop)
    except Exception:
        continue

    if logden_prop > logden_max:
        logden_max = logden_prop
        p_max = p_prop.copy()
        n_accept += 1
        history_logden.append(logden_max)
        history_params.append(p_max.copy())

    if i % print_every == 0:
        elapsed = time.time() - t0
        print(f"[{SEED_IDX}]  iter={i:6d}  max_logden={logden_max:.6f}"
              f"  accepts={n_accept}  elapsed={elapsed:.0f}s")

elapsed = time.time() - t0
print(f"[{SEED_IDX}] Done. {N_ITER} iters in {elapsed:.0f}s")
print(f"[{SEED_IDX}] Final max logden: {logden_max:.6f}")
print(f"[{SEED_IDX}] Final p_max: {p_max}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('/nfs/home/svu/e1498138/emri_search/work/search/greedy_pure_results', exist_ok=True)
savepath = f'/nfs/home/svu/e1498138/emri_search/work/search/greedy_pure_results/seed{SEED_IDX:02d}.pkl'
results = {
    'seed_idx':         SEED_IDX,
    'history_logden':   np.array(history_logden),
    'history_params':   np.array(history_params),
    'p_max_final':      p_max,
    'logden_max_final': logden_max,
    'cov_prop':         cov_prop,
    'SCALE':            SCALE,
    'N_ITER':           N_ITER,
    'n_accept':         n_accept,
}
with open(savepath, 'wb') as f:
    pickle.dump(results, f)
print(f"[{SEED_IDX}] Saved to {savepath}")
