"""
Per-evaluation timing benchmark for the coherent time-maximized f-stat.

Mirrors the exact setup of paris1_noise.py / run_lhs_noise.py, then times
single log_density() calls so you can size a feasible brute-force N before
committing to a Sobol resolution.

Usage:
    python bench_eval_noise.py                 # default 20 timed evals
    python bench_eval_noise.py --n-eval 50     # more samples -> tighter stats
    python bench_eval_noise.py --n-warmup 5    # discard first K (JIT / caches)

Reports per-eval seconds (mean/median/min/max) and projects wall-clock for a
range of candidate N. GPU work is synchronized before each timing stop so the
numbers reflect real device time, not just kernel-launch latency.
"""

import argparse
import os
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n-eval",   type=int, default=20, help="Timed evaluations")
parser.add_argument("--n-warmup", type=int, default=3,  help="Warmup evals to discard")
parser.add_argument("--seed",     type=int, default=42, help="Sampling seed for probe points")
parser.add_argument("--cov-path", type=str,
                    default="/home/svu/e1498138/emri_search/work/cov_matrix_noise_1yr.pkl",
                    help="Fisher covariance pickle for peak-width comparison ('' to skip)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup (identical to run_lhs_noise.py)
# ---------------------------------------------------------------------------
dir_work = '/home/svu/e1498138/emri_search/work'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

import few
from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_timemax_noise import LogLike

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 3 / 12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# GPU sync helper — cupy device barrier if present, else no-op
try:
    import cupy as cp
    def _sync():
        cp.cuda.runtime.deviceSynchronize()
except Exception:
    def _sync():
        pass

# Source parameters (same as run_lhs_noise.py)
m1 = 1e6
m2 = 1e1
a = 0.7
p0 = 9
e0 = 0.4
xI0 = 1.0
dist = 4.5
qS = np.pi
phiS = 0.
qK = 0.
phiK = 0.
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

n_vals = np.arange(-1, 6)
ell = 2

print('Initializing LogLike...')
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
print('LogLike initialized.')

# Prior bounds (same as run_lhs_noise.py)
param_ranges = [
    (5.6,  6.4),
    (0.8,  1.3),
    (0.3,  0.99),
    (8.0,  11.0),
    (0.2,  0.5),
]
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])


def log_density_one(params_5):
    """Single-point log_density (physical 5D: logm1, logm2, a, p0, e0)."""
    logm1, logm2, a_i, p0_i, e0_i = params_5
    try:
        return loglike_obj(np.array([
            10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0
        ]))
    except Exception as exc:
        print(f"  [warn] eval failed: {exc}")
        return -np.inf


# ---------------------------------------------------------------------------
# Probe points: uniform-random within the prior (representative of a scan)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
n_total = args.n_warmup + args.n_eval
probes = prior_lo + rng.random((n_total, 5)) * (prior_hi - prior_lo)

# ---------------------------------------------------------------------------
# Warmup (JIT compilation, GPU memory pools, mode caches)
# ---------------------------------------------------------------------------
print(f"\nWarmup: {args.n_warmup} evals (discarded)...")
for i in range(args.n_warmup):
    _ = log_density_one(probes[i])
    _sync()

# ---------------------------------------------------------------------------
# Timed evals
# ---------------------------------------------------------------------------
print(f"Timing: {args.n_eval} evals...")
times = np.empty(args.n_eval)
finite = 0
for j in range(args.n_eval):
    p = probes[args.n_warmup + j]
    _sync()
    t0 = time.perf_counter()
    val = log_density_one(p)
    _sync()
    times[j] = time.perf_counter() - t0
    if np.isfinite(val):
        finite += 1

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
mean_t = float(times.mean())
print("\n" + "=" * 60)
print("Per-eval timing (seconds)")
print("=" * 60)
print(f"  mean   = {mean_t:.4f}")
print(f"  median = {float(np.median(times)):.4f}")
print(f"  min    = {float(times.min()):.4f}")
print(f"  max    = {float(times.max()):.4f}")
print(f"  std    = {float(times.std()):.4f}")
print(f"  finite = {finite}/{args.n_eval}")

print("\n" + "=" * 60)
print("Projected wall-clock for a brute-force Sobol scan (single GPU)")
print("=" * 60)
print(f"{'N points':>12} | {'~per dim':>9} | {'wall time':>14}")
print("-" * 42)
for N in [int(1e5), int(3.2e5), int(1e6), int(3.2e6), int(1e7)]:
    secs = N * mean_t
    per_dim = N ** (1.0 / 5.0)
    if secs < 3600:
        pretty = f"{secs/60:.1f} min"
    elif secs < 86400:
        pretty = f"{secs/3600:.1f} h"
    else:
        pretty = f"{secs/86400:.1f} d"
    print(f"{N:>12,} | {per_dim:>9.1f} | {pretty:>14}")

print("\nNote: 'per dim' = N**(1/5), the effective grid spacing in 5D.")
print("Compare that spacing to your Fisher-predicted peak width: if the")
print("peak is narrower than the spacing, no amount of Sobol resolution")
print("you can afford will resolve it -- keep the adaptive stage.")

# ---------------------------------------------------------------------------
# Peak-width vs. Sobol-spacing comparison from the Fisher covariance
# ---------------------------------------------------------------------------
param_names = ["log10 m1", "log10 m2", "a", "p0", "e0"]


def _extract_cov(obj):
    """Coerce a variety of pickle layouts into an ndarray covariance."""
    if isinstance(obj, dict):
        for key in ("cov", "covariance", "cov_matrix", "C", "Sigma"):
            if key in obj:
                obj = obj[key]
                break
        else:
            # fall back to the first square-array value
            for v in obj.values():
                arr = np.asarray(v)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    obj = arr
                    break
    if isinstance(obj, (tuple, list)):
        for v in obj:
            arr = np.asarray(v)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                obj = arr
                break
    return np.asarray(obj)


if args.cov_path:
    import pickle
    print("\n" + "=" * 60)
    print("Peak-width (1 sigma) vs. Sobol spacing  -- CAN the peak be resolved?")
    print("=" * 60)
    try:
        with open(args.cov_path, "rb") as f:
            cov_obj = pickle.load(f)
        cov = _extract_cov(cov_obj)
        print(f"Loaded covariance from: {args.cov_path}")
        print(f"  shape = {cov.shape}   (assuming order {param_names})")
        if cov.shape[0] < 5:
            raise ValueError(f"covariance has <5 dims ({cov.shape}); cannot compare")

        sigma = np.sqrt(np.abs(np.diag(cov)))[:5]
        prior_range = prior_hi - prior_lo

        print("\nNOTE: this covariance is for T=1yr; the search runs T=3mo, whose")
        print("peak is WIDER (roughly x2-4 per dim), so this is the optimistic case.")

        print(f"\n{'param':>9} | {'1sig width':>11} | {'prior rng':>9} | "
              f"{'pts across peak @ N='}")
        header_Ns = [int(1e5), int(1e6), int(1e7)]
        print(f"{'':>9} | {'':>11} | {'':>9} | " +
              " ".join(f"{N:>8.0e}" for N in header_Ns))
        print("-" * 72)
        for k in range(5):
            spacings = [prior_range[k] / (N ** (1.0 / 5.0)) for N in header_Ns]
            # points landing within +-1 sigma of the peak along this axis
            across = [max(0.0, 2 * sigma[k] / s) for s in spacings]
            print(f"{param_names[k]:>9} | {sigma[k]:>11.2e} | "
                  f"{prior_range[k]:>9.2f} | " +
                  " ".join(f"{a:>8.1e}" for a in across))

        # Overall: expected quasi-random hits inside the 5D 1-sigma ellipsoid
        print("\nExpected Sobol points inside the 5D 1-sigma ellipsoid:")
        # ellipsoid volume / prior box volume * N
        from math import pi
        det = np.linalg.det(cov[:5, :5])
        vol_ellipsoid = (pi ** 2.5 / 3.75) * np.sqrt(np.abs(det))  # V of 5D unit-sigma ellipsoid
        vol_box = float(np.prod(prior_range))
        frac = vol_ellipsoid / vol_box
        print(f"  1-sigma ellipsoid volume fraction of prior box = {frac:.3e}")
        for N in [int(1e5), int(1e6), int(1e7), int(1e9)]:
            print(f"    N={N:>12,}  ->  ~{frac * N:.2e} points inside the peak")
        print("\nRule of thumb: you need this >= a few to resolve the peak by")
        print("brute force. If it stays << 1 at feasible N, the adaptive stage")
        print("is doing essential work -- do not drop it.")
    except Exception as exc:
        print(f"  [skip] could not load/parse covariance: {exc}")
