"""
External LHS evaluation script for intrinsic_ffunc_3mth_snr32.
Generates LHS samples over the prior and evaluates log_density in batches,
saving checkpoints so the job can be interrupted and resumed.

Usage:
    python run_lhs_snr32.py                  # fresh run
    python run_lhs_snr32.py --resume         # resume from latest checkpoint

Output:
    lhs_snr32_final.pkl   ->  (physical_points, logden)  ready for run_sampling()

Checkpoint files:
    lhs_snr32_ckpt_NNNNN.pkl  saved every --save-every batches
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from smt.sampling_methods import LHS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n-samples",   type=int,   default=int(1e5),  help="Total LHS samples")
parser.add_argument("--batch-size",  type=int,   default=100,        help="Samples per batch")
parser.add_argument("--save-every",  type=int,   default=10,         help="Save checkpoint every N batches")
parser.add_argument("--seed",        type=int,   default=42,         help="LHS random seed")
parser.add_argument("--outdir",      type=str,   default="./lhs_snr32_checkpoints", help="Checkpoint directory")
parser.add_argument("--resume",      action="store_true",            help="Resume from latest checkpoint")
args = parser.parse_args()

args.outdir = os.path.abspath(args.outdir)   # fix: resolve before chdir
os.makedirs(args.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup (mirrors intrinsic_ffunc_3mth_snr32.ipynb)
# ---------------------------------------------------------------------------
os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')

import few
import GWfuncs
import loglike_pure

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

use_gpu     = True
force_backend = "cuda12x"
dt          = 10
T           = 3 / 12

print(f"dt={dt}s  T={T}yr")

inspiral_kwargs = {"func": "KerrEccEqFlux", "DENSE_STEPPING": 0, "include_minus_m": False}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs_comb = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep  = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

print("Building waveform generators...")
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, frame="detector",
    inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_comb, use_gpu=use_gpu)

waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, frame="detector",
    inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_sep, use_gpu=use_gpu)

gwf = GWfuncs.GravWaveAnalysis(T, dt)

# Source parameters
m1, m2, a, p0, e0 = 1e6, 1e1, 0.7, 9, 0.4
xI0 = 1.0;  dist = 1.8
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true  = [np.log10(m1), np.log10(m2), a, p0, e0]

n_vals = np.arange(-1, 6)
ell = 2

print("Initializing loglike...")
loglike_obj = loglike_pure.LogLike(
    params_star, waveform_gen_comb, gwf,
    verbose=False, waveform_gen_sep=waveform_gen_sep,
    ell=ell, n_vals=n_vals, M_mode=None)

data     = loglike_obj.signal
data_snr = gwf.rhostat(data)
print(f"SNR = {data_snr}")

# Prior bounds (same as notebook)
param_ranges = [
    (5.6,  6.4),
    (0.8,  1.3),
    (0.3,  0.99),
    (8.0,  11.0),
    (0.2,  0.5),
]
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])


def prior_transform(u):
    """Map [0,1]^5 -> physical parameter space."""
    u = np.atleast_2d(u)
    return prior_lo + u * (prior_hi - prior_lo)


def log_density(params):
    params = np.asarray(params)
    n = params.shape[0]
    out = np.full(n, -np.inf)
    for i in range(n):
        try:
            logm1, logm2, a_i, p0_i, e0_i = params[i]
            out[i] = loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception as exc:
            print(f"  [warn] sample {i} failed: {exc}")
    return out


# ---------------------------------------------------------------------------
# Generate the full LHS grid (deterministic given seed)
# ---------------------------------------------------------------------------
ndim     = 5
n_total  = args.n_samples
xlimits  = np.column_stack([np.zeros(ndim), np.ones(ndim)])
sampling = LHS(xlimits=xlimits, random_state=args.seed)
unit_pts = np.clip(sampling(n_total), 0.0, 1.0)           # shape (n_total, 5) in [0,1]
phys_pts = prior_transform(unit_pts)                       # shape (n_total, 5) physical

# ---------------------------------------------------------------------------
# Resume or fresh start
# ---------------------------------------------------------------------------
logden   = np.full(n_total, np.nan)
start_idx = 0

if args.resume:
    ckpts = sorted(glob.glob(os.path.join(args.outdir, "ckpt_*.pkl")))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from checkpoint: {latest}")
        with open(latest, "rb") as f:
            ckpt = pickle.load(f)
        # Sanity check: same grid?
        assert ckpt["n_total"] == n_total and ckpt["seed"] == args.seed, \
            "Checkpoint grid mismatch — check --n-samples / --seed"
        logden    = ckpt["logden"]
        start_idx = ckpt["next_idx"]
        print(f"Resuming from sample {start_idx}/{n_total}")
    else:
        print("No checkpoint found, starting fresh.")

# ---------------------------------------------------------------------------
# Evaluate in batches with checkpoints
# ---------------------------------------------------------------------------
batch_size  = args.batch_size
save_every  = args.save_every
n_batches   = (n_total - start_idx + batch_size - 1) // batch_size

print(f"Evaluating {n_total - start_idx} remaining samples "
      f"in {n_batches} batches of {batch_size}  "
      f"(checkpoint every {save_every} batches)")

t0 = time.time()
batch_count = 0

for i in range(start_idx, n_total, batch_size):
    end = min(i + batch_size, n_total)
    logden[i:end] = log_density(phys_pts[i:end])
    batch_count += 1

    # Progress
    done  = end
    total = n_total
    elapsed = time.time() - t0
    rate = (done - start_idx) / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else float("inf")
    print(f"  [{done}/{total}]  "
          f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  "
          f"eta={remaining:.0f}s  "
          f"finite={np.sum(np.isfinite(logden[:done]))}")

    # Checkpoint
    if batch_count % save_every == 0 or end == n_total:
        ckpt_path = os.path.join(args.outdir, f"ckpt_{end:06d}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "n_total":  n_total,
                "seed":     args.seed,
                "logden":   logden,
                "next_idx": end,
                "phys_pts": phys_pts,   # included so the checkpoint is self-contained
            }, f)
        print(f"  -> checkpoint saved: {ckpt_path}")

# ---------------------------------------------------------------------------
# Save final output  (physical_points, logden)  — ready for run_sampling()
# ---------------------------------------------------------------------------
out_path = os.path.join(args.outdir, "/scratch/e1498138/lhs/nonoise/lhs_5e5.pkl")
with open(out_path, "wb") as f:
    pickle.dump((phys_pts, logden), f)

print(f"\nDone!  Final output: {out_path}")
print(f"Total finite evaluations: {np.sum(np.isfinite(logden))} / {n_total}")
