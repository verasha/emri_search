"""
External LHS evaluation script (coherent log-likelihood) for the Mojito-light
EMRI_G source, over the 7-dim extended prior (logm1, logm2, a, p0, e0, qS, phiS).
Generates LHS samples over the prior and evaluates log_density (full coherent
log-likelihood via LogLike) in batches, saving checkpoints so the job can be
interrupted and resumed.

Usage:
    python run_lhs_noise_ext.py                  # fresh run
    python run_lhs_noise_ext.py --resume         # resume from latest checkpoint

Output:
    final.pkl   ->  (physical_points, logden)  ready for run_sampling()

Checkpoint files:
    ckpt_NNNNNN.pkl  saved every --save-every batches
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
parser.add_argument("--n-samples",   type=int,   default=int(1e3),  help="Total LHS samples")
parser.add_argument("--batch-size",  type=int,   default=10,         help="Samples per batch")
parser.add_argument("--save-every",  type=int,   default=10,         help="Save checkpoint every N batches")
parser.add_argument("--seed",        type=int,   default=42,         help="LHS random seed")
parser.add_argument("--outdir",      type=str,   default="/scratch/e1498138/paper/ext/emri_g/lhs_coherent",
                    help="Checkpoint directory")
parser.add_argument("--resume",      action="store_true",            help="Resume from latest checkpoint")
args = parser.parse_args()

args.outdir = os.path.abspath(args.outdir)
os.makedirs(args.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup (mirrors paris1_sc_ext.py / run_lhs_noise_sc_ext.py)
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
T = 3/12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Mojito light EMRI_G
m1 = 6.72e5
m2 = 9.84e1
a = 0.5
p0 = 15.7117
e0 = 0.7440
xI0 = 1.0
dist = 4.755
qS = 0.5906
phiS = 3.5808
qK = 1.1707
phiK = 3.8495
Phi_phi0 = 3.1940
Phi_theta0 = 3.3780
Phi_r0 = 0.6038

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

data_snr = float(gwf.rhostat_timemax(loglike_obj.signal).get())
print(f'SNR (time-max): {data_snr:.4f}')

# Prior bounds (same as run_lhs_noise_sc_ext.py / paris1_sc_ext.py)
param_ranges = [
    (5.6,  6.4),
    (1.7,  2.3),
    (0.3,  0.99),
    (13.5, 16.5),
    (0.6,  0.9),
    (0.0,  np.pi),      # qS: source colatitude
    (0.0,  2*np.pi),    # phiS: source azimuth
]
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])


def prior_transform(u):
    u = np.atleast_2d(u)
    return prior_lo + u * (prior_hi - prior_lo)


def log_density(params):
    params = np.asarray(params)
    n = params.shape[0]
    out = np.full(n, -np.inf)
    for i in range(n):
        try:
            logm1, logm2, a_i, p0_i, e0_i, qS_i, phiS_i = params[i]
            out[i] = loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS_i, phiS_i, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception as exc:
            print(f"  [warn] sample {i} failed: {exc}")
    return out


# ---------------------------------------------------------------------------
# Generate the full LHS grid (deterministic given seed)
# ---------------------------------------------------------------------------
ndim     = 7
n_total  = args.n_samples
xlimits  = np.column_stack([np.zeros(ndim), np.ones(ndim)])
sampling = LHS(xlimits=xlimits, random_state=args.seed)
unit_pts = np.clip(sampling(n_total), 0.0, 1.0)
phys_pts = prior_transform(unit_pts)

# ---------------------------------------------------------------------------
# Resume or fresh start
# ---------------------------------------------------------------------------
logden    = np.full(n_total, np.nan)
start_idx = 0

if args.resume:
    ckpts = sorted(glob.glob(os.path.join(args.outdir, "ckpt_*.pkl")))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from checkpoint: {latest}")
        with open(latest, "rb") as f:
            ckpt = pickle.load(f)
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

    done    = end
    elapsed = time.time() - t0
    rate    = (done - start_idx) / elapsed if elapsed > 0 else 0
    remaining = (n_total - done) / rate if rate > 0 else float("inf")
    print(f"  [{done}/{n_total}]  "
          f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  "
          f"eta={remaining:.0f}s  "
          f"finite={np.sum(np.isfinite(logden[:done]))}")

    if batch_count % save_every == 0 or end == n_total:
        ckpt_path = os.path.join(args.outdir, f"ckpt_{end:06d}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "n_total":  n_total,
                "seed":     args.seed,
                "logden":   logden,
                "next_idx": end,
                "phys_pts": phys_pts,
            }, f)
        print(f"  -> checkpoint saved: {ckpt_path}")

# ---------------------------------------------------------------------------
# Save final output  (physical_points, logden)  — ready for run_sampling()
# ---------------------------------------------------------------------------
out_path = os.path.join(args.outdir, "final.pkl")
with open(out_path, "wb") as f:
    pickle.dump((phys_pts, logden), f)

print(f"\nDone!  Final output: {out_path}")
print(f"Total finite evaluations: {np.sum(np.isfinite(logden))} / {n_total}")
