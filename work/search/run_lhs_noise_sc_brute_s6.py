"""
LHS brute-force evaluation script: computes the semi-coherent statistic S_N
for each template, sampling broad priors over *all* free parameters except
Phi_theta0 (which is degenerate with Phi_phi0/Phi_r0 mod the orbital
frequencies here). dist is included in the search for completeness, even
though S_N = <d|h>_N / sqrt(<h|h>_N) is in principle invariant to an
overall amplitude rescaling of h (dist just rescales <h|h>_N and <d|h>_N
by the same factor), so it should not move the recovered S_N.

S_N(theta) = <d|h(theta)>_N / sqrt(<h|h>_N)
           = SNR_semicoherent(d, h(theta), N_seg)   (arXiv:2205.08702 eqs. 34-35)

Per-segment maximization over an overall time shift AND phase (phase_max=True).

Sky/spin polar angles (qS, qK) are sampled uniform in cosine to avoid
polar-cap oversampling; azimuthal/phase angles are sampled uniform in
[0, 2*pi).

Usage:
    python run_lhs_noise_sc_brute.py                  # fresh run
    python run_lhs_noise_sc_brute.py --resume         # resume from latest checkpoint

Output:
    final.pkl  ->  (physical_points, det_snr)  ready for inspection / seeding PARIS
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
parser.add_argument("--n-samples",  type=int, default=int(1e5), help="Total LHS samples")
parser.add_argument("--batch-size", type=int, default=10,       help="Samples per batch")
parser.add_argument("--save-every", type=int, default=10,       help="Save checkpoint every N batches")
parser.add_argument("--seed",       type=int, default=42,       help="LHS random seed")
parser.add_argument("--n-segs",     type=int, default=4,        help="Number of time segments for S_N")
parser.add_argument("--outdir",     type=str,
                    default="/scratch/e1498138/lhs/noise/sc/ckpt_brute_1e5",
                    help="Checkpoint directory")
parser.add_argument("--resume",     action="store_true", help="Resume from latest checkpoint")
args = parser.parse_args()

args.outdir = os.path.abspath(args.outdir)
os.makedirs(args.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
dir_work = '/home/svu/e1498138/emri_search/work'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

import few
from GWfuncs_noise import GravWaveAnalysis, build_waveform_response

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt      = 5
T       = 12/12
N_segs  = args.n_segs
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_segs}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source (Mojito light EMRI_G)
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
Phi_theta0 = 3.3780     # fixed: not searched
Phi_r0 = 0.6038

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

# Generate signal + noise
print('Generating signal + noise...')
h_true = gwf.xp.array(waveform_response(
    m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0,
    T=T, dt=dt,
))
signal = h_true + gwf.generate_colored_noise(seed=42)
print('Signal generated.')

# Context: injected SNR vs the S_N noise floor for this segmentation
hf_true = gwf.freq_wave(h_true)
rho_true = float(gwf.xp.sqrt(gwf.inner(hf_true, hf_true)))
N_per = gwf.N // N_segs
floor_est = np.sqrt(2.0 * np.log(N_per) * N_segs)
print(f"Injected rho = {rho_true:.2f}   "
      f"S_N noise floor estimate ~ {floor_est:.1f}   "
      f"(need rho well above this for a clear peak)")
sn_true = float(gwf.SNR_semicoherent(signal, h_true, N_seg=N_segs, phase_max=True))
print(f"S_{N_segs} at true params (with noise): {sn_true:.2f}")
del h_true, hf_true

# ---------------------------------------------------------------------------
# Prior bounds — broad, covering the full physical/astrophysical range for
# each searched parameter. Phi_theta0 is held fixed (see header).
# ---------------------------------------------------------------------------
param_ranges = [
    (5.6,   6.4),        # log10(m1): primary mass [Msun]
    (1.7,  2.3),         # log10(m2): secondary mass [Msun]
    (0.3,   0.999),        # a: primary spin
    (13.5, 16.5),        # p0: initial semi-latus rectum
    (0.6,   0.9),          # e0: initial eccentricity
    (3,   5),         # dist: luminosity distance [Gpc]
    (-1.0,  1.0),          # cos(qS): source colatitude
    (0.0,   2*np.pi),      # phiS: source azimuth
    (-1.0,  1.0),          # cos(qK): spin colatitude
    (0.0,   2*np.pi),      # phiK: spin azimuth
    (0.0,   2*np.pi),      # Phi_phi0: initial azimuthal phase
    (0.0,   2*np.pi),      # Phi_r0: initial radial phase
]
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])


def prior_transform(u):
    u = np.atleast_2d(u)
    return prior_lo + u * (prior_hi - prior_lo)


# ---------------------------------------------------------------------------
# S_N: sum_i max_{tau, phase} Re/|.| (<d_i|h_i(tau)>) / sqrt(<h|h>_N)
# ---------------------------------------------------------------------------
def compute_det_snr(params):
    """
    params: (n, 12) array of
        [log10_m1, log10_m2, a, p0, e0, dist, cos_qS, phiS, cos_qK,
         phiK, Phi_phi0, Phi_r0]
    xI0 is held fixed (only xI0 = 1 is supported by the waveform model).
    returns: (n,) array of S_N values (nan on failure)
    """
    params = np.asarray(params)
    n = params.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        try:
            (logm1, logm2, a_i, p0_i, e0_i, dist_i,
             cos_qS_i, phiS_i, cos_qK_i, phiK_i, Phi_phi0_i, Phi_r0_i) = params[i]
            qS_i = np.arccos(cos_qS_i)
            qK_i = np.arccos(cos_qK_i)
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist_i, qS_i, phiS_i, qK_i, phiK_i,
                Phi_phi0_i, Phi_theta0, Phi_r0_i,
                T=T, dt=dt,
            ))
            out[i] = float(gwf.SNR_semicoherent(signal, h_temp, N_seg=N_segs, phase_max=True))
        except Exception as exc:
            print(f"  [warn] sample {i} failed: {exc}")
    return out


# ---------------------------------------------------------------------------
# Generate the full LHS grid (deterministic given seed)
# ---------------------------------------------------------------------------
ndim     = 12
n_total  = args.n_samples
xlimits  = np.column_stack([np.zeros(ndim), np.ones(ndim)])
sampling = LHS(xlimits=xlimits, random_state=args.seed)
unit_pts = np.clip(sampling(n_total), 0.0, 1.0)
phys_pts = prior_transform(unit_pts)

# ---------------------------------------------------------------------------
# Resume or fresh start
# ---------------------------------------------------------------------------
det_snr   = np.full(n_total, np.nan)
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
        assert ckpt.get("n_segs", N_segs) == N_segs, \
            "Checkpoint N_segs mismatch — check --n-segs"
        det_snr   = ckpt["det_snr"]
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

t0          = time.time()
batch_count = 0

for i in range(start_idx, n_total, batch_size):
    end = min(i + batch_size, n_total)
    det_snr[i:end] = compute_det_snr(phys_pts[i:end])
    batch_count += 1

    done      = end
    elapsed   = time.time() - t0
    rate      = (done - start_idx) / elapsed if elapsed > 0 else 0
    remaining = (n_total - done) / rate if rate > 0 else float("inf")
    finite    = np.sum(np.isfinite(det_snr[:done]))
    max_snr   = float(np.nanmax(det_snr[:done])) if finite > 0 else float("nan")
    min_snr   = float(np.nanmin(det_snr[:done])) if finite > 0 else float("nan")
    print(f"  [{done}/{n_total}]  "
          f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  "
          f"eta={remaining:.0f}s  "
          f"finite={finite}  det_snr=[{min_snr:.3f}, {max_snr:.3f}]")

    if batch_count % save_every == 0 or end == n_total:
        ckpt_path = os.path.join(args.outdir, f"ckpt_{end:06d}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "n_total":  n_total,
                "seed":     args.seed,
                "n_segs":   N_segs,
                "det_snr":  det_snr,
                "next_idx": end,
                "phys_pts": phys_pts,
            }, f)
        print(f"  -> checkpoint saved: {ckpt_path}")

# ---------------------------------------------------------------------------
# Save final output  (physical_points, det_snr)
# ---------------------------------------------------------------------------
out_path = os.path.join(args.outdir, "final.pkl")
with open(out_path, "wb") as f:
    pickle.dump((phys_pts, det_snr), f)

finite = np.sum(np.isfinite(det_snr))
print(f"\nDone!  Final output: {out_path}")
print(f"Finite evaluations: {finite} / {n_total}")
print(f"Max det_snr: {float(np.nanmax(det_snr)):.4f}")
best = phys_pts[np.nanargmax(det_snr)]
print(f"Best point: logm1={best[0]:.4f}  logm2={best[1]:.4f}  a={best[2]:.4f}  "
      f"p0={best[3]:.4f}  e0={best[4]:.4f}  dist={best[5]:.4f}  "
      f"cos_qS={best[6]:.4f}  phiS={best[7]:.4f}  cos_qK={best[8]:.4f}  "
      f"phiK={best[9]:.4f}  Phi_phi0={best[10]:.4f}  Phi_r0={best[11]:.4f}")
