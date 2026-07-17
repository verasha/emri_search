"""
Semi-coherent LHS: Lambda = sum_k z_k^2

z_k = Re(<d|h_k>) / rho_k  where h_k is the template windowed to segment k.

The inner products are computed via the pre-computed time-domain kernel q_t:
    Re(<d|h_k>) = 4 * sum_{ch,t in k} q_t[ch][t] * h[ch][t]
    q_t = irfft(conj(signal_fft) / PSD)   [full-length, computed once]

This avoids per-segment FFTs entirely and is correct for colored noise.
Using short-segment FFTs (the naive approach) causes massive spectral leakage
because the segment PSD has coarser frequency resolution than the colored noise.

Normalization: rho_k^2 ~ rho^2 / N_segs  (valid for slowly-varying EMRI amplitude)

Usage:
    python run_lhs_semicoherent.py             # fresh run
    python run_lhs_semicoherent.py --resume    # resume from latest checkpoint
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
parser.add_argument("--n-samples",  type=int, default=int(1e6))
parser.add_argument("--batch-size", type=int, default=10)
parser.add_argument("--save-every", type=int, default=100)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--n-segs",     type=int, default=60,
                    help="Number of time segments (basin widens as N_segs^4)")
parser.add_argument("--outdir",     type=str,
                    default="/scratch/e1498138/lhs/noise/semicoherent/ckpt_1e6_s60")
parser.add_argument("--resume",     action="store_true")
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
dt      = 10
T       = 3 / 12
N_segs  = args.n_segs
print(f"dt={dt}s  T={T}yr  TDI={tdi_gen}  N_segs={N_segs}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis (full T)...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

N     = gwf.N
N_seg = N // N_segs
print(f"N={N}  N_seg={N_seg}  T_seg={N_seg*dt/86400:.2f} days")

# Source parameters
m1 = 1e6;  m2 = 1e1;  a = 0.7;  p0 = 9;  e0 = 0.4
xI0 = 1.0; dist = 3
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5

print('Generating signal + noise...')
signal = gwf.xp.array(waveform_response(
    m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt,
)) + gwf.generate_colored_noise(seed=42)
print(f'Signal shape: {signal.shape}')

# Pre-compute signal_fft and the time-domain matched filter kernel q_t.
# Re(<d|h_k>) = 4 * sum_{ch, t in segment_k} q_t[ch][t] * h[ch][t]
# q_t = irfft(conj(signal_fft) / PSD)  — full-length, DC zeroed.
# This accounts for the colored noise correctly without spectral leakage.
print('Pre-computing signal_fft and q_t...')
signal_fft = gwf.freq_wave(signal)   # (n_chan, N//2+1)

Q_f = gwf.xp.zeros((gwf.n_chan, gwf.N // 2 + 1), dtype=gwf.xp.complex128)
for ch in range(gwf.n_chan):
    Q_f[ch, 1:] = gwf.xp.conj(signal_fft[ch, 1:]) / gwf.PSD[ch]   # DC stays 0

q_t = gwf.xp.stack([
    gwf.xp.fft.irfft(Q_f[ch], n=gwf.N)
    for ch in range(gwf.n_chan)
])  # (n_chan, N)
print('Done.')

# ---------------------------------------------------------------------------
# Prior bounds
# ---------------------------------------------------------------------------
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
    return prior_lo + np.atleast_2d(u) * (prior_hi - prior_lo)

# ---------------------------------------------------------------------------
# Semi-coherent statistic
# ---------------------------------------------------------------------------
def compute_lambda(params):
    """
    params: (n, 5) array of [log10_m1, log10_m2, a, p0, e0]
    returns: (n,) Lambda = sum_k z_k^2

    Per-segment inner product (correct for colored noise):
        Re(<d|h_k>) = 4 * sum_{ch, t in k} q_t[ch][t] * h[ch][t]
    Normalization: rho_k ~ rho / sqrt(N_segs)  (uniform amplitude approximation)
    """
    params = np.asarray(params)
    out = np.full(params.shape[0], np.nan)
    for i in range(params.shape[0]):
        try:
            logm1, logm2, a_i, p0_i, e0_i = params[i]
            h = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt,
            ))
            h_fft = gwf.freq_wave(h)
            rho   = float(gwf.xp.sqrt(gwf.inner(h_fft, h_fft)))
            if rho == 0:
                continue
            rho_k = rho / np.sqrt(N_segs)   # per-segment SNR approximation

            Lambda = 0.0
            for k in range(N_segs):
                num_k = 4.0 * float(gwf.xp.real(
                    gwf.xp.sum(q_t[:, k*N_seg:(k+1)*N_seg]
                               * h[:, k*N_seg:(k+1)*N_seg])
                ))
                z_k = num_k / rho_k
                Lambda += z_k ** 2
            out[i] = Lambda
        except Exception as exc:
            print(f"  [warn] sample {i}: {exc}")
    return out

# ---------------------------------------------------------------------------
# Sanity check at true params and small offsets
# ---------------------------------------------------------------------------
noise_floor = float(N_segs)
thresh_5sig = noise_floor + 5.0 * np.sqrt(2.0 * noise_floor)

print('\n--- Sanity check ---')
true_phys = np.array([[np.log10(m1), np.log10(m2), a, p0, e0]])
L_true = compute_lambda(true_phys)[0]
print(f"True params:   Lambda={L_true:.2f}")
print(f"Noise floor:   {noise_floor:.0f}   5-sigma threshold: {thresh_5sig:.1f}")
print(f"Expected signal (dist=3, rho~19): ~{19**2 + N_segs:.0f}")

# Small offsets — should still be >> noise floor if inside semi-coherent basin
for delta, label in [
    ([0.005, 0, 0, 0,    0],    "logM1+0.005"),
    ([0,     0, 0, 0.1,  0],    "p0+0.1"),
    ([0,     0, 0, 0,    0.02], "e0+0.02"),
]:
    L_off = compute_lambda(true_phys + np.array(delta))[0]
    print(f"  {label:15s}: Lambda={L_off:.2f}")
print('--- End sanity check ---\n')

# ---------------------------------------------------------------------------
# Generate full LHS grid (deterministic given seed)
# ---------------------------------------------------------------------------
ndim     = 5
n_total  = args.n_samples
xlimits  = np.column_stack([np.zeros(ndim), np.ones(ndim)])
sampling = LHS(xlimits=xlimits, random_state=args.seed)
unit_pts = np.clip(sampling(n_total), 0.0, 1.0)
phys_pts = prior_transform(unit_pts)

# ---------------------------------------------------------------------------
# Resume or fresh start
# ---------------------------------------------------------------------------
Lambda_arr = np.full(n_total, np.nan)
start_idx  = 0

if args.resume:
    ckpts = sorted(glob.glob(os.path.join(args.outdir, "ckpt_*.pkl")))
    if ckpts:
        latest = ckpts[-1]
        print(f"Resuming from: {latest}")
        with open(latest, "rb") as f:
            ckpt = pickle.load(f)
        assert ckpt["n_total"] == n_total and ckpt["seed"] == args.seed, \
            "Checkpoint mismatch — check --n-samples / --seed"
        Lambda_arr = ckpt["Lambda"]
        start_idx  = ckpt["next_idx"]
        print(f"Resuming from {start_idx}/{n_total}")
    else:
        print("No checkpoint found, starting fresh.")

# ---------------------------------------------------------------------------
# Evaluate in batches with checkpoints
# ---------------------------------------------------------------------------
batch_size  = args.batch_size
save_every  = args.save_every
n_batches   = (n_total - start_idx + batch_size - 1) // batch_size
print(f"Evaluating {n_total-start_idx} samples in {n_batches} batches of {batch_size}  "
      f"(checkpoint every {save_every} batches)")

t0          = time.time()
batch_count = 0

for i in range(start_idx, n_total, batch_size):
    end = min(i + batch_size, n_total)
    Lambda_arr[i:end] = compute_lambda(phys_pts[i:end])
    batch_count += 1

    done    = end
    elapsed = time.time() - t0
    rate    = (done - start_idx) / elapsed if elapsed > 0 else 0
    eta     = (n_total - done) / rate if rate > 0 else float('inf')
    finite  = int(np.sum(np.isfinite(Lambda_arr[:done])))
    n_above = int(np.sum(Lambda_arr[:done] > thresh_5sig))
    max_L   = float(np.nanmax(Lambda_arr[:done])) if finite > 0 else float('nan')
    print(f"  [{done}/{n_total}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  "
          f"eta={eta:.0f}s  finite={finite}  above_thresh={n_above}  max_L={max_L:.1f}")

    if batch_count % save_every == 0 or end == n_total:
        ckpt_path = os.path.join(args.outdir, f"ckpt_{end:07d}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({
                "n_total":  n_total,
                "seed":     args.seed,
                "n_segs":   N_segs,
                "Lambda":   Lambda_arr,
                "next_idx": end,
                "phys_pts": phys_pts,
            }, f)
        print(f"  -> {ckpt_path}")

# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------
out_path = os.path.join(args.outdir, "final.pkl")
with open(out_path, "wb") as f:
    pickle.dump((phys_pts, Lambda_arr), f)

finite  = int(np.sum(np.isfinite(Lambda_arr)))
n_above = int(np.sum(Lambda_arr > thresh_5sig))
print(f"\nDone!  {out_path}")
print(f"Finite={finite}/{n_total}  above_thresh={n_above}  "
      f"max_Lambda={float(np.nanmax(Lambda_arr)):.2f}")
best = phys_pts[np.nanargmax(Lambda_arr)]
print(f"Best:  logM={best[0]:.4f}  logmu={best[1]:.4f}  a={best[2]:.4f}  "
      f"p0={best[3]:.4f}  e0={best[4]:.4f}")
print(f"True:  logM={np.log10(m1):.4f}  logmu={np.log10(m2):.4f}  a={a:.4f}  "
      f"p0={p0:.4f}  e0={e0:.4f}")
