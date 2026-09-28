"""
LHS evaluation script: computes the semi-coherent statistic S_N for each template.

S_N(theta) = <d|h(theta)>_N / sqrt(<h|h>_N)
           = SNR_semicoherent(d, h(theta), N_seg)   (arXiv:2205.08702 eqs. 34-35)

Per-segment maximization over an overall time shift only (no phase max),
on whitened time series. Wider basin than the coherent X statistics, at the
cost of an elevated noise floor ~ sqrt(2 ln(N/N_seg) * N_seg): the injection
needs rho >~ sqrt(N_seg) * sqrt(2 ln(N/N_seg)) to stand out.

No chi_sq suppression, no mode selection.

Configured by editing the literals below -- source, prior box, N_seg -- which
is what makes it useful for one-off exploration. It reaches GWfuncs_noise via
the sys.path insert further down, so it runs from this subdirectory unchanged.

Usage:
    python lhs_sc.py --n-segs 4 --outdir /scratch/.../lhs_sc_s4

Output:
    final.pkl  ->  (physical_points, det_snr)  ready for inspection / seeding PARIS

There is no checkpointing: the run either finishes or is repeated. Checkpoints
wrote the full phys_pts array (5.6 MB at 1e5 samples) once every 100 samples,
1000 files and 6.4 GB for a single run, none of it pruned -- and resuming only
asserted n_total and seed, never the prior box, so editing param_ranges (which
is how this script is configured) and resuming silently glued det_snr from the
old box onto phys_pts from the new one.
"""

import argparse
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
parser.add_argument("--seed",       type=int, default=42,       help="LHS random seed")
parser.add_argument("--n-segs",     type=int, default=4,        help="Number of time segments for S_N")
parser.add_argument("--outdir",     type=str,
                    default="/scratch/e1498138/paper/emri_a_snr30/lhs_sc",
                    help="Output directory (results go to scratch, not the repo)")
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
from few.utils.geodesic import get_separatrix
from GWfuncs_noise import GravWaveAnalysis, build_waveform_response


def icrs_to_ecliptic(qK_icrs, phiK_icrs, eps_deg=23.43929111):
    """
    Convert a sky direction (colatitude, longitude) from the ICRS/equatorial
    frame (qK = pi/2 - dec, phiK = ra) to the ecliptic frame (SSB frame used
    by the LISA response, is_ecliptic_latitude=False convention) via the
    standard equatorial<->ecliptic rotation with J2000 mean obliquity eps.
    """
    dec = np.pi / 2 - qK_icrs
    ra = phiK_icrs
    eps = np.deg2rad(eps_deg)

    sin_beta = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(ra) * np.sin(eps)
    beta = np.arcsin(sin_beta)

    y = np.cos(dec) * np.sin(ra) * np.cos(eps) + np.sin(dec) * np.sin(eps)
    x = np.cos(dec) * np.cos(ra)
    lam = np.arctan2(y, x) % (2 * np.pi)

    qK_ecl = np.pi / 2 - beta
    phiK_ecl = lam
    return qK_ecl, phiK_ecl

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt      = 10
T       = 1.4
#1.4190 # plunge for EMRI B 
#0.9497 # plunge for emri A 
N_segs  = args.n_segs
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_segs}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen,
                                            pad_output=True)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)


# Mojito light EMRI_G
# m1 = 6.72e5
# m2 = 9.84e1
# a = 0.5
# p0 = 15.7117
# e0 = 0.7440
# xI0 = 1.0
# dist = 4.755
# qS = 0.5906
# phiS = 3.5808
# qK = 1.1707
# phiK = 3.8495
# Phi_phi0 = 3.1940
# Phi_theta0 = 3.3780
# Phi_r0 = 0.6038

# Mojito light EMRI_C
# m1 = 3.54e6
# m2 = 8.01e1
# a = 0.950
# p0 = 7.3890
# e0 = 0.3160
# xI0 = 1.0000
# dist = 4.858
# qS = 2.0420
# phiS = 5.7873
# qK, phiK = icrs_to_ecliptic(1.6633, 1.7431)  # catalog qK/phiK are in ICRS, convert to ecliptic
# Phi_phi0 = 2.4044
# Phi_theta0 = 0.2850
# Phi_r0 = 0.9743

# Mojito light EMRI_F
# m1 = 1.00e7
# m2 = 1.00e1
# a = 0.998
# p0 = 2.1200
# e0 = 0.4250
# xI0 = 1.0
# dist = 4.313
# qS = 0.6280
# phiS = 1.2561
# qK, phiK = icrs_to_ecliptic(2.7000, 2.6200)  # catalog qK/phiK are in ICRS, convert to ecliptic
# Phi_phi0 = 4.1163
# Phi_theta0 = 1.4444
# Phi_r0 = 0.9810

# # Mojito Light EMRI A (rescaled)
# m1 = 5.35e5
# m2 = 9.51
# a = 0.988
# p0 = 13.5739
# e0 = 0.0156
# xI0 = -1.0
# dist = 2.081234 #1.067 NOTE: rescaled to SNR30
# qS = 1.0706
# phiS = 4.7079
# qK, phiK = icrs_to_ecliptic(0.7247, 4.0398 )  # catalog qK/phiK are in ICRS, convert to ecliptic
# Phi_phi0 = 2.1550
# Phi_theta0 = 0.4544
# Phi_r0 = 1.1248

# Mojito Light EMRI B (rescaled)
m1 = 2.80e6
m2 = 1.54e1
a = 0.973
p0 = 4.8621
e0 =  0.1403
xI0 = 1.0
dist = 20.650702 # 1.869  NOTE: rescaled to SNR30
qS = 1.4100
phiS = 5.2356
qK, phiK = icrs_to_ecliptic(2.8302 ,   5.7136 )  # catalog qK/phiK are in ICRS, convert to ecliptic
Phi_phi0 = 3.9359
Phi_theta0 =  0.3677
Phi_r0 =   4.8738

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
# Prior bounds
# ---------------------------------------------------------------------------


# EMRI A 
# param_ranges = [
#     (5.3,  6.2),
#     (0.6,  1.3),
#     (0.7,  0.999),
#     (12,  15.1),
#     (0.005,  0.2),
#     (-1.0,  1.0),       # cos(qS): source colatitude, sampled uniform in cosine
#     (0.0,  2*np.pi),    # phiS: source azimuth
# ]

# EMRI B
param_ranges = [
    (6.0,  6.9),
    (0.8,  1.5),
    (0.7,  0.999),
    (3.0, 8.0),
    (0.005, 0.5),
    (-1.0,  1.0),       # cos(qS): source colatitude, sampled uniform in cosine
    (0.0,  2*np.pi),    # phiS: source azimuth
]

# EMRI_G
# param_ranges = [
#     (5.6,  6.4),
#     (1.7,  2.3),
#     (0.3,  0.99),
#     (13.5,  16.5),
#     (0.6,  0.9),
#     (-1.0,  1.0),       # cos(qS): source colatitude, sampled uniform in cosine
#     (0.0,  2*np.pi),    # phiS: source azimuth
# ]



# EMRI_C
# param_ranges = [
#     # (6.15,  6.95),
#     # (1.60,  2.20),
#     # (0.70,  0.999),
#     # (5.89,  8.89),
#     # (0.17,  0.47),
#     (6.54808, 6.54911),
#     (1.90229, 1.90370),
#     (0.94932, 0.95003),
#     (7.38774, 7.39225),
#     (0.31562, 0.31965),
#     (-1.0,  1.0),       # cos(qS): source colatitude, sampled uniform in cosine
#     (0.0,  2*np.pi),    # phiS: source azimuth
# ]

# EMRI_F
# param_ranges = [
#     (6.6,   7.4),
#     (0.7,   1.3),
#     (0.95,  0.999),
#     (1.8,   2.6),
#     (0.3,   0.55),
#     (-1.0,  1.0),       # cos(qS): source colatitude, sampled uniform in cosine
#     (0.0,  2*np.pi),    # phiS: source azimuth
# ]



prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])


def prior_transform(u):
    u = np.atleast_2d(u)
    return prior_lo + u * (prior_hi - prior_lo)


# ---------------------------------------------------------------------------
# S_N: sum_i max_tau Re(<d_i|h_i(tau)>) / sqrt(<h|h>_N)
# ---------------------------------------------------------------------------
SEPARATRIX_MARGIN = 0.1  # p0 must clear the Kerr separatrix by this much


def compute_det_snr(params):
    """
    params: (n, 7) array of [log10_m1, log10_m2, a, p0, e0, cos_qS, phiS]
    returns: (n,) array of S_N values (nan on failure)
    """
    params = np.asarray(params)
    n = params.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        try:
            logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = params[i]
            p_sep = get_separatrix(a_i, e0_i, xI0)
            if p0_i < p_sep + SEPARATRIX_MARGIN:
                print(f"  [warn] sample {i} skipped: p0={p0_i:.4f} within "
                      f"{SEPARATRIX_MARGIN} of separatrix={p_sep:.4f}")
                continue
            qS_i = np.arccos(cos_qS_i)
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS_i, phiS_i, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                T=T, dt=dt,
            ))
            out[i] = float(gwf.SNR_semicoherent(signal, h_temp, N_seg=N_segs, phase_max=True))
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

det_snr = np.full(n_total, np.nan)

# ---------------------------------------------------------------------------
# Evaluate in batches
# ---------------------------------------------------------------------------
batch_size = args.batch_size
n_batches  = (n_total + batch_size - 1) // batch_size

print(f"Evaluating {n_total} samples in {n_batches} batches of {batch_size}")

t0 = time.time()

for i in range(0, n_total, batch_size):
    end = min(i + batch_size, n_total)
    det_snr[i:end] = compute_det_snr(phys_pts[i:end])

    done      = end
    elapsed   = time.time() - t0
    rate      = done / elapsed if elapsed > 0 else 0
    remaining = (n_total - done) / rate if rate > 0 else float("inf")
    finite    = np.sum(np.isfinite(det_snr[:done]))
    max_snr   = float(np.nanmax(det_snr[:done])) if finite > 0 else float("nan")
    min_snr   = float(np.nanmin(det_snr[:done])) if finite > 0 else float("nan")
    print(f"  [{done}/{n_total}]  "
          f"elapsed={elapsed:.0f}s  rate={rate:.1f}/s  "
          f"eta={remaining:.0f}s  "
          f"finite={finite}  det_snr=[{min_snr:.3f}, {max_snr:.3f}]")

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
      f"p0={best[3]:.4f}  e0={best[4]:.4f}  cos_qS={best[5]:.4f}  phiS={best[6]:.4f}")
