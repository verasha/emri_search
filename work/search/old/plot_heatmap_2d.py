"""
2D logden heatmap in p0-e0 space with connection lines overlaid.

Usage:
    python plot_heatmap_2d.py
    python plot_heatmap_2d.py --dim-x 3 --dim-y 4 --n-grid 30
    python plot_heatmap_2d.py --dim-x 2 --dim-y 3   # a vs p0

Output:
    heatmap_2d_<dimx>_<dimy>.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dim-x",  type=int, default=3,  help="x-axis parameter index (0-4)")
parser.add_argument("--dim-y",  type=int, default=4,  help="y-axis parameter index (0-4)")
parser.add_argument("--n-grid", type=int, default=30, help="Grid resolution per axis")
parser.add_argument("--outdir",  type=str, default="./plots", help="Output directory")
parser.add_argument("--fix-pt", type=str, default="true",
                    choices=["true", "sec1", "sec2"],
                    help="Reference point used to fix non-plotted dims (true/sec1/sec2)")
args = parser.parse_args()

args.outdir = os.path.abspath(args.outdir)   # resolve before chdir
os.makedirs(args.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import few
import GWfuncs
import loglike_timemax

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

use_gpu       = True
force_backend = "cuda12x"
dt            = 10
T             = 3 / 12

inspiral_kwargs  = {"func": "KerrEccEqFlux", "DENSE_STEPPING": 0, "include_minus_m": False}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs       = {"force_backend": force_backend}
sum_kwargs_comb  = {"force_backend": force_backend, "pad_output": True}

print("Building waveform generator...")
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, frame="detector",
    inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_comb, use_gpu=use_gpu)

gwf = GWfuncs.GravWaveAnalysis(T, dt)

# Source parameters
m1, m2, a, p0, e0 = 1e6, 1e1, 0.7, 9, 0.4
xI0 = 1.0;  dist = 1.8
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true  = [np.log10(m1), np.log10(m2), a, p0, e0]

n_vals = np.arange(-1, 6)
ell    = 2

print("Initializing loglike...")
loglike_obj = loglike_timemax.LogLikeTimeMax(
    params_star, waveform_gen_comb, gwf,
    verbose=False, ell=ell, n_vals=n_vals, M_mode=None)

param_ranges = [
    (5.6,  6.4),
    (0.8,  1.3),
    (0.3,  0.99),
    (8.0,  11.0),
    (0.2,  0.5),
]
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])

labels = [r'$\log_{10}(m_1)$', r'$\log_{10}(m_2)$', r'$a$', r'$p_0$', r'$e_0$']


def log_density_safe(params):
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
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Secondary points and connection lines
# ---------------------------------------------------------------------------
start1    = np.array([5.87693589, 0.85550847, 0.49798554, 10.12743171, 0.48748133])
start2    = np.array([5.93979891, 0.9842598,  0.55618277,  9.82410481,  0.41788778])
true_pt   = np.array(param_true)

def make_line(start, n_points=102):
    direction = true_pt - start
    t_lows, t_highs = [], []
    for i in range(5):
        if direction[i] > 0:
            t_lows.append((prior_lo[i] - start[i]) / direction[i])
            t_highs.append((prior_hi[i] - start[i]) / direction[i])
        elif direction[i] < 0:
            t_lows.append((prior_hi[i] - start[i]) / direction[i])
            t_highs.append((prior_lo[i] - start[i]) / direction[i])
        else:
            t_lows.append(-np.inf)
            t_highs.append(np.inf)
    t_min = max(t_lows)
    t_max = min(t_highs)
    t_vals = np.sort(np.append(np.linspace(t_min, t_max, n_points), [0.0, 1.0]))
    return start[:, np.newaxis] + t_vals * direction[:, np.newaxis]

line_pts1 = make_line(start1)
line_pts2 = make_line(start2)

# ---------------------------------------------------------------------------
# 2D grid evaluation
# ---------------------------------------------------------------------------
dim_x = args.dim_x
dim_y = args.dim_y
n_grid = args.n_grid

# Key points to guarantee are in the grid
_key_x = [true_pt[dim_x], start1[dim_x], start2[dim_x]]
_key_y = [true_pt[dim_y], start1[dim_y], start2[dim_y]]

x_vals = np.unique(np.sort(np.concatenate([
    np.linspace(param_ranges[dim_x][0], param_ranges[dim_x][1], n_grid),
    _key_x
])))
y_vals = np.unique(np.sort(np.concatenate([
    np.linspace(param_ranges[dim_y][0], param_ranges[dim_y][1], n_grid),
    _key_y
])))
XX, YY = np.meshgrid(x_vals, y_vals)

# Choose reference point for non-plotted dims
_fix_map = {"true": param_true, "sec1": list(start1), "sec2": list(start2)}
fix_pt = _fix_map[args.fix_pt]

_n_pts = XX.size
grid_pts = np.tile(fix_pt, (_n_pts, 1))
grid_pts[:, dim_x] = XX.ravel()
grid_pts[:, dim_y] = YY.ravel()

import pickle as _pkl
_grid_cache = os.path.join(args.outdir, f"heatmap_grid_{dim_x}_{dim_y}_{n_grid}_{args.fix_pt}.pkl")

if os.path.exists(_grid_cache):
    print(f"Loading cached grid: {_grid_cache}")
    with open(_grid_cache, 'rb') as _f:
        XX, YY, logden_grid = _pkl.load(_f)
else:
    print(f"Evaluating {_n_pts} grid points...")
    logden_grid = log_density_safe(grid_pts).reshape(XX.shape)
    with open(_grid_cache, 'wb') as _f:
        _pkl.dump((XX, YY, logden_grid), _f)
    print(f"Grid cached: {_grid_cache}")
print("Done.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.pcolormesh(XX, YY, logden_grid, cmap='viridis', shading='auto')
plt.colorbar(im, ax=ax, label='logden')

ax.plot(line_pts1[dim_x], line_pts1[dim_y], '-',  color='cyan',   linewidth=2,  label='Secondary 1 line')
ax.plot(line_pts2[dim_x], line_pts2[dim_y], '-',  color='orange', linewidth=2,  label='Secondary 2 line')

ax.scatter(true_pt[dim_x],  true_pt[dim_y],  color='red',    s=200, zorder=5, marker='*', label='True Point')
ax.scatter(start1[dim_x],   start1[dim_y],   color='cyan',   s=100, zorder=5, marker='o', label='Secondary 1')
ax.scatter(start2[dim_x],   start2[dim_y],   color='orange', s=100, zorder=5, marker='o', label='Secondary 2')

ax.set_xlabel(labels[dim_x], fontsize=12)
ax.set_ylabel(labels[dim_y], fontsize=12)
ax.set_title(f"fix-pt: {args.fix_pt}", fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.2)
plt.tight_layout()

out_path = os.path.join(args.outdir, f"heatmap_2d_{dim_x}_{dim_y}_{args.fix_pt}.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

# --- No-lines version ---
fig2, ax2 = plt.subplots(figsize=(7, 6))
im2 = ax2.pcolormesh(XX, YY, logden_grid, cmap='viridis', shading='auto')
plt.colorbar(im2, ax=ax2, label='logden')

ax2.scatter(true_pt[dim_x], true_pt[dim_y], facecolors='none', edgecolors='red',    s=200, zorder=5, linewidths=2, marker='o', label='True Point')
ax2.scatter(start1[dim_x],  start1[dim_y],  facecolors='none', edgecolors='cyan',   s=150, zorder=5, linewidths=2, marker='o', label='Secondary 1')
ax2.scatter(start2[dim_x],  start2[dim_y],  facecolors='none', edgecolors='orange', s=150, zorder=5, linewidths=2, marker='o', label='Secondary 2')

ax2.set_xlabel(labels[dim_x], fontsize=12)
ax2.set_ylabel(labels[dim_y], fontsize=12)
ax2.set_title(f"fix-pt: {args.fix_pt}", fontsize=10)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(alpha=0.2)
plt.tight_layout()

out_path2 = os.path.join(args.outdir, f"heatmap_2d_{dim_x}_{dim_y}_{args.fix_pt}_nolines.png")
plt.savefig(out_path2, dpi=150)
print(f"Saved: {out_path2}")
