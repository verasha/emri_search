"""
Converted from study/paris1_noise_s12.ipynb.

Loads the paris1 semi-coherent (noise, 1yr, N_seg=12) sampler state and
produces the same diagnostic plots as the notebook, saved to work/search/paper/.
"""

import numpy as np
import few
import os
import sys
import pickle

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_timemax_noise import LogLike

import parismc
import cupy as cp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import corner

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
outdir = os.path.join(dir_work, 'search', 'paper', 'paris1_noise_s12')
os.makedirs(outdir, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(outdir, name)
    fig.savefig(path)
    print(f'  -> saved {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Setup (waveform, noise, likelihood)
# ---------------------------------------------------------------------------
use_gpu = True
tdi_gen = 1
dt = 5
T = 12/12
N_SEGS = 12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_SEGS}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source parameters
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
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

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


def log_density(params):
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i = params[i]
        try:
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                T=T, dt=dt,
            ))
            out[i] = float(gwf.SNR_semicoherent(loglike_obj.signal, h_temp, N_seg=N_SEGS))
        except Exception:
            pass
    return out


def prior_transform(u):
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99]
    p0lim = [8.0, 11.0]
    e0lim = [0.2, 0.5]
    t = np.zeros_like(u)
    t[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    t[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    t[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    t[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]
    t[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    return t


def inverse_prior_transform(params):
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99]
    p0lim = [8.0, 11.0]
    e0lim = [0.2, 0.5]
    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    return u


# ---------------------------------------------------------------------------
# Load sampler state
# ---------------------------------------------------------------------------
dir_scratch = '/scratch/e1498138/'
savepath = dir_scratch + 'paris1_sc/int_1yr_s12'

print(f'Loading sampler state from {savepath}...')
sampler = parismc.Sampler.load_state(savepath + '/sampler_state.pkl')

samples, weights = sampler.get_samples_with_weights(flatten=True)
proc_pt = sampler.searched_points_list
logden_list = sampler.searched_log_densities_list

maxld_pt1 = prior_transform(proc_pt[0][np.argmax(logden_list)].reshape(1, -1))
print('Max logden point (physical):', maxld_pt1)
print('log_density at MAP:', log_density(maxld_pt1))
print('log_density at truth:', log_density([param_true]))

param_ranges = [(5.6, 6.4), (0.8, 1.3), (0.3, 0.99), (8.0, 11.0), (0.2, 0.5)]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

all_logden = np.concatenate(logden_list)
sorted_idx = np.argsort(all_logden).flatten()
top10_idx = sorted_idx[-10:][::-1]
top10_pts = prior_transform(proc_pt[0][top10_idx])

for k, avg in enumerate(top10_pts.T.mean(axis=1)):
    print(f'  top10 avg[{k}] = {avg}')

# ---------------------------------------------------------------------------
# Plot 1: full corner plot with MAP overlay
# ---------------------------------------------------------------------------
print('Plotting corner...')
labels = [r'$\log_{10}(m_1)$', r'$\log_{10}(m_2)$', r'$a$', r'$p_0$', r'$e_0$']
fig = corner.corner(
    samples,
    weights=weights,
    labels=labels,
    truths=param_true,
    truth_color='red',
    color='green',
    show_titles=True,
    label_kwargs={"fontsize": 10},
    title_kwargs={"fontsize": 12},
    quantiles=[0.16, 0.5, 0.84],
    smooth=True,
    bins=30,
    plot_datapoints=False,
    hist_kwargs={"density": True, 'linewidth': 2.5},
    linewidth=2.5,
    fill_contours=True,
    range=param_ranges,
)
corner.overplot_points(fig, maxld_pt1.reshape(1, -1),
                       color='blue', marker='*', ms=10,
                       reverse=False)
savefig(fig, 'corner_full.png')

# ---------------------------------------------------------------------------
# Plot 2: Fisher 2-sigma ellipse vs recovered point (logm1 vs p0)
# ---------------------------------------------------------------------------
print('Plotting Fisher ellipse...')
cov_path = os.path.join(dir_work, 'cov_matrix_noise_1yr.pkl')
with open(cov_path, 'rb') as f:
    cov = pickle.load(f)


def draw_cov_ellipse(ax, cx, cy, cov2d, n_sigma=1, **kwargs):
    vals, vecs = np.linalg.eigh(cov2d)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ell = Ellipse(xy=(cx, cy),
                   width=2 * n_sigma * np.sqrt(vals[0]),
                   height=2 * n_sigma * np.sqrt(vals[1]),
                   angle=angle, **kwargs)
    ax.add_patch(ell)


_pt = np.asarray(param_true).ravel()
_map = np.asarray(maxld_pt1).ravel()
sigs = np.sqrt(np.diag(cov))

j, i = 0, 3   # x = log10(m1), y = p0
cov2d = cov[np.ix_([j, i], [j, i])]

xlo = min(_pt[j], _map[j]); xhi = max(_pt[j], _map[j])
ylo = min(_pt[i], _map[i]); yhi = max(_pt[i], _map[i])
xpad = (xhi - xlo) * 0.05 + 3 * sigs[j]
ypad = (yhi - ylo) * 0.05 + 3 * sigs[i]

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_xlim(xlo - xpad, xhi + xpad)
ax.set_ylim(ylo - ypad, yhi + ypad)

ax.axvline(_pt[j], color='red', lw=1.0, alpha=0.5, zorder=1)
ax.axhline(_pt[i], color='red', lw=1.0, alpha=0.5, zorder=1)
ax.plot(_pt[j], _pt[i], '+', color='red', ms=13, mew=2.5, zorder=3)
ax.plot(_map[j], _map[i], '*', color='blue', ms=13, zorder=4)

draw_cov_ellipse(ax, _pt[j], _pt[i], cov2d,
                  n_sigma=2, facecolor='none', edgecolor='green', lw=3, zorder=2)

ax.set_xlabel(r'$\log m_1$')
ax.set_ylabel(r'$p_0$')

legend_handles = [
    Line2D([0], [0], color='red', marker='+', ls='', ms=12, mew=2.5, alpha=0.5, label='Injection'),
    Line2D([0], [0], color='blue', marker='*', ls='', ms=13, label='Recovered'),
    Line2D([0], [0], color='green', ls='-', lw=3, label='Fisher 2σ'),
]
ax.legend(handles=legend_handles, frameon=False, fontsize=12)
plt.tight_layout()
savefig(fig, 'fisher_ellipse_logm1_p0.png')

# ---------------------------------------------------------------------------
# Plot 3: pairwise injection-vs-MAP grid (full prior range per panel)
# ---------------------------------------------------------------------------
print('Plotting pairwise injection-vs-MAP grid...')
plt.rcParams.update({
    'font.size': 13, 'axes.labelsize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

labels = [r'$\log m_1$', r'$\log m_2$', r'$a$', r'$p_0$', r'$e_0$']
ndim = len(labels)

param_true_arr = np.asarray(param_true).ravel()
maxld_pt1_arr = np.asarray(maxld_pt1).ravel()

fig, axes = plt.subplots(ndim, ndim, figsize=(9, 9))

for i in range(ndim):
    for j in range(ndim):
        ax = axes[i, j]
        if j >= i:
            ax.set_visible(False)
            continue

        ax.axvline(param_true_arr[j], color='red', lw=1.0, alpha=0.7, zorder=1)
        ax.axhline(param_true_arr[i], color='red', lw=1.0, alpha=0.7, zorder=1)
        ax.plot(param_true_arr[j], param_true_arr[i], '+', color='red',
                ms=13, mew=2.5, zorder=3)

        ax.plot(maxld_pt1_arr[j], maxld_pt1_arr[i], '*', color='blue',
                ms=13, zorder=4)

        ax.set_xlim(*param_ranges[j])
        ax.set_ylim(*param_ranges[i])

        if j == 0:
            ax.set_ylabel(labels[i])
        else:
            ax.set_yticklabels([])
        if i == ndim - 1:
            ax.set_xlabel(labels[j])
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.set_xticklabels([])

handles = [
    Line2D([0], [0], color='red', marker='+', ls='', ms=12, mew=2.5, label='Injection'),
    Line2D([0], [0], color='blue', marker='*', ls='', ms=13, label='Recovered'),
]
fig.legend(handles=handles, loc='upper right', frameon=False, fontsize=13)
fig.tight_layout()
savefig(fig, 'pairwise_injection_vs_map.png')

# ---------------------------------------------------------------------------
# Plot 4: top-10 points corner overlay
# ---------------------------------------------------------------------------
print('Plotting top10 corner overlay...')
ranges = param_ranges
true_params = np.array([6.0, 1.0, 0.7, 9.0, 0.4])

_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']

all_pts = np.vstack([top10_pts, [true_params]])
fig = corner.corner(all_pts, labels=labels, show_titles=False,
                     plot_datapoints=False, plot_density=False, plot_contours=False, ranges=ranges,
                     hist_kwargs=dict(alpha=0, lw=0))

for i, pt in enumerate(top10_pts):
    corner.overplot_points(fig, pt.reshape(1, -1), color=_cycle[i % len(_cycle)], marker=markers_list[i], ms=8)

corner.overplot_points(fig, np.array([true_params]), color='black', marker='+', ms=10)

axes = np.array(fig.axes).reshape((ndim, ndim))
for row in range(ndim):
    for col in range(ndim):
        if col <= row:
            axes[row, col].set_xlim(ranges[col])
        if col < row:
            axes[row, col].set_ylim(ranges[row])

ld_vals = all_logden[top10_idx]
handles = [Line2D([0], [0], color=_cycle[i % len(_cycle)], marker=markers_list[i], ls='', ms=7,
                   label=f"pt {i+1} (ld={ld_vals[i]:.3f})")
           for i in range(10)]
handles.append(Line2D([0], [0], color='black', marker='+', ls='', ms=8, label='true'))
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=7)
savefig(fig, 'top10_corner.png')

# ---------------------------------------------------------------------------
# Plot 5: top-N corner overlays for N = 10, 20, 50, 100
# ---------------------------------------------------------------------------
print('Plotting topN corner overlays...')


def plot_topN(N, color='steelblue', marker='o', ms=5):
    idx = sorted_idx[-N:][::-1]
    pts = prior_transform(proc_pt[0][idx])
    ld_vals = all_logden[idx]

    all_pts = np.vstack([pts, [true_params]])
    fig = corner.corner(all_pts, labels=labels, show_titles=False,
                         plot_datapoints=False, plot_density=False, plot_contours=False, ranges=ranges,
                         hist_kwargs=dict(alpha=0, lw=0))

    corner.overplot_points(fig, pts, color=color, marker=marker, ms=ms, alpha=0.6)
    corner.overplot_points(fig, np.array([true_params]), color='black', marker='+', ms=12, zorder=10)

    axes_arr = np.array(fig.axes).reshape((ndim, ndim))
    for row in range(ndim):
        for col in range(ndim):
            if col <= row:
                axes_arr[row, col].set_xlim(ranges[col])
            if col < row:
                axes_arr[row, col].set_ylim(ranges[row])

    handles = [
        Line2D([0], [0], color=color, marker=marker, ls='', ms=7,
               label=f'top {N}  (ld: {ld_vals[-1]:.3f} - {ld_vals[0]:.3f})'),
        Line2D([0], [0], color='black', marker='+', ls='', ms=9, label='true'),
    ]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8)
    fig.suptitle(f'Top {N} LHS points', fontsize=10, y=1.01)
    savefig(fig, f'top{N}_corner.png')


for N in [10, 20, 50, 100]:
    plot_topN(N)

# ---------------------------------------------------------------------------
# Plot 6: 1D log-density slices along the MAP -> truth connecting line
# ---------------------------------------------------------------------------
print('Plotting connection-line 1D slices...')
proc1_maxld_pt_1d = maxld_pt1[0]
true_pt = np.array(param_true)

n_points = 50
t_values = np.linspace(0, 1, n_points)
line_points_proc1 = proc1_maxld_pt_1d[:, np.newaxis] + t_values * (true_pt - proc1_maxld_pt_1d)[:, np.newaxis]

logden_theory_proc1 = log_density(np.array(line_points_proc1).T)

fig_1d, axs_1d = plt.subplots(1, 5, figsize=(20, 4))
labels_1d = [r'$\log_{10}(m_1)$', r'$\log_{10}(m_2)$', r'$a$', r'$p_0$', r'$e_0$']
for dim in range(5):
    ax = axs_1d[dim]
    ax.plot(line_points_proc1[dim], logden_theory_proc1, '-',
            color='blue', alpha=0.5, linewidth=2, label='Computed f-stat')
    ax.axvline(proc1_maxld_pt_1d[dim], color='blue', linestyle='--',
               alpha=0.5, label='Proc1 Max Logden Point')
    ax.axvline(param_true[dim], color='red', linestyle='--',
               alpha=0.5, label='True Point')
    ax.set_xlabel(labels_1d[dim], fontsize=12)
    ax.grid(True, alpha=0.3)
axs_1d[0].set_ylabel('logden', fontsize=12)
axs_1d[-1].legend()
fig_1d.tight_layout()
savefig(fig_1d, 'connection_line_1d_slices.png')

# ---------------------------------------------------------------------------
# Plot 7: full prior-range 1D slice along logm1, through MAP and truth
# ---------------------------------------------------------------------------
print('Plotting full-prior-range logm1 slice...')
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])

start = proc1_maxld_pt_1d
direction = true_pt - proc1_maxld_pt_1d

i = 0  # logm1
if direction[i] > 0:
    t_lo_logm1 = (prior_lo[i] - start[i]) / direction[i]
    t_hi_logm1 = (prior_hi[i] - start[i]) / direction[i]
else:
    t_lo_logm1 = (prior_hi[i] - start[i]) / direction[i]
    t_hi_logm1 = (prior_lo[i] - start[i]) / direction[i]

n_points = 200
t_values_logm1 = np.linspace(t_lo_logm1, t_hi_logm1, n_points)

t_true = (true_pt[i] - start[i]) / direction[i]
secondary_point = proc1_maxld_pt_1d.copy()
t_secondary = (secondary_point[i] - start[i]) / direction[i]

t_values_logm1 = np.sort(np.concatenate([t_values_logm1, [t_true, t_secondary]]))
line_points_logm1 = start[:, np.newaxis] + t_values_logm1 * direction[:, np.newaxis]

logden_tm = log_density(line_points_logm1.T)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(line_points_logm1[0], logden_tm, '-', color='blue', linewidth=2, label='f-stat (1yr) timemax')
ax.axvline(proc1_maxld_pt_1d[0], color='blue', linestyle='--', alpha=0.5, label='Max Logden Point')
ax.axvline(param_true[0], color='red', linestyle='--', alpha=0.5, label='True Point')
ax.set_xlabel(r'$\log_{10}(m_1)$', fontsize=12)
ax.set_ylabel('logden', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
savefig(fig, 'full_prior_range_logm1_slice.png')

print(f'\nDone. All plots saved to {outdir}')
