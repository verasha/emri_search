"""
Fisher parallelotope ellipse study.

Shows the Fisher-based prior ellipse (from compute_fisher_parallelotope)
centered on the anneal12 best-fit point, overlaid on the anneal12 posterior
corner plot. Marks both the true parameters and the anneal12 best-fit point.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse as MplEllipse
import corner

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

import os
import sys

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_timemax
import parismc

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

from misc import compute_fisher_parallelotope
from lisatools.sensitivity import CornishLISASens

# ─────────────────────────────────────────────
# A. Setup (same as paris2 / ellipse_prior_study)
# ─────────────────────────────────────────────

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu       = True
force_backend = "cuda12x"
dt = 10;  T = 3/12
print(f"dt={dt}s, T={T} years")

inspiral_kwargs  = {"func": 'KerrEccEqFlux', "DENSE_STEPPING": 0, "include_minus_m": False}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs       = {"force_backend": force_backend}
sum_kwargs_comb  = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep   = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

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

m1 = 1e6;  m2 = 1e1;  a = 0.7;  p0 = 9;  e0 = 0.4
xI0 = 1.0; dist = 1.8
qS = np.pi; phiS = 0.; qK = 0.; phiK = 0.
Phi_phi0 = 0.4; Phi_theta0 = 0.0; Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true  = [np.log10(m1), np.log10(m2), a, p0, e0]

loglike_obj = loglike_timemax.LogLikeTimeMax(
    params_star, waveform_gen_comb, gwf,
    verbose=False, waveform_gen_sep=waveform_gen_sep,
    ell=2, n_vals=np.arange(-1, 6), M_mode=None
)
data_snr = float(gwf.rhostat(loglike_obj.signal))
print(f'SNR: {data_snr}')

# ─────────────────────────────────────────────
# B. Load anneal12 → posterior samples + best-fit point
# ─────────────────────────────────────────────

_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

def log_density(params):
    raise RuntimeError("stub")

def prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

S_ANNEAL12 = 30.0

sampler_a12 = parismc.Sampler.load_state(
    './intrinsic_ffunc_3mth_snr32_anneal12/sampler_state.pkl'
)

all_pts_u  = sampler_a12.searched_points_list[0]
all_logden = sampler_a12.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
a12_best   = prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'Anneal12 best-fit (S=1 loglike={all_logden[maxld_idx]/S_ANNEAL12:.4f}): {a12_best}')

samples_a12, weights_a12 = sampler_a12.get_samples_with_weights(flatten=True)
weights_a12 = weights_a12 / weights_a12.sum()
rng = np.random.default_rng(0)
idx_rs = rng.choice(len(samples_a12), size=50_000, replace=True, p=weights_a12)
posterior_samples = samples_a12[idx_rs]   # annealed S=30 posterior samples
del sampler_a12, samples_a12, weights_a12, idx_rs

# ─────────────────────────────────────────────
# C. Fisher parallelotope centered on anneal12 best-fit
# ─────────────────────────────────────────────

# required by compute_fisher_parallelotope (scope bug in misc.py)
channels     = ["A"]
noise_kwargs = {'sens_fn': CornishLISASens, 'return_type': 'PSD'}

N_SIGMA_PRIOR = 3.0

fisher_params_a12 = list(params_star)
fisher_params_a12[0] = 10**a12_best[0]
fisher_params_a12[1] = 10**a12_best[1]
fisher_params_a12[2] = a12_best[2]
fisher_params_a12[3] = a12_best[3]
fisher_params_a12[4] = a12_best[4]

print('Computing Fisher matrix...')
Q, b, meta = compute_fisher_parallelotope(
    ctx             = {'T': T, 'dt': dt},
    fisher_params   = fisher_params_a12,
    params_to_infer = ['m1', 'm2', 'a', 'p0', 'e0'],
    additional_kwargs = {},
    use_gpu         = True,
    _TARGET_SNR     = data_snr,
    prior_sigma_range = N_SIGMA_PRIOR,
    using_evec      = False,
)
print(f'Fisher meta: {meta}')

# Convert to log-mass space
b_log = b.copy()
b_log[0] = b[0] / (fisher_params_a12[0] * np.log(10))
b_log[1] = b[1] / (fisher_params_a12[1] * np.log(10))

sigma_fisher = b_log / N_SIGMA_PRIOR
cov_fisher   = np.diag(sigma_fisher**2)   # diagonal (axis-aligned)
print(f'Fisher 1-sigma (log-mass space): {sigma_fisher}')

# Mahalanobis distances
cov_inv  = np.diag(1.0 / sigma_fisher**2)
diff_true = np.array(param_true) - a12_best
maha_true = np.sqrt(diff_true @ cov_inv @ diff_true)
print(f'Mahalanobis (anneal12 best -> true): {maha_true:.3f}σ')

# ─────────────────────────────────────────────
# D. Corner plot
# ─────────────────────────────────────────────

param_names  = [r'$\log_{10}(m_1)$', r'$\log_{10}(m_2)$', r'$a$', r'$p_0$', r'$e_0$']
param_ranges = [(5.6, 6.4), (0.8, 1.3), (0.3, 0.99), (8.0, 11.0), (0.2, 0.5)]
d = 5

# KDE-inflate covariance for visual matching
smooth_sigma = 1.0
bins         = 40
bin_widths   = np.array([(hi - lo) / bins for lo, hi in param_ranges])
cov_plot     = cov_fisher + np.diag((smooth_sigma * bin_widths)**2)

fig = corner.corner(
    posterior_samples,
    labels=param_names,
    truths=param_true,
    truth_color='red',
    color='steelblue',
    show_titles=True,
    label_kwargs={"fontsize": 10},
    title_kwargs={"fontsize": 10},
    quantiles=[0.16, 0.5, 0.84],
    smooth=True,
    bins=bins,
    plot_datapoints=False,
    hist_kwargs={"density": True, 'linewidth': 2.0},
    linewidth=2.0,
    fill_contours=True,
    range=param_ranges,
)

# Overplot anneal12 best-fit as blue dashed lines
corner.overplot_lines(fig, a12_best, color='blue', lw=1.2, ls='--')

# Draw Fisher ellipses (1σ, 2σ, 3σ) on each off-diagonal panel
def draw_cov_ellipse_2d(ax, mu_2d, cov_2x2, n_sigma, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    angle  = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width  = 2 * n_sigma * np.sqrt(eigvals[1])
    height = 2 * n_sigma * np.sqrt(eigvals[0])
    ell = MplEllipse(xy=mu_2d, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)

axes = np.array(fig.axes).reshape(d, d)
sigmas_to_draw = [1, 2, 3]
colors_sigma   = ['orange', 'darkorange', 'chocolate']

for i in range(d):
    for j in range(i):
        ax = axes[i, j]
        cov_2x2 = cov_plot[np.ix_([j, i], [j, i])]
        mu_2d   = a12_best[[j, i]]
        for ns, col in zip(sigmas_to_draw, colors_sigma):
            draw_cov_ellipse_2d(
                ax, mu_2d, cov_2x2, n_sigma=ns,
                fill=False, edgecolor=col, linewidth=1.5,
                linestyle='--', zorder=5
            )

# Legend
blue_patch  = mpatches.Patch(color='steelblue',  label='Anneal12 posterior (S=30)')
orange_line = plt.Line2D([0], [0], color='orange',     ls='--', lw=1.5, label='Fisher ellipse 1σ')
do_line     = plt.Line2D([0], [0], color='darkorange',  ls='--', lw=1.5, label='Fisher ellipse 2σ')
ch_line     = plt.Line2D([0], [0], color='chocolate',   ls='--', lw=1.5, label='Fisher ellipse 3σ')
blue_line   = plt.Line2D([0], [0], color='blue',  ls='--', lw=1.2, label='Anneal12 best-fit')
red_line    = plt.Line2D([0], [0], color='red',   ls='--', lw=1.2, label='True params')
fig.legend(handles=[blue_patch, orange_line, do_line, ch_line, blue_line, red_line],
           loc='upper right', fontsize=9)
fig.suptitle(f'Fisher prior ellipse (centred on anneal12 best-fit)\n'
             f'Mahalanobis(best→true) = {maha_true:.2f}σ', fontsize=11)

outpath = './plots/fisher_ellipse_corner.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath}')
