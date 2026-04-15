"""
Ellipsoidal prior study for PARIS search.

Compares:
  1. Box prior (flat prior)
  2. Ellipsoidal prior derived from weighted covariance of paris2_2 samples

Sections:
  A. Full setup (loglike_pure / GWfuncs_pure, non-phasemax)
  B. Load paris2_2 samples + compute weighted covariance
  C. Ellipse construction helpers
  D. Corner plot: ellipse samples vs paris2_2 posterior
  E. Volume efficiency: ellipse/box ratio as function of dimension d
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import gamma
import corner
import pickle

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.constants import YRSID_SI

import os
import sys

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_timemax
import parismc
import cupy as cp

# ─────────────────────────────────────────────
# A. Full setup (loglike_timemax / GWfuncs — matches paris2_2)
# ─────────────────────────────────────────────

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 3/12
print(f"Using dt={dt}s, T={T} years")

inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}
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

n_vals = np.arange(-1, 6)
ell    = 2

loglike_obj = loglike_timemax.LogLikeTimeMax(
    params_star, waveform_gen_comb, gwf,
    verbose=False, waveform_gen_sep=waveform_gen_sep,
    ell=ell, n_vals=n_vals, M_mode=None
)

data     = loglike_obj.signal
data_snr = gwf.rhostat(data)
print('SNR:', data_snr)

S = 1

def log_density(params):
    params = np.asarray(params)
    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)
    for i in range(n_samples):
        logm1, logm2, a_i, p0_i, e0_i = params[i]
        try:
            loglike = S * loglike_obj(np.array([
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            loglike = -np.inf
        log_likes[i] = loglike
    return log_likes

def prior_transform(u):
    logm1lim = [5.6, 6.4];  logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99];     p0lim = [8.0, 11.0];  e0lim = [0.2, 0.5]
    transformed = np.zeros_like(u)
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    transformed[:, 2] = (alim[1]     - alim[0])     * u[:, 2] + alim[0]
    transformed[:, 3] = (p0lim[1]    - p0lim[0])    * u[:, 3] + p0lim[0]
    transformed[:, 4] = (e0lim[1]    - e0lim[0])    * u[:, 4] + e0lim[0]
    return transformed

def inverse_prior_transform(params):
    logm1lim = [5.6, 6.4];  logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99];     p0lim = [8.0, 11.0];  e0lim = [0.2, 0.5]
    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0])     / (alim[1]     - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0])    / (p0lim[1]    - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0])    / (e0lim[1]    - e0lim[0])
    return u

print('Setup done.')

# ─────────────────────────────────────────────
# B. Load paris2_2 samples & compute covariance
# ─────────────────────────────────────────────

param_names  = [r'$\log_{10}(m_1)$', r'$\log_{10}(m_2)$', r'$a$', r'$p_0$', r'$e_0$']
param_ranges = [(5.6, 6.4), (0.8, 1.3), (0.3, 0.99), (8.0, 11.0), (0.2, 0.5)]
prior_lo = np.array([r[0] for r in param_ranges])
prior_hi = np.array([r[1] for r in param_ranges])
d = 5

sampler = parismc.Sampler.load_state(
    './search/intrinsic_ffunc_3mth_snr32_paris2_2/sampler_state.pkl'
)

# Best-fit (maxld) point in physical space
_pts_u   = sampler.searched_points_list[0]
_logdens = sampler.searched_log_densities_list[0]
mu_center = prior_transform(_pts_u[np.argmax(_logdens)].reshape(1, -1))[0]
print(f'paris2_2 maxld = {np.max(_logdens):.4f}')
print(f'mu_center (maxld): {mu_center}')

# Importance-weight resampling from the posterior
samples, weights = sampler.get_samples_with_weights(flatten=True)
weights = weights / weights.sum()
n_resample = 50_000
rng_resample = np.random.default_rng(0)
idx = rng_resample.choice(len(samples), size=n_resample, replace=True, p=weights)
posterior_samples = samples[idx]

cov_paris2_2 = np.cov(posterior_samples.T)   # keep annealed (tighter)

print("mu_center (maxld):         ", mu_center)
print("Posterior 1-sigma (diag):  ", np.sqrt(np.diag(cov_paris2_2)))

# corner.corner with smooth=True applies gaussian_filter(sigma=smooth) on the
# 2D histogram in bin units. Each bin has width = range/bins in physical space.
# This adds (smooth * bin_width)^2 of apparent variance in each dimension.
# To visually match the corner contours, inflate the ellipse covariance by this.
smooth_sigma = 1.0
bins         = 40
bin_widths   = np.array([(hi - lo) / bins for lo, hi in param_ranges])
cov_plot     = cov_paris2_2 + np.diag((smooth_sigma * bin_widths) ** 2)

print("KDE bin widths:            ", bin_widths)
print("True 1-sigma:              ", np.sqrt(np.diag(cov_paris2_2)))
print("Apparent 1-sigma (w/ KDE): ", np.sqrt(np.diag(cov_plot)))

# ─────────────────────────────────────────────
# C. Ellipse helpers
# ─────────────────────────────────────────────

def sample_ellipse_uniform(mu, cov, n_samples, n_sigma=3.0, rng=None):
    """Sample uniformly inside the n_sigma-sigma ellipse via Muller's method."""
    if rng is None:
        rng = np.random.default_rng()
    d = len(mu)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_samples, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    r = rng.uniform(0, 1, n_samples) ** (1.0 / d)
    pts = mu + n_sigma * r[:, None] * (L @ z.T).T
    return pts

def ellipse_volume(cov, n_sigma=3.0):
    d = cov.shape[0]
    return (np.pi ** (d / 2) / gamma(d / 2 + 1)
            * n_sigma**d * np.sqrt(np.linalg.det(cov)))

def box_volume_from_cov(cov, n_sigma=3.0):
    return np.prod(2 * n_sigma * np.sqrt(np.diag(cov)))

def box_volume_prior(param_ranges):
    return np.prod([hi - lo for lo, hi in param_ranges])

# ─────────────────────────────────────────────
# D. Corner plot: 2D marginal ellipse patches vs paris2_2 posterior
# ─────────────────────────────────────────────

from matplotlib.patches import Ellipse as MplEllipse

def draw_cov_ellipse_2d(ax, mu_2d, cov_2x2, n_sigma, **kwargs):
    """Draw the n_sigma confidence ellipse from a 2x2 covariance on ax."""
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    # eigvals sorted ascending; eigvecs[:, i] is the i-th eigenvector
    angle  = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width  = 2 * n_sigma * np.sqrt(eigvals[1])   # major axis
    height = 2 * n_sigma * np.sqrt(eigvals[0])   # minor axis
    ell = MplEllipse(xy=mu_2d, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)

fig = corner.corner(
    posterior_samples,
    labels=param_names,
    truths=param_true,
    truth_color='red',
    color='green',
    show_titles=True,
    label_kwargs={"fontsize": 10},
    title_kwargs={"fontsize": 10},
    quantiles=[0.16, 0.5, 0.84],
    smooth=True,
    bins=40,
    plot_datapoints=False,
    hist_kwargs={"density": True, 'linewidth': 2.0},
    linewidth=2.0,
    fill_contours=True,
    range=param_ranges,
)
corner.overplot_lines(fig, mu_center, color='blue', lw=1.2, ls='--')

# Draw 2D marginal ellipses (1σ, 2σ, 3σ) on each off-diagonal panel
axes = np.array(fig.axes).reshape(d, d)
sigmas_to_draw = [1, 2, 3]
colors_sigma   = ['orange', 'darkorange', 'chocolate']

for i in range(d):
    for j in range(i):
        ax = axes[i, j]
        cov_2x2 = cov_plot[np.ix_([j, i], [j, i])]
        mu_2d   = mu_center[[j, i]]
        for ns, col in zip(sigmas_to_draw, colors_sigma):
            draw_cov_ellipse_2d(
                ax, mu_2d, cov_2x2, n_sigma=ns,
                fill=False, edgecolor=col, linewidth=1.5,
                linestyle='--', zorder=5
            )

green_patch = mpatches.Patch(color='green', label='paris2_2 posterior')
orange_line = plt.Line2D([0], [0], color='orange',    ls='--', lw=1.5, label='Ellipse 1σ')
do_line     = plt.Line2D([0], [0], color='darkorange', ls='--', lw=1.5, label='Ellipse 2σ')
ch_line     = plt.Line2D([0], [0], color='chocolate',  ls='--', lw=1.5, label='Ellipse 3σ')
blue_line   = plt.Line2D([0], [0], color='blue',  ls='--', label='paris2_2 maxld point')
red_line    = plt.Line2D([0], [0], color='red',   ls='--', label='True params')
fig.legend(handles=[green_patch, orange_line, do_line, ch_line, blue_line, red_line],
           loc='upper right', fontsize=9)
fig.suptitle('paris2_2 posterior vs Ellipsoidal prior (2D marginals)', fontsize=12)
plt.savefig('./search/plots/ellipse_prior_corner_2.png', dpi=150, bbox_inches='tight')
print("Saved: ./search/plots/ellipse_prior_corner_2.png")

# ─────────────────────────────────────────────
# E. Volume efficiency: ellipse/box ratio vs dimension d
# ─────────────────────────────────────────────

dims = np.arange(1, 21)
n_sigma_eff = 3.0
ratios = []
for dd in dims:
    V_ell = (np.pi ** (dd / 2) / gamma(dd / 2 + 1)) * n_sigma_eff**dd
    V_box = (2 * n_sigma_eff) ** dd
    ratios.append(V_ell / V_box)
ratios = np.array(ratios)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(dims, ratios, 'o-', color='royalblue', lw=2, ms=6)
ax2.axvline(d, color='red', ls='--', label=f'Our d={d}')
ax2.annotate(f'  d={d}: {ratios[d-1]*100:.2f}% of box',
             xy=(d, ratios[d-1]), xytext=(d+0.5, ratios[d-1]*3),
             fontsize=10, color='red')
ax2.set_xlabel('Number of dimensions', fontsize=12)
ax2.set_ylabel(r'$V_{\rm ellipse} / V_{\rm box}$', fontsize=12)
ax2.set_title(f'Volume efficiency: {n_sigma_eff:.0f}σ ellipse vs bounding box', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig('./search/plots/ellipse_volume_efficiency_2.png', dpi=150, bbox_inches='tight')
print("Saved: ./search/plots/ellipse_volume_efficiency_2.png")

print("\n=== Volume summary ===")
for ns in [1.0, 3.0, 5.0]:
    V_ell   = ellipse_volume(cov_paris2_2, n_sigma=ns)
    V_box_c = box_volume_from_cov(cov_paris2_2, n_sigma=ns)
    V_prior = box_volume_prior(param_ranges)
    print(f"  n_sigma={ns:.0f}:  V_ellipse={V_ell:.3e}  V_cov_box={V_box_c:.3e}"
          f"  V_prior={V_prior:.3e}  ellipse/prior={V_ell/V_prior*100:.6f}%")

print("\nDone.")
