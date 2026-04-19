"""
Check LHS-in-Cholesky-space seeding:
  - Generate LHS in [-1,1]^5, filter by ||z|| <= 1
  - Transform to physical space via mu_center + N_SIGMA * L @ z
  - Corner plot: filtered points + dashed 3sigma ellipse + hypercube bounds + true params
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse as MplEllipse
from scipy.special import gamma
from smt.sampling_methods import LHS
import corner
import sys
import os

sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/parismc')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
import parismc

# ── paris2 setup ──────────────────────────────────────────────────────────────
_p2_lo = np.array([5.6, 0.8, 0.3, 8.0, 0.2])
_p2_hi = np.array([6.4, 1.3, 0.99, 11.0, 0.5])

def log_density(x): pass
def prior_transform_p2(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

import __main__
__main__.log_density       = log_density
__main__.prior_transform   = prior_transform_p2

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sampler_2 = parismc.Sampler.load_state(
    './intrinsic_ffunc_3mth_snr32_paris2/sampler_state.pkl'
)

all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
mu_center  = prior_transform_p2(all_pts_u[np.argmax(all_logden)].reshape(1, -1))[0]

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng = np.random.default_rng(0)
idx = rng.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx].T)
del sampler_2, samples_p2, weights_p2, idx

# ── ellipse + hypercube definition ────────────────────────────────────────────
N_SIGMA_PRIOR = 3.0
sigma_diag    = np.sqrt(np.diag(cov_posterior))
ellipse_lo    = np.clip(mu_center - N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
ellipse_hi    = np.clip(mu_center + N_SIGMA_PRIOR * sigma_diag, _p2_lo, _p2_hi)
cov_inv       = np.linalg.inv(cov_posterior)
_L            = np.linalg.cholesky(cov_posterior)

param_names = [r'$\log_{10}(m_1)$', r'$\log_{10}(m_2)$', r'$a$', r'$p_0$', r'$e_0$']
param_true  = [6.0, 1.0, 0.7, 9.0, 0.4]
d = 5

# ── generate LHS in Cholesky space, filter by sphere ─────────────────────────
N_LHS = int(1e5)
_lhs  = LHS(xlimits=np.column_stack([-np.ones(d), np.ones(d)]))
lhs_z = _lhs(N_LHS)
mask  = np.sum(lhs_z ** 2, axis=1) <= 1.0
lhs_z_in   = lhs_z[mask]
lhs_phys   = mu_center + N_SIGMA_PRIOR * (_L @ lhs_z_in.T).T
print(f'LHS points inside ellipse: {mask.sum()} / {N_LHS}  ({100*mask.mean():.1f}%)')

# verify all are inside ellipse via Mahalanobis
diffs  = lhs_phys - mu_center
maha2  = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
print(f'Max Mahalanobis distance: {np.sqrt(maha2.max()):.4f}  (should be <= {N_SIGMA_PRIOR})')

# ── helper: draw 2D marginal ellipse ─────────────────────────────────────────
def draw_cov_ellipse_2d(ax, mu_2d, cov_2x2, n_sigma, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    angle  = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width  = 2 * n_sigma * np.sqrt(eigvals[1])
    height = 2 * n_sigma * np.sqrt(eigvals[0])
    ell = MplEllipse(xy=mu_2d, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)

# ── corner plot ───────────────────────────────────────────────────────────────
param_ranges = [(lo, hi) for lo, hi in zip(ellipse_lo, ellipse_hi)]

fig = corner.corner(
    lhs_phys,
    labels=param_names,
    truths=param_true,
    truth_color='red',
    color='steelblue',
    show_titles=True,
    label_kwargs={"fontsize": 10},
    title_kwargs={"fontsize": 10},
    plot_datapoints=True,
    plot_density=True,
    fill_contours=False,
    range=param_ranges,
    data_kwargs={"alpha": 0.1, "ms": 1},
)

axes = np.array(fig.axes).reshape(d, d)

# draw 3sigma ellipse (dashed) and hypercube bounds on each 2D panel
for i in range(d):
    for j in range(i):
        ax = axes[i, j]
        cov_2x2 = cov_posterior[np.ix_([j, i], [j, i])]
        mu_2d   = mu_center[[j, i]]
        draw_cov_ellipse_2d(ax, mu_2d, cov_2x2, n_sigma=N_SIGMA_PRIOR,
                            fill=False, edgecolor='orange', linewidth=1.8,
                            linestyle='--', zorder=5, label='3σ ellipse')
        # hypercube bounds
        ax.axvline(ellipse_lo[j], color='gray', ls=':', lw=1.2)
        ax.axvline(ellipse_hi[j], color='gray', ls=':', lw=1.2)
        ax.axhline(ellipse_lo[i], color='gray', ls=':', lw=1.2)
        ax.axhline(ellipse_hi[i], color='gray', ls=':', lw=1.2)

# mu_center crosshair
corner.overplot_lines(fig, mu_center, color='blue', lw=1.0, ls='--')

# legend
blue_patch  = mpatches.Patch(color='steelblue', label='LHS inside ellipse')
orange_line = plt.Line2D([0], [0], color='orange', ls='--', lw=1.8, label=f'{N_SIGMA_PRIOR:.0f}σ ellipse')
gray_line   = plt.Line2D([0], [0], color='gray',   ls=':',  lw=1.2, label='hypercube bounds')
blue_line   = plt.Line2D([0], [0], color='blue',   ls='--', lw=1.0, label='paris2 maxld (mu_center)')
red_line    = plt.Line2D([0], [0], color='red',    ls='--', lw=1.0, label='true params')
fig.legend(handles=[blue_patch, orange_line, gray_line, blue_line, red_line],
           loc='upper right', fontsize=9)
fig.suptitle('LHS seeds (Cholesky-space filtered) vs 3σ ellipse & hypercube', fontsize=11)

outpath = './plots/lhs_ellipse_check.png'
os.makedirs('./plots', exist_ok=True)
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved: {outpath}')
