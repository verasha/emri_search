"""
Compare the actual top logden point (global argmax over the fully-evaluated
LHS grid) across three box/ellipsoid runs, in one corner plot vs the true
injection.
"""
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import corner

files = [
    ('/scratch/e1498138/paris5_noise/lhs_f.pkl',            'N=5e5 (paris5, 3sig)'),
    ('/scratch/e1498138/box/15sig_5e5_ellipsoid/lhs_f.pkl',  'N=5e5 (box, 15sig)'),
    ('/scratch/e1498138/box/20sig_5e5_ellipsoid/lhs_f.pkl',  'N=5e5 (box, 20sig)'),
]

param_names = ['logm1', 'logm2', 'a', 'p0', 'e0']
m1, m2, a, p0, e0 = 1e6, 1e1, 0.7, 9.0, 0.4
true_point = np.array([np.log10(m1), np.log10(m2), a, p0, e0])

all_phys = []
best_points = []
best_lds = []
labels = []

for path, label in files:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    lhs_phys = np.asarray(d['lhs_phys'])
    log_densities = np.asarray(d['log_densities'])
    finite = np.isfinite(log_densities)
    best_idx = np.argmax(np.where(finite, log_densities, -np.inf))
    best_point = lhs_phys[best_idx]
    best_ld = log_densities[best_idx]

    print(f'{label}: actual top logden={best_ld:.4f}  '
          f'point=' + ', '.join(f'{n}={v:.5f}' for n, v in zip(param_names, best_point)))

    all_phys.append(lhs_phys)
    best_points.append(best_point)
    best_lds.append(best_ld)
    labels.append(label)

# ── Corner plot: pooled points set the frame, overplot the injection + the
# actual top-logden point from each file ────────────────────────────────────
pooled = np.vstack(all_phys)
lo = pooled.min(axis=0)
hi = pooled.max(axis=0)
pad = 0.05 * (hi - lo)
box_range = list(zip(lo - pad, hi + pad))

fig = corner.corner(
    pooled,
    labels=param_names,
    range=box_range,
    plot_datapoints=False,
    plot_density=False,
    plot_contours=False,
    hist_kwargs=dict(alpha=0, lw=0),
)

n_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
legend_handles = [plt.Line2D([], [], color='black', marker='+', linestyle='', ms=12, label='injection')]
for label, best_point, best_ld, color in zip(labels, best_points, best_lds, n_colors):
    corner.overplot_points(fig, best_point.reshape(1, -1), color=color, marker='o', ms=8)
    pt_str = ', '.join(f'{n}={v:.4f}' for n, v in zip(param_names, best_point))
    legend_handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='',
                                     label=f'{label}: actual top logden={best_ld:.3f} ({pt_str})'))
corner.overplot_points(fig, true_point.reshape(1, -1), color='black', marker='+', ms=14)

fig.suptitle('Actual top logden per run vs injection', fontsize=10)
fig.legend(handles=legend_handles, loc='upper right')

plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plotdir, exist_ok=True)
plotpath = os.path.join(plotdir, 'corner_compare_top_logden.png')
fig.savefig(plotpath, dpi=150)
plt.close(fig)
print(f'Saved corner plot to {plotpath}')
