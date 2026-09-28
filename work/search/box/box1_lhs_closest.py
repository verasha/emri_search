"""
Fast basin-containment check for the box1 box (mu +/- N_SIGMA*sigma, uniform).
Loads paris4_noise/int_3mth_noise_6 to get the box center and per-dim posterior
sigma, then LHS-samples the box at increasing N (1e5, 5e5, 1e6) WITHOUT
evaluating log_density (that's the expensive step), and reports the LHS point
closest to the true injection at each N (per-dimension sigma-normalized distance;
cov_posterior here is nearly singular, so a full Mahalanobis metric is ill-conditioned).
"""
import argparse
import numpy as np
import pickle
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

import few
from smt.sampling_methods import LHS

parser = argparse.ArgumentParser()
parser.add_argument("--n-sigma", type=float, default=20.0,
                     help="Half-width of the box in units of paris4_noise's posterior sigma")
args = parser.parse_args()
N_SIGMA = args.n_sigma
print(f"Using N_sigma={N_SIGMA}")

os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/search/')

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_pure_noise import LogLike
import parismc

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("WARNING")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12 / 12
print(f"dt={dt}s  T={T}yr  TDI gen={tdi_gen}")

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

m1, m2, a, p0, e0, xI0 = 1e6, 1e1, 0.7, 9.0, 0.4, 1.0
dist, qS, phiS, qK, phiK = 4.5, np.pi, 0., 0., 0.
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5
params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]

n_vals = np.arange(-1, 6)
ell = 2

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
print(f'SNR (rhostat): {float(gwf.SNR(gwf.freq_wave(loglike_obj.signal))):.4f}')


_p2_lo = np.array([5.94889, 0.99336, 0.62652, 8.74104, 0.37427])
_p2_hi = np.array([6.03054, 1.13038, 0.99000, 9.68094, 0.41300])

def _stub_prior_transform(u):
    return _p2_lo + (_p2_hi - _p2_lo) * u

def log_density(params):
    raise RuntimeError("stub")

def prior_transform(u):
    return _stub_prior_transform(u)

import __main__
__main__.log_density    = log_density
__main__.prior_transform = prior_transform

paris4_noise_path = '/scratch/e1498138/paris4_sc/int_1yr_s6/sampler_state.pkl'
print(f'Loading paris4_noise sampler from {paris4_noise_path}...')
sampler_2 = parismc.Sampler.load_state(paris4_noise_path)

all_pts_u  = sampler_2.searched_points_list[0]
all_logden = sampler_2.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = _stub_prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'paris4_noise maxld: {all_logden[maxld_idx]:.4f}')
print(f'paris4_noise maxld point: {mu_center}')

samples_p2, weights_p2 = sampler_2.get_samples_with_weights(flatten=True)
weights_p2 = weights_p2 / weights_p2.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p2), size=50_000, replace=True, p=weights_p2)
cov_posterior = np.cov(samples_p2[idx_rs].T)
print('paris4_noise posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_2, samples_p2, weights_p2, idx_rs

# ── Box bounds (uniform, no clipping to the old p2 box) ────────────────────────

param_names = ['logm1', 'logm2', 'a', 'p0', 'e0']
sigma_diag = np.sqrt(np.diag(cov_posterior))
box_lo = mu_center - N_SIGMA * sigma_diag
box_hi = mu_center + N_SIGMA * sigma_diag

print(f'Box bounds (N_sigma={N_SIGMA}):')
for i, name in enumerate(param_names):
    print(f'  {name}: [{box_lo[i]:.5f}, {box_hi[i]:.5f}]  '
          f'mu={mu_center[i]:.5f}  sigma={sigma_diag[i]:.5f}')

# ── True injection point (in the same logm1/logm2/a/p0/e0 space as mu_center) ──

true_point = np.array([np.log10(m1), np.log10(m2), a, p0, e0])

print(f'True injection point: {true_point}')
dist_from_center = np.sqrt(np.sum(((mu_center - true_point) / sigma_diag) ** 2))
print(f'Per-dim-normalized distance from mu_center to true point: {dist_from_center:.4f} sigma')

# Per-dimension sigma-normalized Euclidean distance. cov_posterior is nearly
# singular here (posterior lives on a ~1D ridge in 5D), so a true Mahalanobis
# distance with the full inverse covariance blows up in the degenerate
# directions and gives a meaningless ranking. This diagonal-only normalization
# is well-conditioned and matches the (axis-aligned) box itself.
def sigma_normalized_distance(points, ref, sigma_diag):
    diff = (points - ref) / sigma_diag
    return np.sqrt(np.sum(diff ** 2, axis=1))

# ── LHS at increasing N: log_density only on the N_CLOSEST nearest points ──────
# (the injection's likelihood surface is an extremely narrow coherent spike, so
# the single nearest-by-parameter-distance point can easily miss it entirely;
# evaluating a neighborhood around it and taking the best logden is more robust)

ndim = 5
N_LIST = [int(1e5), int(5e5), int(1e6)]
N_CLOSEST = 1000

summary = []
all_closest_phys = []
best_points = []
best_lds = []
for N_LHS in N_LIST:
    print(f'\nGenerating {N_LHS} LHS points in {ndim}D box...')
    _lhs_sampler = LHS(xlimits=np.column_stack([np.zeros(ndim), np.ones(ndim)]))
    lhs_u    = np.clip(_lhs_sampler(N_LHS), 0.0, 1.0)
    lhs_phys = box_lo + (box_hi - box_lo) * lhs_u

    dists = sigma_normalized_distance(lhs_phys, true_point, sigma_diag)
    n_closest   = min(N_CLOSEST, N_LHS)
    closest_idx = np.argpartition(dists, n_closest - 1)[:n_closest]

    print(f'Evaluating log_density on the {n_closest} closest points (N={N_LHS})...')
    closest_log_densities = np.full(n_closest, -np.inf)
    for j, idx in enumerate(closest_idx):
        logm1_i, logm2_i, a_i, p0_i, e0_i = lhs_phys[idx]
        try:
            closest_log_densities[j] = loglike_obj(np.array([
                10**logm1_i, 10**logm2_i, a_i, p0_i, e0_i,
                xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            closest_log_densities[j] = -np.inf

    best_j       = np.argmax(closest_log_densities)
    best_idx     = closest_idx[best_j]
    best_point   = lhs_phys[best_idx]
    best_dist    = dists[best_idx]
    best_ld      = closest_log_densities[best_j]
    sigma_offsets = (best_point - true_point) / sigma_diag

    print(f'Top logden among {n_closest} closest points to injection (N={N_LHS}):')
    for i, name in enumerate(param_names):
        print(f'  {name}: {best_point[i]:.5f}  (true={true_point[i]:.5f}, '
              f'offset={sigma_offsets[i]:+.3f} sigma)')
    print(f'Per-dim-normalized distance: {best_dist:.4f} sigma')
    print(f'log_density: {best_ld:.6f}')

    savepath = f'/scratch/e1498138/box/{N_SIGMA:.0f}sig_closest/N{N_LHS}.pkl'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    save_data = {
        'lhs_u':                 lhs_u,
        'lhs_phys':              lhs_phys,
        'box_lo':                box_lo,
        'box_hi':                box_hi,
        'mu_center':             mu_center,
        'cov_posterior':         cov_posterior,
        'sigma_diag':            sigma_diag,
        'true_point':            true_point,
        'closest_idx':           closest_idx,
        'closest_log_densities': closest_log_densities,
        'best_idx':              best_idx,
        'best_dist':             best_dist,
        'best_point':            best_point,
        'best_log_density':      best_ld,
        'N_SIGMA':               N_SIGMA,
        'N_LHS':                 N_LHS,
        'N_CLOSEST':             n_closest,
        'T':                     T,
        'dt':                    dt,
    }
    with open(savepath, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'Saved to {savepath}')

    summary.append((N_LHS, best_dist, sigma_offsets, best_ld))
    all_closest_phys.append(lhs_phys[closest_idx])
    best_points.append(best_point)
    best_lds.append(best_ld)

# ── Summary across N ────────────────────────────────────────────────────────────

print(f'\n=== Summary (N_sigma={N_SIGMA}) ===')
print(f'{"N":>10}  {"dist (sigma)":>16}  {"log_density":>14}  per-dim sigma offsets')
for N_LHS, best_dist, sigma_offsets, best_ld in summary:
    offsets_str = ', '.join(f'{name}={o:+.3f}' for name, o in zip(param_names, sigma_offsets))
    print(f'{N_LHS:>10}  {best_dist:>16.4f}  {best_ld:>14.4f}  {offsets_str}')

# ── Corner plot: prior/box space, injection vs top logden point per N ─────────
# corner.corner needs more sample rows than dims to build its histogram
# skeleton, so use the pooled evaluated closest-points (invisible histograms/
# contours) purely to set up the axes/frame, then overplot one marker per N
# plus the injection cross.
box_range = list(zip(box_lo, box_hi))
fig = corner.corner(
    np.vstack(all_closest_phys),
    labels=param_names,
    range=box_range,
    plot_datapoints=False,
    plot_density=False,
    plot_contours=False,
    hist_kwargs=dict(alpha=0, lw=0),
)
n_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
legend_handles = [plt.Line2D([], [], color='black', marker='+', linestyle='', ms=12, label='injection')]
for N_LHS, best_point, best_ld, color in zip(N_LIST, best_points, best_lds, n_colors):
    corner.overplot_points(fig, best_point.reshape(1, -1),
                           color=color, marker='o', ms=8)
    pt_str = ', '.join(f'{name}={v:.4f}' for name, v in zip(param_names, best_point))
    legend_handles.append(plt.Line2D([], [], color=color, marker='o', linestyle='',
                                     label=f'N={N_LHS:.0e}: logden={best_ld:.3f} ({pt_str})'))
corner.overplot_points(fig, true_point.reshape(1, -1), color='black', marker='+', ms=14)
fig.suptitle(f'N_sigma={N_SIGMA:.0f} for 1 yr templates', fontsize=10)
fig.legend(handles=legend_handles, loc='upper right')

plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(plotdir, exist_ok=True)
plotpath = os.path.join(plotdir, f'corner_{N_SIGMA:.0f}sig_1yr.png')
fig.savefig(plotpath, dpi=150)
plt.close(fig)
print(f'Saved corner plot to {plotpath}')
