import numpy as np
import few
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')  # headless: save figures to files only
import matplotlib.pyplot as plt
import corner

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_timemax_noise import LogLike

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

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


print('Setting up ParisMC...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,
    alpha=int(1e3),
    trail_size=int(1e3),
    boundary_limiting=True,
    use_beta=True,
    integral_num=int(1e5),
    gamma=500,
    exclude_scale_z=np.inf,
    use_pool=False,
    keep_dead_processes=True,
    seed=6342,
    merge_type='distance',
)

ndim = 5
n_seed = 10
sigma = 0.01
init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config,
)

print('Getting LHS points...')
dir_scratch = '/scratch/e1498138'
lhs_path = '/scratch/e1498138/lhs/noise/semicoherent/ckpt_1e5_1yr/final.pkl'

with open(lhs_path, 'rb') as f:
    phys_pts, det_snr = pickle.load(f)

valid = np.isfinite(det_snr)
external_lhs_points = inverse_prior_transform(phys_pts[valid])
external_lhs_log_densities = det_snr[valid]
print(f'Loaded {valid.sum()} / {len(det_snr)} finite LHS evaluations.')

savepath = dir_scratch + f'/paris1_sc/int_1yr_s{N_SEGS}'

plot_dir = os.path.join(savepath, 'plots')
os.makedirs(plot_dir, exist_ok=True)

labels = [r'$\log_{10} m_1$', r'$\log_{10} m_2$', r'$a$', r'$p_0$', r'$e_0$']
param_ranges = [(5.6, 6.4), (0.8, 1.3), (0.3, 0.99), (8.0, 11.0), (0.2, 0.5)]


def make_plots(sampler, tag):
    """Save corner plot, log-density traces, and best-point scatter as PNGs."""
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    if len(samples) == 0:
        return

    # Weighted corner plot with true parameters marked
    fig = corner.corner(
        samples, weights=weights, labels=labels, range=param_ranges,
        truths=param_true, truth_color='red', show_titles=True,
    )
    fig.savefig(os.path.join(plot_dir, f'corner_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Per-process log-density (semi-coherent SNR) traces
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(sampler.n_proc):
        n = sampler.element_num_list[j]
        ld = sampler.searched_log_densities_list[j][:n]
        ax.plot(ld, alpha=0.6, lw=0.5, label=f'proc {j} (max {sampler.max_logden_list[j]:.2f})')
    ax.set_xlabel('sample index')
    ax.set_ylabel('log density (semi-coherent SNR)')
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(os.path.join(plot_dir, f'logden_trace_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Searched points coloured by log density, all 2D parameter pairs
    samples_list, _ = sampler.get_samples_with_weights(flatten=False)
    pts = np.concatenate([samples_list[j][:sampler.element_num_list[j]] for j in range(sampler.n_proc)])
    lds = np.concatenate([sampler.searched_log_densities_list[j][:sampler.element_num_list[j]]
                          for j in range(sampler.n_proc)])
    order = np.argsort(lds)  # draw best points on top
    pairs = [(i, k) for i in range(ndim) for k in range(i + 1, ndim)]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, (i, k) in zip(axes.ravel(), pairs):
        sc = ax.scatter(pts[order, i], pts[order, k], c=lds[order], s=2, cmap='viridis')
        ax.plot(param_true[i], param_true[k], 'r*', ms=12)
        ax.set_xlim(param_ranges[i])
        ax.set_ylim(param_ranges[k])
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[k])
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'searched_points_{tag}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved plots ({tag}) to {plot_dir}')


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()
        make_plots(sampler, f'iter{i:05d}')


print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(5000),
    savepath=savepath,
    print_iter=10,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
print('Savepath:', savepath)

make_plots(sampler, 'final')

# Report the best point found
best_proc = int(np.argmax(sampler.max_logden_list))
n = sampler.element_num_list[best_proc]
lds = sampler.searched_log_densities_list[best_proc][:n]
best_idx = int(np.argmax(lds))
best_u = sampler.searched_points_list[best_proc][best_idx][None, :]
best_phys = sampler.apply_prior_transform(best_u, prior_transform)[0]
print(f'Best log density (SNR): {lds[best_idx]:.4f}')
print('Best point:', dict(zip(['logm1', 'logm2', 'a', 'p0', 'e0'], best_phys.tolist())))
print('True point:', dict(zip(['logm1', 'logm2', 'a', 'p0', 'e0'], param_true)))
