"""Average of the top-k samples (by log density) from the paris1_sc int_1yr_s12 run.

For k = 10, 50, 100: prints the averaged physical point and the average logden
of those top-k samples. No GPU / waveform setup needed.
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


dir_scratch = '/scratch/e1498138/'
savepath = dir_scratch + 'paris1_sc/int_1yr_s12'

param_names = ['logm1', 'logm2', 'a', 'p0', 'e0']




print(f'Loading sampler state from {savepath}/sampler_state.pkl ...')
sampler = parismc.Sampler.load_state(savepath + '/sampler_state.pkl')

# Gather all valid searched points and log densities (arrays are preallocated,
# only the first element_num entries of each process are valid)
pts_list, lds_list = [], []
for j in range(sampler.n_proc):
    n = sampler.element_num_list[j]
    pts_list.append(sampler.searched_points_list[j][:n])
    lds_list.append(sampler.searched_log_densities_list[j][:n])

# Include samples from merged/dead processes if they were kept
archived_pts = getattr(sampler, 'archived_points', None) or []
archived_lds = getattr(sampler, 'archived_log_densities', None) or []
for p_arch, ld_arch in zip(archived_pts, archived_lds):
    pts_list.append(p_arch)
    lds_list.append(ld_arch)

pts = np.concatenate(pts_list)
lds = np.concatenate(lds_list)
phys = prior_transform(pts)

finite = np.isfinite(lds)
print(f'Total samples: {len(lds)} ({finite.sum()} finite logden, '
      f'{sampler.n_proc} live + {len(archived_lds)} archived processes)')

order = np.argsort(lds[finite])[::-1]
phys_f = phys[finite]
lds_f = lds[finite]

header = '  '.join(f'{n:>10}' for n in param_names)
print(f'\n{"":>14}{header}  {"logden":>10}')
print(f'{"true":>14}' + '  '.join(f'{v:10.5f}' for v in param_true))

best = order[0]
print(f'{"best sample":>14}' + '  '.join(f'{v:10.5f}' for v in phys_f[best])
      + f'  {lds_f[best]:10.5f}')

for k in (10, 50, 100):
    top = order[:k]
    avg_pt = phys_f[top].mean(axis=0)
    avg_ld = lds_f[top].mean()
    print(f'{f"top {k} avg":>14}' + '  '.join(f'{v:10.5f}' for v in avg_pt)
          + f'  {avg_ld:10.5f}')
