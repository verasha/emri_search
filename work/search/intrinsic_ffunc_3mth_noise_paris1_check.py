import numpy as np
import few
import os
import sys
import pickle

dir_work = '/nfs/home/svu/e1498138/localgit/FEWNEW/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_timemax_noise import LogLike
# from loglike_phasemax_noise import LogLike

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 10
T = 3/12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

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
dist = 5 #scale by 2-3x
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

data_snr = float(gwf.rhostat_timemax(loglike_obj.signal).get())
print(f'SNR (time-max): {data_snr:.4f}')


def log_density(params):
    params = np.asarray(params)
    log_likes = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        logm1, logm2, a, p0, e0 = params[i]
        try:
            loglike = loglike_obj(np.array([
                10**logm1, 10**logm2, a, p0, e0,
                xI0, dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0
            ]))
        except Exception:
            loglike = -np.inf
        log_likes[i] = loglike
    return log_likes


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
)

ndim = 5
n_seed = 1
sigma = 1e-5
init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config,
)

# print('Preparing LHS samples...')
# sampler.prepare_lhs_samples(lhs_num=int(1e3), batch_size=10)
# print('Done preparing LHS samples.')


dir_scratch = '/scratch/e1498138/'

# lhspath='lhs/lhs_noise_paris1_1e4_pm_checkpoints/lhs_noise_paris1_final.pkl'  

# filepath_lhs= dir_scratch+lhspath

# with open(filepath_lhs, 'rb') as f:
#     _phys_pts, _logden = pickle.load(f)

# Start from true pt
external_lhs_points        = inverse_prior_transform(np.array([param_true]))
external_lhs_log_densities = log_density(prior_transform(external_lhs_points))
print('Starting point (phys):', param_true)
print('Starting log_density:', external_lhs_log_densities)

#3=tests
#4=with 1e5 lhs
#5=with 1e4lhs, nowwith timemax fix
#6=phasemax only
# check: start from true pt 
savepath = dir_scratch + 'paris1_noise/int_3mth_noise_check'

def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()

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
