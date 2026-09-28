import argparse
import numpy as np
import few
import os
import sys
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--n-sigma", type=float, default=3.0,
                     help="Half-width of the box in units of paris4_noise's posterior sigma")
args = parser.parse_args()
N_SIGMA = args.n_sigma
print(f"Using N_sigma={N_SIGMA}")

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_pure_noise import LogLike

import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12/12
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source parameters
#the original parameters, just for reference, not the ones that correspond to SNR20
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

# paris4_noise maxLD point + per-dim posterior sigma (diag), from
# work/search/logs/lhs_paris5.out
mu    = np.array([5.99582573, 0.99810132, 0.69144752, 9.04888015, 0.40169474])
sigma = np.array([0.00170288, 0.00074718, 0.00352764, 0.02022939, 0.00066724])
box_lo = mu - N_SIGMA * sigma
box_hi = mu + N_SIGMA * sigma
print(f"Box bounds (N_sigma={N_SIGMA}):")
for name, lo, hi in zip(["logm1", "logm2", "a", "p0", "e0"], box_lo, box_hi):
    print(f"  {name}: [{lo:.5f}, {hi:.5f}]")

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
    t = box_lo + (box_hi - box_lo) * u
    return t


def inverse_prior_transform(params):
    params = np.asarray(params)
    u = (params - box_lo) / (box_hi - box_lo)
    return u


print('Done setting up log-likelihood and prior.')
print('Setting up ParisMC sampler...')

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
    merge_type='distance'
)
print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')

ndim   = 5
n_seed = 10
init_cov      = np.eye(ndim) * 1e-5
init_cov_list = [init_cov] * n_seed

print('Done setting up initial covariance matrix.')
print('Initializing sampler...')

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Done initializing sampler.')

print('Evaluating log_density on ellipse LHS points...')
import pickle

dir_scratch='/scratch/e1498138'
savepath = f'{dir_scratch}/box/{N_SIGMA:.0f}sig/lhs_f.pkl'

with open(savepath, 'rb') as f:
    data = pickle.load(f)

external_lhs_points = data['lhs_u']
external_lhs_log_densities = data['log_densities']

def callback(sampler, i):
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

print('Running paris3 sampling...')
filepath=f'{dir_scratch}/box/box1_{N_SIGMA:.0f}sig'


sampler.run_sampling(
    num_iterations=int(5e4),
    savepath=filepath,
    print_iter=100,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities
    # stop_max_ld_stable_iters=int(1e4)
)
print('Done.')
