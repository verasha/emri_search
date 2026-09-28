import numpy as np
import few
import os
import sys
import pickle

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_pure_noise import LogLike

sys.path.insert(0, "/nfs/home/svu/e1498138/parismc_dev")
import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12/12 #NOTE: changed!
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
# # Source
# m1 = 1e6
# m2 = 1e1
# a = 0.7
# p0 = 9
# e0 = 0.4
# xI0 = 1.0
# dist = 4.5
# qS = np.pi
# phiS = 0.
# qK = 0.
# phiK = 0.
# Phi_phi0 = 0.4
# Phi_theta0 = 0.0
# Phi_r0 = 0.5    

# Mojito light EMRI_G
m1 = 6.72e5
m2 = 9.84e1
a = 0.5
p0 = 15.7117
e0 = 0.7440
xI0 = 1.0
dist = 4.755
qS = 0.5906
phiS = 3.5808
qK = 1.1707
phiK = 3.8495
Phi_phi0 = 3.1940
Phi_theta0 = 3.3780
Phi_r0 = 0.6038

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, np.cos(qS), phiS]

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
    """
    Coherent (pure) f-statistic (loglike_pure_noise.LogLike), batched.

    Waveform generation still runs one row at a time inside
    LogLike.log_density_batch (few has no batched API over source
    parameters), but every rfft/inner-product/chi-square step downstream
    of generation is computed once across the whole (B, ndim) batch instead
    of once per row.
    """
    params = np.asarray(params)
    B = params.shape[0]
    logm1, logm2, a_i, p0_i, e0_i, cosqS_i, phiS_i = (params[:, j] for j in range(7))
    qS_i = np.arccos(cosqS_i)
    theta_batch = np.stack([
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        np.full(B, xI0), np.full(B, dist), qS_i, phiS_i,
        np.full(B, qK), np.full(B, phiK),
        np.full(B, Phi_phi0), np.full(B, Phi_theta0), np.full(B, Phi_r0),
    ], axis=1)
    try:
        return loglike_obj.log_density_batch(theta_batch)
    except Exception:
        return np.full(B, -np.inf)


def prior_transform(u):
    # from stage1_anneal
    # logm1lim = [5.82507, 5.82762]
    # logm2lim = [1.99252, 1.99310]
    # alim = [0.49334, 0.50078]
    # p0lim = [15.70599, 15.75367]
    # e0lim = [0.74388, 0.74418]
    # cosqSlim = [0.79926, 0.86149]    
    # phiSlim = [3.48384, 3.66905]

    logm1lim = [5.82663, 5.82804]
    logm2lim = [1.99282, 1.99316]
    alim =  [0.49785, 0.50193]
    p0lim = [15.69935, 15.72547]
    e0lim = [0.74392, 0.74409]
    cosqSlim = [0.80724, 0.85206]
    phiSlim =  [3.49931, 3.62436]
    t = np.zeros_like(u)
    t[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]
    t[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]
    t[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    t[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0]
    t[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    t[:, 5] = (cosqSlim[1] - cosqSlim[0]) * u[:, 5] + cosqSlim[0]
    t[:, 6] = (phiSlim[1] - phiSlim[0]) * u[:, 6] + phiSlim[0]
    return t


def inverse_prior_transform(params):
    logm1lim = [5.82663, 5.82804]
    logm2lim = [1.99282, 1.99316]
    alim =  [0.49785, 0.50193]
    p0lim = [15.69935, 15.72547]
    e0lim = [0.74392, 0.74409]
    cosqSlim = [0.80724, 0.85206]
    phiSlim =  [3.49931, 3.62436]
    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0] = (params[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (params[:, 1] - logm2lim[0]) / (logm2lim[1] - logm2lim[0])
    u[:, 2] = (params[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (params[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (params[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5] = (params[:, 5] - cosqSlim[0]) / (cosqSlim[1] - cosqSlim[0])
    u[:, 6] = (params[:, 6] - phiSlim[0]) / (phiSlim[1] - phiSlim[0])
    return u


print('Setting up ParisMC...')
config = parismc.SamplerConfig(
    # merge_type='distance' (not the published-release default, but the
    # right fit for this narrow, single-mode local refinement — see prior
    # discussion) + handover off, since handover is only defined/useful
    # alongside the weight/density-based merge criteria ('single'/'sameruler').
    merge_type='distance',
    handover_size=0,        # 0.2.0 default is 100 (sample handover ON)

    alpha=int(1e3),
    trail_size=int(1e3),
    boundary_limiting=True,
    use_beta=True,
    integral_num=int(1e5),
    gamma=500,
    exclude_scale_z=np.inf,
    use_pool=False,
    parallel_eval=True,
    keep_dead_processes=True,
    density_selfcheck=False
)

ndim = 7
n_seed = 10
sigma = 1e-2
init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config,
)

# Published-release merge behaviour + parallelisation only — fail loudly if
# a parismc_dev default drifts back under us instead of silently changing
# merge behaviour.
assert sampler.merge_type == 'distance'
assert sampler.handover_size == 0
assert sampler.parallel_eval is True
assert sampler.sameruler_cloud == 1 and sampler.single_cloud == 1
assert sampler.enforce_trail_size is False
assert sampler.epoch_evidence is False

print('Getting LHS points...')
dir_scratch = '/scratch/e1498138'

lhs_path = dir_scratch + '/paper/ext/emri_g/f_lhs_dev/lhs_f_box_1e4.pkl'

with open(lhs_path, 'rb') as f:
    lhs_data = pickle.load(f)
phys_pts = np.asarray(lhs_data['lhs_phys'])
det_snr = np.asarray(lhs_data['log_densities'])

valid = np.isfinite(det_snr)
external_lhs_points = inverse_prior_transform(phys_pts[valid])
external_lhs_log_densities = det_snr[valid]
print(f'Loaded {valid.sum()} / {len(det_snr)} finite LHS evaluations.')
savepath = dir_scratch + f'/paper/ext/emri_g/stage3_f_dev/'


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()
    if i % 50 == 0 and i > 0:
        # log_density's batch size varies iteration to iteration (it tracks
        # how many candidates land in-bounds), so cupy has to build a new
        # cuFFT plan for each distinct batch shape. Left alone, the plan
        # cache and the memory pool both fragment over hundreds of
        # iterations until an allocation fails with CUFFT_INTERNAL_ERROR;
        # clearing the plan cache (which releases the workspace memory it
        # was holding) before freeing pool blocks keeps both defragmented.
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()


print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(5e4),
    savepath=savepath,
    print_iter=10,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
print('Savepath:', savepath)

# parallel_eval is bit-identical to the serial path only if no process
# exhausts its trail_size draw pool within an iteration; this counter is 0
# on every target tested in the dev-package migration notes.
assert sampler.pool_refill_count == 0
print('pool_refill_count:', sampler.pool_refill_count)
