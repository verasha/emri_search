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

sys.path.insert(0, "/nfs/home/svu/e1498138/parismc_dev")
import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12/12 #NOTE: changed!
N_SEGS = 6  # must match --n-segs used to build the LHS seed
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_SEGS}")

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
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    h_stack, idx_ok = [], []
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = params[i]
        try:
            qS_i = np.arccos(cos_qS_i)
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS_i, phiS_i, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                T=T, dt=dt,
            ))
            h_stack.append(h_temp)
            idx_ok.append(i)
        except Exception:
            pass
    if h_stack:
        h_batch = gwf.xp.stack(h_stack, axis=0)  # (b, n_chan, N)
        try:
            snr = gwf.SNR_semicoherent_batch(loglike_obj.signal, h_batch, N_seg=N_SEGS, phase_max=True)
        except Exception as e:
            # Transient cuFFT plan/workspace failures (e.g. CUFFT_INTERNAL_ERROR) can
            # still slip through the periodic plan-cache clear in `callback` since batch
            # shapes vary every call. Recover by clearing here and dropping this batch
            # (-inf) rather than losing the whole multi-hour run to one bad FFT call.
            print(f'log_density batch of {h_batch.shape[0]} points failed ({e!r}); '
                  f'clearing cuFFT plan cache and treating batch as -inf.')
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
            return out
        for k, i in enumerate(idx_ok):
            out[i] = float(snr[k])
    return out


print('Getting LHS points...')
dir_scratch = '/scratch/e1498138'

lhs_path = dir_scratch + '/paper/ext/emri_g/paris3_lhs_s6_dev/lhs_s6.pkl'

with open(lhs_path, 'rb') as f:
    lhs_data = pickle.load(f)

# Use the same ellipsoid box the LHS grid was drawn in (see paris3_lhs_s6_ext.py)
# so external_lhs_points/prior_transform are consistent with lhs_u below.
ellipse_lo = lhs_data['ellipse_lo']
ellipse_hi = lhs_data['ellipse_hi']
print('Prior bounds (from paris3_lhs_s6_ext ellipsoid box):')
param_names = ['logm1', 'logm2', 'a', 'p0', 'e0', 'cosqS', 'phiS']
for name, lo, hi in zip(param_names, ellipse_lo, ellipse_hi):
    print(f'  {name}: [{lo:.5f}, {hi:.5f}]')


def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u


def inverse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)


print('Setting up ParisMC...')
config = parismc.SamplerConfig(
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
    seed=6342,
    merge_type='single',   # 0.2.0 default is 'sameruler'
    handover_size=0,       # 0.2.0 default is 100 (sample handover ON)
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

# Verify published-release behaviour was restored (see parismc migration notes).
assert sampler.merge_type == 'single'
assert sampler.handover_size == 0
assert sampler.parallel_eval is True
assert sampler.sameruler_cloud == 1 and sampler.single_cloud == 1
assert sampler.enforce_trail_size is False
assert sampler.epoch_evidence is False

lhs_u = lhs_data['lhs_u']
log_densities = lhs_data['log_densities']

valid = np.isfinite(log_densities)
external_lhs_points = lhs_u[valid]
external_lhs_log_densities = log_densities[valid]
print(f'Loaded {valid.sum()} / {len(log_densities)} finite LHS evaluations.')
savepath = dir_scratch + f'/paper/ext/emri_g/stage2_merge_s6_dev'


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
    num_iterations=int(1e5),
    savepath=savepath,
    print_iter=10,
    callback=callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
print('Savepath:', savepath)

# parallel_eval is bit-identical to serial as long as no process exhausts its
# trail_size draw pool within an iteration; this counter is how you know it
# did not happen (it is 0 on every target tested).
assert sampler.pool_refill_count == 0

# Published-equivalent evidence (0.2.0 defaults to the 'win-matched' ruler).
logZ = sampler.compute_log_evidence(ruler='hist')
print('logZ (hist ruler):', logZ)
