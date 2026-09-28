"""
Resume a saved paris3_sc_ext_dev_alt sampler and continue sampling.

Usage:
  python resume_paris3_sc_ext_dev_alt.py
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

sys.path.insert(0, "/nfs/home/svu/e1498138/parismc_dev")
import parismc
import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12 / 12
N_SEGS = 6  # must match --n-segs used to build the LHS seed
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}, N_segs={N_SEGS}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=True, tdi_gen=tdi_gen)

print('Building GravWaveAnalysis...')
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

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
            print(f'log_density batch of {h_batch.shape[0]} points failed ({e!r}); '
                  f'clearing cuFFT plan cache and treating batch as -inf.')
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
            return out
        for k, i in enumerate(idx_ok):
            out[i] = float(snr[k])
    return out


# ── Load the LHS grid + ellipsoid prior bounds (must match the original run) ──
lhs_path = '/scratch/e1498138/paper/ext/emri_g/paris3_lhs_s6_dev/lhs_s6.pkl'
print(f'Loading LHS grid from {lhs_path}...')
with open(lhs_path, 'rb') as f:
    lhs_data = pickle.load(f)

ellipse_lo = lhs_data['ellipse_lo']
ellipse_hi = lhs_data['ellipse_hi']
print('Prior bounds (from paris3_lhs_s6_ext ellipsoid box):')
param_names = ['logm1', 'logm2', 'a', 'p0', 'e0', 'cosqS', 'phiS']
for name, lo, hi in zip(param_names, ellipse_lo, ellipse_hi):
    print(f'  {name}: [{lo:.5f}, {hi:.5f}]')


def prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u


print('Done setting up log-likelihood and prior.')

# Load saved sampler state
dir_scratch = '/scratch/e1498138'
savepath = dir_scratch + '/paper/ext/emri_g/stage2_merge_s12_dev_alt'
state_path = savepath + '/sampler_state.pkl'
print(f'Loading sampler state from: {state_path}')

if not os.path.isfile(state_path):
    print(f"Sampler state not found at: {state_path}")
    print("Please run paris3_sc_ext_dev_alt.py first.")
    exit(1)

sampler = parismc.Sampler.load_state(state_path)

# Rebind functions (log_density_func_original can't be pickled, see the
# checkpoint TypeError paris3_sc_ext_dev_alt.py prints on every save_state).
try:
    sampler.log_density_func_original = log_density
    sampler.prior_transform = prior_transform
    sampler.log_density_func = sampler.transformed_log_density_func
except Exception as e:
    print(f"Warning: Could not rebind functions: {e}")

print('Done loading sampler.')
print(f"Sampler ndim: {sampler.ndim}")
print(f"Sampler n_seed: {sampler.n_seed}")
print(f"Sampler current_iter: {getattr(sampler, 'current_iter', None)}")

# Verify published-release behaviour survived the checkpoint round-trip.
assert sampler.merge_type == 'single'
assert sampler.handover_size == 0


def callback(sampler, i):
    if i % 500 == 0 and i > 0:
        sampler.save_state()
    if i % 50 == 0 and i > 0:
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()


# Continue sampling
print('Resuming paris3_sc_ext_dev_alt sampling...')
out_dir = dir_scratch + '/paper/ext/emri_g/stage2_dev_res'

sampler.run_sampling(
    num_iterations=int(2e4),
    savepath=out_dir,
    print_iter=10,
    callback=callback,
)
print('Done.')
print('Savepath:', out_dir)

assert sampler.pool_refill_count == 0

logZ = sampler.compute_log_evidence(ruler='hist')
print('logZ (hist ruler):', logZ)
