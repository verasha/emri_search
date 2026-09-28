"""
Anneal counterpart of paris3_sc_brute.py (cf. paris4_sc_ext.py vs paris3_sc_ext.py,
paris6_ext.py vs paris5_ext.py).

Continues from the converged EMRI_G "brute" (12-dim: intrinsic + dist, cos_qS,
phiS, cos_qK, phiK, Phi_phi0, Phi_r0 -- xI0 and Phi_theta0 fixed) posterior in
paper/ext/emri_g/stage2_merge_brute/sampler_state.pkl, with a single seed and
an S-annealing schedule on the semicoherent (phase-maxed) SNR statistic,
instead of the multi-seed LHS-seeded merge search paris3_sc_brute.py runs.

Same priors as the merging stage (paris3_sc_brute.py) -- NOT a re-derived
N-sigma box -- so this anneal run explores the identical prior volume the
merge stage did, just starting from its converged point/covariance instead of
the full LHS grid.

best_fit and the initial covariance are hardcoded below (MAP point and
bootstrap-resampled posterior covariance from stage2_merge_brute's
sampler_state.pkl, same recipe as emri_g_stage2_merge_brute.ipynb cell 21) --
re-paste fresh values here if stage2_merge_brute is ever rerun.
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
T = 12/12  # matches paris3_sc_brute.py
N_SEGS = 6  # must match --n-segs used to build the brute-force LHS seed
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
Phi_theta0 = 3.3780     # fixed: not searched
Phi_r0 = 0.6038

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, dist, np.cos(qS), phiS, np.cos(qK), phiK, Phi_phi0, Phi_r0]

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

print("Setting up log_density and prior functions...")

# S schedule: jump to next S after stuck_iters of no improvement
S_schedule  = [3.0, 10.0, 30.0]
stuck_iters = 10000

# annealing dict
anneal_state = {
    'S':              S_schedule[0],
    'stage':          0,         # index into S_schedule
    'ref_max_ld':     None,      # max_ld at last check
    'ref_iter':       0,
    'stuck_count':    0,
}


def log_density(params):
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    h_stack, idx_ok = [], []
    for i in range(params.shape[0]):
        (logm1, logm2, a_i, p0_i, e0_i, dist_i,
         cos_qS_i, phiS_i, cos_qK_i, phiK_i, Phi_phi0_i, Phi_r0_i) = params[i]
        try:
            qS_i = np.arccos(cos_qS_i)
            qK_i = np.arccos(cos_qK_i)
            h_temp = gwf.xp.array(waveform_response(
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist_i, qS_i, phiS_i, qK_i, phiK_i,
                Phi_phi0_i, Phi_theta0, Phi_r0_i,
                T=T, dt=dt,
            ))
            h_stack.append(h_temp)
            idx_ok.append(i)
        except Exception:
            pass
    if h_stack:
        h_batch = gwf.xp.stack(h_stack, axis=0)  # (b, n_chan, N)
        snr = gwf.SNR_semicoherent_batch(loglike_obj.signal, h_batch, N_seg=N_SEGS, phase_max=True)
        for k, i in enumerate(idx_ok):
            out[i] = float(snr[k]) * anneal_state['S']
    return out


# Same priors as the merging stage (paris3_sc_brute.py) -- same bounds as
# run_lhs_noise_sc_brute.py. xI0 and Phi_theta0 held fixed (not searched);
# sky/spin polar angles sampled uniform in cosine.
LOGM1_LIM = [5.81854, 5.83713]
LOGM2_LIM = [1.99115, 1.99507]
A_LIM     = [0.49296, 0.51048]
P0_LIM    = [15.53112, 15.88282]
E0_LIM    = [0.74301, 0.74536]
DIST_LIM  = [3,  5]
COSQS_LIM = [-1.0, 1.0]
PHIS_LIM  = [0.0,  2 * np.pi]
COSQK_LIM = [-1.0, 1.0]
PHIK_LIM  = [0.0,  2 * np.pi]
PHIPHI0_LIM = [0.0, 2 * np.pi]
PHIR0_LIM   = [0.0, 2 * np.pi]


def prior_transform(u):
    t = np.zeros_like(u)
    t[:, 0]  = (LOGM1_LIM[1] - LOGM1_LIM[0]) * u[:, 0] + LOGM1_LIM[0]
    t[:, 1]  = (LOGM2_LIM[1] - LOGM2_LIM[0]) * u[:, 1] + LOGM2_LIM[0]
    t[:, 2]  = (A_LIM[1] - A_LIM[0]) * u[:, 2] + A_LIM[0]
    t[:, 3]  = (P0_LIM[1] - P0_LIM[0]) * u[:, 3] + P0_LIM[0]
    t[:, 4]  = (E0_LIM[1] - E0_LIM[0]) * u[:, 4] + E0_LIM[0]
    t[:, 5]  = (DIST_LIM[1] - DIST_LIM[0]) * u[:, 5] + DIST_LIM[0]
    t[:, 6]  = (COSQS_LIM[1] - COSQS_LIM[0]) * u[:, 6] + COSQS_LIM[0]
    t[:, 7]  = (PHIS_LIM[1] - PHIS_LIM[0]) * u[:, 7] + PHIS_LIM[0]
    t[:, 8]  = (COSQK_LIM[1] - COSQK_LIM[0]) * u[:, 8] + COSQK_LIM[0]
    t[:, 9]  = (PHIK_LIM[1] - PHIK_LIM[0]) * u[:, 9] + PHIK_LIM[0]
    t[:, 10] = (PHIPHI0_LIM[1] - PHIPHI0_LIM[0]) * u[:, 10] + PHIPHI0_LIM[0]
    t[:, 11] = (PHIR0_LIM[1] - PHIR0_LIM[0]) * u[:, 11] + PHIR0_LIM[0]
    return t


def inverse_prior_transform(params):
    params = np.asarray(params)
    u = np.zeros_like(params)
    u[:, 0]  = (params[:, 0] - LOGM1_LIM[0]) / (LOGM1_LIM[1] - LOGM1_LIM[0])
    u[:, 1]  = (params[:, 1] - LOGM2_LIM[0]) / (LOGM2_LIM[1] - LOGM2_LIM[0])
    u[:, 2]  = (params[:, 2] - A_LIM[0]) / (A_LIM[1] - A_LIM[0])
    u[:, 3]  = (params[:, 3] - P0_LIM[0]) / (P0_LIM[1] - P0_LIM[0])
    u[:, 4]  = (params[:, 4] - E0_LIM[0]) / (E0_LIM[1] - E0_LIM[0])
    u[:, 5]  = (params[:, 5] - DIST_LIM[0]) / (DIST_LIM[1] - DIST_LIM[0])
    u[:, 6]  = (params[:, 6] - COSQS_LIM[0]) / (COSQS_LIM[1] - COSQS_LIM[0])
    u[:, 7]  = (params[:, 7] - PHIS_LIM[0]) / (PHIS_LIM[1] - PHIS_LIM[0])
    u[:, 8]  = (params[:, 8] - COSQK_LIM[0]) / (COSQK_LIM[1] - COSQK_LIM[0])
    u[:, 9]  = (params[:, 9] - PHIK_LIM[0]) / (PHIK_LIM[1] - PHIK_LIM[0])
    u[:, 10] = (params[:, 10] - PHIPHI0_LIM[0]) / (PHIPHI0_LIM[1] - PHIPHI0_LIM[0])
    u[:, 11] = (params[:, 11] - PHIR0_LIM[0]) / (PHIR0_LIM[1] - PHIR0_LIM[0])
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
    keep_dead_processes=True
)

print('Done setting up ParisMC sampler.')
print('Setting up best_fit + initial covariance (from stage2_merge_brute MAP)...')

dir_scratch = '/scratch/e1498138/'

# MAP point from stage2_merge_brute, physical units:
# [logm1, logm2, a, p0, e0, dist, cosqS, phiS, cosqK, phiK, Phi_phi0, Phi_r0]
best_fit = [5.82823584, 1.99321534, 0.50249727, 15.68750185, 0.74337348,
            4.34912446, 0.82592136, 3.60967665, 0.69796553, 3.09941436,
            3.40361281, 4.7843498]

# Bootstrap-resampled posterior covariance from stage2_merge_brute, same
# recipe as emri_g_stage2_merge_brute.ipynb cell 21
paris1_cov = np.array([
    [ 3.09289827e-03,  2.88144155e-03,  9.44270379e-03,
     -3.08061296e-03, -2.79633645e-03, -3.36817780e-03,
      1.41492616e-04, -4.39888287e-05,  2.20492877e-03,
      3.63874819e-04, -2.00954328e-03,  1.73452563e-03],
    [ 2.88144155e-03,  3.77522827e-03,  8.60701011e-03,
     -2.74917466e-03, -3.02113700e-03, -4.21683883e-03,
      5.49524188e-04,  6.14385131e-04,  2.14347176e-03,
      2.39527514e-03,  7.55217707e-04,  2.22306355e-03],
    [ 9.44270379e-03,  8.60701011e-03,  2.89162526e-02,
     -9.43380974e-03, -8.45359769e-03, -1.03102272e-02,
      3.74233353e-04, -3.05238711e-04,  6.84990319e-03,
      5.55249320e-04, -6.46105244e-03,  4.83979095e-03],
    [-3.08061296e-03, -2.74917466e-03, -9.43380974e-03,
      3.08528434e-03,  2.73930585e-03,  3.24959874e-03,
     -9.49782397e-05,  1.19614756e-04, -2.19316021e-03,
     -1.24180967e-04,  2.27914496e-03, -1.65049502e-03],
    [-2.79633645e-03, -3.02113700e-03, -8.45359769e-03,
      2.73930585e-03,  2.69633590e-03,  3.38940983e-03,
     -2.97484475e-04, -2.38248708e-04, -1.98261624e-03,
     -1.19151136e-03,  7.52308833e-04, -2.06102717e-03],
    [-3.36817780e-03, -4.21683883e-03, -1.03102272e-02,
      3.24959874e-03,  3.38940983e-03,  9.83744614e-03,
     -4.67859117e-04, -4.26779575e-04, -4.17437526e-04,
      1.26245673e-03, -3.56861816e-03, -1.54324996e-03],
    [ 1.41492616e-04,  5.49524188e-04,  3.74233353e-04,
     -9.49782397e-05, -2.97484475e-04, -4.67859117e-04,
      6.98569272e-04,  2.28597162e-04,  1.09636867e-03,
      3.00937104e-04,  1.97246827e-03,  1.07346361e-03],
    [-4.39888287e-05,  6.14385131e-04, -3.05238711e-04,
      1.19614756e-04, -2.38248708e-04, -4.26779575e-04,
      2.28597162e-04,  6.18002745e-04, -1.45997965e-04,
      2.00073114e-03,  2.24711180e-03,  1.52319705e-03],
    [ 2.20492877e-03,  2.14347176e-03,  6.84990319e-03,
     -2.19316021e-03, -1.98261624e-03, -4.17437526e-04,
      1.09636867e-03, -1.45997965e-04,  9.20194996e-03,
      6.06782693e-04, -1.39972349e-04,  1.44830232e-04],
    [ 3.63874819e-04,  2.39527514e-03,  5.55249320e-04,
     -1.24180967e-04, -1.19151136e-03,  1.26245673e-03,
      3.00937104e-04,  2.00073114e-03,  6.06782693e-04,
      1.31225483e-02,  6.77173550e-03,  4.85581499e-03],
    [-2.00954328e-03,  7.55217707e-04, -6.46105244e-03,
      2.27914496e-03,  7.52308833e-04, -3.56861816e-03,
      1.97246827e-03,  2.24711180e-03, -1.39972349e-04,
      6.77173550e-03,  2.29687771e-02,  9.73627138e-03],
    [ 1.73452563e-03,  2.22306355e-03,  4.83979095e-03,
     -1.65049502e-03, -2.06102717e-03, -1.54324996e-03,
      1.07346361e-03,  1.52319705e-03,  1.44830232e-04,
      4.85581499e-03,  9.73627138e-03,  1.59065021e-02],
])

print('best_fit (from stage2_merge_brute MAP):', best_fit)

# Change to the search directory
dir_search = os.path.join(dir_work, 'search')
os.chdir(dir_search)
sys.path.insert(0, dir_search)

ndim = 12
n_seed = 1  # start already merged

init_cov_list = [paris1_cov / anneal_state['S']]

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

external_lhs_points        = inverse_prior_transform(np.array([best_fit]))
external_lhs_log_densities = log_density(prior_transform(external_lhs_points))
print('Starting point (phys):', best_fit)
print(f'Starting log_density (S={anneal_state["S"]}):', external_lhs_log_densities)


_stop_flag = [False]

def anneal_callback(sampler, i):
    global anneal_state
    state = anneal_state

    # initialise ref on first call
    if state['ref_max_ld'] is None:
        state['ref_max_ld'] = sampler.max_logden_list[0]
        state['ref_iter']   = i
        return

    current_max = sampler.max_logden_list[0]
    stage       = state['stage']
    S           = S_schedule[stage]

    # reset stuck clock whenever max_ld improves
    if current_max > state['ref_max_ld']:
        state['ref_max_ld'] = current_max
        state['ref_iter']   = i
        print(f"S={S} improved -> {current_max:.5f} at iter {i}", flush=True)
        return

    # check if stuck for stuck_iters
    if i - state['ref_iter'] >= stuck_iters:
        # JUMP
        if stage < len(S_schedule) - 1:
            new_S = S_schedule[stage + 1]
            scale = new_S / S
            for j in range(sampler.n_proc):
                n = sampler.element_num_list[j]
                sampler.searched_log_densities_list[j][:n] *= scale
                sampler.max_logden_list[j] *= scale
            for k in range(len(sampler.archived_log_densities)):
                sampler.archived_log_densities[k] *= scale
            sampler.loglike_normalization *= scale
            state['stage']      = stage + 1
            state['S']          = new_S
            state['ref_max_ld'] = sampler.max_logden_list[0]
            state['ref_iter']   = i
            print(f"Stuck {stuck_iters} iters at S={S}. Jumping -> S={new_S} at iter {i}", flush=True)
        else:
            # STOPPING
            print(f"Stuck {stuck_iters} iters at S={S} (final stage). Stopping at iter {i}.", flush=True)
            _stop_flag[0] = True


def combined_callback(sampler, i):
    anneal_callback(sampler, i)
    if _stop_flag[0]:
        sampler.stop_sampling = True
    if i % 1000 == 0 and i > 0:
        sampler.save_state()

savepath = dir_scratch + 'paper/ext/emri_g/stage2_anneal_brute/'

print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(1e5),
    savepath=savepath,
    print_iter=100,
    callback=combined_callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done running sampling.')
print('Savepath:', savepath)
