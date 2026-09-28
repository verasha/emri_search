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
T = 12/12 #NOTE: changed!
N_SEGS = 12  # must match --n-segs used to build the brute-force LHS seed
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

            out[i] = float(gwf.SNR_semicoherent(loglike_obj.signal, h_temp, N_seg=N_SEGS, phase_max=True))*anneal_state['S']
        except Exception:
            pass
    return out


# Broad priors 

LOGM1_LIM = [5.6,  6.4]
LOGM2_LIM = [1.7,  2.3]
A_LIM     = [0.3,  0.999]
P0_LIM    = [13.5,  16.5]
E0_LIM    = [0.6,  0.9]
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
    seed=6342
)

ndim = 12
n_seed = 1 # start already merged
paris1_cov = np.array([[ 1.45551163e-04,  3.88667655e-05,  4.61994907e-04,
         -7.14840509e-04, -4.74738455e-05,  2.54531172e-03,
          8.59486858e-05,  1.11894340e-04,  3.39863499e-05,
         -7.64240680e-05, -4.56601007e-04,  5.22877368e-04],
        [ 3.88667655e-05,  1.44647481e-05,  1.23143541e-04,
         -1.88834477e-04, -1.43898110e-05,  4.84085940e-04,
          2.59396856e-05,  3.81064390e-05,  3.87393102e-06,
         -1.97346453e-05, -1.41210447e-04,  1.57900949e-04],
        [ 4.61994907e-04,  1.23143541e-04,  1.48961672e-03,
         -2.30303477e-03, -1.51342319e-04,  8.15179075e-03,
          2.84626164e-04,  3.43319731e-04,  1.53464782e-04,
         -2.48412550e-04, -1.51533169e-03,  1.68389987e-03],
        [-7.14840509e-04, -1.88834477e-04, -2.30303477e-03,
          3.57061103e-03,  2.33692829e-04, -1.28725268e-02,
         -4.25971787e-04, -5.61528710e-04, -1.75128988e-04,
          3.33079195e-04,  2.27157362e-03, -2.58431611e-03],
        [-4.74738455e-05, -1.43898110e-05, -1.51342319e-04,
          2.33692829e-04,  1.92112167e-05, -6.38135297e-04,
         -2.61458164e-05, -5.17435775e-05, -1.42597302e-06,
          6.70600632e-06,  1.55431680e-04, -1.80893038e-04],
        [ 2.54531172e-03,  4.84085940e-04,  8.15179075e-03,
         -1.28725268e-02, -6.38135297e-04,  4.16635421e-01,
         -8.74134580e-03,  6.77263857e-03, -2.61600036e-02,
         -6.60726750e-03,  6.10945038e-02, -2.04310198e-02],
        [ 8.59486858e-05,  2.59396856e-05,  2.84626164e-04,
         -4.25971787e-04, -2.61458164e-05, -8.74134580e-03,
          1.24781480e-03, -4.19311210e-04,  1.54877118e-03,
          4.83562000e-04, -4.60384317e-03,  1.36264907e-03],
        [ 1.11894340e-04,  3.81064390e-05,  3.43319731e-04,
         -5.61528710e-04, -5.17435775e-05,  6.77263857e-03,
         -4.19311210e-04,  1.87874749e-03, -1.79493602e-03,
          1.73190873e-03,  1.75880627e-03,  1.25106069e-03],
        [ 3.39863499e-05,  3.87393102e-06,  1.53464782e-04,
         -1.75128988e-04, -1.42597302e-06, -2.61600036e-02,
          1.54877118e-03, -1.79493602e-03,  1.03732718e-02,
         -1.94165339e-03, -9.92534396e-03,  3.66934645e-03],
        [-7.64240680e-05, -1.97346453e-05, -2.48412550e-04,
          3.33079195e-04,  6.70600632e-06, -6.60726750e-03,
          4.83562000e-04,  1.73190873e-03, -1.94165339e-03,
          6.22104273e-03, -1.18171202e-03, -7.82195658e-04],
        [-4.56601007e-04, -1.41210447e-04, -1.51533169e-03,
          2.27157362e-03,  1.55431680e-04,  6.10945038e-02,
         -4.60384317e-03,  1.75880627e-03, -9.92534396e-03,
         -1.18171202e-03,  2.86318620e-02, -8.30183206e-03],
        [ 5.22877368e-04,  1.57900949e-04,  1.68389987e-03,
         -2.58431611e-03, -1.80893038e-04, -2.04310198e-02,
          1.36264907e-03,  1.25106069e-03,  3.66934645e-03,
         -7.82195658e-04, -8.30183206e-03,  1.01137361e-02]])

init_cov_list = [paris1_cov / anneal_state['S']]

sampler = parismc.Sampler(
    ndim=ndim,
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config,
)

best_fit = [5.82783552,  1.99310637,  0.50130309, 15.70697013,  0.74418544,
         3.97688723,  0.79852328,  3.74416169,  0.22894789,  4.09004734,
         2.34095757,  3.62570792]  

external_lhs_points        = inverse_prior_transform(np.array([best_fit]))
external_lhs_log_densities = log_density(prior_transform(external_lhs_points))
print('Starting point (phys):', best_fit)
print(f'Starting log_density (S={anneal_state["S"]}):', external_lhs_log_densities)

dir_scratch = '/scratch/e1498138'
savepath = dir_scratch + f'/paper/ext/emri_g/stage1_anneal_brute'


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


print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(1e5),
    savepath=savepath,
    print_iter=10,
    callback=combined_callback,
    external_lhs_points=external_lhs_points,
    external_lhs_log_densities=external_lhs_log_densities,
)
print('Done.')
print('Savepath:', savepath)
