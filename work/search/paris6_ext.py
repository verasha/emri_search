"""
Anneal counterpart of paris5_ext.py (cf. paris4_sc_ext.py vs paris3_sc_ext.py).

Continues from a converged EMRI_G point/covariance (cf.
paper/emri_g/emri_g_stage2_f.ipynb, maxld_pt1 from stage2_f_smol) with a
single seed and an S-annealing schedule on the coherent (pure) f-statistic
log_density, instead of the multi-seed LHS-seeded merge search paris5_ext.py
runs.
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
from loglike_pure_noise import LogLike

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

def icrs_to_ecliptic(qK_icrs, phiK_icrs, eps_deg=23.43929111):
    """
    Convert a sky direction (colatitude, longitude) from the ICRS/equatorial
    frame (qK = pi/2 - dec, phiK = ra) to the ecliptic frame (SSB frame used
    by the LISA response, is_ecliptic_latitude=False convention) via the
    standard equatorial<->ecliptic rotation with J2000 mean obliquity eps.
    """
    dec = np.pi / 2 - qK_icrs
    ra = phiK_icrs
    eps = np.deg2rad(eps_deg)

    sin_beta = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(ra) * np.sin(eps)
    beta = np.arcsin(sin_beta)

    y = np.cos(dec) * np.sin(ra) * np.cos(eps) + np.sin(dec) * np.sin(eps)
    x = np.cos(dec) * np.cos(ra)
    lam = np.arctan2(y, x) % (2 * np.pi)

    qK_ecl = np.pi / 2 - beta
    phiK_ecl = lam
    return qK_ecl, phiK_ecl

# Mojito light EMRI_G (matches paper/emri_g/emri_g_stage2_f.ipynb)
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
    """
    Coherent (pure) f-statistic (loglike_pure_noise.LogLike), row-by-row,
    scaled by the current anneal temperature anneal_state['S'].
    """
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
    for i in range(params.shape[0]):
        logm1, logm2, a_i, p0_i, e0_i, cosqS_i, phiS_i = params[i]
        try:
            qS_i = np.arccos(cosqS_i)
            theta = [
                10**logm1, 10**logm2, a_i, p0_i, e0_i,
                xI0, dist, qS_i, phiS_i, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
            ]
            out[i] = float(loglike_obj(theta)) * anneal_state['S']
        except Exception:
            pass
    return out


def prior_transform(u):
    # EMRI_G, matches emri_g_stage2_f.ipynb box
    logm1lim = [5.82507, 5.82762]
    logm2lim = [1.99252, 1.99310]
    alim = [0.49334, 0.50078]
    p0lim = [15.70599, 15.75367]
    e0lim = [0.74388, 0.74418]
    cosqSlim = [0.79926, 0.86149]
    phiSlim = [3.48384, 3.66905]

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
    logm1lim = [5.82507, 5.82762]
    logm2lim = [1.99252, 1.99310]
    alim = [0.49334, 0.50078]
    p0lim = [15.70599, 15.75367]
    e0lim = [0.74388, 0.74418]
    cosqSlim = [0.79926, 0.86149]
    phiSlim = [3.48384, 3.66905]
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
print('Setting up initial covariance matrix...')

# Change to the search directory
dir_search =  os.path.join(dir_work, 'search')
os.chdir(dir_search)
sys.path.insert(0, dir_search)

ndim = 7
n_seed = 1  # start already merged

# from emri_g_stage2_f.ipynb (stage2_f_smol), proc 1 max-logden point covariance
paris1_cov = np.array([[ 1.46574105e-03,  9.28309168e-04,  1.51754396e-03,
        -1.53019328e-03, -1.22003834e-03, -1.38093364e-03,
         4.40818899e-05],
       [ 9.28309168e-04,  2.29792024e-03,  7.81466297e-04,
        -7.47141886e-04, -1.54771384e-03, -1.30267796e-03,
         6.77638469e-03],
       [ 1.51754396e-03,  7.81466297e-04,  1.60141169e-03,
        -1.61152517e-03, -1.17593695e-03, -1.44113962e-03,
        -8.72683680e-04],
       [-1.53019328e-03, -7.47141886e-04, -1.61152517e-03,
         1.63097341e-03,  1.17246631e-03,  1.41196808e-03,
         8.93352318e-04],
       [-1.22003834e-03, -1.54771384e-03, -1.17593695e-03,
         1.17246631e-03,  1.37773930e-03,  1.19332330e-03,
        -3.39732735e-03],
       [-1.38093364e-03, -1.30267796e-03, -1.44113962e-03,
         1.41196808e-03,  1.19332330e-03,  2.62747710e-02,
         1.26248162e-02],
       [ 4.40818899e-05,  6.77638469e-03, -8.72683680e-04,
         8.93352318e-04, -3.39732735e-03,  1.26248162e-02,
         5.25966276e-02]])

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

# from emri_g_stage2_f.ipynb (stage2_f_smol), proc 1 max-logden point
best_fit = [ 5.82595158,  1.99267966,  0.49587826, 15.73817815,  0.74416625,
        0.82400395,  3.55596108]


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
dir_scratch='/scratch/e1498138/'

savepath = dir_scratch+'paper/ext/emri_g/stage2_f_anneal/'

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
