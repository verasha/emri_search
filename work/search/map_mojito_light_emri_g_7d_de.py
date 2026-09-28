"""
MAP optimization for the Mojito light EMRI_G source, seed=42 noise realization,
restricted to the SAME 7-dim subspace searched by the ParisMC run in
work/search/paper/emri_g/emri_g_stage2_f_anneal.ipynb:

    [log10(m1), log10(m2), a, p0, e0, cos(qS), phiS]

-- scipy.optimize.differential_evolution instead of Nelder-Mead/greedy (cf.
map_mojito_light_emri_g_7d_greedy.py (greedy hill-climb) and
map_mojito_light_emri_c_7d_de.py, the EMRI_C counterpart of this script).

Bounded by the SAME tight local box used by emri_g_stage2_f_anneal.ipynb's
prior_transform/inverse_prior_transform:

    logm1lim  = [5.82507, 5.82762]
    logm2lim  = [1.99252, 1.99310]
    alim      = [0.49334, 0.50078]
    p0lim     = [15.70599, 15.75367]
    e0lim     = [0.74388, 0.74418]
    cosqSlim  = [0.79926, 0.86149]
    phiSlim   = [3.48384, 3.66905]

dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 are held FIXED at their true
values, exactly like the `log_density` function in that notebook does. This
makes the recovered MAP point directly comparable to `maxld_pt1` from the
notebook (same parametrization, same fixed nuisance params, same likelihood
slice).

Uses the "pure" (non-amplitude-maximized) f-statistic from loglike_pure_noise.LogLike:

    f_stat(theta) = X_scalar * exp(-1/2 * beta * chi_sq)
    X_scalar = Re<d|h> / rho_h

DE is a global, population-based, derivative-free optimizer -- unlike NM's
single local simplex, it maintains a population of candidates across the
whole box and mutates/recombines them each generation, which tends to be
more robust to a rough/noisy objective (mode-selection jaggedness in
loglike_pure_noise.LogLike) than NM's simplex reflections. polish=False
since a final L-BFGS-B polish step relies on the objective being locally
smooth, which is exactly what NM already struggled with here.
"""
import numpy as np
import few
import os
import sys
from scipy.optimize import differential_evolution

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_pure_noise import LogLike

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 12 / 12  # matches emri_g_stage2_f_anneal.ipynb / fisher_mojito_light_g.py
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

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

# 7-dim parametrization, matching maxld_pt1 in emri_g_stage2_f_anneal.ipynb:
# [log10(m1), log10(m2), a, p0, e0, cos(qS), phiS]
param_true = np.array([
    np.log10(m1), np.log10(m2), a, p0, e0, np.cos(qS), phiS,
])

print('Generating data (signal + seed=42 colored noise) and pure-f-stat LogLike...')
n_vals = np.arange(-1, 6)  # n from -1 to 5, matches paris1*/diag_beta_vs_T convention
loglike_obj = LogLike(
    params=params_star,
    waveform_response=waveform_response,
    gwf=gwf,
    add_noise=True,
    seed=42,
    verbose=False,
    ell=2,
    n_vals=n_vals,
    M_mode=None,
)
print('Data generated.')


def full_params(x):
    # dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 fixed at truth -- same
    # nuisance values used by log_density() in emri_g_stage2_f_anneal.ipynb.
    logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = x
    qS_i = np.arccos(cos_qS_i)
    return [10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS_i, phiS_i, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0]


def lnL(theta):
    return loglike_obj(theta)


def neg_lnL(x):
    # Points where waveform generation itself raises (e.g. plunge, e0 out of
    # physical range, |cos(qS)| > 1 -- shouldn't happen inside the box below,
    # but guard anyway) are penalized with a large finite value.
    try:
        val = lnL(full_params(x))
    except Exception:
        return 1e10
    if not np.isfinite(val):
        return 1e10
    return -val


lnL_true = lnL(full_params(param_true))
print(f"lnL at injected (true) params: {lnL_true:.6g}")

# ── Prior box -- matches emri_g_stage2_f_anneal.ipynb's prior_transform box ──
logm1lim = [5.82507, 5.82762]
logm2lim = [1.99252, 1.99310]
alim     = [0.49334, 0.50078]
p0lim    = [15.70599, 15.75367]
e0lim    = [0.74388, 0.74418]
cosqSlim = [0.79926, 0.86149]
phiSlim  = [3.48384, 3.66905]

bounds = [logm1lim, logm2lim, alim, p0lim, e0lim, cosqSlim, phiSlim]
names  = ['log10(m1)', 'log10(m2)', 'a', 'p0', 'e0', 'cos(qS)', 'phiS']

print('\nRunning differential_evolution over the box:')
for name, b in zip(names, bounds):
    print(f"  {name:12s}: {b}")

checkpoint_path = '/scratch/e1498138/map_mojito_light_g_7d_de_seed42_checkpoint.npz'
checkpoint_every = 100
step_counter = {'n': 0}


def checkpoint_callback(xk, convergence=None):
    step_counter['n'] += 1
    if step_counter['n'] % checkpoint_every == 0:
        fval = neg_lnL(xk)
        np.savez(
            checkpoint_path,
            x_map=xk,
            params_map=np.array(full_params(xk), dtype=float),
            param_true=param_true,
            lnL_true=lnL_true,
            lnL_map=-fval,
            nit=step_counter['n'],
            bounds=np.array(bounds, dtype=float),
        )
        param_str = ', '.join(f"{n}={v: .6f}" for n, v in zip(names, xk))
        print(f"[checkpoint] step {step_counter['n']}: lnL={-fval:.6g}  {param_str}")
        print(f"[checkpoint] saved to {checkpoint_path}")
    return False


result_de = differential_evolution(
    neg_lnL,
    bounds=bounds,
    seed=42,
    maxiter=1000,
    popsize=20,
    tol=1e-10,
    mutation=(0.5, 1.5),
    recombination=0.7,
    polish=False,
    updating='deferred',
    workers=1,
    disp=True,
    callback=checkpoint_callback,
)
print(f"\nDE result: x={result_de.x}, lnL={-result_de.fun:.6g}, "
      f"nit={result_de.nit}, nfev={result_de.nfev}")

x_map = result_de.x
params_map = full_params(x_map)

print("\n=== MAP result (7-dim, differential evolution, matching ParisMC search subspace) ===")
for name, true_v, map_v in zip(names, param_true, x_map):
    print(f"  {name:12s}: true={true_v: .6f}  MAP={map_v: .6f}  diff={map_v - true_v: .3e}")
print(f"\nlnL(true) = {lnL_true:.6g}")
print(f"lnL(MAP)  = {-result_de.fun:.6g}")

np.savez(
    '/scratch/e1498138/map_mojito_light_g_7d_de_seed42.npz',
    x_map=x_map,
    params_map=np.array(params_map, dtype=float),
    param_true=param_true,
    lnL_true=lnL_true,
    lnL_map=-result_de.fun,
    nit=result_de.nit,
    nfev=result_de.nfev,
    bounds=np.array(bounds, dtype=float),
)
print("\nSaved to /scratch/e1498138/map_mojito_light_g_7d_de_seed42.npz")
