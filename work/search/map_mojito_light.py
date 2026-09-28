"""
MAP optimization for the Mojito light EMRI_G source, seed=42 noise realization.

Uses the "pure" (non-amplitude-maximized) f-statistic from loglike_pure_noise.LogLike:

    f_stat(theta) = X_scalar * exp(-1/2 * beta * chi_sq)
    X_scalar = Re<d|h> / rho_h

NOTE: unlike the real Gaussian lnL (Re<d,h> - 1/2<h,h>), this statistic is
normalized by rho_h and is invariant to dist -- dist is NOT actually
constrained by this objective, so its recovered value should not be trusted.

Maximized with scipy.optimize.minimize(method='Nelder-Mead'), a local,
derivative-free, unconstrained optimizer -- no prior box needs to be defined.
Since it's local-only, it's started at the injected ("true") params, which is
where the MAP is expected to sit (within noise-induced offset).

Free params (13): logm1, logm2, a, p0, e0, dist, qS, phiS, qK, phiK,
Phi_phi0, Phi_theta0, Phi_r0.

Fixed: xI0 -- equatorial-only waveform model (KerrEccEqFlux), xI0=+-1 is not
a continuous parameter.
"""
import numpy as np
import few
import os
import sys
from scipy.optimize import minimize

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
T = 1.0
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
param_true = np.array([
    np.log10(m1), np.log10(m2), a, p0, e0, dist, qS, phiS,
    qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,
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
    (logm1, logm2, a_i, p0_i, e0_i, dist_i, qS_i, phiS_i,
     qK_i, phiK_i, Phi_phi0_i, Phi_theta0_i, Phi_r0_i) = x
    return [10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist_i, qS_i, phiS_i, qK_i, phiK_i,
            Phi_phi0_i, Phi_theta0_i, Phi_r0_i]


def lnL(theta):
    return loglike_obj(theta)


def neg_lnL(x):
    # No bounds -- Nelder-Mead is unconstrained. Only guard against points
    # where waveform generation itself raises (e.g. plunge, e0 out of
    # physical range) by penalizing with a large finite value so the simplex
    # steps back rather than crashing.
    try:
        val = lnL(full_params(x))
    except Exception:
        return 1e10
    if not np.isfinite(val):
        return 1e10
    return -val


lnL_true = lnL(full_params(param_true))
print(f"lnL at injected (true) params: {lnL_true:.6g}")

print('Running Nelder-Mead from the injected (true) params...')
result_nm = minimize(
    neg_lnL,
    x0=param_true,
    method='Nelder-Mead',
    options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 20000, 'maxfev': 20000, 'adaptive': True},
)
print(f"NM result: x={result_nm.x}, lnL={-result_nm.fun:.6g}")

x_map = result_nm.x
params_map = full_params(x_map)

print("\n=== MAP result ===")
names = ['log10(m1)', 'log10(m2)', 'a', 'p0', 'e0', 'dist', 'qS', 'phiS',
          'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']
for name, true_v, map_v in zip(names, param_true, x_map):
    print(f"  {name:12s}: true={true_v: .6f}  MAP={map_v: .6f}  diff={map_v - true_v: .3e}")
print(f"\nlnL(true) = {lnL_true:.6g}")
print(f"lnL(MAP)  = {-result_nm.fun:.6g}")

np.savez(
    '/scratch/e1498138/map_mojito_light_pure_seed42.npz',
    x_map=x_map,
    params_map=np.array(params_map, dtype=float),
    param_true=param_true,
    lnL_true=lnL_true,
    lnL_map=-result_nm.fun,
)
print("\nSaved to /scratch/e1498138/map_mojito_light_pure_seed42.npz")
