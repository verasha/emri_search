"""
MAP optimization for the Mojito light EMRI_C source, seed=42 noise realization,
restricted to the SAME 7-dim subspace searched by the ParisMC run in
work/search/paper/emri_c/emri_c_stage3_f.ipynb:

    [log10(m1), log10(m2), a, p0, e0, cos(qS), phiS]

dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 are held FIXED at their true
values, exactly like the `log_density` function in that notebook does. This
makes the recovered MAP point directly comparable to `maxld_pt1` from the
notebook (same parametrization, same fixed nuisance params, same likelihood
slice) -- unlike map_mojito_light_emri_c.py's full 13-dim MAP, which lets the
nuisance params float and therefore sits at a different point in the 7-dim
subspace for reasons unrelated to how well the 7-dim search itself performs.

Uses the "pure" (non-amplitude-maximized) f-statistic from loglike_pure_noise.LogLike:

    f_stat(theta) = X_scalar * exp(-1/2 * beta * chi_sq)
    X_scalar = Re<d|h> / rho_h

Maximized with scipy.optimize.minimize(method='Nelder-Mead'), started at the
injected ("true") params in this 7-dim parametrization.
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
dt = 10
T = 23/12  # matches emri_c_stage3_f.ipynb
print(f"Using dt={dt}s, T={T}yr, TDI gen={tdi_gen}")

print('Building ResponseWrapper...')
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

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


# Mojito light EMRI_C
m1 = 3.54e6
m2 = 8.01e1
a = 0.950
p0 = 7.3890
e0 = 0.3160
xI0 = 1.0000
dist = 4.858
qS = 2.0420
phiS = 5.7873
qK, phiK = icrs_to_ecliptic(1.6633, 1.7431)  # catalog qK/phiK are in ICRS, convert to ecliptic
Phi_phi0 = 2.4044
Phi_theta0 = 0.2850
Phi_r0 = 0.9743

params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

# 7-dim parametrization, matching maxld_pt1 in emri_c_stage3_f.ipynb:
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
    # nuisance values used by log_density() in emri_c_stage3_f.ipynb.
    logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = x
    qS_i = np.arccos(cos_qS_i)
    return [10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist, qS_i, phiS_i, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0]


def lnL(theta):
    return loglike_obj(theta)


def neg_lnL(x):
    # No bounds -- Nelder-Mead is unconstrained. Only guard against points
    # where waveform generation itself raises (e.g. plunge, e0 out of
    # physical range, |cos(qS)| > 1) by penalizing with a large finite value
    # so the simplex steps back rather than crashing.
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

print("\n=== MAP result (7-dim, matching ParisMC search subspace) ===")
names = ['log10(m1)', 'log10(m2)', 'a', 'p0', 'e0', 'cos(qS)', 'phiS']
for name, true_v, map_v in zip(names, param_true, x_map):
    print(f"  {name:12s}: true={true_v: .6f}  MAP={map_v: .6f}  diff={map_v - true_v: .3e}")
print(f"\nlnL(true) = {lnL_true:.6g}")
print(f"lnL(MAP)  = {-result_nm.fun:.6g}")

np.savez(
    '/scratch/e1498138/map_mojito_light_c_7d_pure_seed42.npz',
    x_map=x_map,
    params_map=np.array(params_map, dtype=float),
    param_true=param_true,
    lnL_true=lnL_true,
    lnL_map=-result_nm.fun,
)
print("\nSaved to /scratch/e1498138/map_mojito_light_c_7d_pure_seed42.npz")
