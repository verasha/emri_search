"""
MAP optimization for the Mojito light EMRI_C source, seed=42 noise realization
-- greedy Gaussian-proposal hill-climb instead of Nelder-Mead.

Uses the "pure" (non-amplitude-maximized) f-statistic from loglike_pure_noise.LogLike:

    f_stat(theta) = X_scalar * exp(-1/2 * beta * chi_sq)
    X_scalar = Re<d|h> / rho_h

NOTE: unlike the real Gaussian lnL (Re<d,h> - 1/2<h,h>), this statistic is
normalized by rho_h and is invariant to dist -- dist is NOT actually
constrained by this objective, so its recovered value should not be trusted.

map_mojito_light_emri_c.py's Nelder-Mead run, started at the injected params,
did not converge to an improvement over lnL(true) here -- i.e. we're already
standing at the foot of the mountain and NM can't find the way up. That's a
sign the pure-f-stat surface is rough/noisy at the scale NM's simplex moves
on (mode selection in loglike_pure_noise.LogLike is not smooth), which throws
off simplex reflections/contractions. A greedy hill-climb sidesteps that: at
each step draw a proposal from a small, fixed-covariance Gaussian centred on
the current best point, and only move there if the logden improves (see
work/search/old/greedy_pure_opt.py / greedy_pure_opt_alt.py for the pattern
this follows). If accepts stall for a while, shrink the step size and keep
climbing with finer steps -- same idea as walking up the mountain with
smaller and smaller strides as you approach the peak.

Free params (13): logm1, logm2, a, p0, e0, dist, qS, phiS, qK, phiK,
Phi_phi0, Phi_theta0, Phi_r0.

Fixed: xI0 -- equatorial-only waveform model (KerrEccEqFlux), xI0=+-1 is not
a continuous parameter.

Settings (dt, T, tdi_gen) match work/search/paper/emri_c/emri_c_stage3_f.ipynb
so this MAP result is directly comparable to that notebook's search runs and
to map_mojito_light_emri_c.py's NM result.
"""
import numpy as np
import few
import os
import sys
import time

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


def eval_loglike(x):
    # Points where waveform generation itself raises (e.g. plunge, e0 out of
    # physical range) are treated as -inf so the greedy step just rejects
    # them, rather than crashing the loop.
    try:
        val = lnL(full_params(x))
    except Exception:
        return -np.inf
    if not np.isfinite(val):
        return -np.inf
    return val


lnL_true = eval_loglike(param_true)
print(f"lnL at injected (true) params: {lnL_true:.6g}")

# ── Greedy hill-climb ────────────────────────────────────────────────────────
# Proposal: N(p_max, cov_prop), cov_prop diagonal with per-parameter step
# sizes tuned to the natural scale of each dimension (small, since we start
# right at the injected point and only expect a noise-induced offset). No
# prior box needed -- eval_loglike() rejects unphysical/failed points itself.
names = ['log10(m1)', 'log10(m2)', 'a', 'p0', 'e0', 'dist', 'qS', 'phiS',
          'qK', 'phiK', 'Phi_phi0', 'Phi_theta0', 'Phi_r0']

step0 = np.array([
    1e-4,   # log10(m1)
    1e-3,   # log10(m2)
    1e-4,   # a
    1e-4,   # p0
    1e-4,   # e0
    1e-2,   # dist
    1e-3,   # qS
    1e-3,   # phiS
    1e-3,   # qK
    1e-3,   # phiK
    1e-3,   # Phi_phi0
    1e-3,   # Phi_theta0
    1e-3,   # Phi_r0
])

N_ITER        = 20_000
print_every   = 200
stuck_patience = 500    # shrink step size after this many iters with no accept
shrink_factor  = 0.5
step_floor     = 1e-3 * step0  # don't shrink below this fraction of step0
rng = np.random.default_rng(42)

p_max = param_true.copy()
logden_max = lnL_true
step = step0.copy()

history_logden = [logden_max]
history_params = [p_max.copy()]
n_accept = 0
stuck_count = 0

print(f"\nRunning greedy hill-climb for {N_ITER} iterations...")
t0 = time.time()

for i in range(1, N_ITER + 1):
    cov_prop = np.diag(step**2)
    p_prop = rng.multivariate_normal(p_max, cov_prop)

    logden_prop = eval_loglike(p_prop)

    if logden_prop > logden_max:
        logden_max = logden_prop
        p_max = p_prop
        n_accept += 1
        stuck_count = 0
        history_logden.append(logden_max)
        history_params.append(p_max.copy())
    else:
        stuck_count += 1
        if stuck_count >= stuck_patience:
            step = np.maximum(step * shrink_factor, step_floor)
            stuck_count = 0

    if i % print_every == 0:
        elapsed = time.time() - t0
        print(f"  iter={i:6d}  max_logden={logden_max:.6f}  accepts={n_accept}"
              f"  step(diag)={step}  elapsed={elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\nDone. {N_ITER} iters in {elapsed:.0f}s ({N_ITER / elapsed:.1f}/s)")
print(f"Total accepts: {n_accept}")

x_map = p_max
params_map = full_params(x_map)

print("\n=== MAP result (greedy hill-climb) ===")
for name, true_v, map_v in zip(names, param_true, x_map):
    print(f"  {name:12s}: true={true_v: .6f}  MAP={map_v: .6f}  diff={map_v - true_v: .3e}")
print(f"\nlnL(true) = {lnL_true:.6g}")
print(f"lnL(MAP)  = {logden_max:.6g}")

np.savez(
    '/scratch/e1498138/map_mojito_light_c_greedy_seed42.npz',
    x_map=x_map,
    params_map=np.array(params_map, dtype=float),
    param_true=param_true,
    lnL_true=lnL_true,
    lnL_map=logden_max,
    history_logden=np.array(history_logden),
    history_params=np.array(history_params),
    step_final=step,
    n_accept=n_accept,
)
print("\nSaved to /scratch/e1498138/map_mojito_light_c_greedy_seed42.npz")
