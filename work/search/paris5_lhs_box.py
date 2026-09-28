"""
Precompute LHS grid inside a user-defined box for the paris5 noise search,
scoring each point with the coherent (pure) f-statistic (loglike_obj(theta),
from loglike_pure_noise.LogLike) — same scoring approach as paris5_lhs_ext.py,
but the box bounds are predefined directly instead of being derived from an
N-sigma ellipsoid around a stage1_anneal sampler_state.pkl.

Edit box_lo / box_hi below to set the LHS sampling box per dimension
[logm1, logm2, a, p0, e0, qS, phiS].
"""
import numpy as np
import pickle
import time
import os
import sys

import few
from smt.sampling_methods import LHS

os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/search/')

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_pure_noise import LogLike

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("WARNING")

use_gpu = True
tdi_gen = 1
dt=5
T=14/12
# dt = 10
# T = 23 / 12
print(f"dt={dt}s  T={T}yr  TDI gen={tdi_gen}")

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source (Mojito light EMRI_G)

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


# def icrs_to_ecliptic(qK_icrs, phiK_icrs, eps_deg=23.43929111):
#     """
#     Convert a sky direction (colatitude, longitude) from the ICRS/equatorial
#     frame (qK = pi/2 - dec, phiK = ra) to the ecliptic frame (SSB frame used
#     by the LISA response, is_ecliptic_latitude=False convention) via the
#     standard equatorial<->ecliptic rotation with J2000 mean obliquity eps.
#     """
#     dec = np.pi / 2 - qK_icrs
#     ra = phiK_icrs
#     eps = np.deg2rad(eps_deg)

#     sin_beta = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(ra) * np.sin(eps)
#     beta = np.arcsin(sin_beta)

#     y = np.cos(dec) * np.sin(ra) * np.cos(eps) + np.sin(dec) * np.sin(eps)
#     x = np.cos(dec) * np.cos(ra)
#     lam = np.arctan2(y, x) % (2 * np.pi)

#     qK_ecl = np.pi / 2 - beta
#     phiK_ecl = lam
#     return qK_ecl, phiK_ecl

# # Mojito light EMRI_C
# m1 = 3.54e6
# m2 = 8.01e1
# a = 0.950
# p0 = 7.3890
# e0 = 0.3160
# xI0 = 1.0000
# dist = 4.858
# qS = 2.0420
# phiS = 5.7873
# qK, phiK = icrs_to_ecliptic(1.6633, 1.7431)  # catalog qK/phiK are in ICRS, convert to ecliptic
# Phi_phi0 = 2.4044
# Phi_theta0 = 0.2850
# Phi_r0 = 0.9743


params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, np.cos(qS), phiS]

n_vals = np.arange(-1, 6)
ell = 2

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
print(f'SNR (rhostat): {float(gwf.SNR(gwf.freq_wave(loglike_obj.signal))):.4f}')


def f_score(phys_point):
    """Coherent (pure) f-statistic at a physical point [logm1, logm2, a, p0, e0, cosqS, phiS]."""
    logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = phys_point
    qS_i = np.arccos(cos_qS_i)
    theta = [
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        xI0, dist, qS_i, phiS_i, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
    ]
    return float(loglike_obj(theta))


# ── User-defined box ────────────────────────────────────────────────────────
# 

param_names = ['logm1', 'logm2', 'a', 'p0', 'e0', 'cosqS', 'phiS']
# from stage1_anneal
box_lo = np.array([5.82507, 1.99252, 0.49334, 15.70599, 0.74388, 0.79926, 3.48384])
box_hi = np.array([5.82762, 1.99310, 0.50078, 15.75367, 0.74418, 0.86149, 3.66905])

# from stage2_anneal dev version (much tighter)
# box_lo = np.array([5.82663, 1.99282, 0.49785, 15.69935, 0.74392, 0.80724, 3.49931])
# box_hi = np.array([5.82804, 1.99316, 0.50193, 15.72547, 0.74409, 0.85206, 3.62436])


# emri c, from stage1_anneal_s12
# logm1   : [6.54900, 6.54901]  width=0.00001  (1.22% of full prior width)
# logm2   : [1.90363, 1.90364]  width=0.00001  (0.95% of full prior width)
# a       : [0.95000, 0.95001]  width=0.00001  (1.83% of full prior width)
# p0      : [7.38894, 7.38903]  width=0.00009  (1.90% of full prior width)
# e0      : [0.31599, 0.31601]  width=0.00002  (0.60% of full prior width)
# cos_qS  : [-0.47014, -0.44492]  width=0.02522  (1.26% of full prior width)
# phiS    : [5.77627, 5.80239]  width=0.02613  (0.42% of full prior width)


# box_lo = np.array([6.54900, 1.90363, 0.95000, 7.38894, 0.31599, -0.47014, 5.77627])
# box_hi = np.array([6.54901, 1.90364, 0.95001, 7.38903, 0.31601, -0.44492, 5.80239])
print('LHS box bounds:')
for name, lo, hi in zip(param_names, box_lo, box_hi):
    print(f'  {name}: [{lo:.5f}, {hi:.5f}]')


def box_prior_transform(u):
    return box_lo + (box_hi - box_lo) * u


def inverse_box_prior_transform(params):
    return (np.asarray(params) - box_lo) / (box_hi - box_lo)


# ── Generate LHS uniformly in the box ──────────────────────────────────────

ndim  = 7
N_LHS = int(1e4)

print(f'Generating {N_LHS} LHS points in {ndim}D box...')
_lhs_sampler = LHS(xlimits=np.column_stack([box_lo, box_hi]))
lhs_phys     = _lhs_sampler(N_LHS)
lhs_phys     = np.clip(lhs_phys, box_lo, box_hi)
n_inside     = N_LHS
lhs_u    = np.clip(
    np.array([inverse_box_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)

# ── Evaluate coherent f-statistic ──────────────────────────────────────────────

print(f'Evaluating coherent f-statistic on {n_inside} points...')
log_densities = np.full(n_inside, -np.inf)
t0 = time.time()

for i in range(n_inside):
    if i % 500 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (n_inside - i) / rate
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    try:
        ld = f_score(lhs_phys[i])
        log_densities[i] = ld if np.isfinite(ld) else -np.inf
    except Exception:
        log_densities[i] = -np.inf

elapsed = time.time() - t0
print(f'Done. {n_inside} evals in {elapsed:.0f}s ({n_inside/elapsed:.1f}/s)')

n_finite = np.isfinite(log_densities).sum()
print(f'Finite log_densities: {n_finite} / {n_inside}')
if n_finite > 0:
    print(f'Max log_density: {np.max(log_densities[np.isfinite(log_densities)]):.6f}')
    print(f'Mean log_density (finite): {np.mean(log_densities[np.isfinite(log_densities)]):.6f}')

# ── Save ──────────────────────────────────────────────────────────────────────

savepath = f'/scratch/e1498138/paper/ext/emri_g/stage2_f_14mth/lhs_final.pkl'
os.makedirs(os.path.dirname(savepath), exist_ok=True)
save_data = {
    'lhs_u':          lhs_u,
    'lhs_phys':       lhs_phys,
    'log_densities':  log_densities,
    'box_lo':         box_lo,
    'box_hi':         box_hi,
    'N_LHS':          N_LHS,
    'n_inside':       n_inside,
    'statistic':      'coherent_f (loglike_pure_noise)',
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
