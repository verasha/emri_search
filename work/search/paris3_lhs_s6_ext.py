"""
Precompute LHS grid inside the paris2_sc_ext (stage1_anneal_s12) posterior ellipsoid
for the paris3 noise search, scoring each point with the S6 semi-coherent statistic
(gwf.SNR_semicoherent(signal, h, N_seg=6, phase_max=True)).

This continues from paris2_sc_ext.py / paper/ext/emri_g/stage1_anneal_s12.ipynb: the
source, LogLike, and 7-dim (logm1, logm2, a, p0, e0, cos_qS, phiS) parameter space
match that stage, and the ellipsoid center/covariance are loaded from its sampler
state (paper/ext/emri_g/stage1_anneal_s12/sampler_state.pkl).

For all 7 dims the LHS box is the N-sigma bounding box around the stage1_anneal_s12
posterior (clipped to the paris2_sc_ext prior box), including cos_qS and phiS —
unlike the earlier (pre-"ext") paris3_lhs_s6.py, sky localization here IS already
informed by paris2_sc_ext's converged single-mode posterior.

Loads paper/ext/emri_g/stage1_anneal_s12 to get ellipse center and covariance.
Saves unit-cube points, physical points, S6 log_densities, and prior bounds.
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
from loglike_timemax_noise import LogLike
sys.path.insert(0, "/nfs/home/svu/e1498138/parismc_dev")

import parismc

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("WARNING")

use_gpu = True
tdi_gen = 1
dt = 10
T = 23/12 #NOTE: changed!
print(f"dt={dt}s  T={T}yr  TDI gen={tdi_gen}")

# ── Semi-coherent statistic configuration ─────────────────────────────────────
N_SEG = 6  # S6: number of segments for the semi-coherent statistic

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

# Source (Mojito light EMRI_G, matches paris2_sc_ext.py)
# m1 = 6.72e5
# m2 = 9.84e1
# a = 0.5
# p0 = 15.7117
# e0 = 0.7440
# xI0 = 1.0
# dist = 4.755
# qS = 0.5906
# phiS = 3.5808
# qK = 1.1707
# phiK = 3.8495
# Phi_phi0 = 3.1940
# Phi_theta0 = 3.3780
# Phi_r0 = 0.6038


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


params_star = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
               Phi_phi0, Phi_theta0, Phi_r0]
param_true = [np.log10(m1), np.log10(m2), a, p0, e0, np.cos(qS), phiS]

n_vals = np.arange(-1, 6)
ell = 2

# LogLike is used only to build the injected data (signal + noise) in .signal.
# The S6 score below reads loglike_obj.signal directly and does not call
# loglike_obj(...), so mode selection here only affects the injection, not scoring.
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

# Injected data time series (signal + colored noise), time-domain (n_chan, N).
data_signal = loglike_obj.signal


def s6_score(phys_point):
    """S6 semi-coherent statistic at a physical point [logm1, logm2, a, p0, e0, cos_qS, phiS]."""
    logm1, logm2, a_i, p0_i, e0_i, cos_qS_i, phiS_i = phys_point
    qS_i = np.arccos(cos_qS_i)
    h = gwf.xp.array(waveform_response(
        10**logm1, 10**logm2, a_i, p0_i, e0_i,
        xI0, dist, qS_i, phiS_i, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt,
    ))
    return float(gwf.SNR_semicoherent(data_signal, h, N_seg=N_SEG, phase_max=True))


# ── Load paper/ext/emri_g/stage1_anneal_s12 for ellipse center and covariance ─
# Real prior_transform / log_density from paris2_sc_ext_dev.py (the script that
# produced this checkpoint), passed explicitly so Sampler.load_state has working
# callables regardless of whether the checkpoint embeds them by value.

_p1_lo = np.array([5.6, 1.7, 0.3, 13.5, 0.6, -1.0, 0.0])
_p1_hi = np.array([6.4, 2.3, 0.99, 16.5, 0.9, 1.0, 2 * np.pi])

# S schedule from paris2_sc_ext_dev.py; only matters if log_density is actually
# re-evaluated (it isn't below -- mu_center/cov come from stored searched points
# and log-densities), so the exact stage reached at checkpoint time is not needed.
S_schedule = [3.0, 10.0, 30.0]
anneal_state = {'S': S_schedule[0]}

def log_density(params):
    params = np.asarray(params)
    out = np.full(params.shape[0], -np.inf)
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
            out[i] = float(gwf.SNR_semicoherent(data_signal, h_temp, N_seg=12, phase_max=True)) * anneal_state['S']
        except Exception:
            pass
    return out

def prior_transform(u):
    u = np.asarray(u)
    t = np.zeros_like(u)
    t[:, 0] = (_p1_hi[0] - _p1_lo[0]) * u[:, 0] + _p1_lo[0]
    t[:, 1] = (_p1_hi[1] - _p1_lo[1]) * u[:, 1] + _p1_lo[1]
    t[:, 2] = (_p1_hi[2] - _p1_lo[2]) * u[:, 2] + _p1_lo[2]
    t[:, 3] = (_p1_hi[3] - _p1_lo[3]) * u[:, 3] + _p1_lo[3]
    t[:, 4] = (_p1_hi[4] - _p1_lo[4]) * u[:, 4] + _p1_lo[4]
    t[:, 5] = (_p1_hi[5] - _p1_lo[5]) * u[:, 5] + _p1_lo[5]
    t[:, 6] = (_p1_hi[6] - _p1_lo[6]) * u[:, 6] + _p1_lo[6]
    return t

stage1_anneal_path = '/scratch/e1498138/paper/ext/emri_g/stage1_anneal_s12_dev/sampler_state.pkl'
print(f'Loading stage1_anneal_s12 sampler from {stage1_anneal_path}...')
sampler_1 = parismc.Sampler.load_state(
    stage1_anneal_path,
    log_density_func=log_density,
    prior_transform=prior_transform,
)

all_pts_u  = sampler_1.searched_points_list[0]
all_logden = sampler_1.searched_log_densities_list[0]
maxld_idx  = np.argmax(all_logden)
mu_center  = prior_transform(all_pts_u[maxld_idx].reshape(1, -1))[0]
print(f'stage1_anneal_s12 maxld: {all_logden[maxld_idx]:.4f}')
print(f'stage1_anneal_s12 maxld point: {mu_center}')

samples_p1, weights_p1 = sampler_1.get_samples_with_weights(flatten=True)
weights_p1 = weights_p1 / weights_p1.sum()
rng_rs = np.random.default_rng(0)
idx_rs = rng_rs.choice(len(samples_p1), size=50_000, replace=True, p=weights_p1)
cov_posterior = np.cov(samples_p1[idx_rs].T)
print('stage1_anneal_s12 posterior 1-sigma (diag):', np.sqrt(np.diag(cov_posterior)))

del sampler_1, samples_p1, weights_p1, idx_rs

# ── Ellipsoid bounding box ─────────────────────────────────────────────────────
# e0 has high Fisher information → tight posterior at the (possibly wrong) mode.
# Use N_SIGMA_E0 >> N_SIGMA_OTHER so true e0=0.744 is safely inside the LHS volume.
# Unlike the pre-"ext" paris3_lhs_s6.py, cos_qS/phiS ARE restricted here since
# paris2_sc_ext already converged on a single sky-localized mode.

param_names = ['logm1', 'logm2', 'a', 'p0', 'e0', 'cos_qS', 'phiS']
N_SIGMA = 30.0
N_SIGMA_PER_DIM = np.array([N_SIGMA, N_SIGMA, N_SIGMA,
                             N_SIGMA, N_SIGMA, N_SIGMA, N_SIGMA])

sigma_diag = np.sqrt(np.diag(cov_posterior))
ellipse_lo = np.clip(mu_center - N_SIGMA_PER_DIM * sigma_diag, _p1_lo, _p1_hi)
ellipse_hi = np.clip(mu_center + N_SIGMA_PER_DIM * sigma_diag, _p1_lo, _p1_hi)

print(f'Ellipse prior bounds (per-dim sigma):')
for i, name in enumerate(param_names):
    print(f'  {name}: [{ellipse_lo[i]:.5f}, {ellipse_hi[i]:.5f}]  '
          f'mu={mu_center[i]:.5f}  sigma={sigma_diag[i]:.5f}  N_sigma={N_SIGMA_PER_DIM[i]:.0f}')

def ellipse_prior_transform(u):
    return ellipse_lo + (ellipse_hi - ellipse_lo) * u

def inverse_ellipse_prior_transform(params):
    return (np.asarray(params) - ellipse_lo) / (ellipse_hi - ellipse_lo)

# ── Generate LHS uniformly in the N-sigma box ──────────────────────────────────
# No ellipsoid truncation: points fill the axis-aligned box [ellipse_lo,
# ellipse_hi] (mu ± N_sigma * sigma_diag per dim), not the correlated ellipsoid.

ndim  = 7
N_LHS = int(1e5)

print(f'Generating {N_LHS} LHS points in {ndim}D box...')
_lhs_sampler = LHS(xlimits=np.column_stack([ellipse_lo, ellipse_hi]))
lhs_phys     = _lhs_sampler(N_LHS)
lhs_phys     = np.clip(lhs_phys, _p1_lo, _p1_hi)
n_inside     = N_LHS
lhs_u    = np.clip(
    np.array([inverse_ellipse_prior_transform(p) for p in lhs_phys]), 0.0, 1.0
)

# ── Evaluate S6 semi-coherent statistic ────────────────────────────────────────

print(f'Evaluating S{N_SEG} semi-coherent statistic on {n_inside} points...')
log_densities = np.full(n_inside, -np.inf)
t0 = time.time()

for i in range(n_inside):
    if i % 500 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta  = (n_inside - i) / rate
        print(f'  [{i}/{n_inside}]  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s')

    try:
        ld = s6_score(lhs_phys[i])
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

savepath = f'/scratch/e1498138/paper/ext/emri_g/paris3_lhs_s{N_SEG}_dev/lhs_s{N_SEG}.pkl'
os.makedirs(os.path.dirname(savepath), exist_ok=True)
save_data = {
    'lhs_u':          lhs_u,
    'lhs_phys':       lhs_phys,
    'log_densities':  log_densities,
    'ellipse_lo':     ellipse_lo,
    'ellipse_hi':     ellipse_hi,
    'mu_center':      mu_center,
    'cov_posterior':     cov_posterior,
    'sigma_diag':        sigma_diag,
    'N_SIGMA_PER_DIM':   N_SIGMA_PER_DIM,
    'N_SIGMA':       N_SIGMA,
    'N_LHS':             N_LHS,
    'n_inside':       n_inside,
    'N_SEG':          N_SEG,
    'statistic':      f'SNR_semicoherent(N_seg={N_SEG}, phase_max=True)',
    'T':              T,
    'dt':             dt,
}
with open(savepath, 'wb') as f:
    pickle.dump(save_data, f)
print(f'Saved to {savepath}')
