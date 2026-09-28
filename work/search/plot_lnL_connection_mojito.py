"""
lnL "connection plot": evaluate the Gaussian lnL along the straight line
connecting the injected (true) params to the Nelder-Mead MAP result found by
map_mojito_light.py, for the Mojito light EMRI_G source (seed=42 noise).

theta(t) = param_true + t * (x_map - param_true)

t=0 -> true params, t=1 -> MAP. Extends a bit past both ends to check for
overshoot / secondary peaks along the segment, which would indicate the NM
result isn't a clean local max in this direction.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import few
import os
import sys

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5
T = 3/12

waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

xI0 = 1.0

data = np.load('/scratch/e1498138/map_mojito_light_seed42.npz')
param_true = data['param_true']
x_map = data['x_map']
lnL_true_saved = float(data['lnL_true'])
lnL_map_saved = float(data['lnL_map'])

# Regenerate the exact same data (signal + seed=42 colored noise)
m1, m2, a, p0, e0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = param_true
m1, m2 = 10**m1, 10**m2

signal = gwf.xp.array(waveform_response(
    m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt,
))
noise = gwf.generate_colored_noise(seed=42)
signal = signal + noise
signal_fft = gwf.freq_wave(signal)


def full_params(x):
    (logm1, logm2, a_i, p0_i, e0_i, dist_i, qS_i, phiS_i,
     qK_i, phiK_i, Phi_phi0_i, Phi_theta0_i, Phi_r0_i) = x
    return [10**logm1, 10**logm2, a_i, p0_i, e0_i,
            xI0, dist_i, qS_i, phiS_i, qK_i, phiK_i,
            Phi_phi0_i, Phi_theta0_i, Phi_r0_i]


def lnL(theta):
    (m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t,
     qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t) = theta
    h = gwf.xp.array(waveform_response(
        m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t,
        qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t, T=T, dt=dt,
    ))
    h_fft = gwf.freq_wave(h)
    dh = gwf.inner(signal_fft, h_fft)
    hh = gwf.inner(h_fft, h_fft)
    return float(dh - 0.5 * hh)


def lnL_at_t(t):
    theta = param_true + t * (x_map - param_true)
    try:
        return lnL(full_params(theta))
    except Exception:
        return np.nan


t_vals = np.linspace(-0.3, 1.3, 81)
lnL_vals = np.array([lnL_at_t(t) for t in t_vals])

print("t, lnL:")
for t, l in zip(t_vals, lnL_vals):
    print(f"  {t: .3f}  {l: .4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t_vals, lnL_vals, '-o', ms=3, lw=1)
ax.axvline(0.0, color='C1', ls='--', label=f'true (t=0), lnL={lnL_true_saved:.3f}')
ax.axvline(1.0, color='C2', ls='--', label=f'MAP (t=1), lnL={lnL_map_saved:.3f}')
ax.set_xlabel('t  (theta = true + t*(MAP - true))')
ax.set_ylabel('lnL')
ax.set_title('lnL along the true -> MAP connecting line\nMojito light EMRI_G, seed=42')
ax.legend()
fig.tight_layout()

outpath = dir_work + 'search/lnL_connection_mojito.png'
fig.savefig(outpath, dpi=150)
print(f"\nSaved plot to {outpath}")
