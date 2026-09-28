"""
Diagnostic: does the beta-gated F-stat veto (loglike_timemax_noise.LogLike)
fire at the TRUE injected parameters, and if so, at which observation
durations T does it stop firing?

For each T in T_values, rebuilds the ResponseWrapper/GravWaveAnalysis/LogLike
exactly as in paris1_noise_ext.py, then replicates the internals of
LogLike.__call__ evaluated at the true parameters to print rho_tot,
rho_dom_M, alpha, beta, and the final log-density (float or -inf).
"""

import numpy as np
import few
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from loglike_timemax_noise import LogLike

import cupy as cp

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
tdi_gen = 1
dt = 5

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

n_vals = np.arange(-1, 6)
ell = 2

T_values = [3/12, 6/12, 9/12, 1.0]

# groups to inspect the periodogram for: index 0 -> n=-1, index 1 -> n=0
group_indices_to_plot = [0, 1]
spectrum_T_to_plot = [3/12, 1.0]
spectra = {}  # spectra[T][group_idx] = (freqs_hz, power)

print(f'{"T (yr)":>8} {"rho_tot":>10} {"rho_dom_M":>10} {"alpha":>8} {"beta":>10} {"logl":>12}')
for T in T_values:
    print(f'Building for T={T}...')
    waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)
    gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

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

    theta_template = params_star
    selected = loglike_obj.selected_labels
    waveform_combined = loglike_obj._generate_selected_waveforms(theta_template, selected)

    h_temp = gwf.xp.array(waveform_response(*theta_template, T=T, dt=dt))
    h_temp_fft = gwf.wave_fft(h_temp)
    mode_ffts = [gwf.wave_fft(waveform_combined[i]) for i in range(waveform_combined.shape[0])]

    rho_tot = gwf.xp.sqrt(gwf.inner_timemax_f(h_temp_fft, h_temp_fft))
    rho_m = gwf.xp.array([gwf.xp.sqrt(gwf.inner_timemax_f(hf, hf)) for hf in mode_ffts])
    max_idx = int(rho_m.argmax())
    alpha = float(rho_m[max_idx] / rho_tot)
    beta = float(gwf.calc_beta(rho_m[max_idx], rho_tot))

    logl = loglike_obj(theta_template)

    rho_tot_f = float(rho_tot)
    rho_dom_f = float(rho_m[max_idx])
    print(f'{T:8.4f} {rho_tot_f:10.4f} {rho_dom_f:10.4f} {alpha:8.4f} {beta:10.4f} {logl:12.4f}')
    print(f'  all mode rho: {[float(r) for r in rho_m]}')
    print(f'  n_modes in dominant group: {len(loglike_obj.flattened_modes)}, groups: {len(selected)}')

    if any(abs(T - Tp) < 1e-9 for Tp in spectrum_T_to_plot):
        N = h_temp.shape[-1]
        freqs = gwf.xp.fft.fftfreq(N, d=dt)
        pos = slice(1, N // 2 + 1)
        freqs_pos = freqs[pos]
        freqs_pos_np = freqs_pos.get() if hasattr(freqs_pos, 'get') else freqs_pos

        group_spectra = {}
        for gi in group_indices_to_plot:
            hf = mode_ffts[gi]
            power = gwf.xp.zeros(freqs_pos.shape, dtype=gwf.xp.float64)
            for ci in range(gwf.n_chan):
                power += gwf.xp.abs(hf[ci][pos]) ** 2 / (0.5 * gwf.PSD[ci])
            power_np = power.get() if hasattr(power, 'get') else power
            group_spectra[gi] = (freqs_pos_np, power_np)
        spectra[T] = group_spectra

print('Building power spectrum comparison plot...')
fig, axes = plt.subplots(1, len(group_indices_to_plot), figsize=(6 * len(group_indices_to_plot), 5), squeeze=False)
axes = axes[0]
group_n_labels = {0: 'n=-1', 1: 'n=0'}
for ax, gi in zip(axes, group_indices_to_plot):
    for T in spectrum_T_to_plot:
        freqs_np, power_np = spectra[T][gi]
        ax.loglog(freqs_np, power_np, label=f'T={T:.2f}yr', alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'$|H(f)|^2 / (0.5\, S_n(f))$ per channel-summed')
    ax.set_title(f'Group {group_n_labels.get(gi, gi)} periodogram')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

fig.tight_layout()
outpath = dir_work + 'search/diag_beta_vs_T_spectrum.png'
fig.savefig(outpath, dpi=150)
print(f'Saved spectrum comparison plot to: {outpath}')

print('Done.')
