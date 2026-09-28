"""
Two-sided spectrogram of a plunging EMRI waveform (full mode content) with the
ell=2 frequency tracks, summed over m, for each retained harmonic n overlaid.

Adapted from work/search/old/plot_spectogram_PLUNGE_SNR30.ipynb (cell 33,
"Spectogram only"), which differs from the ".../plot_spectogram_PLUNGE_SNR30
copy.ipynb" version used for plot_spectrogram_plunge_l2m2.py: it plots the
FULL waveform (all ell/m modes, not restricted to ell=2), uses a two-sided
(fftshift'd) STFT, and times/frequency tracks are referenced to plunge
(t - t_plunge) rather than to the start of the signal.
"""
import os
import shutil
import sys

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal
from collections import defaultdict

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.trajectory.inspiral import EMRIInspiral
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.constants import YRSID_SI, MTSUN_SI

_work_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.chdir(_work_dir)
sys.path.insert(0, _work_dir)

import GWfuncs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
use_gpu = True
force_backend = "cuda12x"
dt = 10.0   # seconds
T = 2.0     # years

inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}
amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs = {"force_backend": force_backend, "pad_output": True}

waveform_gen = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

# Source parameters (plunging case)
m1, m2, a, p0, e0 = 1e6, 3e1, 0.7, 11.7, 0.4  # p0=11.7 plunges; original was 15
xI0 = 1.0
dist = 2.0  # Gpc
qS, phiS, qK = 0.5, 1.0, 2.67
phiK = phiS + np.pi / 3
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5

params = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

gwf = GWfuncs.GravWaveAnalysis(T, dt)

# ---------------------------------------------------------------------------
# Full waveform (all modes)
# ---------------------------------------------------------------------------
print("Generating waveform...")
h = waveform_gen(*params, T=T, dt=dt)
h_real = h.real

# ---------------------------------------------------------------------------
# Trajectory -> fundamental frequencies along the inspiral
# ---------------------------------------------------------------------------
print("Computing trajectory and fundamental frequencies...")
traj_gen = EMRIInspiral(func='KerrEccEqFlux')
t_arr, p_arr, e_arr, x_arr, Phi_phi_arr, Phi_theta_arr, Phi_r_arr = traj_gen(
    m1, m2, a, p0, e0, xI0, T=T, dt=dt, upsample=True,
    Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0,
)

freqs_list = []
for p_val, e_val in zip(p_arr, e_arr):
    Om_phi, Om_theta, Om_r = get_fundamental_frequencies(a, p_val, e_val, xI0)
    freqs_list.append([Om_phi, Om_theta, Om_r])
freqs_arr = np.array(freqs_list)

M = m1 + m2
Msec = M * MTSUN_SI
freq_phi = freqs_arr[:, 0] / (2 * np.pi * Msec)
freq_r = freqs_arr[:, 2] / (2 * np.pi * Msec)

# (ell, m, n) mode frequency tracks, all ell/m, n in [-1, 5]
modes = [(l, m, n) for l in range(2, 4) for m in range(-l, l + 1) for n in range(-1, 6)]
mode_freqs = [m * freq_phi + n * freq_r for (l, m, n) in modes]

# Sum frequency tracks over m for each (ell, n)
mode_groups = defaultdict(list)
for i, (ell, m, n) in enumerate(modes):
    mode_groups[(ell, n)].append(mode_freqs[i])
summed_freqs = {(ell, n): np.sum(freq_list, axis=0) for (ell, n), freq_list in mode_groups.items()}

# ---------------------------------------------------------------------------
# Two-sided STFT spectrogram, referenced to plunge time
# ---------------------------------------------------------------------------
fs = 1.0 / dt
nperseg = 2 ** 16
noverlap = nperseg * 3 // 4

f, t, Zxx = signal.stft(h_real.get(), fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
Zxx = scipy.fft.fftshift(Zxx, axes=0)
f = scipy.fft.fftshift(f)

t_traj_years = t_arr - t_arr[-1]  # time from plunge

log_amp = np.log10(np.abs(Zxx))
max_val = np.max(log_amp)
vmin_val = max_val - 6  # 6 orders of magnitude dynamic range

t_spec_seconds = t - t[-1]  # time from plunge
extent = [t_spec_seconds[0], t_spec_seconds[-1], f[0], f[-1]]

# ---------------------------------------------------------------------------
# Plot: spectrogram + ell=2 (summed over m) frequency tracks for each n
# PRD (Physical Review D / REVTeX) style, single-column width
# ---------------------------------------------------------------------------
have_latex = shutil.which('latex') is not None
plt.rcParams.update({
    "text.usetex": have_latex,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.figsize": (3.4, 2.6),   # single-column PRD width, inches (matches check_f.py)
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(4.3, 3.3))  # extra width offsets the colorbar so the
                                             # imshow panel itself ends up ~square
im = ax.imshow(log_amp,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap='Greys',
                vmin=vmin_val,
                vmax=max_val)

n_values_summed = sorted(set(n for (ell, n) in summed_freqs.keys()))
cmap = cm.get_cmap('summer')
colors_summed = cmap(np.linspace(0, 0.8, len(n_values_summed)))

for (ell, n), freq in summed_freqs.items():
    if ell == 2:
        color_idx = n_values_summed.index(n)
        ax.plot(t_traj_years, freq, color=colors_summed[color_idx],
                 linewidth=1.5, linestyle='dashed', alpha=1, label=f'$n={n}$')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("log A")

ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time from plunge [s]')
ax.set_ylim(-0.04, 0.04)

ax.legend(loc='lower right', ncol=2, fontsize=6.5, columnspacing=0.8, handlelength=1.4,
          framealpha=0.85, facecolor='white', edgecolor='none')
fig.tight_layout()
fig.savefig('spectrogram_plunge_summed_l2.pdf')
plt.show()
