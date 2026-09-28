"""
Spectrogram of a plunging EMRI waveform (l=2 modes only, wide n-range) with
the (ell=2, m=2) instantaneous-frequency tracks for each n overlaid.

Adapted from the "Spectogram" cell in
work/search/old/plot_spectogram_PLUNGE_SNR30 copy.ipynb (cell 33).
"""
import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import signal

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

# Source parameters
m1, m2, a, p0, e0 = 1e6, 1e1, 0.7, 9, 0.4
xI0 = 1.0
dist = 1.8  # Gpc
qS, phiS, qK, phiK = np.pi, 0.0, 0.0, 0.0
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5

params = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

gwf = GWfuncs.GravWaveAnalysis(T, dt)

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

# ---------------------------------------------------------------------------
# Waveform (l=2 modes only, wide n-range) -> STFT spectrogram
# ---------------------------------------------------------------------------
print("Generating waveform (l=2 modes only)...")
modes_l2_all = [(2, m, n) for m in range(-2, 3) for n in range(-55, 56)]
h = waveform_gen(*params, T=T, dt=dt, mode_selection=modes_l2_all, include_minus_mkn=False)
h_real = h.real

fs = 1.0 / dt
nperseg = 2 ** 16
noverlap = nperseg * 3 // 4
f, t, Zxx = signal.stft(h_real.get(), fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=True)

t_traj_years = t_arr - t_arr[0]

log_amp = np.log10(np.abs(Zxx))
max_val = np.max(log_amp)
vmin_val = max_val - 6  # 6 orders of magnitude dynamic range

t_spec_seconds = t - t[0]
extent = [t_spec_seconds[0], t_spec_seconds[-1], f[0], f[-1]]

# ---------------------------------------------------------------------------
# Plot: spectrogram + (ell=2, m=2) frequency tracks for each n
# PRD (Physical Review D / REVTeX) style, double-column width
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

fig, ax = plt.subplots()
im = ax.imshow(log_amp,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap='Greys',
                vmin=vmin_val,
                vmax=max_val)

n_values = np.arange(-1, 6)
cmap = cm.get_cmap('winter')
colors = cmap(np.linspace(0, 1, len(n_values)))

for i, (ell, m, n) in enumerate(modes):
    if ell == 2 and m == 2:
        color_idx = n + 1
        ax.plot(t_traj_years, mode_freqs[i], linestyle='dashed', color=colors[color_idx],
                 linewidth=1.5, alpha=0.8, label=f'$n={n}$')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\log_{10}$(amp)" if have_latex else "log10(amp)")

ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [s]')
ax.set_title(r'Spectrogram vs.\ frequency tracks, $\ell=2$, $m=2$' if have_latex
             else 'Spectrogram vs. frequency tracks, l=2, m=2')
ax.set_ylim(-0.005, 0.01)

ax.legend(loc='best')
fig.tight_layout()
fig.savefig('spectrogram_plunge_l2m2.pdf')
plt.show()
