import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.constants import YRSID_SI, MTSUN_SI
import os
import sys

# Changing directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

# Configuration
use_gpu = True
force_backend = "cuda12x"
dt = 10.0  # seconds
T = 2.0  # years (use 2 years for full evolution)

# Waveform generator setup
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}

amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs = {
    "force_backend": force_backend,
    "pad_output": True,
}

waveform_gen = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu
)

# Source parameters (same as in your file)
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 15.0
e0 = 0.4
xI0 = 1.0
dist = 0.25  # Gpc
qS = 0.5
phiS = 1.0
qK = 1.0
phiK = phiS + np.pi/3
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

print("Generating waveform...")
h = waveform_gen(*params, T=T, dt=dt)

# Use h_I channel (index 1)
h_signal = h[1]

print(f"Signal length: {len(h_signal)} samples")
print(f"Duration: {len(h_signal) * dt / YRSID_SI:.2f} years")

# Generate trajectory to get fundamental frequencies
print("Computing fundamental frequencies...")
from few.trajectory.inspiral import EMRIInspiral

traj_gen = EMRIInspiral(func='KerrEccEqFlux')
t_arr, p_arr, e_arr, x_arr, Phi_phi_arr, Phi_theta_arr, Phi_r_arr = traj_gen(
    m1, m2, a, p0, e0, xI0, T=T, dt=dt
)

# Get fundamental frequencies along trajectory
freqs_list = []
for p_val, e_val in zip(p_arr, e_arr):
    Om_phi, Om_theta, Om_r = get_fundamental_frequencies(a, p_val, e_val, xI0)
    freqs_list.append([Om_phi, Om_theta, Om_r])

freqs_arr = np.array(freqs_list)
M_total_sec = (m1 + m2) * MTSUN_SI  # Total mass in seconds
freq_phi = freqs_arr[:, 0] / (2 * np.pi * M_total_sec)  # Convert to Hz
freq_theta = freqs_arr[:, 1] / (2 * np.pi * M_total_sec)
freq_r = freqs_arr[:, 2] / (2 * np.pi * M_total_sec)

# Define the four strong harmonic modes (l, m, n, k)
# Typically these are combinations like (2, 2, 0, 0), (2, 1, 0, 0), etc.
# For equatorial orbits, the four strongest are often:
modes = [
    (2, 0, 0),  # m*Omega_phi + n*Omega_theta + k*Omega_r
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
]

# Compute mode frequencies
mode_freqs = []
for m, n, k in modes:
    f_mode = m * freq_phi + n * freq_theta + k * freq_r
    mode_freqs.append(f_mode)

# Figure 1: Spectrogram with frequency trajectories
print("Computing spectrogram...")
fs = 1.0 / dt  # Sampling frequency
nperseg = 2**16  # Window size for STFT
noverlap = nperseg * 3 // 4

f, t, Sxx = signal.spectrogram(h_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

# Convert time to years and adjust to end at t_p
t_years = t / YRSID_SI
T_total = len(h_signal) * dt / YRSID_SI
t_years_adjusted = t_years - T_total  # t_p - 2y to t_p

# Interpolate trajectory times to spectrogram times
t_traj_years = t_arr - t_arr[-1]  # Adjust to end at t_p = 0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Spectrogram
ax1.pcolormesh(t_years_adjusted, f, np.log10(Sxx + 1e-30),
               shading='gouraud', cmap='gray', vmin=-15, vmax=-8)

# Overlay frequency trajectories for four strong modes
colors = ['red', 'red', 'red', 'red']
for i, (m, n, k) in enumerate(modes):
    f_interp = np.interp(t_years_adjusted, t_traj_years, mode_freqs[i])
    ax1.plot(t_years_adjusted, f_interp, color=colors[i], linewidth=2, alpha=0.8)

ax1.set_ylabel('Frequency [Hz]')
ax1.set_xlabel('Time')
ax1.set_xlim([t_years_adjusted[0], t_years_adjusted[-1]])
ax1.set_ylim([0, 0.01])

# Set x-ticks
t_p = 0
xticks = [t_p - 2, t_p - 1, t_p]
xticklabels = ['$t_p - 2y$', '$t_p - 1y$', '$t_p$']
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)

# Plot 2: Cross-sections at different times
times_before_plunge = [2.0, 3.0/12.0, 1.0/365.25]  # 2y, 3m, 1d in years
colors_cross = ['red', 'green', 'blue']
labels_cross = ['$t_p - 2y$', '$t_p - 3m$', '$t_p - 1d$']

for time_before, color, label in zip(times_before_plunge, colors_cross, labels_cross):
    # Find closest time index
    t_target = -time_before
    idx = np.argmin(np.abs(t_years_adjusted - t_target))

    # Get STFT at this time
    ax2.plot(f, Sxx[:, idx], color=color, linewidth=2, label=label, alpha=0.7)

ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Short-time Fourier amplitude [$10^{-7}$ s]')
ax2.set_xlim([0, 0.01])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrogram_analysis.png', dpi=300, bbox_inches='tight')
print("Saved spectrogram_analysis.png")
plt.show()
