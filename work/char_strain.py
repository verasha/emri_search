import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/svu/e1498138/emri_search/work')

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response

try:
    import cupy as cp
    xp = cp
    use_gpu = True
    print("Using GPU (CuPy)")
except ImportError:
    xp = np
    use_gpu = False
    print("Using CPU (NumPy)")

# ── shared parameters ─────────────────────────────────────────────────────────
T  = 3 / 12   # years
dt = 10.0     # seconds

m2   = 10.0
a    = 0.7
e0   = 0.5
xI0  = 1.0
dist = 1.8    # Gpc
qS, phiS           = np.pi, 0.0
qK, phiK           = 0.0, 0.0
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5

# single panel: leftmost case
m1, p0 = 3e5, 16.9

# ── build objects ─────────────────────────────────────────────────────────────
print("Building waveform response and analysis objects...")
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=1)
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=1)

freqs  = np.fft.rfftfreq(gwf.N, gwf.dt)[1:]   # Hz, drop DC
to_np  = lambda arr: arr.get() if use_gpu else np.asarray(arr)
psd_np = to_np(gwf.PSD)                        # (3, N_freq)

# LISA noise curve: h_n = sqrt(f * S_n), average A+E
h_n = np.sqrt(freqs * 0.5 * (psd_np[0] + psd_np[1]))

# ── generate signal ───────────────────────────────────────────────────────────
wave_params = [m1, m2, a, p0, e0, xI0, dist,
               qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

print(f"Generating signal: m1={m1:.0e}, p0={p0}...")
signal   = xp.array(waveform_response(*wave_params, T=T, dt=dt))
signal_f = gwf.freq_wave(signal)
sig_np   = to_np(signal_f)[:, 1:]   # (3, N_freq)

# per-channel characteristic strain: h_c = 2f|h~|
h_c_A = 2 * freqs * np.abs(sig_np[0])
h_c_E = 2 * freqs * np.abs(sig_np[1])

# ── plot ──────────────────────────────────────────────────────────────────────
floor = 1e-24
fig, ax = plt.subplots(figsize=(6, 5))

ax.fill_between(freqs, floor, h_c_A, color='royalblue', alpha=0.8, label='A channel')
ax.loglog(freqs, h_n, color='black', lw=1.5, label='LISA PSD')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([1e-4, 1e-1])
ax.set_ylim([1e-23, None])
ax.set_xlabel(r'$f$ (Hz)', fontsize=12)
ax.set_ylabel('Characteristic Strain', fontsize=12)

exp  = int(np.floor(np.log10(m1)))
coef = m1 / 10**exp
ax.set_title(rf'$({coef:.0f} \times 10^{exp},\,{m2:.0f},\,{p0},\,{e0})$', fontsize=11)

ax.legend(fontsize=10, loc='upper right')
ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
plt.savefig('char_strain_emri.png', dpi=150)
print("Saved char_strain_emri.png")
plt.show()
