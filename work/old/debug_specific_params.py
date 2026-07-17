#!/usr/bin/env python3
"""
Debug script for the specific parameter set causing broadcasting error:
m1=1.21e+05, m2=1.52e+01, a=0.678, p0=11.49, e0=0.363
"""

import numpy as np
import cupy as cp
import few
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum 
from few.utils.ylm import GetYlms
from few.waveform import FastKerrEccentricEquatorialFlux
import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2

print("=== DEBUG: Specific Parameter Set Analysis ===")

# Problem parameters from error log
m1, m2, a, p0, e0 = 1.21e+05, 1.52e+01, 0.678, 11.49, 0.363
xI0 = 1.0
theta = np.pi/3
phi = np.pi/2

# Time parameters
dt = 10
T = 1.0
dist = 1.0
use_gpu = True

print(f"Parameters: m1={m1:.2e}, m2={m2:.2e}, a={a:.3f}, p0={p0:.2f}, e0={e0:.3f}")
print(f"Time params: dt={dt}, T={T}")

# Initialize components
print("\n=== Initializing Components ===")

# Waveform generator
inspiral_kwargs={
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False, 
    "use_gpu": use_gpu,
    "force_backend": "cuda12x"
}

amplitude_kwargs = {"force_backend": "cuda12x"}
Ylm_kwargs = {"force_backend": "cuda12x"}
sum_kwargs = {"force_backend": "cuda12x", "pad_output": True}

waveform_gen = FastKerrEccentricEquatorialFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

# Individual components
traj = EMRIInspiral(func=KerrEccEqFlux, force_backend="cuda12x", use_gpu=use_gpu)
amp = AmpInterpKerrEccEq(force_backend="cuda12x")
interpolate_mode_sum = InterpolatedModeSum(force_backend="cuda12x")
ylm_gen = GetYlms(include_minus_m=False, force_backend="cuda12x")

print("\n=== Step 1: Generate Full Waveform ===")
h_temp = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=dt, T=T)
print(f"h_temp shape: {h_temp.shape}")
print(f"h_temp length: {len(h_temp)}")

print("\n=== Step 2: Generate Trajectory ===")
(t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(m1, m2, a, p0, e0, xI0, T=T, dt=dt)
print(f"Trajectory length: {len(t)}")
print(f"t shape: {t.shape}")
print(f"t range: [{t[0]:.6f}, {t[-1]:.6f}]")
print(f"t spacing (first few): {np.diff(t[:5])}")
t_gpu = cp.asarray(t)

print("\n=== Step 3: Generate Amplitudes ===")
teuk_modes = amp(a, p, e, x)
print(f"teuk_modes shape: {teuk_modes.shape}")

print("\n=== Step 4: Check Trajectory vs Full Waveform Timing ===")
expected_waveform_length = int(T / dt) + 1
print(f"Expected waveform length from T/dt: {expected_waveform_length}")
print(f"Actual waveform length: {len(h_temp)}")
print(f"Length ratio: {len(h_temp) / expected_waveform_length:.2f}")

print("\n=== Step 5: Check Spline Data ===")
print(f"traj.integrator_spline_t shape: {traj.integrator_spline_t.shape}")
print(f"traj.integrator_spline_phase_coeff shape: {traj.integrator_spline_phase_coeff.shape}")

print("\n=== Step 6: Get Ylms ===")
ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]
print(f"ylms shape: {ylms.shape}")

print("\n=== Step 7: Test Single Mode Interpolation ===")
# Pick first mode for testing
idx = 0
l = amp.l_arr[idx]
m = amp.m_arr[idx]
n = amp.n_arr[idx]

print(f"Testing mode ({l},{m},{n}) at index {idx}")

if m >= 0:
    teuk_modes_single = teuk_modes[:, [idx]]
    ylms_single = ylms[[idx]]
    m_arr = amp.m_arr[[idx]]
else:
    pos_m_mask = (amp.l_arr == l) & (amp.m_arr == -m) & (amp.n_arr == n)
    pos_m_idx = cp.where(pos_m_mask)[0][0]
    teuk_modes_single = (-1)**l * cp.conj(teuk_modes[:, [pos_m_idx]])
    ylms_single = ylms[[idx]]
    m_arr = cp.abs(amp.m_arr[[idx]])

print(f"teuk_modes_single shape: {teuk_modes_single.shape}")
print(f"ylms_single shape: {ylms_single.shape}")
print(f"m_arr shape: {m_arr.shape}")
print(f"amp.l_arr[[idx]] shape: {amp.l_arr[[idx]].shape}")
print(f"amp.n_arr[[idx]] shape: {amp.n_arr[[idx]].shape}")

print("\n=== Step 8: Attempt interpolate_mode_sum (This should fail) ===")
try:
    waveform = interpolate_mode_sum(
        t_gpu, teuk_modes_single, ylms_single,
        traj.integrator_spline_t, traj.integrator_spline_phase_coeff[:, [0, 2]],
        amp.l_arr[[idx]], m_arr, amp.n_arr[[idx]], 
        dt=dt, T=T
    )
    print(f"SUCCESS: waveform shape: {waveform.shape}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # Print detailed shape information for debugging
    print("\n=== Detailed Shape Debug ===")
    print(f"t_gpu: {t_gpu.shape}, dtype: {t_gpu.dtype}")
    print(f"teuk_modes_single: {teuk_modes_single.shape}, dtype: {teuk_modes_single.dtype}")
    print(f"ylms_single: {ylms_single.shape}, dtype: {ylms_single.dtype}")
    print(f"traj.integrator_spline_t: {traj.integrator_spline_t.shape}, dtype: {traj.integrator_spline_t.dtype}")
    print(f"traj.integrator_spline_phase_coeff[:, [0, 2]]: {traj.integrator_spline_phase_coeff[:, [0, 2]].shape}")
    print(f"amp.l_arr[[idx]]: {amp.l_arr[[idx]].shape}, dtype: {amp.l_arr[[idx]].dtype}")
    print(f"m_arr: {m_arr.shape}, dtype: {m_arr.dtype}")
    print(f"amp.n_arr[[idx]]: {amp.n_arr[[idx]].shape}, dtype: {amp.n_arr[[idx]].dtype}")

print("\n=== Step 9: Check Time Grid Expectations ===")
# What time grid does interpolate_mode_sum expect to create?
expected_output_length = int(T / dt) + 1
expected_times = np.linspace(0, T, expected_output_length)
print(f"Expected output time grid length: {expected_output_length}")
print(f"Expected times range: [{expected_times[0]:.6f}, {expected_times[-1]:.6f}]")
print(f"Expected dt: {expected_times[1] - expected_times[0]:.6f}")

print("\n=== Analysis Complete ===")