print("Starting imports...")
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

print("Importing few...")
import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum 


from few.utils.ylm import GetYlms

from few import get_file_manager

from few.waveform import FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

import os
import sys

# # Change to the desired directory
# os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

# # Add it to Python path
# sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

print("Importing GWfuncs...")
import localgit.FEWNEW.work.GWfuncs_backup as GWfuncs_backup
# import gc
# import pickle
print("Importing cupy...")
import cupy as cp

print("Configuring few...")
# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

print("Importing dynesty...")
import dynesty

# GPU configuration and missing variables
use_gpu = True
dist = 1.0  # Distance in Gpc
dt = 10     # Time step
T = 1.0     # Total time

# Check GPU availability
if not cuda.is_available():
    print("Warning: CUDA not available, falling back to CPU")
    use_gpu = False

print("Setting up waveform generator...")
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
        "use_gpu" : use_gpu,
        "force_backend": "cuda12x"  # Force GPU
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    # "use_gpu" : use_gpu
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    "pad_output": False,
    # "use_gpu" : use_gpu
}

print("Creating FastKerrEccentricEquatorialFlux...")
# Kerr eccentric flux
waveform_gen = FastKerrEccentricEquatorialFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

#Generating data (true)

m1_o = 1e5
m2_o = 1e1
a_o = 0.3
p0_o = 40
e0_o = 0.1
xI_o = 1.0
theta_o = np.pi/3  # polar viewing angle
phi_o = np.pi/2  # azimuthal viewing angle

print("Generating true waveform...")
h_true = waveform_gen(m1_o, m2_o, a_o, p0_o, e0_o, xI_o, theta_o, phi_o, dist=dist, dt=dt, T=1)

print("Creating GWfuncs analysis object...")
N_data = int(len(h_true))  # Use data length as reference
gwf = GWfuncs_backup.GravWaveAnalysis(N=N_data, dt=dt)

print("Initializing trajectory and amplitude generators...")
# Initialize trajectory and amplitude generators
traj = EMRIInspiral(func=KerrEccEqFlux, force_backend="cuda12x", use_gpu=use_gpu)
amp = AmpInterpKerrEccEq(force_backend="cuda12x")
interpolate_mode_sum = InterpolatedModeSum(force_backend="cuda12x")
ylm_gen = GetYlms(include_minus_m=False, force_backend="cuda12x")

print("All initialization complete! Starting parameter_space_search_example...")
parameter_space_search_example(n_samples=2)

def loglike(params):
    """Log-likelihood function for the nested sampling."""
    # Convert theta to parameters
    m1, m2, a, p0, e0, xI0, theta, phi = params

    # Generate template waveform with the current parameters
    h_temp = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=dt, T=1)

    # Calculate the factor for normalization
    factor = gwf.dist_factor(dist, m2)

    # Generate trajectory
    (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(m1, m2, a, p0, e0, xI0, T=T, dt=dt)
    t_gpu = cp.asarray(t)

    # Get amplitudes along trajectory
    teuk_modes = amp(a, p, e, x)

    # Get Ylms
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    # Calculate power for all modes
    m0mask = amp.m_arr_no_mask != 0
    total_power = gwf.calc_power(teuk_modes, ylms, m0mask)

    # Get mode labels
    mode_labels = [f"({l},{m},{n})" for l,m,n in zip(amp.l_arr, amp.m_arr, amp.n_arr)]

    # Change num of selected modes here 
    M_mode = 10

    # Get top M indices
    top_indices_gpu = gwf.xp.argsort(total_power)[-M_mode:][::-1]  # Top M indices in descending order
    top_indices = top_indices_gpu.get().tolist()  # Convert to CPU list only once

    # Pick modes based on top M power contributions
    mp_modes = [mode_labels[idx] for idx in top_indices]
    top_indices = [mode_labels.index(mode) for mode in mp_modes]

    # Generate hm_arr for top modes
    waveform_per_mode = []
    for idx in top_indices:
        l = amp.l_arr[idx]
        m = amp.m_arr[idx]
        n = amp.n_arr[idx]

        if m >= 0:
            teuk_modes_single = teuk_modes[:, [idx]]
            ylms_single = ylms[[idx]]
            m_arr = amp.m_arr[[idx]]
        else:
            pos_m_mask = (amp.l_arr == l) & (amp.m_arr == -m) & (amp.n_arr == n)
            pos_m_idx = gwf.xp.where(pos_m_mask)[0][0]
            teuk_modes_single = (-1)**l * gwf.xp.conj(teuk_modes[:, [pos_m_idx]])
            ylms_single = ylms[[idx]]
            m_arr = gwf.xp.abs(amp.m_arr[[idx]]) 

        waveform = interpolate_mode_sum(
            t_gpu, teuk_modes_single, ylms_single,
            traj.integrator_spline_t, traj.integrator_spline_phase_coeff[:, [0, 2]],
            amp.l_arr[[idx]], m_arr, amp.n_arr[[idx]], 
            dt=dt, T=T
        )
        waveform_per_mode.append(waveform / factor)

    # Calculate rho_m
    rho_m = gwf.rhostat_modes(waveform_per_mode)
    # Calculate Xm 
    X_modes = gwf.Xmstat(waveform_per_mode, rho_m)

    # Calculate X_scalar
    Xdotrho = gwf.xp.sum(X_modes * rho_m)
    rho_norm = gwf.xp.sqrt(gwf.xp.sum(rho_m**2))
    X_scalar = Xdotrho / rho_norm

    # Calculate optimal SNR of most dominant mode by power
    rho_dom_M = gwf.rhostat(waveform_per_mode[0])

    # Calculate total rho 
    rho_tot = gwf.rhostat(h_temp)

    # Calculate alpha with numerical stability
    alpha = rho_dom_M / (rho_tot + 1e-30)  # Add small epsilon to avoid division by zero
    alpha = gwf.xp.clip(alpha, 1e-10, 0.99)  # Clip to avoid edge cases

    # Calculate chi sq
    chi_sq = gwf.xp.linalg.norm(X_modes - rho_m)**2

    # Calculate f statistic using alternate formula: f = X * (alpha*rho)^(-chi_sq^2/((1-alpha^2)*rho^2))
    # This avoids the problematic exp(log(...)) construction
    base = alpha * rho_tot
    base = gwf.xp.clip(base, 1e-30, gwf.xp.inf)  # Ensure positive base
    
    exponent = -chi_sq**2 / ((1 - alpha**2) * rho_tot**2 + 1e-30)  # Add epsilon to denominator
    exponent = gwf.xp.clip(exponent, -700, 700)  # Prevent overflow
    
    f_stat = X_scalar * (base ** exponent)

    return f_stat
    

def prior_transform(params):
    ## TODO ##
    return ###

# Parameter space search example
def parameter_space_search_example(n_samples=10):
    """Example parameter space search"""
    print(f"Running parameter space search with {n_samples} samples...")
    
    # Parameter ranges (adjust as needed)
    m1_range = (1e4, 1e6)
    m2_range = (10, 100)  
    a_range = (0.1, 0.9)
    p0_range = (10, 100)
    e0_range = (0.01, 0.5)
    xI0_range = (0.5, 1.0)
    theta_range = (0, np.pi)
    phi_range = (0, 2*np.pi)
    
    # Generate random parameter samples
    np.random.seed(42)  # For reproducibility
    
    for i in range(n_samples):
        # Sample parameters uniformly
        params = [
            np.random.uniform(*m1_range),
            np.random.uniform(*m2_range), 
            np.random.uniform(*a_range),
            np.random.uniform(*p0_range),
            np.random.uniform(*e0_range),
            np.random.uniform(*xI0_range),
            np.random.uniform(*theta_range),
            np.random.uniform(*phi_range)
        ]
        
        try:
            # Evaluate likelihood
            f_stat = loglike(params)
            print(f"Sample {i+1}/{n_samples}: f_stat = {f_stat}")
                
        except Exception as e:
            print(f"Error in evaluation {i+1}: {e}")
            continue
    
    print("Parameter space search completed!")

