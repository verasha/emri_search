import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float64, complex128
from numba.cuda import jit as cuda_jit
import math

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

import localgit.FEWNEW.work.GWfuncs_backup as GWfuncs_backup
# import gc
# import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

import dynesty

# GPU configuration and missing variables
use_gpu = True
dist = 1.0  # Distance in Gpc
dt = 10     # Time step
T = 0.1     # Total time

# Check GPU availability
if not cuda.is_available():
    print("Warning: CUDA not available, falling back to CPU")
    use_gpu = False

# GPU kernel functions for parallel computation
@cuda_jit
def power_calculation_kernel(teuk_modes, ylms, m0mask, power_out):
    """GPU kernel for calculating power of modes in parallel"""
    idx = cuda.grid(1)
    if idx < teuk_modes.shape[1] and m0mask[idx]:
        power_sum = 0.0
        for t_idx in range(teuk_modes.shape[0]):
            mode_val = teuk_modes[t_idx, idx]
            ylm_val = ylms[idx]
            power_sum += (mode_val.real * ylm_val.real + mode_val.imag * ylm_val.imag) ** 2
        power_out[idx] = power_sum

@cuda_jit
def mode_processing_kernel(teuk_modes, ylms, l_arr, m_arr, n_arr, waveform_out):
    """GPU kernel for processing individual modes in parallel"""
    idx = cuda.grid(1)
    if idx < teuk_modes.shape[1]:
        # Process each mode
        for t_idx in range(teuk_modes.shape[0]):
            mode_val = teuk_modes[t_idx, idx]
            ylm_val = ylms[idx]
            waveform_out[t_idx, idx] = mode_val * ylm_val

@cuda_jit
def chi_sq_kernel(X_modes, rho_m, chi_sq_out):
    """GPU kernel for calculating chi-squared values"""
    idx = cuda.grid(1)
    if idx < X_modes.shape[0]:
        diff = X_modes[idx] - rho_m[idx]
        chi_sq_out[idx] = diff.real * diff.real + diff.imag * diff.imag

@cuda_jit
def f_statistic_kernel(X_scalar, beta, chi_sq, f_stat_out):
    """GPU kernel for calculating f-statistic"""
    idx = cuda.grid(1)
    if idx < chi_sq.shape[0]:
        f_exp = -0.5 * beta * chi_sq[idx]
        f_stat_out[idx] = X_scalar * math.exp(f_exp)

# Initialize missing objects needed for the analysis
traj = EMRIInspiral(func=KerrEccEqFlux, force_backend="cuda12x", use_gpu=use_gpu)
amp = AmpInterpKerrEccEq(force_backend="cuda12x")
interpolate_mode_sum = InterpolatedModeSum(force_backend="cuda12x")
ylm_gen = GetYlms(include_minus_m=False, force_backend="cuda12x")

# Initialize factor and other missing variables
factor = 1.0  # Normalization factor
mp_modes = []  # Most powerful modes list
mode_labels = []  # Mode labels list

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

h_true = waveform_gen(m1_o, m2_o, a_o, p0_o, e0_o, xI_o, theta_o, phi_o, dist=dist, dt=dt, T=0.1)

N = int(len(h_true)) 
gwf = GWfuncs_backup.GravWaveAnalysis(N=N,dt=dt)

# Define structured array for storing mode information
mode_dtype = np.dtype([
    ('m1', 'f8'), ('m2', 'f8'), ('a', 'f8'), ('p0', 'f8'), 
    ('e0', 'f8'), ('xI0', 'f8'), ('theta', 'f8'), ('phi', 'f8'),
    ('l', 'i4'), ('m', 'i4'), ('n', 'i4'), 
    ('power_fraction', 'f8'), ('rank', 'i4'),
    ('f_statistic', 'f8'), ('total_power', 'f8')
])

# Global storage for mode data
mode_database = []

def loglike_gpu(params, save_modes=True):
    """GPU-parallelized log-likelihood function for the nested sampling."""
    # Convert theta to parameters
    m1, m2, a, p0, e0, xI0, theta, phi = params

    # Generate template waveform with the current parameters
    h_temp = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=dt, T=1)

    # Generate trajectory
    (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(m1, m2, a, p0, e0, xI0, T=T, dt=dt)

    # Get amplitudes along trajectory
    teuk_modes = amp(a, p, e, x)

    # Get Ylms
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    # Calculate power for all modes using GPU parallelization
    m0mask = amp.m_arr_no_mask != 0
    
    # Allocate GPU memory for power calculation
    power_gpu = cuda.device_array(teuk_modes.shape[1], dtype=np.float64)
    
    # Configure GPU kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (teuk_modes.shape[1] + threads_per_block - 1) // threads_per_block
    
    # Launch GPU kernel for power calculation
    power_calculation_kernel[blocks_per_grid, threads_per_block](
        cuda.as_cuda_array(teuk_modes), 
        cuda.as_cuda_array(ylms), 
        cuda.as_cuda_array(m0mask), 
        power_gpu
    )
    
    # Copy result back to host
    total_power = power_gpu.copy_to_host()
    total_power_sum = np.sum(total_power)

    # Change num of selected modes here 
    M_mode = 3

    # Get top M indices
    top_indices_gpu = gwf.xp.argsort(total_power)[-M_mode:][::-1]  # Top M indices in descending order
    
    if use_gpu:
        top_indices = top_indices_gpu.get().tolist()  # Convert to CPU list only once
    else:
        top_indices = top_indices_gpu.tolist()

    # Generate hm_arr for top modes using GPU parallelization
    waveform_per_mode = []
    
    # Allocate GPU arrays for batch processing
    waveform_gpu = cuda.device_array((teuk_modes.shape[0], len(top_indices)), dtype=np.complex128)
    
    # Configure kernel for mode processing
    mode_threads_per_block = 64
    mode_blocks_per_grid = (len(top_indices) + mode_threads_per_block - 1) // mode_threads_per_block
    
    for i, idx in enumerate(top_indices):
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
            t, teuk_modes_single, ylms_single,
            traj.integrator_spline_t, traj.integrator_spline_phase_coeff[:, [0, 2]],
            amp.l_arr[[idx]], m_arr, amp.n_arr[[idx]], 
            dt=dt, T=T
        )
        waveform_per_mode.append(waveform / factor)

    # Calculate rho_m using GPU
    rho_m = gwf.rhostat_modes(waveform_per_mode)
    # Calculate Xm using GPU
    X_modes = gwf.Xmstat(waveform_per_mode, rho_m)

    # Calculate X_scalar
    Xdotrho = gwf.xp.sum(X_modes * rho_m)
    rho_norm = gwf.xp.sqrt(gwf.xp.sum(rho_m**2))
    X_scalar = Xdotrho / rho_norm

    # Calculate optimal SNR of most dominant mode by power
    rho_dom_M = gwf.rhostat(waveform_per_mode[0])

    # Calculate total rho 
    rho_tot = gwf.rhostat(h_temp)

    # Calculate alpha
    alpha = rho_dom_M / rho_tot

    # Calculate beta 
    beta_num = 2 * gwf.xp.log(alpha * rho_tot)
    beta_denom = (1-alpha**2) * rho_tot**2 
    beta = beta_num / beta_denom

    # Calculate chi sq using GPU kernel
    chi_sq_gpu = cuda.device_array(len(X_modes), dtype=np.float64)
    chi_blocks = (len(X_modes) + threads_per_block - 1) // threads_per_block
    
    chi_sq_kernel[chi_blocks, threads_per_block](
        cuda.as_cuda_array(X_modes), 
        cuda.as_cuda_array(rho_m), 
        chi_sq_gpu
    )
    
    chi_sq = chi_sq_gpu.copy_to_host()

    # Calculate f statistic using GPU kernel
    f_stat_gpu = cuda.device_array(len(chi_sq), dtype=np.float64)
    
    f_statistic_kernel[chi_blocks, threads_per_block](
        X_scalar, beta, cuda.as_cuda_array(chi_sq), f_stat_gpu
    )
    
    f_stat_values = f_stat_gpu.copy_to_host()
    f_stat = np.sum(f_stat_values)

    # Save dominant mode information
    if save_modes:
        for i, idx in enumerate(top_indices):
            l = int(amp.l_arr[idx])
            m_val = int(amp.m_arr[idx])
            n = int(amp.n_arr[idx])
            power_frac = float(total_power[idx] / total_power_sum)
            
            # Create mode entry
            mode_entry = np.array([(
                m1, m2, a, p0, e0, xI0, theta, phi,
                l, m_val, n, power_frac, i+1,
                float(f_stat), float(total_power_sum)
            )], dtype=mode_dtype)
            
            mode_database.append(mode_entry[0])

    return f_stat

# Keep original CPU version for comparison
def loglike(params):
    """Original CPU log-likelihood function for the nested sampling."""
    # Convert theta to parameters
    m1, m2, a, p0, e0, xI0, theta, phi = params

    # Generate template waveform with the current parameters
    h_temp = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=dt, T=1)

    # Generate trajectory
    (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = traj(m1, m2, a, p0, e0, xI0, T=T, dt=dt)

    # Get amplitudes along trajectory
    teuk_modes = amp(a, p, e, x)

    # Get Ylms
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    # Calculate power for all modes
    m0mask = amp.m_arr_no_mask != 0
    total_power = gwf.calc_power(teuk_modes, ylms, m0mask)

    # Change num of selected modes here 
    M_mode = 3

    # Get top M indices
    top_indices_gpu = gwf.xp.argsort(total_power)[-M_mode:][::-1]  # Top M indices in descending order
    if use_gpu:
        top_indices = top_indices_gpu.get().tolist()  # Convert to CPU list only once
    else:
        top_indices = top_indices_gpu.tolist()

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
            t, teuk_modes_single, ylms_single,
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

    # Calculate alpha
    alpha = rho_dom_M / rho_tot

    # Calculate beta 
    beta_num = 2 * gwf.xp.log(alpha * rho_tot)
    beta_denom = (1-alpha**2) * rho_tot**2 
    beta = beta_num / beta_denom

    # Calculate chi sq
    chi_sq = gwf.xp.abs(X_modes - rho_m)**2

    # Calculate f statistic
    f_exp = -0.5 * beta * chi_sq 
    f_stat = X_scalar * gwf.xp.exp(f_exp)

    return f_stat
    

# File I/O functions for mode data
def save_mode_database(filename='mode_database.npy'):
    """Save the mode database to a NumPy binary file"""
    if mode_database:
        mode_array = np.array(mode_database, dtype=mode_dtype)
        np.save(filename, mode_array)
        print(f"Saved {len(mode_database)} mode entries to {filename}")
    else:
        print("No mode data to save")

def load_mode_database(filename='mode_database.npy'):
    """Load mode database from a NumPy binary file"""
    try:
        mode_array = np.load(filename)
        print(f"Loaded {len(mode_array)} mode entries from {filename}")
        return mode_array
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

def clear_mode_database():
    """Clear the global mode database"""
    global mode_database
    mode_database = []
    print("Mode database cleared")

def get_dominant_modes_summary():
    """Get summary statistics of dominant modes"""
    if not mode_database:
        print("No mode data available")
        return None
    
    mode_array = np.array(mode_database, dtype=mode_dtype)
    
    # Most common dominant modes (rank=1)
    dominant_only = mode_array[mode_array['rank'] == 1]
    
    print("\n=== Dominant Mode Analysis ===")
    print(f"Total parameter sets: {len(np.unique(mode_array[['m1', 'm2', 'a', 'p0', 'e0', 'xI0', 'theta', 'phi']]))}")
    print(f"Total mode entries: {len(mode_array)}")
    
    if len(dominant_only) > 0:
        # Most frequent (l,m,n) combinations for dominant modes
        lmn_combos = np.array([f"({l},{m},{n})" for l,m,n in zip(dominant_only['l'], dominant_only['m'], dominant_only['n'])])
        unique_lmn, counts = np.unique(lmn_combos, return_counts=True)
        
        print("\nMost common dominant modes:")
        for lmn, count in zip(unique_lmn[np.argsort(counts)[::-1]][:5], np.sort(counts)[::-1][:5]):
            print(f"  {lmn}: {count} times ({100*count/len(dominant_only):.1f}%)")
        
        print(f"\nPower fraction stats for dominant modes:")
        print(f"  Mean: {np.mean(dominant_only['power_fraction']):.3f}")
        print(f"  Std:  {np.std(dominant_only['power_fraction']):.3f}")
        print(f"  Min:  {np.min(dominant_only['power_fraction']):.3f}")
        print(f"  Max:  {np.max(dominant_only['power_fraction']):.3f}")
    
    return mode_array

# Parameter space search example
def parameter_space_search_example(n_samples=100):
    """Example parameter space search that saves dominant mode info"""
    print(f"Running parameter space search with {n_samples} samples...")
    
    # Clear previous results
    clear_mode_database()
    
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
        # Sample parameters uniformly (you might want priors here)
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
            # Evaluate likelihood and save modes
            f_stat = loglike_gpu(params, save_modes=True)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{n_samples} evaluations")
                
        except Exception as e:
            print(f"Error in evaluation {i+1}: {e}")
            continue
    
    # Save results
    save_mode_database('parameter_search_modes.npy')
    
    # Show summary
    get_dominant_modes_summary()

def prior_transform(params):
    ## TODO ##
    return ###

parameter_space_search_example(n_samples=100)
