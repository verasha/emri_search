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

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

from few.utils.geodesic import get_fundamental_frequencies

from few.utils.constants import YRSID_SI
from smt.sampling_methods import LHS


import os
import sys

# Changing directory to FEWNEW/work
# to import stuffs
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglikealt
import modeselectoralt
import parismc
# import gc
import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration 
use_gpu = True
force_backend = "cuda12x"  
dt = 10     # Time step
T = 1/12     # Total time
print(f"Using dt = {dt} seconds, T = {T} years")

print('Initializing waveform generator...')
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": force_backend # Force GPU
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": force_backend,  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs_comb = {
    "force_backend": force_backend,  # Force GPU
    "pad_output": True,
}

sum_kwargs_sep = {
    "force_backend": force_backend,  # Force GPU
    "pad_output": True,
    "separate_modes": True,
}

print("Creating GenerateEMRIWaveform class...")
# Kerr eccentric flux
waveform_gen_comb = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, 
    frame='detector',
    inspiral_kwargs=inspiral_kwargs, 
    amplitude_kwargs=amplitude_kwargs, 
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_comb,
    use_gpu=use_gpu
)

# Kerr eccentric flux
waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, 
    frame='detector',
    inspiral_kwargs=inspiral_kwargs, 
    amplitude_kwargs=amplitude_kwargs, 
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_sep,
    use_gpu=use_gpu
)


print('Done initializing waveform generator.')

print("Creating GravWaveAnalysis class...")
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")


# Source parameters
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 15
e0 = 0.4 
# NOTE: BELOW FIXED
xI0 = 1.0
dist = 0.25 # Gpc
# Polar and azimuthal angles .. detector frame
# S = Solar system barycenter
# K = spin angular momentum of the MBH
qS = 0.5 
phiS = 1 
qK = 1 #fixed
phiK = phiS + np.pi/3
# Phases
Phi_phi0 = 0.4
Phi_theta0 = 0.0 # equatorial
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

# n-indexed mode selection parameters
n_vals = np.arange(-1, 3)  # n from -1 to 2
ell = 2  # quadrupole

# NOTE: change verbose argument for debugging
# Using n-indexed mode selection
loglike_obj = loglikealt.LogLike(
    params_star,
    waveform_gen_comb,
    gwf,
    verbose=False,
    waveform_gen_sep=waveform_gen_sep,
    ell=ell,
    n_vals=n_vals,
    M_mode=None  # No SNR filtering, use all n-groups
)

print('Done initializing loglike class.')
print('Calculating SNR...')
data = loglike_obj.signal
data_snr = gwf.rhostat(data)
print('SNR calculated:', data_snr)
print("Setting up log_density and prior functions...")


with open('cov_matrix_2yr.pkl', 'rb') as f:
    cov_matrix = pickle.load(f)

sigmas = np.sqrt(np.diag(cov_matrix))
print("2 year sigmas:", sigmas)
def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0] 
    log_likes = np.zeros(n_samples)


    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2

        loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
        log_likes[i] = 10*loglike #NOTE: scaled 10x

    return log_likes

def prior_transform(u):
    n=1000
    logm1lim = [param_true[0] - n*sigmas[0], param_true[0] + n*sigmas[0]]
    logm2lim = [param_true[1] - n*sigmas[1], param_true[1] + n*sigmas[1]]
    alim = [param_true[2] - n*sigmas[2], min(param_true[2] + n*sigmas[2], 0.999)]  # a must be <1
    p0lim = [param_true[3] - n*sigmas[3], param_true[3] + n*sigmas[3]]
    e0lim = [param_true[4] - n*sigmas[4], param_true[4] + n*sigmas[4]]

    transformed = np.zeros_like(u)

    # Uniform in log for masses

    # m1
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]

    # m2
    transformed[:, 1] = (logm2lim[1] - logm2lim[0]) * u[:, 1] + logm2lim[0]

    # Linear in others 

    # a
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]

    # p0
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0] 

    # e0
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]

    
    return transformed

    

print('Done setting up log-likelihood and prior.')

# Change to the search directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

# Load saved sampler
state_path = './intrinsic_ffunc_1mth_moremodes/sampler_state.pkl'
print(f'Loading sampler state from: {state_path}')

sampler = parismc.Sampler.load_state(state_path)

# Rebind the functions
try:
    sampler.log_density_func_original = log_density
    if hasattr(sampler, "prior_transform") and sampler.prior_transform is not None:
        sampler.prior_transform = prior_transform
    # If prior_transform was set, ensure transformed log-density hook is in place
    if getattr(sampler, "prior_transform", None) is not None:
        sampler.log_density_func = sampler.transformed_log_density_func
    else:
        sampler.log_density_func = sampler.log_density_func_original
except Exception as e:
    print(f"Warning: Could not rebind functions: {e}")
    pass

print('Done loading sampler.')
print(f"Sampler ndim: {sampler.ndim}")
print(f"Sampler n_seed: {sampler.n_seed}")
print(f"Sampler current_iter: {getattr(sampler, 'current_iter', None)}")

# Continue sampling
print('Resuming sampling...')
more_iters = int(5e5)
out_dir = './intrinsic_ffunc_1mth_moremodes_resumed/'

sampler.run_sampling(
    num_iterations=more_iters,
    savepath=out_dir,
    print_iter=100,
    stop_dlogZ=0.1
)
print('Done resuming sampling.')

# print('Setting up ParisMC sampler...')
# config = parismc.SamplerConfig(
#     merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
#     alpha=int(1e5),                    # Use recent samples for weighting. 
#     trail_size=int(1e3),          # Maximum trials per iteration
#     boundary_limiting=True,        # Enable boundary constraints
#     use_beta=True,                # Use beta correction for boundaries
#     integral_num=int(1e5),        # MC samples for beta estimation
#     gamma=500,                    # Covariance update frequency NOTE: changed from 100
#     exclude_scale_z=10,       # No exclusion based on weights
#     use_pool=False,               # Set to True for multiprocessing
#     # n_pool=4                      # Number of processes (if use_pool=True)
# )

# print('Done setting up ParisMC sampler.')
# print('Setting up initial covariance matrix...')

# # Change to the search directory
# os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
# sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

# # with open('cov_matrix_2yr.pkl', 'rb') as f:
# #     cov_matrix = pickle.load(f)

# ndim = 5
# # n_seed = 10
# n_seed = 100

# # init_cov_list = [cov_matrix for _ in range(n_seed)]
# sigma = 1e-5
# init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]

# print('Done setting up initial covariance matrix.')

# print('Initializing sampler...')
# sampler = parismc.Sampler(
#     ndim=ndim, 
#     n_seed=n_seed,
#     log_density_func=log_density,
#     init_cov_list=init_cov_list,
#     prior_transform=prior_transform,
#     config=config
# )
# print('Done initializing sampler.')

# print('Preparing LHS samples...')
# sampler.prepare_lhs_samples(lhs_num=int(1e5), batch_size=10)

# print('Done preparing LHS samples.')

# print('Running sampling...')
# sampler.run_sampling(
#     num_iterations=int(1e5), 
#     savepath='./intrinsic_ffunc_1mth_moremodes/',
#     print_iter=100, # Print progress every n iterations

# )
# print('Done running sampling.')
