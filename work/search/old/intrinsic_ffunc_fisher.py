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
import loglikebasic
import modeselector
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
T = 0.25     # Total time

# -----------------------------
# PARIS global context (picklable functions require module scope)
# -----------------------------
# Fisher-parallelotope affine prior (primary for this script)
# _PARIS_AFFINE_CENTER = None       # type: Optional[np.ndarray]
# _PARIS_AFFINE_Q = None            # type: Optional[np.ndarray]
# _PARIS_AFFINE_B = None            # type: Optional[np.ndarray]
# # _PARIS_DIM = None                 # type: Optional[int]

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
p0 = 7.5
e0 = 0.4 
## NOTE: BELOW THIS ALL ARE FIXED
xI0 = 1.0
dist = 0.5 # Gpc
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

# NOTE: change verbose argument for debugging
loglike_obj = loglikebasic.LogLike(params_star, waveform_gen_comb, gwf, M_init=5, verbose=False, waveform_gen_sep=waveform_gen_sep, noise_weighted=True)
print('Done initializing loglike class.')
print("Setting up log_density and prior functions...")
print('Calculating SNR...')
data = loglike_obj.signal
data_snr = gwf.rhostat(data)
print('SNR calculated:', data_snr)
print("Setting up log_density and prior functions...")
# target_snr = 82.472929428421

def log_density(params):
    params = np.asarray(params)

    n_samples = params.shape[0] 
    log_likes = np.zeros(n_samples)


    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1 = 10**logm1
        m2 = 10**logm2

        loglike = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
        log_likes[i] = loglike

    return log_likes

# NOTE: OLDIEEE
def prior_transform(u):

    # sampling_test => 30x fisher box
    # logm1lim = [5.995531126784557, 6.004468873215443]
    # logm2lim = [1.4746191268598186, 1.4796233825795062]
    # alim = [0.6919479173260448, 0.7080520826739551]
    # p0lim = [7.455582230927566, 7.544417769072434]
    # e0lim = [0.3980771809772245, 0.40192281902277555]

    # box1 = 3 x box of sampling_test= 90x fisher box
    logm1lim = [5.9865933803536695, 6.0134066196463305]
    logm2lim = [1.469614871140131, 1.4846276382991939]
    alim = [0.6758437519781343, 0.7241562480218656]
    p0lim = [7.366746692782698, 7.633253307217302]
    e0lim = [0.3942315429316735, 0.40576845706832654]

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

# def prior_transform(u):
#     global _PARIS_AFFINE_CENTER, _PARIS_AFFINE_Q, _PARIS_AFFINE_B
#     u = np.asarray(u, dtype=float)
#     center = _PARIS_AFFINE_CENTER
#     Q = _PARIS_AFFINE_Q
#     b = _PARIS_AFFINE_B
#     dim = Q.shape[0]

#     def map_one(u1):
#         t = 2.0 * np.asarray(u1)[:dim] - 1.0
#         return center + Q @ (b * t)
#     if u.ndim == 1:
#         theta = map_one(u)
#         return theta
#     else:
#         out = np.zeros((u.shape[0], dim), dtype=float)
#         for i in range(u.shape[0]):
#             out[i] = map_one(u[i])
#         return out
    
print('Done setting up log-likelihood and prior.')
print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
    alpha=10000,                    # Use recent samples for weighting. 
    trail_size=int(1e3),          # Maximum trials per iteration
    boundary_limiting=True,        # Enable boundary constraints
    use_beta=True,                # Use beta correction for boundaries
    integral_num=int(1e5),        # MC samples for beta estimation
    gamma=500,                    # Covariance update frequency NOTE: changed from 100
    exclude_scale_z=10,       # No exclusion based on weights
    use_pool=False,               # Set to True for multiprocessing
    n_pool=4                      # Number of processes (if use_pool=True)
)

print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')

# Change to the search directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search')

with open('cov_matrix_intrinsic_new.pkl', 'rb') as f:
    cov_matrix = pickle.load(f)

ndim = 5
n_seed = 10

init_cov_list = [cov_matrix for _ in range(n_seed)]

print('Done setting up initial covariance matrix.')
# print('Setting up LHS samples')
# prior_sigma_range = 75
# evals, evecs = np.linalg.eigh(cov_matrix)
# _PARIS_AFFINE_CENTER = np.array(param_true)  
# _PARIS_AFFINE_Q = evecs
# _PARIS_AFFINE_B = prior_sigma_range * np.sqrt(evals)

# n_samples = int(1e5)
# xlimits = np.column_stack([
#         np.zeros(ndim, dtype=float),
#         np.ones(ndim, dtype=float),
#     ])
# sampling = LHS(xlimits=xlimits)
# lhs_points = np.clip(sampling(n_samples), 0.0, 1.0)

# # Fisher ellipse truncation
# n_before = lhs_points.shape[0]
# t = 2.0 * lhs_points - 1.0
# keep_mask = np.sum(t * t, axis=1) <= 1.0
# lhs_points = lhs_points[keep_mask]
# n_after = lhs_points.shape[0]

# print(f"[LHS] before: {n_before}, after ellipse: {n_after}")

# physical_points = prior_transform(lhs_points)
# lhs_vals = log_density(physical_points)
# largest_phys = np.max(physical_points, axis=0)
# smallest_phys = np.min(physical_points, axis=0)
# midpoint_phys = 0.5 * (largest_phys + smallest_phys)
# print("[LHS] physical max coords:", np.array2string(
#     largest_phys,
#     formatter={'float_kind': lambda x: f"{x:.12g}"},
#     separator=', '
# ))
# print("[LHS] physical min coords:", np.array2string(
#     smallest_phys,
#     formatter={'float_kind': lambda x: f"{x:.12g}"},
#     separator=', '
# ))
# print("[LHS] physical mid coords:", np.array2string(
#     midpoint_phys,
#     formatter={'float_kind': lambda x: f"{x:.12g}"},
#     separator=', '
# ))

# point_blocks = []
# point_blocks.append(lhs_points)
# log_blocks = []
# log_blocks.append(np.asarray(lhs_vals, dtype=float).reshape(-1))

# external_lhs_points = np.vstack(point_blocks) 
# external_lhs_log_densities = np.concatenate(log_blocks)

# print('Done setting up LHS samples.')

print('Initializing sampler...')
sampler = parismc.Sampler(
    ndim=ndim, 
    n_seed=n_seed,
    log_density_func=log_density,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Done initializing sampler.')

print('Preparing LHS samples...')
sampler.prepare_lhs_samples(lhs_num=int(1e6), batch_size=10)

print('Done preparing LHS samples.')

print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(5e5), 
    savepath='./intrinsic_ffunc_box1/',
    print_iter=100, # Print progress every n iterations
    # external_lhs_points=external_lhs_points,
    # external_lhs_log_densities=external_lhs_log_densities,
)
print('Done running sampling.')
