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

import os
import sys

# Change to the desired directory
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

# Add it to Python path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglike
import modeselector
import parismc
# import gc
# import pickle
import cupy as cp

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")


# GPU configuration 
use_gpu = True
force_backend = "cuda12x"  
dt = 10     # Time step
T = 0.25     # Total time

print('Initializing waveform generator...')
# keyword arguments for inspiral generator 
inspiral_kwargs={
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0, #change to 1/True for uniform sampling
        "include_minus_m": False, 
}

# keyword arguments for inspiral generator 
amplitude_kwargs = {
    "force_backend": "cuda12x" # Force GPU
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    # "assume_positive_m": True  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "force_backend": "cuda12x",  # Force GPU
    "pad_output": True,
}

print("Creating GenerateEMRIWaveform class...")
# Kerr eccentric flux
waveform_gen = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux, 
    frame='detector',
    inspiral_kwargs=inspiral_kwargs, 
    amplitude_kwargs=amplitude_kwargs, 
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu
)

print('Done initializing waveform generator.')

#Generating data (true)

# Source parameters
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4 
xI0 = 1.0 #NOTE: fixed
dist = 3 # Gpc
# Polar and azimuthal angles .. detector frame
# S = Solar system barycenter
# K = spin angular momentum of the MBH
qS = 0.5 
phiS = 1 
qK = 1 #NOTE: fixed
phiK = phiS + np.pi/3
# Phases
Phi_phi0 = 0.4
Phi_theta0 = 0.0 # NOTE: fixed
Phi_r0 = 0.5

print('Generating data signal...')
data = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
print('Done generating data signal.')


print('Setting up GWFuncs...')
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)
print('Done setting up GWFuncs.')

print('Setting up log-likelihood and prior...')
def loglike(params):
    # start_time = time.time()
    params = np.asarray(params)

    n_samples = params.shape[0]
    # print(f'Processing {n_samples} samples')
    log_likes = np.zeros(n_samples)

    for i in range(n_samples):
        # sample_start = time.time()
        # try:
        m1, m2, a, p0, e0, dist, qS, phiS, Phi_phi0, Phi_r0 = params[i]
        phiK = phiS + np.pi/3

        # htemp_start = time.time()
        htemp = waveform_gen(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                            Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
        
        # htemp_end = time.time()
        # print(f'    Waveform generation time: {htemp_end - htemp_start:.4f}s')

        # loglike_start = time.time()
        res = data - htemp
        res_f = gwf.freq_wave(res)
        inner_res = gwf.inner(res_f, res_f)
        log_likes[i] = -0.5 * inner_res
            # loglike_end = time.time()
            # print(f'    Log-likelihood computation time: {loglike_end - loglike_start:.4f}s')

            # sample_end = time.time()
            # print(f'  Sample {i+1}/{n_samples}: {sample_end - sample_start:.4f}s')

        # except Exception as e:
        #     # sample_end = time.time()
            # print(f'  Sample {i+1}/{n_samples}: ERROR - {e}')
        #     log_likes[i] = -np.inf

    # end_time = time.time()
    # print(f'Total batch time: {end_time - start_time:.4f}s')
    return log_likes

    
def prior_transform(u):
    # 3 SIGMA
    # m1lim = [9.9999999984e+05, 1.0000000002e+06]
    # m2lim = [2.9999906018e+01, 3.0000093982e+01]
    # alim = [6.9987915412e-01, 7.0012084588e-01]
    # p0lim = [7.4993352066e+00, 7.5006647934e+00]
    # e0lim = [3.9996851706e-01, 4.0003148294e-01]
    # distlim = [2.9386476938e+00, 3.0613523062e+00]
    # qSlim = [4.5998172393e-01, 5.4001827607e-01]
    # phiSlim = [9.6625687330e-01, 1.0337431267e+00]
    # Phiphilim = [3.7348894300e-01, 4.2651105700e-01]
    # Phirlim = [4.8726834297e-01, 5.1273165703e-01]

    # 5 SIGMA
    m1lim = [9.9999999974e+05, 1.0000000003e+06]
    m2lim = [2.9999843364e+01, 3.0000156636e+01]
    alim = [6.9979859020e-01, 7.0020140980e-01]
    p0lim = [7.4988920111e+00, 7.5011079889e+00]
    e0lim = [3.9994752843e-01, 4.0005247157e-01]
    distlim = [2.8977461563e+00, 3.1022538437e+00]
    qSlim =  [4.3330287321e-01, 5.6669712679e-01]
    phiSlim = [9.4376145549e-01, 1.0562385445e+00]
    Phiphilim = [3.5581490499e-01, 4.4418509501e-01]
    Phirlim = [4.7878057162e-01, 5.2121942838e-01]

    transformed = np.zeros_like(u)

    # Log-uniform for masses

    # m1
    transformed[:, 0] = 10**(np.log10(m1lim[0]) + u[:, 0] * (np.log10(m1lim[1]) - np.log10(m1lim[0])))

    # m2
    transformed[:, 1] = 10**(np.log10(m2lim[0]) + u[:, 1] * (np.log10(m2lim[1]) - np.log10(m2lim[0])))
    
    # Uniform for other parameters

    # a
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]

    # p0
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0] 

    # e0
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]

    # dist 
    transformed[:, 5] = (distlim[1] - distlim[0]) * u[:, 5] + distlim[0]

    # qS
    transformed[:, 6] = (qSlim[1] - qSlim[0]) * u[:, 6] + qSlim[0]

    # phiS
    transformed[:, 7] = (phiSlim[1] - phiSlim[0]) * u[:, 7] + phiSlim[0]

    # Phi_phi0
    transformed[:, 8] = (Phiphilim[1] - Phiphilim[0]) * u[:, 8] + Phiphilim[0]

    # Phi_r0
    transformed[:, 9] = (Phirlim[1] - Phirlim[0]) * u[:, 9] + Phirlim[0]

    
    return transformed

print('Done setting up log-likelihood and prior.')
print('Setting up ParisMC sampler...')
config = parismc.SamplerConfig(
    merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
    alpha=1000,                    # Use recent samples for weighting. NOTE: can change to 10k
    trail_size=int(1e3),          # Maximum trials per iteration
    boundary_limiting=True,        # Enable boundary constraints
    use_beta=True,                # Use beta correction for boundaries
    integral_num=int(1e5),        # MC samples for beta estimation
    gamma=100,                    # Covariance update frequency
    exclude_scale_z=np.inf,       # No exclusion based on weights
    use_pool=False,               # Set to True for multiprocessing
    n_pool=4                      # Number of processes (if use_pool=True)
)
print('Done setting up ParisMC sampler.')
print('Setting up initial covariance matrix...')
import pickle
with open('cov_matrix.pkl', 'rb') as f:
    cov_matrix = pickle.load(f)

ndim = 10
n_seed = 10  # Number of initial walkers/chains. can just change to 1

init_cov_list = [cov_matrix for _ in range(n_seed)]

print('Done setting up initial covariance matrix.')
print('Initializing sampler...')
sampler = parismc.Sampler(
    ndim=ndim, 
    n_seed=n_seed,
    log_density_func=loglike,
    init_cov_list=init_cov_list,
    prior_transform=prior_transform,
    config=config
)
print('Done initializing sampler.')
print('Preparing LHS samples...')
sampler.prepare_lhs_samples(lhs_num=int(1e5), batch_size=100) #NOTE: change to 10-100
print('Done preparing LHS samples.')
print('Running sampling...')
sampler.run_sampling(
    num_iterations=int(1e5),  #NOTE: 
    savepath='./likelihoodtest3_results/',
    print_iter=100 # Print progress every n iterations
)
print('Done running sampling.')