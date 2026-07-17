"""
Iterative Fisher-guided LHS sampling
Implements shrinking Fisher-box prior refinement through repeated sampling cycles
"""

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

# Changing directory to FEWNEW/work to import stuffs
os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglikebasic
import modeselector
import parismc
import pickle
import cupy as cp

from tqdm import tqdm
from stableemrifisher.fisher.fisher import StableEMRIFisher
from lisatools.sensitivity import get_sensitivity, CornishLISASens


# ============================================================================
# CONFIGURATION
# ============================================================================

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# GPU configuration
use_gpu = True
force_backend = "cuda12x"
dt = 10     # Time step
T = 0.25     # Total time

# Iteration parameters
n_outer_iterations = 5  # Number of shrinking iterations
n_lhs_samples = 20       # Number of LHS samples per iteration
n_optim_iterations = 50  # Number of optimization iterations per sample
shrink_factor = 0.8      # Shrinkage factor per iteration
fisher_box_scale = 30    # Initial Fisher box scale (30x Fisher)

# ============================================================================
# WAVEFORM SETUP
# ============================================================================

print('Initializing waveform generator...')

# keyword arguments for inspiral generator
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}

amplitude_kwargs = {
    "force_backend": force_backend
}

Ylm_kwargs = {
    "force_backend": force_backend,
}

sum_kwargs_comb = {
    "force_backend": force_backend,
    "pad_output": True,
}

sum_kwargs_sep = {
    "force_backend": force_backend,
    "pad_output": True,
    "separate_modes": True,
}

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

# ============================================================================
# PROBLEM SETUP
# ============================================================================

print("Creating GravWaveAnalysis class...")
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

print("Initializing loglike class...")
# Source parameters
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4
xI0 = 1.0
dist = 0.5  # Gpc
qS = 0.5
phiS = 1
qK = 1
phiK = phiS + np.pi/3
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

loglike_obj = loglikebasic.LogLike(params_star, waveform_gen_comb, gwf, M_init=5,
                                   verbose=False, waveform_gen_sep=waveform_gen_sep,
                                   noise_weighted=True)
print('Done initializing loglike class.')

print('Calculating SNR...')
data = loglike_obj.signal
data_snr = gwf.rhostat(data)
print('SNR calculated:', data_snr)

# Check log-likelihood at true parameters
print('\n' + '='*70)
print('VERIFYING LOG-LIKELIHOOD AT TRUE PARAMETERS')
print('='*70)
print(f'True parameters: {param_true}')
loglike_at_true = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
print(f'Log-likelihood at true parameters: {loglike_at_true}')
print(f'Expected log-likelihood: ~82')
if abs(loglike_at_true - 82) > 10:
    print('WARNING: Log-likelihood at true parameters is NOT ~82!')
    print('This suggests a problem with the likelihood calculation or setup')
print('='*70 + '\n')

# ============================================================================
# FISHER MATRIX SETUP
# ============================================================================

waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_comb,
    use_gpu=use_gpu
)

waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(frame='detector')

sef = StableEMRIFisher(
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    stats_for_nerds=False, #change for verbose output
    use_gpu=use_gpu,
    deriv_type='stable',
    noise_model=get_sensitivity,
    noise_kwargs={'sens_fn': CornishLISASens, 'return_type': 'PSD'},
    channels=["A"]
)

# Fisher calculation parameters
der_order = 4
Ndelta = 8
stability_plot = False
param_names = ['m1', 'm2', 'a', 'p0', 'e0']
fixed_params = [xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_jacobian(m1_val, m2_val):
    """Compute Jacobian for log-scaling masses at given m1, m2 values"""
    J = np.eye(5)
    J[0, 0] = m1_val * np.log(10)
    J[1, 1] = m2_val * np.log(10)
    return J

def log_density(params):
    """Compute log-likelihood for parameter samples"""
    params = np.asarray(params)
    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)

    for i in range(n_samples):
        logm1, logm2, a, p0, e0 = params[i]
        m1_val = 10**logm1
        m2_val = 10**logm2

        loglike = loglike_obj(np.array([m1_val, m2_val, a, p0, e0, xI0, dist,
                                       qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
        log_likes[i] = loglike

    return log_likes


class PriorTransform:
    """Picklable prior transform class with given limits"""
    def __init__(self, limits):
        logm1lim, logm2lim, alim, p0lim, e0lim = limits
        self.logm1lim = logm1lim
        self.logm2lim = logm2lim
        self.alim = alim
        self.p0lim = p0lim
        self.e0lim = e0lim

    def __call__(self, u):
        transformed = np.zeros_like(u)
        transformed[:, 0] = (self.logm1lim[1] - self.logm1lim[0]) * u[:, 0] + self.logm1lim[0]
        transformed[:, 1] = (self.logm2lim[1] - self.logm2lim[0]) * u[:, 1] + self.logm2lim[0]
        transformed[:, 2] = (self.alim[1] - self.alim[0]) * u[:, 2] + self.alim[0]
        transformed[:, 3] = (self.p0lim[1] - self.p0lim[0]) * u[:, 3] + self.p0lim[0]
        transformed[:, 4] = (self.e0lim[1] - self.e0lim[0]) * u[:, 4] + self.e0lim[0]
        return transformed


def calculate_fishers(physical_points):
    """Calculate Fisher matrices for given physical parameter points"""
    # Transform log masses to linear
    trans_points = physical_points.copy()
    trans_points[:, 0] = 10**trans_points[:, 0]
    trans_points[:, 1] = 10**trans_points[:, 1]

    # Prepare parameter lists
    pars_list = [list(params) + fixed_params for params in trans_points]

    # Calculate Fisher matrices
    Fishers = []
    Fishers_scaled = []
    for i, params in enumerate(tqdm(pars_list, desc="Computing Fishers")):
        Fisher = sef(*params, param_names=param_names,
                    T=T, dt=dt,
                    der_order=der_order,
                    Ndelta=Ndelta,
                    stability_plot=stability_plot,
                    live_dangerously=False)
        Fishers.append(Fisher)

        # Scale Fisher matrix using Jacobian specific to this point
        m1_val = trans_points[i, 0]
        m2_val = trans_points[i, 1]
        J = get_jacobian(m1_val, m2_val)
        Fishers_scaled.append(J.T @ Fisher @ J)

    return Fishers_scaled


def compute_new_limits(best_point, fisher_matrix, current_scale):
    """
    Compute new prior limits based on best point and Fisher matrix

    Parameters:
    - best_point: Best parameter point found (in log-mass space)
    - fisher_matrix: Scaled Fisher matrix at best point
    - current_scale: Current Fisher box scale

    Returns:
    - New limits for prior_transform
    """
    cov = np.linalg.inv(fisher_matrix)
    sigmas = np.sqrt(np.diag(cov))

    # Fisher box = 2.5 * sigma
    # current_scale x fisher box = current_scale * 2.5 * sigma
    limits = []

    for i, val in enumerate(best_point):
        prior_low = val - current_scale * 2.5 * sigmas[i]
        prior_high = val + current_scale * 2.5 * sigmas[i]
        limits.append([prior_low, prior_high])

    return limits


def run_optimization_round(lhs_points, physical_points, fishers_scaled, prior_transform,
                          n_iterations=50):
    """
    Run optimization around LHS points using Gaussian proposals

    Returns:
    - sampler: The ParisMC sampler with results
    - best_point: Best point found (unit cube)
    - best_physical_point: Best point in physical space
    - best_log_density: Best log-likelihood
    """
    # Prepare covariance matrices (0.01 * Fisher^-1)
    cov_matrices = [np.linalg.inv(0.01*F) for F in fishers_scaled]
    # cov_matrices = [np.linalg.inv(F) for F in fishers_scaled]
    init_cov_list = cov_matrices

    # Setup sampler
    config = parismc.SamplerConfig(gamma=n_iterations)

    sampler = parismc.Sampler(
        ndim=5,
        n_seed=len(lhs_points),
        log_density_func=log_density,
        init_cov_list=init_cov_list,
        prior_transform=prior_transform,
        config=config
    )

    # Compute log-densities for LHS points
    lhs_logden = log_density(physical_points)

    # Run sampling
    print(f'Running optimization for {n_iterations} iterations...')
    sampler.run_sampling(
        num_iterations=n_iterations,
        savepath='./lhs_iteration/',
        print_iter=10,
        external_lhs_points=lhs_points,
        external_lhs_log_densities=lhs_logden
    )

    # Extract best point (searched_log_densities_list is flat, searched_points_list[0] contains all points)
    best_idx = np.argmax(sampler.searched_log_densities_list)
    best_point = sampler.searched_points_list[0][best_idx]
    best_physical_point = prior_transform(np.array([best_point]))[0]
    best_log_density = np.max(sampler.searched_log_densities_list)

    return sampler, best_point, best_physical_point, best_log_density


# ============================================================================
# MAIN ITERATIVE PROCEDURE
# ============================================================================

def main():
    """Main iterative Fisher-guided LHS sampling procedure"""

    # Initial prior limits (30x Fisher box)
    initial_limits = [
        [5.995531126784557, 6.004468873215443],   # logm1
        [1.4746191268598186, 1.4796233825795062], # logm2
        [0.6919479173260448, 0.7080520826739551], # a
        [7.455582230927566, 7.544417769072434],   # p0
        [0.3980771809772245, 0.40192281902277555] # e0
    ]

    current_limits = initial_limits
    current_scale = fisher_box_scale

    # Storage for results
    all_results = []

    # Unit cube limits for LHS
    ndim = 5
    xlimits = np.column_stack([np.zeros(ndim, dtype=float), np.ones(ndim, dtype=float)])

    print(f"\n{'='*70}")
    print(f"Starting iterative Fisher-guided LHS sampling")
    print(f"Number of outer iterations: {n_outer_iterations}")
    print(f"Shrink factor: {shrink_factor} (final scale: {fisher_box_scale * shrink_factor**n_outer_iterations:.4f}x)")
    print(f"{'='*70}\n")

    for iter_num in range(n_outer_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iter_num + 1}/{n_outer_iterations}")
        print(f"Current Fisher box scale: {current_scale:.4f}x")
        print(f"Current limits:")
        print(f"  logm1: [{current_limits[0][0]:.10f}, {current_limits[0][1]:.10f}]")
        print(f"  logm2: [{current_limits[1][0]:.10f}, {current_limits[1][1]:.10f}]")
        print(f"  a:     [{current_limits[2][0]:.10f}, {current_limits[2][1]:.10f}]")
        print(f"  p0:    [{current_limits[3][0]:.10f}, {current_limits[3][1]:.10f}]")
        print(f"  e0:    [{current_limits[4][0]:.10f}, {current_limits[4][1]:.10f}]")
        print(f"{'='*70}\n")

        # Step 1: Generate LHS samples
        print(f"Step 1: Generating {n_lhs_samples} LHS samples...")
        sampling = LHS(xlimits=xlimits)
        lhs_points = np.clip(sampling(n_lhs_samples), 0.0, 1.0)

        # Transform to physical space
        prior_transform = PriorTransform(current_limits)
        physical_points = prior_transform(lhs_points)
        print(f"LHS samples generated in current prior region")

        # Step 2: Calculate Fisher matrices
        print(f"\nStep 2: Computing Fisher matrices for {n_lhs_samples} samples...")
        fishers_scaled = calculate_fishers(physical_points)

        # Save intermediate results
        with open(f'fisher_lhs_iter{iter_num+1}.pkl', 'wb') as f:
            pickle.dump((lhs_points, physical_points, fishers_scaled), f)

        # Step 3: Optimize around LHS points
        print(f"\nStep 3: Optimizing around LHS points...")
        sampler, best_point, best_physical_point, best_log_density = run_optimization_round(
            lhs_points, physical_points, fishers_scaled, prior_transform,
            n_iterations=n_optim_iterations
        )

        print(f"\nBest point found:")
        print(f"  Log-density: {float(best_log_density):.6e}")
        print(f"  Parameters (best):  {best_physical_point}")
        print(f"  Parameters (true):  {param_true}")
        print(f"  Difference:         {[best_physical_point[i] - param_true[i] for i in range(5)]}")
        print(f"  Distance to true:   {math.dist(param_true, best_physical_point):.6f}")

        # Additional diagnostics: check LHS log-densities
        lhs_logden_current = log_density(physical_points)
        print(f"\nLHS sample diagnostics:")
        print(f"  Best LHS log-density:    {float(np.max(lhs_logden_current)):.6e}")
        print(f"  Worst LHS log-density:   {float(np.min(lhs_logden_current)):.6e}")
        print(f"  Mean LHS log-density:    {float(np.mean(lhs_logden_current)):.6e}")
        print(f"  Improvement from LHS:    {float(best_log_density - np.max(lhs_logden_current)):.6e}")

        # Check distance of LHS samples to true params
        lhs_distances = [math.dist(param_true, physical_points[i]) for i in range(len(physical_points))]
        print(f"  Closest LHS to true:     {min(lhs_distances):.6f}")
        print(f"  Farthest LHS from true:  {max(lhs_distances):.6f}")

        # Store results
        iter_results = {
            'iter_num': iter_num + 1,
            'current_scale': current_scale,
            'lhs_points': lhs_points,
            'physical_points': physical_points,
            'fishers_scaled': fishers_scaled,
            'best_point': best_point,
            'best_physical_point': best_physical_point,
            'best_log_density': best_log_density,
            'sampler': sampler,
            'current_limits': current_limits
        }
        all_results.append(iter_results)

        # Save overall progress
        with open(f'iterative_results_iter{iter_num+1}.pkl', 'wb') as f:
            pickle.dump(all_results, f)

        # Step 4: Calculate Fisher at best point and update limits for next iteration
        if iter_num < n_outer_iterations - 1:  # Don't need to update after last iteration
            print(f"\nStep 4: Computing Fisher at best point and updating prior...")

            # Calculate Fisher at best point
            best_fishers = calculate_fishers(np.array([best_physical_point]))
            best_fisher = best_fishers[0]

            # Update scale for next iteration
            current_scale *= shrink_factor

            # Compute new limits
            new_limits = compute_new_limits(best_physical_point, best_fisher, current_scale)
            current_limits = new_limits

            print(f"New prior limits computed for next iteration")
            print(f"Next Fisher box scale: {current_scale:.4f}x")

            # Check if true parameters are still within new limits
            param_names_list = ['logm1', 'logm2', 'a', 'p0', 'e0']
            all_in_range = True
            for i in range(5):
                in_range = new_limits[i][0] <= param_true[i] <= new_limits[i][1]
                if not in_range:
                    print(f"  WARNING: True {param_names_list[i]}={param_true[i]:.6f} is OUTSIDE new range [{new_limits[i][0]:.6f}, {new_limits[i][1]:.6f}]")
                    all_in_range = False

            if all_in_range:
                print(f"  True parameters still within search region")
            else:
                print(f"  True parameters have LEFT the search region")

    print(f"\n{'='*70}")
    print(f"COMPLETED ALL {n_outer_iterations} ITERATIONS")
    print(f"{'='*70}\n")

    # Save final results
    with open('final_iterative_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print("Final results saved to 'final_iterative_results.pkl'")

    # Print summary
    print("\nSummary of iterations:")
    print(f"{'Iter':<6} {'Scale':<10} {'Best Log-Density':<20} {'Distance to True':<20}")
    print("-" * 70)
    for result in all_results:
        dist_to_true = math.dist(param_true, result['best_physical_point'])
        print(f"{result['iter_num']:<6} {result['current_scale']:<10.4f} "
              f"{float(result['best_log_density']):<20.6e} {dist_to_true:<20.6f}")

    return all_results


if __name__ == "__main__":
    results = main()
