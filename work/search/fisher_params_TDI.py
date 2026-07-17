import os
import sys
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt

import few

from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import KerrEccEqFlux
from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.ylm import GetYlms
from few import get_file_manager
from few.utils.geodesic import get_fundamental_frequencies
from few.utils.constants import YRSID_SI
from few.waveform import (
    GenerateEMRIWaveform,
    FastSchwarzschildEccentricFlux,
    FastKerrEccentricEquatorialFlux,
)

# TDI response
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

# Change to the desired directory
os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')

import GWfuncs
import cupy as cp

import stableemrifisher
print(stableemrifisher.__file__)
from tqdm import tqdm
from stableemrifisher.fisher.fisher import StableEMRIFisher
from stableemrifisher.utils import inner_product

# tune few configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# ---------------------------------------------------------------------------
# GPU / observation configuration
# ---------------------------------------------------------------------------
use_gpu = True
dt = 5     # Time step
T = 1       # Total observation time (years)

ctx = {'T': T, 'dt': dt}

# ---------------------------------------------------------------------------
# Waveform generator setup
# ---------------------------------------------------------------------------
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
    "err": 1e-15,
}

amplitude_kwargs = {
    "force_backend": "cuda12x",
}

Ylm_kwargs = {
    "force_backend": "cuda12x",
}

sum_kwargs = {
    "force_backend": "cuda12x",
    "pad_output": True,
}

waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

# waveform generator setup (source frame; the response handles projection)
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False)

# ---------------------------------------------------------------------------
# Source parameters
# ---------------------------------------------------------------------------
m1 = 1e6
m2 = 1e1
a = 0.7
p0 = 9
e0 = 0.4
xI0 = 1.0
dist = 4.5
qS = np.pi
phiS = 0.
qK = 0.
phiK = 0.
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

param_dict = {
    'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'xI0': xI0,
    'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
    'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
}

param_names = ['m1', 'm2', 'a', 'p0', 'e0']

# ---------------------------------------------------------------------------
# StableEMRIFisher with TDI ResponseWrapper
# ---------------------------------------------------------------------------
# TDI channels and matched noise PSDs
channels = [A1TDISens, E1TDISens, T1TDISens]
noise_kwargs = [{"sens_fn": ch} for ch in channels]

sef = StableEMRIFisher(
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=dict(
        Tobs=ctx['T'],
        t0=10000.0,
        dt=ctx['dt'],
        index_lambda=8,
        index_beta=7,
        flip_hx=True,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        force_backend="cuda12x" if use_gpu else "cpu",
        order=20,
        tdi="1st generation",
        tdi_chan="AET",
    ),
    stats_for_nerds=True,
    use_gpu=use_gpu,
    deriv_type='stable',
    noise_model=get_sensitivity,
    noise_kwargs=noise_kwargs,
    channels=channels,
    T=ctx['T'],
    dt=ctx['dt'],
    stability_plot=False,
    der_order=6,
    Ndelta=12,
    plunge_check=True,
    return_derivatives=False,
)

# ---------------------------------------------------------------------------
# Fisher calculation
# ---------------------------------------------------------------------------
emri_kwargs = {"T": ctx['T'], "dt": ctx['dt']}

Fisher = sef(
    wave_params=param_dict,
    param_names=param_names,
    live_dangerously=False,
    stability_plot=True,
    der_order=8,
    Ndelta=16,
    return_derivatives=False,
)

print("Fisher shape:", Fisher.shape)

# CHAIN RULE FOR FISHER (log-scale m1, m2)
J_mx = np.eye(len(param_names))
J_mx[0, 0] = m1
J_mx[1, 1] = m2

Fisher_scaled = J_mx.T @ Fisher @ J_mx
print("Fisher:")
print(Fisher_scaled)
cov = np.linalg.inv(Fisher_scaled)
print("Covariance matrix:")
print(cov)

with open('cov_matrix_TDI_AET.pkl', 'wb') as f:
    pickle.dump(cov, f)

# Covariance ellipse plot
from stableemrifisher.plot import CovEllipsePlot
CovEllipsePlot(cov)
plt.savefig('cov_ellipse_TDI_AET.png', dpi=150, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Mahalanobis distance of recovered points from the true source
# ---------------------------------------------------------------------------
# The points below are in (log10 m1, log10 m2, a, p0, e0) coordinates, so we
# build the Fisher in that same basis: d(theta_i)/d(log10 x) = x * ln(10).
ln10 = np.log(10.0)
J_log10 = np.eye(len(param_names))
J_log10[0, 0] = m1 * ln10
J_log10[1, 1] = m2 * ln10
Fisher_log10 = J_log10.T @ Fisher @ J_log10   # inverse covariance in log10 basis

# True (fiducial) point in the same coordinates
theta_true = np.array([np.log10(m1), np.log10(m2), a, p0, e0])


def mahalanobis(point):
    """sqrt( dtheta^T Fisher dtheta ), dtheta = point - true, in log10 basis."""
    d = np.asarray(point, float).ravel() - theta_true
    return float(np.sqrt(d @ Fisher_log10 @ d))


points = {
    'fstat': np.array([6.10267641, 0.9230586, 0.4502283, 8.42052417, 0.41391971]),
    'X':     np.array([5.63862259, 1.1727771, 0.84499254, 8.8912361, 0.24532288]),
    's12':   np.array([5.9897139, 1.06187119, 0.84993711, 9.21099004, 0.39363801]),
}

print("\nMahalanobis distance from true source (log10 m1, log10 m2, a, p0, e0):")
print(f"  true point: {theta_true}")
for label, pt in points.items():
    print(f"  {label:6s} d_M = {mahalanobis(pt):.4f}  (point = {pt})")
