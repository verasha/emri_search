"""
Compare Fisher-matrix conditioning at p0=8 vs p0=7.7 (both near-plunge, 9mth
observation) to check how localization changes as p0 approaches the
separatrix within a fixed, shorter observation window.

Same a, e0, dist for both points so the comparison isolates the effect of p0.
"""
import os
import sys
import pickle

import numpy as np

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

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')

import GWfuncs
import cupy as cp

import stableemrifisher
from stableemrifisher.fisher.fisher import StableEMRIFisher

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
dt = 5
T = 9 / 12

inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
    "err": 1e-15,
}
amplitude_kwargs = {"force_backend": "cuda12x"}
Ylm_kwargs = {"force_backend": "cuda12x"}
sum_kwargs = {"force_backend": "cuda12x", "pad_output": True}

waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False)

# Fixed across both cases so the comparison isolates p0
m1 = 1e6
m2 = 1e1
a = 0.7
e0 = 0.4
xI0 = 1.0
dist = 6.0
qS = np.pi
phiS = 0.
qK = 0.
phiK = 0.
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

param_names = ['m1', 'm2', 'a', 'p0', 'e0']

channels = [A1TDISens, E1TDISens, T1TDISens]
noise_kwargs = [{"sens_fn": ch} for ch in channels]

sef = StableEMRIFisher(
    waveform_class=waveform_class,
    waveform_class_kwargs=waveform_class_kwargs,
    waveform_generator=waveform_generator,
    waveform_generator_kwargs=waveform_generator_kwargs,
    ResponseWrapper=ResponseWrapper,
    ResponseWrapper_kwargs=dict(
        Tobs=T,
        t0=10000.0,
        dt=dt,
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
    T=T,
    dt=dt,
    stability_plot=False,
    der_order=6,
    Ndelta=12,
    plunge_check=True,
    return_derivatives=False,
)

ln10 = np.log(10.0)


def fisher_at(p0):
    param_dict = {
        'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'xI0': xI0,
        'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
        'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
    }
    Fisher = sef(
        wave_params=param_dict,
        param_names=param_names,
        live_dangerously=False,
        stability_plot=False,
        der_order=8,
        Ndelta=16,
        return_derivatives=False,
    )
    # log10(m1), log10(m2) chain rule
    J = np.eye(len(param_names))
    J[0, 0] = m1 * ln10
    J[1, 1] = m2 * ln10
    Fisher_log10 = J.T @ Fisher @ J
    cov = np.linalg.inv(Fisher_log10)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    return Fisher_log10, cov, sigma, corr


results = {}
for p0 in [8.0, 7.7]:
    print(f"\n=== p0={p0} ===")
    Fisher_log10, cov, sigma, corr = fisher_at(p0)
    idx_a, idx_p0 = param_names.index('a'), param_names.index('p0')
    print(f"sigma (log10m1, log10m2, a, p0, e0): {sigma}")
    print(f"corr(a, p0) = {corr[idx_a, idx_p0]:.4f}")
    print("Full correlation matrix:")
    print(corr)
    results[p0] = dict(Fisher_log10=Fisher_log10, cov=cov, sigma=sigma, corr=corr)

print("\n=== Comparison ===")
for p0 in [8.0, 7.7]:
    idx_a, idx_p0 = param_names.index('a'), param_names.index('p0')
    r = results[p0]
    print(f"p0={p0}:  sigma_a={r['sigma'][idx_a]:.4f}  sigma_p0={r['sigma'][idx_p0]:.4f}  "
          f"corr(a,p0)={r['corr'][idx_a, idx_p0]:.4f}")

with open('fisher_compare_p0_8_7.7_9mth.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nSaved to fisher_compare_p0_8_7.7_9mth.pkl")
