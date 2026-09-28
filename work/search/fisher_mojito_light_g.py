import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

import few

from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

# TDI response
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

os.chdir('/home/svu/e1498138/emri_search/work/')
sys.path.insert(0, '/home/svu/e1498138/emri_search/work/')

import cupy as cp

import stableemrifisher
print(stableemrifisher.__file__)
from stableemrifisher.fisher.fisher import StableEMRIFisher

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

# ---------------------------------------------------------------------------
# GPU / observation configuration — matches GWfuncs_noise.build_waveform_response
# (T=1yr, dt=5s, TDI gen 1 -> AET channels) used throughout the paris4/5 search
# scripts, so the Fisher below is comparable to what those scripts are searching.
# ---------------------------------------------------------------------------
use_gpu = True
dt = 5
T = 12 / 12

ctx = {'T': T, 'dt': dt}

# ---------------------------------------------------------------------------
# Waveform generator setup
# ---------------------------------------------------------------------------
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
    "err": 1e-10,  # 1e-15 (fisher_params_TDI.py's value, tuned for a
                    # closer-to-plunge source) fails to converge for this
                    # source's trajectory within max_iter.
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

waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False)

# ---------------------------------------------------------------------------
# Source parameters — Mojito light EMRI_G (same as paris4/5_ext_dev.py)
# ---------------------------------------------------------------------------
m1 = 6.72e5
m2 = 9.84e1
a = 0.5
p0 = 15.7117
e0 = 0.7440
xI0 = 1.0
dist = 4.755
qS = 0.5906
phiS = 3.5808
qK = 1.1707
phiK = 3.8495
Phi_phi0 = 3.1940
Phi_theta0 = 3.3780
Phi_r0 = 0.6038

param_dict = {
    'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'xI0': xI0,
    'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
    'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
}

# The 7 dims searched by paris4/5_ext_dev.py.
param_names = ['m1', 'm2', 'a', 'p0', 'e0', 'qS', 'phiS']

# ---------------------------------------------------------------------------
# StableEMRIFisher with TDI ResponseWrapper — the noise PSD weighting (AET,
# 1st-generation TDI) is what makes this a noise-aware ("with noise") Fisher,
# i.e. inner products are weighted by get_sensitivity(channel) rather than a
# flat/no-noise metric.
# ---------------------------------------------------------------------------
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
# Fisher calculation, natural (m1, m2, a, p0, e0, qS, phiS) basis
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Chain rule into the search basis actually used by paris4/5_ext_dev.py:
# (log10 m1, log10 m2, a, p0, e0, cosqS, phiS). Fisher_new = J^T Fisher J,
# J = d(old)/d(new) evaluated at the fiducial point; J is diagonal here since
# each new coordinate is a bijective function of the SAME old coordinate.
# ---------------------------------------------------------------------------
ln10 = np.log(10.0)
J = np.eye(len(param_names))
J[0, 0] = m1 * ln10             # d(m1)/d(log10 m1)
J[1, 1] = m2 * ln10             # d(m2)/d(log10 m2)
# a, p0, e0 unchanged (identity)
J[5, 5] = -1.0 / np.sin(qS)     # d(qS)/d(cosqS), qS = arccos(cosqS)
# phiS unchanged (identity)

Fisher_scaled = J.T @ Fisher @ J
cov = np.linalg.inv(Fisher_scaled)

labels = [r'$\log_{10} m_1$', r'$\log_{10} m_2$', r'$a$', r'$p_0$', r'$e_0$',
          r'$\cos q_S$', r'$\phi_S$']

print("Fisher (log10 m1, log10 m2, a, p0, e0, cosqS, phiS basis):")
print(Fisher_scaled)
print("Covariance matrix:")
print(cov)
print("\n1-sigma marginal uncertainties:")
for name, sigma in zip(labels, np.sqrt(np.diag(cov))):
    print(f"  {name:16s} sigma = {sigma:.6g}")

with open('fisher_mojito_light_g.pkl', 'wb') as f:
    pickle.dump({
        'Fisher_natural': Fisher,
        'Fisher_scaled': Fisher_scaled,
        'cov': cov,
        'param_names_natural': param_names,
        'param_names_scaled': ['logm1', 'logm2', 'a', 'p0', 'e0', 'cosqS', 'phiS'],
        'theta_true_scaled': np.array([
            np.log10(m1), np.log10(m2), a, p0, e0, np.cos(qS), phiS,
        ]),
        'param_dict': param_dict,
        'T': T, 'dt': dt,
    }, f)
print("Saved to fisher_mojito_light_g.pkl")

# Covariance ellipse plot, in the search basis.
from stableemrifisher.plot import CovEllipsePlot
CovEllipsePlot(cov)
plt.savefig('cov_ellipse_mojito_light_g.png', dpi=150, bbox_inches='tight')
