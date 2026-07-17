"""
Quick test to check log-likelihood at true parameters
"""
import numpy as np
import sys
import os

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglikebasic

# Configuration
cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
force_backend = "cuda12x"
dt = 10
T = 0.25

# Setup waveform generators (same as main script)
inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
}

amplitude_kwargs = {"force_backend": force_backend}
Ylm_kwargs = {"force_backend": force_backend}
sum_kwargs_comb = {"force_backend": force_backend, "pad_output": True}
sum_kwargs_sep = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

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

# Setup parameters (same as main script)
m1 = 1e6
m2 = 3e1
a = 0.7
p0 = 7.5
e0 = 0.4
xI0 = 1.0
dist = 0.5
qS = 0.5
phiS = 1
qK = 1
phiK = phiS + np.pi/3
Phi_phi0 = 0.4
Phi_theta0 = 0.0
Phi_r0 = 0.5

params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
param_true = [np.log10(m1), np.log10(m2), a, p0, e0]

print("Creating GWfuncs and loglike objects...")
gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)
loglike_obj = loglikebasic.LogLike(params_star, waveform_gen_comb, gwf, M_init=5,
                                   verbose=True, waveform_gen_sep=waveform_gen_sep,
                                   noise_weighted=True)

print("\nCalculating SNR...")
data = loglike_obj.signal
data_snr = gwf.rhostat(data)
print(f"SNR: {data_snr}")

print("\n" + "="*70)
print("Testing log-likelihood at TRUE parameters")
print("="*70)

# Test 1: Call loglike_obj directly with true parameters
print("\nTest 1: Direct call to loglike_obj with true parameters")
params_array = np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0])
loglike_true = loglike_obj(params_array)
print(f"  Log-likelihood: {loglike_true}")

# Test 2: Use log_density function (like in the main script)
print("\nTest 2: Using log_density function")
def log_density(params):
    params = np.asarray(params)
    n_samples = params.shape[0]
    log_likes = np.zeros(n_samples)

    for i in range(n_samples):
        logm1, logm2, a_val, p0_val, e0_val = params[i]
        m1_val = 10**logm1
        m2_val = 10**logm2

        loglike = loglike_obj(np.array([m1_val, m2_val, a_val, p0_val, e0_val, xI0, dist,
                                       qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
        log_likes[i] = loglike

    return log_likes

loglike_via_log_density = log_density(np.array([param_true]))[0]
print(f"  Log-likelihood: {loglike_via_log_density}")

# Test 3: Slightly perturbed parameters
print("\nTest 3: Slightly perturbed parameters (offset by 0.001)")
param_perturbed = np.array(param_true) + 0.001
loglike_perturbed = log_density(np.array([param_perturbed]))[0]
print(f"  Log-likelihood: {loglike_perturbed}")
print(f"  Difference from true: {loglike_perturbed - loglike_true}")

print("\n" + "="*70)
print("Expected log-likelihood at true: ~82")
print(f"Actual log-likelihood at true:   {loglike_true}")
print("="*70)
