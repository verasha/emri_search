"""
Sweep over log10(m1) and compare the three intrinsic-parameter statistics:
  - ln L   : full log-likelihood (loglike, cell-by-cell waveform residual)
  - f      : detection statistic f_stat from loglikebasic.LogLike (shifted by -83
             to line up visually with ln L, as in the original notebook)
  - X      : phase-maximized cross-correlation statistic gwf.Xstat(signal, h_temp)

Adapted from work/sampling_test/check_f.ipynb (recovered from git history at
commit c3eea81^), with the X curve (previously commented out / never computed)
added in.
"""
import gc
import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

import few
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform

_work_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(_work_dir)
sys.path.insert(0, _work_dir)
sys.path.insert(0, os.path.join(_work_dir, 'old'))

import GWfuncs
import loglikebasic

# ---------------------------------------------------------------------------
# GPU / FEW configuration
# ---------------------------------------------------------------------------
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
gc.collect()

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
dt = 10     # Time step
T = 0.25    # Total time
force_backend = "cuda12x"

inspiral_kwargs = {
    "func": 'KerrEccEqFlux',
    "DENSE_STEPPING": 0,
    "include_minus_m": False,
    "err": 1e-15,
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
    use_gpu=use_gpu,
)
waveform_gen_sep = GenerateEMRIWaveform(
    FastKerrEccentricEquatorialFlux,
    frame='detector',
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs_sep,
    use_gpu=use_gpu,
)
gwf = GWfuncs.GravWaveAnalysis(T, dt)

# Source (injection) parameters
m1, m2, a, p0, e0 = 1e6, 3e1, 0.7, 7.5, 0.4
xI0 = 1.0
dist = 0.5  # Gpc
qS, phiS, qK = 0.5, 1, 1
phiK = phiS + np.pi / 3
Phi_phi0, Phi_theta0, Phi_r0 = 0.4, 0.0, 0.5

# ---------------------------------------------------------------------------
# Sweep range on log10(m1) -- hardcoded 100-sigma prior range from a previous
# Fisher covariance run, so we don't need to load the .pkl here
# ---------------------------------------------------------------------------
x = np.linspace(5.999998216302528, 6.000001783697472, 100)

# ---------------------------------------------------------------------------
# f statistic sweep (loglikebasic.LogLike)
# ---------------------------------------------------------------------------
params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
gc.collect()

loglike_obj = loglikebasic.LogLike(
    params_star, waveform_gen_comb, gwf, M_init=5, verbose=True,
    waveform_gen_sep=waveform_gen_sep, noise_weighted=True,
)


def fstat(params):
    logm1, logm2, a_, p0_, e0_ = params
    m1_ = 10 ** logm1
    m2_ = 10 ** logm2
    return loglike_obj(np.array([m1_, m2_, a_, p0_, e0_, xI0, dist, qS, phiS, qK, phiK,
                                  Phi_phi0, Phi_theta0, Phi_r0]))


f_vals = []
for val in x:
    param = [val, np.log10(m2), a, p0, e0]
    f_vals.append(fstat(param))

# ---------------------------------------------------------------------------
# Reference (injected) waveform, used for both ln L and X
# ---------------------------------------------------------------------------
waveform = waveform_gen_comb(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                              Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)


def loglike_and_X(params):
    """Full ln L and the (phase-maximized) X statistic for a template."""
    logm1, logm2, a_, p0_, e0_ = params
    m1_ = 10 ** logm1
    m2_ = 10 ** logm2
    phiK_ = phiS + np.pi / 3

    htemp = waveform_gen_comb(m1_, m2_, a_, p0_, e0_, xI0, dist, qS, phiS, qK, phiK_,
                               Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)

    res = waveform - htemp
    res_f = gwf.freq_wave(res)
    inner_res = gwf.inner(res_f, res_f)
    calc_loglike = -0.5 * inner_res

    calc_X = gwf.Xstat(waveform, htemp)

    return calc_loglike, calc_X


loglike_vals = []
X_vals = []
for val in x:
    param = [val, np.log10(m2), a, p0, e0]
    calc_loglike, calc_X = loglike_and_X(param)
    print(calc_loglike, calc_X)
    loglike_vals.append(calc_loglike)
    X_vals.append(calc_X)

loglike_vals_new = [ll.get() if hasattr(ll, 'get') else ll for ll in loglike_vals]
X_vals_new = [xx.get() if hasattr(xx, 'get') else xx for xx in X_vals]
X_vals_new = [np.real(xx) for xx in X_vals_new]

# All three curves are peak-normalized (shifted so their max is 0) so they're
# directly comparable on the same Delta-ln-L-like axis.
loglike_vals_new = [ll - max(loglike_vals_new) for ll in loglike_vals_new]
f_vals_new = [ff - max(f_vals) for ff in f_vals]
X_vals_new = [xx - max(X_vals_new) for xx in X_vals_new]

# ---------------------------------------------------------------------------
# Plot: ln L, f (shifted), X -- PRD (Physical Review D / REVTeX) style
# ---------------------------------------------------------------------------
have_latex = shutil.which('latex') is not None
plt.rcParams.update({
    "text.usetex": have_latex,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "figure.figsize": (3.4, 2.6),   # single-column PRD width, inches
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig, ax = plt.subplots()
ax.plot(x, loglike_vals_new, c='k', label=r'$\ln L$')
ax.plot(x, f_vals_new, c='r', label=r'$f$')
ax.plot(x, X_vals_new, c='grey', linestyle='dashed', label=r'$X$')

# Posterior bulk: 1D Gaussian 3sigma (99.7%) C.L., Delta chi^2_1dof = 9
# -> Delta ln L = -0.5 * 9 = -4.5
ax.axhline(-4.5, color='tab:blue', linestyle='--', lw=0.8, label='Bulk')

ax.set_ylim(-5, 0)
ax.set_xlabel(r'$\log_{10} m_1$')
ax.set_ylabel(r'func.\ val.' if have_latex else 'func. val.')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('check_f.pdf')
plt.show()
