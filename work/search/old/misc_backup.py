import os
import json
import signal
import time
import argparse
from typing import Tuple, Optional,Dict, Any
from scipy.signal.windows import tukey
import numpy as np

# FEW / waveform & noise
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import generate_PSD, inner_product

from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from few.waveform import GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
import matplotlib.pyplot as plt

try:
    import cupy as cp
    xp=cp
except:
    xp=np
    print("CuPy not found, using NumPy instead.")

__all__ = [
    # Parameter utilities
    "_clip_physical_params_intrinsic",
    "load_startingpoint_param_array",
    # Signal processing
    "compute_fft_with_windowing",
    "add_noise_func",
    # Analysis & metrics
    "calculate_detection_snr_0pa_vs_1pa",
    "calculate_detection_overlap_0pa_vs_1pa",
    "calculate_time_max_0pa_vs_1pa",
    "chi2_match",
    "inner_prod",
    # Fisher
    "compute_fisher_parallelotope",
    "covariance_from_fisher_parallelotope",
    # Plotting & validation
    "plot_time_series_from_fft",
    "check_noise_model_consistency",
]


def _is_pos_def(mat: np.ndarray) -> bool:
    """Return True if matrix is positive-definite via Cholesky test."""
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False

def check_and_clip_prior(priors_range, param_names):
    """Check that prior ranges are consistent with physical bounds and reference values.
    Make sure the param_names are decorated with the same pattern as in the reference dict (e.g., 'm1', 'm2', 'a', etc.) for proper checking."""

    # --- Physical bounds ---
    EMRI_param_ranges = {
        "m1": (0, None),
        "m2": (0, None),
        "a": (-0.999, 0.999),
        "p0": (0, None),
        "e0": (0, 1),
        "xI0": (-1, 1),
        "dist": (0, None),
        "qS": (-np.pi, np.pi),
        "phiS": (-np.pi, np.pi),
        "qK": (-np.pi, np.pi),
        "phiK": (-np.pi, np.pi),
        "Phi_phi0": (-np.pi, np.pi),
        "Phi_theta0": (-np.pi, np.pi),
        "Phi_r0": (-np.pi, np.pi),
    }

    # --- Angular width limits ---
    angular_width_limits = {
        "phiS": 2*np.pi,
        "phiK": 2*np.pi,
        "Phi_phi0": 2*np.pi,
        "Phi_theta0": 2*np.pi,
        "Phi_r0": 2*np.pi,
        "qS": 2*np.pi,
        "qK": 2*np.pi,
    }

    checked_ranges = []

    for i, base_name in enumerate(param_names):
        low, high = priors_range[i]

        # Enforcing physical bounds without circular bound
        if base_name in EMRI_param_ranges:
            ref_low, ref_high = EMRI_param_ranges[base_name]

            if base_name not in angular_width_limits:
                if ref_low is not None:
                    low = max(low, ref_low)
                if ref_high is not None:
                    high = min(high, ref_high)

        # Angular Widths
        if base_name in angular_width_limits:
            max_width = angular_width_limits[base_name]
            width = high - low

            if width > max_width:
                center = 0.5 * (low + high)
                low = center - 0.5 * max_width
                high = center + 0.5 * max_width

        checked_ranges.append([low, high])
    return checked_ranges


def _clip_physical_params_intrinsic(theta: np.ndarray) -> np.ndarray:
    """Clip mapped physical parameters to minimal physical ranges.

    Indices: 0:m1, 1:m2, 2:a, 3:p0, 4:e0, [5:chi2]
    - m1, m2 > 0
    - a in [-0.9999, 0.9999]
    - e0 in (0, 1)
    - chi2 in [-0.99, 0.99] (only when pa_template == '1PA')
    """
    x = np.asarray(theta, dtype=float).copy()
    if x.ndim == 1:
        if x.shape[0] >= 1:
            x[0] = max(x[0], 1e-30)
        if x.shape[0] >= 2:
            x[1] = max(x[1], 1e-30)
        if x.shape[0] >= 3:
            x[2] = np.clip(x[2], -0.9999, 0.9999)
        if x.shape[0] >= 5:
            x[4] = np.clip(x[4], 1e-8, 1 - 1e-8)
   
        return x
    else:
        x[:, 0] = np.maximum(x[:, 0], 1e-30)
        x[:, 1] = np.maximum(x[:, 1], 1e-30)
        if x.shape[1] >= 3:
            x[:, 2] = np.clip(x[:, 2], -0.9999, 0.9999)
        if x.shape[1] >= 5:
            x[:, 4] = np.clip(x[:, 4], 1e-8, 1 - 1e-8)
        return x
    
def compute_fisher_parallelotope(ctx: dict,
                                 fisher_params: list,
                                 params_to_infer: list,
                                 additional_kwargs: dict,
                                 build_waveform_response = None,
                                 use_gpu: bool = True,
                                 _TARGET_SNR: float = None,
                                 prior_sigma_range: float = 20,
                                 using_evec: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build Fisher-based local prior around ``theta0``.

    When ``using_evec`` is True we recover the original Fisher ellipsoid (axes
    given by covariance eigenvectors). When False we form a simple axis-aligned
    box whose half-widths are ``prior_sigma_range * sqrt(diag(F^{-1}))``.

    Returns ``(Q, b, meta)`` where ``Q`` is the basis matrix (orthonormal when
    ``using_evec`` is True, identity otherwise), ``b`` holds the half-lengths,
    and ``meta`` carries diagnostics including ``using_evec``.
    """
    # Prepare waveform generator
    dim = len(params_to_infer)
    if 'waveform_response' in ctx and ctx['waveform_response'] is not None:
        waveform_response = ctx['waveform_response']
    else:
        waveform_response = build_waveform_response(T=ctx['T'], dt=ctx['dt'], use_gpu=use_gpu)

    channels = [A1TDISens, E1TDISens, T1TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    param_names = params_to_infer
    fisher_params = fisher_params

    sef = StableEMRIFisher(waveform_class=SuperKludgeWaveform,
                       waveform_class_kwargs = dict(sum_kwargs=dict(pad_output=False, odd_len=True)),
                       waveform_generator = GenerateEMRIWaveform,
                       waveform_generator_kwargs= dict(return_list=False),
                       ResponseWrapper=ResponseWrapper,
                       ResponseWrapper_kwargs = dict(Tobs=ctx['T'],
                                                    t0=10000.0,
                                                    dt=ctx['dt'],
                                                    index_lambda=8,
                                                    index_beta=7,
                                                    flip_hx=True,
                                                    is_ecliptic_latitude=False,
                                                    remove_garbage="zero",
                                                    orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
                                                    force_backend = "cuda12x" if use_gpu else "cpu",
                                                    order=20,
                                                    tdi="2nd generation",
                                                    tdi_chan="AET"),
                       stats_for_nerds = True, use_gpu = use_gpu,
                       deriv_type='stable',
                       noise_model=get_sensitivity,
                       noise_kwargs=noise_kwargs,
                       channels=channels,
                       T = ctx['T'], dt = ctx['dt'],
                       stability_plot = False,
                       der_order = 6, Ndelta = 12,
                       plunge_check=True, return_derivatives=False
                       )
    emri_kwargs = {"T":ctx['T'], "dt":ctx['dt']}
 
    pars_list_com = list(fisher_params) + [ctx['chi2'],additional_kwargs['evolve_1PA'],additional_kwargs['evolve_primary'],
     additional_kwargs['evolve_2PA'],additional_kwargs['deviation_included'],additional_kwargs['dev_1'],additional_kwargs['dev_2']]
    SNR = sef.SNRcalc_SEF(*pars_list_com,**emri_kwargs,use_gpu=use_gpu)
    print("SNR: ", SNR)
    param_dict = {
    'm1': fisher_params[0],
    'm2': fisher_params[1],
    'a': fisher_params[2],
    'p0': fisher_params[3],
    'e0': fisher_params[4],
    'xI0': fisher_params[5],
    'dist': fisher_params[6],
    'qS': fisher_params[7],
    'phiS': fisher_params[8],
    'qK': fisher_params[9],
    'phiK': fisher_params[10],
    'Phi_phi0': fisher_params[11],
    'Phi_theta0': fisher_params[12],
    'Phi_r0': fisher_params[13]}

    Fisher = sef(wave_params = param_dict,param_names=param_names, add_param_args=additional_kwargs,
            live_dangerously = False, stability_plot = True,der_order = 6, Ndelta = 12,
            )
                   
    try:
        F = np.asarray(Fisher, dtype=float)
        print(f"[FISHER] {repr(F)}")
        print(f"[FISHER_IS_PD] {_is_pos_def(F)}")
        F_inv = np.linalg.inv(F)
        print(f"[FISHER_INV] {repr(F_inv)}")
        print(f"[FISHER_INV_IS_PD] {_is_pos_def(F_inv)}")
        F_std = np.sqrt(np.diag(F_inv))
        print(f"[FISHER_STD] {repr(F_std)}")
    except Exception as e:
        raise RuntimeError(f"Fisher computation failed: {e}")

    emri_flags = {"T": ctx['T'], "dt": ctx['dt'], '1PA': False, 'evolve_primary': False, '2PA': False}

    waveform_tmpl = xp.array(sef.waveform)
    print("shape of waveform_tmpl: ", waveform_tmpl.shape)
    
    PSD_funcs = generate_PSD(waveform=waveform_tmpl, dt=float(ctx['dt']), noise_PSD=get_sensitivity,
                             channels=channels, noise_kwargs=noise_kwargs, use_gpu=bool(use_gpu))
    snr_model = float(np.sqrt(inner_product(waveform_tmpl, waveform_tmpl, PSD_funcs, float(ctx['dt']), use_gpu=bool(use_gpu))))
    print(f"[MODEL] SNR in fisher calculation: {snr_model:.6f}")

    # Scale Fisher to target SNR
    print("Target SNR for scaling: ", _TARGET_SNR)
    scale = (_TARGET_SNR / max(snr_model, 1e-30)) ** 2
    print('F: scale',scale)

    if not using_evec:
        sigma_diag = F_std
        print(f"[DIAG_STD] {repr(sigma_diag)}")
        eigvals = sigma_diag ** 2
        print(f"[DIAG_EIGVALS] {repr(eigvals)}")
        b = prior_sigma_range * sigma_diag
        print(f"[DIAG_B] {repr(b)}")
        meta = {
            'diag_sigma': sigma_diag.tolist(),
            'eigvals': eigvals.tolist(),
            'snr_model': float(snr_model),
            'scale_applied': float(scale),
            'using_evec': False,
        }
        return np.eye(dim), b, meta

    F_scaled = F * scale

    # Regularize and invert to covariance
    dim = F_scaled.shape[0]
    reg = 1e-20 * np.trace(F_scaled) / max(dim, 1)       
    F_scaled = F_scaled + reg * np.eye(dim)     
    print(f"[F_SCALED] {repr(F_scaled)}")
    print(f"[F_SCALED_IS_PD] {_is_pos_def(F_scaled)}")
    try:
        cov = np.linalg.inv(F_scaled)
    except np.linalg.LinAlgError:
        print("[WARN] F_scaled inversion failed; using pseudoinverse")
        cov = np.linalg.pinv(F_scaled)
    print(f"[COV_RAW] {repr(cov)}")
    print(f"[COV_RAW_IS_PD] {_is_pos_def(cov)}")
    cov = 0.5 * (cov + cov.T)
    print(f"[COV_SYM] {repr(cov)}")
    print(f"[COV_SYM_IS_PD] {_is_pos_def(cov)}")
    cov_diag_std = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    print(f"[COV_STD] {repr(cov_diag_std)}")
    evals_raw, evecs = np.linalg.eigh(cov)
    print(f"[COV_EVALS_RAW] {repr(evals_raw)}")
    print(f"[COV_EVECS] {repr(evecs)}")
    eigvals_eigvalsh = np.linalg.eigvalsh(cov)
    print(f"[COV_EIGVALSH] {repr(eigvals_eigvalsh)}")
    evals = evals_raw
    print(f"[COV_EVALS] {repr(evals)}")
    sigma_diag = cov_diag_std
    print(f"[COV_SIGMA_DIAG] {repr(sigma_diag)}")
    b = prior_sigma_range * np.sqrt(evals)
    print(f"[FISHER_EVALS] {repr(evals)}")
    print(f"[FISHER_B] {repr(b)}")

    meta = {
        'diag_sigma': sigma_diag.tolist(),
        'eigvals': evals.tolist(),
        'snr_model': float(snr_model),
        'scale_applied': float(scale),
        'using_evec': True,
        #'reg_added': float(reg),
    }
    return evecs, b, meta

def covariance_from_fisher_parallelotope(Q: np.ndarray, b: np.ndarray, prior_sigma_range: float = 20) -> np.ndarray:
    """Construct covariance matrix from Fisher-parallelotope outputs (Q, b).

    Given that b = 10 * sqrt(eigvals), we recover eigvals = (b/10)^2 and return Q diag(eigvals) Q^T.
    """
    b = np.asarray(b, dtype=float)
    eigvals = (b / prior_sigma_range) ** 2
    cov = Q @ (np.diag(eigvals)) @ Q.T
    cov = 0.5 * (cov + cov.T)
    return cov


def compute_fft_with_windowing(waveform, dt, N,type=None,use_gpu=False,n_channels=3):
    if use_gpu:
        try:
            xp = cp
        except ImportError:
            print("[WARN] CuPy not available, falling back to NumPy")
            xp = np
    else:
        xp = np
    if type == 'tukey':
        window = xp.asarray(tukey(N, 0.01))
        waveform_windowed = waveform * window
        waveform_f = xp.asarray([xp.fft.rfft(waveform_windowed[i]) * dt for i in range(n_channels)])[:,1:]
    else:
        waveform_f = xp.asarray([xp.fft.rfft(waveform[i]) * dt for i in range(n_channels)])[:,1:]
    # window = xp.asarray(tukey(N, 0.01))
    # waveform_windowed = waveform * window
    # waveform_f = xp.asarray([xp.fft.rfft(waveform_windowed[i]) * dt for i in range(n_channels)])[:,1:]
    return waveform_f


def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    """
    Compute noise-weighted inner product using BBHx's standard normalization.

    Uses: ⟨a|b⟩ = 4·Δf·Re[Σ a(f)·b*(f) / Sn(f)]

    Parameters
    ----------
    signal_1_f : array-like
        First signal in frequency domain
    signal_2_f : array-like
        Second signal in frequency domain
    PSD : array-like
        Power spectral density Sn(f)
    delta_f : float
        Frequency spacing (Hz)
    xp : module, optional
        Array module (numpy or cupy). Default: numpy

    Returns
    -------
    float
        Inner product value ⟨signal_1|signal_2⟩
    """
    return 4 * delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj() / PSD))

def inner_prod_without_phase(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    return 4 * delta_f * xp.abs(xp.sum(signal_1_f * signal_2_f.conj() / PSD))


#use detection SNR and also use max phase if phase_max is True
def calculate_detection_overlap_0pa_vs_1pa(m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs,
                    maximize_phase=False,
                    **fixed):
    xp = cp if fixed['use_gpu'] else np

    print(" calculation withoiut noise additoin, using waveform_true_fft_without_noise for signal")

    signal = fixed['waveform_true_fft_without_noise']
    waveform_response = fixed['waveform_response']
    wave_params = [m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs['chi2'],add_kwargs['evolve_1PA'],add_kwargs['evolve_primary'],
                    add_kwargs['evolve_2PA'], add_kwargs['deviation_included'],add_kwargs['dev_1'],add_kwargs['dev_2']]
    
    emri_kwargs =  {"T": fixed['T'], "dt": fixed['dt'],'chi2': add_kwargs['chi2'],'evolve_1PA': add_kwargs['evolve_1PA'],
                    'evolve_primary': add_kwargs['evolve_primary'],'evolve_2PA': add_kwargs['evolve_2PA'],'deviation_included': add_kwargs['deviation_included'],
               'dev_1': add_kwargs['dev_1'], 'dev_2': add_kwargs['dev_2']}
    
    h = xp.array(waveform_response(*wave_params, **emri_kwargs))
    PSD = fixed['PSD']
    h_f = compute_fft_with_windowing(h, fixed['dt'], fixed['N_fiducial'], use_gpu=fixed['use_gpu'], n_channels=3)
    optimal_snr = inner_prod(h_f, h_f, PSD, fixed['delta_f'], xp=cp)
    optimal_snr_x = inner_prod(signal, signal, PSD, fixed['delta_f'], xp=cp)
    denom = xp.sqrt(optimal_snr * optimal_snr_x)

    if (maximize_phase):
        num = inner_prod_without_phase(signal, h_f, PSD, fixed['delta_f'], xp=cp)
    else:  
        num = inner_prod(signal, h_f, PSD, fixed['delta_f'], xp=cp)
    
    snr = num / denom

    if(xp.isnan(snr) or xp.isinf(snr)):
        print(f"[WARN] overlap computation returned {snr}; setting to 0")
        return -np.inf
    # print(snr)
    return float(snr) 

def calculate_detection_snr_0pa_vs_1pa(m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs,
                    maximize_phase=False,
                    **fixed):
    xp = cp if fixed['use_gpu'] else np
    signal = fixed['waveform_true_fft']
    waveform_response = fixed['waveform_response']
    wave_params = [m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs['chi2'],add_kwargs['evolve_1PA'],add_kwargs['evolve_primary'],
                    add_kwargs['evolve_2PA'], add_kwargs['deviation_included'],add_kwargs['dev_1'],add_kwargs['dev_2']]
    emri_kwargs =  {"T": fixed['T'], "dt": fixed['dt'],'chi2': add_kwargs['chi2'],'evolve_1PA': add_kwargs['evolve_1PA'],
                    'evolve_primary': add_kwargs['evolve_primary'],'evolve_2PA': add_kwargs['evolve_2PA'],'deviation_included': add_kwargs['deviation_included'],
               'dev_1': add_kwargs['dev_1'], 'dev_2': add_kwargs['dev_2']}
    h = xp.array(waveform_response(*wave_params, **emri_kwargs))
    PSD = fixed['PSD']
    h_f = compute_fft_with_windowing(h, fixed['dt'], fixed['N_fiducial'], use_gpu=fixed['use_gpu'], n_channels=3)
    optimal_snr = inner_prod(h_f, h_f, PSD, fixed['delta_f'], xp=cp)
    denom = xp.sqrt(optimal_snr)


    if (maximize_phase):
        num = inner_prod_without_phase(signal, h_f, PSD, fixed['delta_f'], xp=cp)
    else:  
        num = inner_prod(signal, h_f, PSD, fixed['delta_f'], xp=cp)
    
    snr = num / denom

    if(xp.isnan(snr) or xp.isinf(snr)):
        print(f"[WARN] SNR computation returned {snr}; setting to 0")
        return -np.inf
    return float(snr) * 10

def timemax_correlation(h1, h2,dt, PSD, xp=np):

    # FFT with dt scaling
    H1 = xp.array([xp.fft.rfft(h1[k]) * dt for k in range(2)])
    H2 = xp.array([xp.fft.rfft(h2[k]) * dt for k in range(2)])
    # print("H1 shape: ", H1.shape
    #       ,"H2 shape: ", H2.shape)
    # print("PSD shape: ", PSD.shape)

    Y = xp.zeros_like(H1)
    for i in range(2):
        Y[i,1:] = H1[i,1:] * xp.conj(H2[i,1:]) / (0.5 * PSD[i])  # Avoid DC component
    # IFFT to time domain with proper normalization
    S =xp.array([xp.fft.irfft(Y[i]) / dt for i in range(2)])
    # Return maximum correlation
    return  xp.max(xp.abs(S))


def calculate_time_max_0pa_vs_1pa(m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs,
                    **fixed):
    xp = cp if fixed['use_gpu'] else np
    signal = fixed['waveform_true_fft']
    waveform_response = fixed['waveform_response']
    wave_params = [m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs['chi2'],add_kwargs['evolve_1PA'],add_kwargs['evolve_primary'],
                    add_kwargs['evolve_2PA'], add_kwargs['deviation_included'],add_kwargs['dev_1'],add_kwargs['dev_2']]
    emri_kwargs =  {"T": fixed['T'], "dt": fixed['dt'],'chi2': add_kwargs['chi2'],'evolve_1PA': add_kwargs['evolve_1PA'],
                    'evolve_primary': add_kwargs['evolve_primary'],'evolve_2PA': add_kwargs['evolve_2PA'],'deviation_included': add_kwargs['deviation_included'],
               'dev_1': add_kwargs['dev_1'], 'dev_2': add_kwargs['dev_2']}
    h = xp.array(waveform_response(*wave_params, **emri_kwargs))
    PSD = fixed['PSD']
    h_f = compute_fft_with_windowing(h, fixed['dt'], fixed['N_fiducial'], use_gpu=fixed['use_gpu'], n_channels=3)
    Y = xp.zeros_like(h_f)
    for i in range(2):
        Y[i,1:] = h_f[i,1:] * xp.conj(signal[i,1:]) / (0.5 * PSD[i,1:])  # Avoid DC component
    # IFFT to time domain with proper normalization
    S =xp.array([xp.fft.irfft(Y[i]) / fixed['dt'] for i in range(2)])
    # Return maximum correlation
    if (xp.max(xp.abs(S)) is xp.nan or xp.max(xp.abs(S)) is xp.inf):
        print(f"[WARN] Time-max correlation computation returned {xp.max(xp.abs(S))}; setting to 0")
        return -np.inf
    return  xp.max(xp.abs(S)) 

def load_startingpoint_param_array(
    filename: Optional[str] = None,
    allow_missing: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Load the starting-point parameters from a .npy file as a dictionary.

    - If `filename` is provided, load from that path if it exists.
    - Otherwise try default repo/CWD locations.
    - If not found and `allow_missing` is True, return None; else raise.
    """
    def _load_npy(path: str) -> Dict[str, Any]:
        return np.load(path, allow_pickle=True).item()

    # Explicit path override
    if filename is not None:
        if os.path.exists(filename):
            return _load_npy(filename)
        if allow_missing:
            return None
        raise FileNotFoundError(f"Starting-point .npy not found at: {filename}")

    # Default locations
    default_paths = [
        "params.npy",  # repo root
        os.path.join(os.getcwd(), "params.npy")  # CWD fallback
    ]

    for path in default_paths:
        if os.path.exists(path):
            return _load_npy(path)

    if allow_missing:
        return None

    raise FileNotFoundError("Starting-point .npy file not found.")

def add_noise_func(signal,PSD,delta_f,delta_t,n_channels,seed=42):

    covariance_noise = [PSD[k] / (2*delta_f) for k in range(n_channels)]
    noise_f = xp.array(generate_colored_noise(covariance_noise, delta_t, seed=seed, return_time_domain=False))
    data_f = signal + noise_f
    return data_f, noise_f


def generate_colored_noise(variance_noise_AET, delta_t, seed=0, return_time_domain=False):
    """
    Generate colored Gaussian noise with specified variance per frequency bin.

    Parameters
    ----------
    variance_noise_AET : list or array
        Variance per frequency bin for each channel
        Shape: (N_channels, N_freq) or list of arrays
    delta_t : float
        Time sampling interval (seconds)
    seed : int, optional
        Random seed. Default: 0
    window_function : array-like, optional
        Time-domain window to apply. Default: None (no window)
    return_time_domain : bool, optional
        If True, return noise in time domain. Default: False (frequency domain)

    Returns
    -------
    array or list
        Colored noise in frequency or time domain
        Shape: (N_channels, N_freq) or (N_channels, N_time)

    Notes
    -----
    Generates Gaussian noise with specified variance and applies optional windowing.
    """
    
    # Detect array module
    if hasattr(variance_noise_AET[0], '__cuda_array_interface__'):
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            xp = np
    else:
        xp = np

    # Set random seed
    if xp is np:
        np.random.seed(seed)
    else:
        xp.random.seed(seed)

    N_channels = len(variance_noise_AET)
    noise_freq = []

    for k in range(N_channels):
        # Generate white noise in frequency domain
        variance = variance_noise_AET[k]
        N_freq = len(variance)
        
        # Complex Gaussian noise
        noise_real = xp.random.normal(0, xp.sqrt(variance / 2), N_freq)
        noise_imag = xp.random.normal(0, xp.sqrt(variance / 2), N_freq)
        noise_f = noise_real + 1j * noise_imag
        
        noise_freq.append(noise_f)

    if return_time_domain:
        # Just transform to time domain
        return xp.asarray([xp.fft.irfft(noise_freq[k] / delta_t) for k in range(N_channels)])
    else:
        return xp.asarray(noise_freq)

def check_noise_model_consistency(PSD, delta_f, delta_t, n_channels, temp_signal=None, seed=42):

    use_gpu = (xp is cp)

    # Stack PSD list → 2-D array (n_channels, N_freq), drop DC bin to match
    # inner_product's internal [:, 1:] slice on the rfft output
    PSD_array = xp.stack([xp.asarray(PSD[k]) for k in range(n_channels)], axis=0)
    covariance_noise = [PSD_array[k] / (2 * delta_f) for k in range(n_channels)]
    rng = xp.random.default_rng(seed)
    seed_array = rng.integers(0, int(1e6), size=500)
    noise_realizations = []

    for s in seed_array:
        noise_realizations.append(xp.array(
            generate_colored_noise(covariance_noise, delta_t, seed=int(s), return_time_domain=True)
        ))

    noise_realizations = xp.array(noise_realizations)   # (N_real, n_channels, N_time)
    N = noise_realizations.shape[-1]

    # inner_product does rfft then [:, 1:], so it works on N//2 bins.
    # Align PSD to exactly those N//2 bins by dropping the DC bin.
    PSD_for_ip = PSD_array[:, 1 : N // 2 + 1]          # shape (n_channels, N//2)
    print(f"Length of PSD: {N // 2 + 1}")

    mean_noise = xp.mean(noise_realizations, axis=0)
    _to_np = lambda arr: arr.get() if use_gpu else np.asarray(arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(_to_np(mean_noise.flatten()), bins=50, alpha=0.7, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--", label="zero")
    axes[0].set_title("Mean of Noise Realizations\n(should be centred on 0)")
    axes[0].set_xlabel("Mean Value")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    T_sig = N
    if temp_signal is None:
        t = xp.linspace(0, N * delta_t, N)
        temp_signal = xp.array([xp.sin(2 * xp.pi * t) for _ in range(n_channels)])
    else:
        T_sig = temp_signal.shape[-1]
        if T_sig > N:
            temp_signal = temp_signal[:, :N]
        elif T_sig < N:
            pad = xp.zeros((n_channels, N - T_sig), dtype=temp_signal.dtype)
            temp_signal = xp.concatenate([temp_signal, pad], axis=-1)

    inner_products = []
    for noise in noise_realizations:
        ip = inner_product(noise, temp_signal, PSD_for_ip, delta_t, use_gpu=use_gpu)
        inner_products.append(float(_to_np(xp.asarray(ip))))

    inner_products     = xp.array(inner_products)
    var_inner_product  = float(_to_np(xp.var(inner_products)))
    mean_inner_product = float(_to_np(xp.mean(inner_products)))
    aa = float(_to_np(xp.asarray(
        inner_product(temp_signal, temp_signal, PSD_for_ip, delta_t, use_gpu=use_gpu)
    )))
    ratio = var_inner_product / aa if aa != 0 else float("nan")

    ip_np = _to_np(inner_products)
    axes[1].hist(ip_np, bins=50, alpha=0.7, color="darkorange", density=True,
                 label="empirical ⟨n, a⟩")
    x = np.linspace(ip_np.min(), ip_np.max(), 300)
    expected_std = np.sqrt(aa)
    gaussian = np.exp(-0.5 * (x / expected_std) ** 2) / (expected_std * np.sqrt(2 * np.pi))
    axes[1].plot(x, gaussian, "r-", lw=2, label=f"N(0, ⟨a,a⟩)  σ={expected_std:.3g}")
    axes[1].axvline(mean_inner_product, color="blue", linestyle="--",
                    label=f"empirical mean={mean_inner_product:.3g}")
    axes[1].set_title(
        f"Inner Product ⟨n, a⟩ Distribution\n"
        f"sqrt(Var[⟨n,a⟩]={np.sqrt(var_inner_product):.4g}  ⟨a,a⟩={aa:.4g}  ratio={ratio:.4f}"
    )
    axes[1].set_xlabel("⟨n, a⟩")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    print("=" * 55)
    print("  Noise model consistency check")
    print("=" * 55)
    print(f"  Backend                : {'CuPy (GPU)' if use_gpu else 'NumPy (CPU)'}")
    print(f"  Realizations           : {len(seed_array)}")
    print(f"  N time samples         : {N}  |  PSD bins passed: {PSD_for_ip.shape[-1]}")
    print(f"  N (template samples)   : {T_sig}  →  aligned to {N}")
    print(f"  Max |mean noise|       : {float(_to_np(xp.abs(mean_noise).max())):.4e}  (expect ≈ 0)")
    print(f"  E[⟨n, a⟩]             : {mean_inner_product:.4e}           (expect ≈ 0)")
    print(f"  Var[⟨n, a⟩]           : {var_inner_product:.6g}")
    print(f"  ⟨a, a⟩                : {aa:.6g}")
    print(f"  Ratio Var/⟨a,a⟩       : {ratio:.6f}          (expect ≈ 1.0)")
    print("=" * 55)

    return {
        "mean_noise":         mean_noise,
        "inner_products":     inner_products,
        "var_inner_product":  var_inner_product,
        "aa":                 aa,
        "ratio":              ratio,
    }

def chi2_match(m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs,
                    maximize_phase=False,
                    **fixed):
    xp = cp if fixed['use_gpu'] else np
    signal = fixed['waveform_true_fft']
    waveform_response = fixed['waveform_response']
    wave_params = [m1, m2, a, p0, e0, Y0, dist, qS,phiS, qK, phiK, 
                    Phi_phi0, Phi_theta0, Phi_r0,add_kwargs['chi2'],add_kwargs['evolve_1PA'],add_kwargs['evolve_primary'],
                    add_kwargs['evolve_2PA'], add_kwargs['deviation_included'],add_kwargs['dev_1'],add_kwargs['dev_2']]
    emri_kwargs =  {"T": fixed['T'], "dt": fixed['dt'],'chi2': add_kwargs['chi2'],'evolve_1PA': add_kwargs['evolve_1PA'],
                    'evolve_primary': add_kwargs['evolve_primary'],'evolve_2PA': add_kwargs['evolve_2PA'],'deviation_included': add_kwargs['deviation_included'],
               'dev_1': add_kwargs['dev_1'], 'dev_2': add_kwargs['dev_2']}
    h = xp.array(waveform_response(*wave_params, **emri_kwargs))
    PSD = fixed['PSD']
    h_f = compute_fft_with_windowing(h, fixed['dt'], fixed['N_fiducial'], use_gpu=fixed['use_gpu'], n_channels=3)
    ip = inner_prod(h_f-signal, h_f-signal, PSD, fixed['delta_f'], xp=cp)
    return -0.5 * ip * 20

def to_numpy(arr):
        return arr.get() if hasattr(arr, 'get') else np.asarray(arr)

def plot_time_series_from_fft(signal_f, dt, title="Time Series"):
    signal_f = xp.array(signal_f)
    signal_f = to_numpy(signal_f)
    signal_t = np.array([np.fft.irfft(signal_f[k]) / dt for k in range(signal_f.shape[0])])
    t = np.arange(signal_t.shape[1]) * dt    
    chan_labels = ['A', 'E', 'T']

    plt.figure(figsize=(12, 6))

    for k in range(signal_t.shape[0]):
        plt.plot(t, signal_t[k], label=f'Channel {chan_labels[k]}')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(loc="upper right")
    plt.show()