import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

# Add work directory to path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglike_timemax

# Import from few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
import few


def setup_waveform_generators(T=3/12, dt=10, use_gpu=True, force_backend="cuda12x"):
    """Setup waveform generators"""

    cfg_set = few.get_config_setter(reset=True)
    cfg_set.set_log_level("info")

    inspiral_kwargs = {
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0,
        "include_minus_m": False,
    }
    amplitude_kwargs = {"force_backend": force_backend}
    Ylm_kwargs       = {"force_backend": force_backend}
    sum_kwargs_comb  = {"force_backend": force_backend, "pad_output": True}
    sum_kwargs_sep   = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

    waveform_gen_comb = GenerateEMRIWaveform(
        FastKerrEccentricEquatorialFlux, frame='detector',
        inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_comb, use_gpu=use_gpu
    )
    waveform_gen_sep = GenerateEMRIWaveform(
        FastKerrEccentricEquatorialFlux, frame='detector',
        inspiral_kwargs=inspiral_kwargs, amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs, sum_kwargs=sum_kwargs_sep, use_gpu=use_gpu
    )
    return waveform_gen_comb, waveform_gen_sep


# Parameter ordering for 8-param extended space
# [logm1, logm2, a, p0, e0, dist, qS, phiS]
PARAM_MAP = {
    'logm1': 0,
    'logm2': 1,
    'a':     2,
    'p0':    3,
    'e0':    4,
    'dist':  5,
    'qS':    6,
    'phiS':  7,
}


def _eval_one(x, loglike_obj, xI0, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0):
    """
    Evaluate fstat for one parameter vector.

    x : [logm1, logm2, a, p0, e0, dist, qS, phiS]
    qS is passed directly to loglike (radians, in [0, pi]).
    """
    try:
        logm1, logm2, a, p0, e0, dist, qS, phiS = x
        m1   = 10**logm1
        m2   = 10**logm2
        return float(loglike_obj(np.array([
            m1, m2, a, p0, e0, xI0, dist, qS, phiS,
            qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
        ])))
    except Exception:
        return float('-inf')


def fstat_ext(params, loglike_obj, xI0, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0):
    """
    Evaluate fstat for one or many parameter vectors.

    params : array [logm1, logm2, a, p0, e0, dist, qS, phiS]
             shape (8,) for single or (N, 8) for batch
    """
    params = np.asarray(params)
    if params.ndim == 1:
        return _eval_one(params, loglike_obj, xI0, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = _eval_one(params[i], loglike_obj, xI0, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
    return out


def plot_1d_fisher_scan(param_name, param_idx, param_range,
                        params_star, param_true_ext,
                        cov_matrix_path='cov_matrix_snr97_ext_ori.pkl',
                        T=3/12, dt=10, n_sigma=None, figsize=(8, 6)):
    """
    1D scan of fstat for a single parameter (intrinsic or extrinsic).

    Parameters
    ----------
    param_name : str
        Key from PARAM_MAP.
    param_idx : int
        Index in param_true_ext (0-7).
    param_range : array
        Values to scan (in transformed coords: logm1, logm2, a, p0, e0,
        dist [Gpc], qS [rad], phiS).
    params_star : tuple
        True source params in FEW order:
        (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
         Phi_phi0, Phi_theta0, Phi_r0)
    param_true_ext : array, shape (8,)
        True values in transformed coords:
        [logm1, logm2, a, p0, e0, dist, qS, phiS]
    """

    with open(cov_matrix_path, 'rb') as f:
        cov_matrix = pickle.load(f)

    print("Setting up waveform generators...")
    waveform_gen_comb, waveform_gen_sep = setup_waveform_generators(T=T, dt=dt)

    print("Creating GravWaveAnalysis...")
    gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

    print("Initializing loglike class...")
    n_vals = np.arange(-1, 6)
    ell = 2
    loglike_obj = loglike_timemax.LogLikeTimeMax(
        params_star,
        waveform_gen_comb,
        gwf,
        verbose=False,
        waveform_gen_sep=waveform_gen_sep,
        ell=ell,
        n_vals=n_vals,
        M_mode=None
    )

    # Fixed params not in the 8-param vector
    (m1, m2, a, p0, e0, xI0, dist,
     qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0) = params_star

    f_stats = np.zeros(len(param_range))
    print(f"Scanning {param_name}...")
    for i, val in enumerate(param_range):
        p = param_true_ext.copy()
        p[param_idx] = val
        f_stats[i] = fstat_ext(p, loglike_obj, xI0, qK, phiK,
                                Phi_phi0, Phi_theta0, Phi_r0)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(param_range)}")

    param_true_val = param_true_ext[param_idx]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(param_range, f_stats, 'b-', lw=2, label='fstat')
    ax.axvline(param_true_val, color='r', linestyle='--', lw=1.5, label='true')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('func val', fontsize=12)
    if n_sigma is not None:
        ax.set_title(f'{param_name} - {n_sigma}*sigma (from fisher)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return fig, ax, f_stats


def plot(param='qS', n_sigma=1, n_points=50, T=3/12, dt=10,
         cov_matrix_path='cov_matrix_snr32.pkl', save_fig=True):
    """
    Plot fstat 1D scan for a single parameter.

    param : str
        One of: 'logm1', 'logm2', 'a', 'p0', 'e0', 'dist', 'qS', 'phiS'
    """
    if param not in PARAM_MAP:
        raise ValueError(f"Unknown parameter: {param}. Choose from {list(PARAM_MAP.keys())}")

    param_idx = PARAM_MAP[param]

    # True source parameters
    m1 = 1e6
    m2 = 1e1
    a = 0.7
    p0 = 9
    e0 = 0.4
    xI0 = 1.0
    dist = 1.8  # Gpc
    qS = np.pi
    phiS = 0.
    qK =  0.
    phiK = 0.
    Phi_phi0 = 0.4
    Phi_theta0 = 0.0
    Phi_r0 = 0.5



    params_star = (m1, m2, a, p0, e0, xI0, dist,
                   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)

    # True values in transformed (search) coordinates
    param_true_ext = np.array([
        np.log10(m1),   # logm1
        np.log10(m2),   # logm2
        a,
        p0,
        e0,
        dist,           # Gpc
        qS,             # qS [rad]
        phiS,
    ])

    with open(cov_matrix_path, 'rb') as f:
        cov_matrix = pickle.load(f)

    sigma    = np.sqrt(cov_matrix[param_idx, param_idx])
    true_val = param_true_ext[param_idx]

    param_range = np.linspace(true_val - n_sigma*sigma,
                              true_val + n_sigma*sigma, n_points)
    param_range = np.sort(np.append(param_range, true_val))

    # Clip dist to physical range [0, 2] Gpc
    if param == 'dist':
        param_range = np.clip(param_range, 0.0, 2.0)

    fig, ax, f_stats = plot_1d_fisher_scan(
        param_name=param,
        param_idx=param_idx,
        param_range=param_range,
        params_star=params_star,
        param_true_ext=param_true_ext,
        cov_matrix_path=cov_matrix_path,
        T=T,
        dt=dt,
        n_sigma=n_sigma,
    )

    if save_fig:
        plt.savefig(f'fstat_snr97_ext_ori_{param}_{n_sigma}.png', dpi=150, bbox_inches='tight')

    plt.show()
    return fig, ax, f_stats


if __name__ == '__main__':
    plot('dist', n_sigma=1000)
