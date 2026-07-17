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
    """Setup waveform generators (same as notebook)"""

    # Tune few configuration
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.set_log_level("info")

    # Inspiral kwargs
    inspiral_kwargs = {
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0,
        "include_minus_m": False,
    }

    # Amplitude kwargs
    amplitude_kwargs = {
        "force_backend": force_backend
    }

    # Ylm kwargs
    Ylm_kwargs = {
        "force_backend": force_backend,
    }

    # Sum kwargs
    sum_kwargs_comb = {
        "force_backend": force_backend,
        "pad_output": True,
    }

    sum_kwargs_sep = {
        "force_backend": force_backend,
        "pad_output": True,
        "separate_modes": True,
    }

    # Create waveform generators
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

    return waveform_gen_comb, waveform_gen_sep


def fstat(params, loglike_obj, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0):
    """
    Calculate F-statistic for given parameters.

    Parameters:
    -----------
    params : array-like
        [logm1, logm2, a, p0, e0]
    loglike_obj : LogLike object
        Initialized log-likelihood object
    xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 : float
        Fixed parameters

    Returns:
    --------
    fstat : float
        F-statistic value (returns log-likelihood, NOT 2*log-likelihood)
    """
    params = np.asarray(params)

    def eval_one(x):
        try:
            logm1, logm2, a, p0, e0 = x
            m1 = 10**logm1
            m2 = 10**logm2

            fstat_val = loglike_obj(np.array([m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]))
            return fstat_val
        except Exception:
            return float('-inf')

    if params.ndim == 1:
        return eval_one(params)

    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = eval_one(params[i])
    return out


def plot_1d_fisher_scan(param_name, param_idx, param_range, params_star, param_true,
                        cov_matrix_path='cov_matrix_snr32.pkl',
                        T=3/12, dt=10, mode_select=None,
                        figsize=(8, 6)):
    """
    Plot 1D scan of F-statistic for a single parameter.

    Parameters:
    -----------
    param_name : str
        Name of parameter being varied (e.g., 'logm2', 'a', 'p0', 'e0')
    param_idx : int
        Index in param_true (0=logm1, 1=logm2, 2=a, 3=p0, 4=e0)
    param_range : array-like
        Values to scan over
    params_star : tuple
        Full parameter tuple (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
    param_true : list
        True values [logm1, logm2, a, p0, e0]
    cov_matrix_path : str
        Path to covariance matrix pickle file
    T : float
        Observation time in years
    dt : float
        Time step in seconds
    mode_select : list
        List of modes to use
    figsize : tuple
        Figure size

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    f_stats : array
        F-statistic values
    """

    # Load covariance matrix for Fisher prediction
    with open(cov_matrix_path, 'rb') as f:
        cov_matrix = pickle.load(f)

    # Setup
    print("Setting up waveform generators...")
    waveform_gen_comb, waveform_gen_sep = setup_waveform_generators(T=T, dt=dt)

    print("Creating GravWaveAnalysis...")
    gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

    print("Initializing loglike class...")
    # n-indexed mode selection parameters
    n_vals = np.arange(-1,6)  # n from -1 to 5
    ell = 2  # quadrupole only

    # NOTE: change verbose argument for debugging
    # Using n-indexed mode selection
    loglike_obj = loglike_timemax.LogLikeTimeMax(
        params_star,
        waveform_gen_comb,
        gwf,
        verbose=False,
        waveform_gen_sep=waveform_gen_sep,
        ell=ell,
        n_vals=n_vals,
        M_mode=None  # No SNR filtering, use all n-groups
    )


    # Arrays to store results
    f_stats = np.zeros(len(param_range))

    # Unpack params_star
    m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params_star

    print(f"Scanning {param_name}...")
    for i, val in enumerate(param_range):
        # Build parameter array [logm1, logm2, a, p0, e0]
        param_arr = param_true.copy()
        param_arr[param_idx] = val

        # Calculate F-statistic using fstat function
        fstat_val = fstat(param_arr, loglike_obj, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
        f_stats[i] = fstat_val

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(param_range)}")

    # Calculate Fisher prediction
    sigma = np.sqrt(cov_matrix[param_idx, param_idx])
    param_true_val = param_true[param_idx]

    # Normalize computed F-stat to peak at 0
    # f_stats_norm = f_stats - np.max(f_stats)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot F-statistic
    ax.plot(param_range, f_stats, 'b-', lw=2, label='fstat')
    ax.axvline(param_true_val, color='r', linestyle='--', lw=1.5, label='true')

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('func val', fontsize=12)
    ax.set_title(param_name, fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig, ax, f_stats


def plot(param='logm2', n_points=50, T=3/12, dt=10,
         cov_matrix_path='cov_matrix_snr32.pkl', save_fig=True):
    """
    Plot F-statistic for a given parameter.

    Parameters:
    -----------
    param : str
        Parameter name: 'logm1', 'logm2', 'a', 'p0', 'e0'
    n_points : int
        Number of points to scan
    T : float
        Observation time in years
    dt : float
        Time step in seconds
    cov_matrix_path : str
        Path to covariance matrix pickle file
    save_fig : bool
        Whether to save the figure

    Returns:
    --------
    fig, ax, f_stats
    """
    # Parameter mapping
    param_map = {
        'logm1': 0,
        'logm2': 1,
        'a': 2,
        'p0': 3,
        'e0': 4
    }

    if param not in param_map:
        raise ValueError(f"Unknown parameter: {param}. Choose from {list(param_map.keys())}")

    param_idx = param_map[param]

    # True parameters
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



    params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
    param_true = np.array([np.log10(m1), np.log10(m2), a, p0, e0])

    # # Load covariance to determine range
    # with open(cov_matrix_path, 'rb') as f:
    #     cov_matrix = pickle.load(f)

    # Create range for parameter
    logm1lim = [5.6, 6.4]
    logm2lim = [0.8, 1.3]
    alim = [0.3, 0.99]
    p0lim = [8.0, 11.0]
    e0lim = [0.2, 0.5]
    lims = [logm1lim, logm2lim, alim, p0lim, e0lim]
    lim = lims[param_idx]
    true_val = param_true[param_idx]
    param_range = np.linspace(lim[0], lim[1], n_points)
    # Add true point and sort
    param_range = np.append(param_range, true_val)
    param_range = np.sort(param_range)

    # Plot
    fig, ax, f_stats = plot_1d_fisher_scan(
        param_name=param,
        param_idx=param_idx,
        param_range=param_range,
        params_star=params_star,
        param_true=param_true,
        cov_matrix_path=cov_matrix_path,
        T=T,
        dt=dt
    )

    if save_fig:
        plt.savefig(f'fstat_snr32_prior_{param}.png', dpi=150, bbox_inches='tight')

    plt.show()

    return fig, ax, f_stats


if __name__ == '__main__':
    plot('e0')
