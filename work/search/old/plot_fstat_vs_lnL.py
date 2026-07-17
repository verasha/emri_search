import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

# Add work directory to path
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
import localgit.FEWNEW.work.GWfuncs_backup2 as GWfuncs_backup2
import loglikegen

# Import from few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
import few

def setup_waveform_generators(T=1/12, dt=10, use_gpu=True, force_backend="cuda12x"):
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


def standard_lnL(params, data_fd, gwf, waveform_gen_comb,
                 xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0):
    """
    Calculate standard log-likelihood: -0.5*<d-h|d-h>

    Parameters:
    -----------
    params : array-like
        [logm1, logm2, a, p0, e0]
    data_fd : array
        Frequency domain data
    gwf : GravWaveAnalysis object
        GW analysis object with inner product method
    waveform_gen_comb : waveform generator
        Combined waveform generator
    xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 : float
        Fixed parameters

    Returns:
    --------
    lnL : float
        Standard log-likelihood value
    """
    params = np.asarray(params)

    def eval_one(x):
        try:
            logm1, logm2, a, p0, e0 = x
            m1 = 10**logm1
            m2 = 10**logm2

            # Generate full combined template waveform
            template_td = waveform_gen_comb(m1, m2, a, p0, e0, xI0, dist,
                                             qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,
                                             T=gwf.T, dt=gwf.dt)

            # FFT to frequency domain
            template_fd = np.fft.rfft(template_td)

            # Compute residual
            residual = data_fd - template_fd

            # Compute -0.5*<d-h|d-h>
            lnL = -0.5 * gwf.inner(residual, residual)

            return lnL
        except Exception as e:
            print(f"Error in standard_lnL: {e}")
            return float('-inf')

    if params.ndim == 1:
        return eval_one(params)

    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = eval_one(params[i])
    return out


def plot_1d_comparison(param_name, param_idx, param_range, params_star, param_true,
                       T=1/12, dt=10, mode_select=None,
                       figsize=(8, 6), n_sigma=None):
    """
    Plot 1D scan comparing F-statistic and standard log-likelihood.

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
    lnL_vals : array
        Standard log-likelihood values
    """

    # Setup
    print("Setting up waveform generators...")
    waveform_gen_comb, waveform_gen_sep = setup_waveform_generators(T=T, dt=dt)

    print("Creating GravWaveAnalysis...")
    gwf = GWfuncs_backup2.GravWaveAnalysis(T, dt)

    # Default mode selection
    if mode_select is None:
        mode_select = [(2, -2, 0), (2, -2, -1), (2, -2, 1), (3, -3, -1), (3, -3, 1)]

    print("Initializing loglike class...")
    loglike_obj = loglikegen.LogLike(params_star,
                                      waveform_gen_comb,
                                      gwf,
                                      mode_select=mode_select,
                                      verbose=False,
                                      waveform_gen_sep=waveform_gen_sep)

    # Get data from loglike object (already generated at true parameters)
    print("Getting data from loglike object...")
    data_td = loglike_obj.signal
    data_fd = np.fft.rfft(data_td)

    # Arrays to store results
    f_stats = np.zeros(len(param_range))
    lnL_vals = np.zeros(len(param_range))

    print(f"Scanning {param_name}...")
    for i, val in enumerate(param_range):
        # Build parameter array [logm1, logm2, a, p0, e0]
        param_arr = param_true.copy()
        param_arr[param_idx] = val

        # Calculate F-statistic
        fstat_val = fstat(param_arr, loglike_obj, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
        f_stats[i] = fstat_val

        # Calculate standard log-likelihood
        lnL_val = standard_lnL(param_arr, data_fd, gwf, waveform_gen_comb,
                               xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
        lnL_vals[i] = lnL_val

        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(param_range)}")

    # Normalize both to peak at 0
    f_stats_norm = f_stats - np.max(f_stats)
    lnL_norm = lnL_vals - np.max(lnL_vals)

    param_true_val = param_true[param_idx]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot both
    ax.plot(param_range, f_stats_norm, 'b-', lw=2, label='F-statistic')
    ax.plot(param_range, lnL_norm, 'r--', lw=2, label='Standard lnL: -0.5*<d-h|d-h>')
    ax.axvline(param_true_val, color='k', linestyle=':', lw=1.5, label='True value')

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Normalized log-likelihood', fontsize=12)
    if n_sigma is not None:
        ax.set_title(f'{param_name} - F-stat vs Standard lnL ({n_sigma}σ)', fontsize=14)
    else:
        ax.set_title(f'{param_name} - F-stat vs Standard lnL', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Calculate width ratio (FWHM comparison)
    def calculate_fwhm(x, y):
        """Calculate full width at half maximum"""
        half_max = np.max(y) / 2
        indices = np.where(y >= half_max)[0]
        if len(indices) > 1:
            return x[indices[-1]] - x[indices[0]]
        return np.nan

    fwhm_fstat = calculate_fwhm(param_range, f_stats_norm + np.max(f_stats_norm))
    fwhm_lnL = calculate_fwhm(param_range, lnL_norm + np.max(lnL_norm))

    if not np.isnan(fwhm_fstat) and not np.isnan(fwhm_lnL):
        broadening_ratio = fwhm_fstat / fwhm_lnL
        print(f"\nFWHM comparison:")
        print(f"  F-stat FWHM: {fwhm_fstat:.6f}")
        print(f"  lnL FWHM: {fwhm_lnL:.6f}")
        print(f"  Broadening ratio (F-stat/lnL): {broadening_ratio:.3f}")

    return fig, ax, f_stats, lnL_vals


def plot(param='logm2', n_sigma=1, n_points=50, T=1/12, dt=10,
         cov_matrix_path='cov_matrix_2yr.pkl', save_fig=True):
    """
    Plot F-statistic vs standard log-likelihood for a given parameter.

    Parameters:
    -----------
    param : str
        Parameter name: 'logm1', 'logm2', 'a', 'p0', 'e0'
    n_sigma : float
        Number of sigma to scan around true value
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
    fig, ax, f_stats, lnL_vals
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

    # True parameters (same as notebook)
    m1 = 1e6
    m2 = 3e1
    a = 0.7
    p0 = 15
    e0 = 0.4
    xI0 = 1.0
    dist = 0.25
    qS = 0.5
    phiS = 1
    qK = 1
    phiK = phiS + np.pi/3
    Phi_phi0 = 0.4
    Phi_theta0 = 0.0
    Phi_r0 = 0.5

    params_star = (m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0)
    param_true = np.array([np.log10(m1), np.log10(m2), a, p0, e0])

    # Load covariance to determine range
    with open(cov_matrix_path, 'rb') as f:
        cov_matrix = pickle.load(f)

    # Create range for parameter
    sigma = np.sqrt(cov_matrix[param_idx, param_idx])
    true_val = param_true[param_idx]
    param_range = np.linspace(true_val - n_sigma*sigma, true_val + n_sigma*sigma, n_points)
    # Add true point and sort
    param_range = np.append(param_range, true_val)
    param_range = np.sort(param_range)

    # Plot
    fig, ax, f_stats, lnL_vals = plot_1d_comparison(
        param_name=param,
        param_idx=param_idx,
        param_range=param_range,
        params_star=params_star,
        param_true=param_true,
        T=T,
        dt=dt,
        n_sigma=n_sigma
    )

    if save_fig:
        plt.savefig(f'fstat_vs_lnL_{param}_{n_sigma}sigma.png', dpi=150, bbox_inches='tight')

    plt.show()

    return fig, ax, f_stats, lnL_vals


if __name__ == '__main__':
    plot('e0', n_sigma=1)
