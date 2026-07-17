import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens, A2TDISens, E2TDISens, T2TDISens
# from SNR_tutorial_utils import LISA_Noise
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits, ESAOrbits


def build_waveform_response(T: float, dt: float, use_gpu: bool = False, tdi_gen: int = 1,
                            pad_output: bool = False):
    if tdi_gen==1:
        tdi_str = "1st generation"
        orbits = EqualArmlengthOrbits(use_gpu=use_gpu)
        tdi_chan="AET"
    elif tdi_gen==2:
        tdi_str = "2nd generation"
        # orbits = EqualArmlengthOrbits(use_gpu=use_gpu)
        # tdi_chan="AET"
        orbits = ESAOrbits(use_gpu=use_gpu)
        tdi_chan="AE"


    waveform_model = GenerateEMRIWaveform(
        FastKerrEccentricEquatorialFlux,
        return_list=False,
        use_gpu=use_gpu,
        sum_kwargs=dict(pad_output=pad_output),
    )

    return ResponseWrapper(
        waveform_gen=waveform_model,
        Tobs=T,
        t0=10000.0,
        dt=dt,
        index_lambda=8,   # phiS
        index_beta=7,     # qS
        flip_hx=True,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=orbits,
        order=20,
        tdi=tdi_str,
        tdi_chan=tdi_chan,
    )

class GravWaveAnalysis:
    """
    A module for GW data analysis that I've compiled.
    NOISE-BASED
    """

    # Physical constants
    Gpc = 3.0856775814913674e+25 # Gigaparsec in meters
    MRSUN_SI = 1476.6250615036158 # Mass-radius in SI units
    MTSUN_SI = 4.925491025543576e-06 # Mass-time conversion factor in seconds
    YRSID_SI = 31558149.763545603 # Number of seconds in 1 astronomical year

    def __init__(self, T=None, dt=None, use_gpu=None, tdi_gen=2):
        """
        Initialize the class with optional parameters.

        Parameters:
        T (float): Total observation time in years.
        dt (float): Time step for the data.
        use_gpu (bool): Force GPU usage. If None, auto-detect.
        """
        self.dt = dt
        self.T = T 

        # Convert T to seconds
        T_sec = T * self.YRSID_SI

        # Calculate number of data points
        # NOTE: NOT the same as delta_T used for mode selection
        self.N = int(T_sec / self.dt)
        
        # Auto-detect or set backend
        if use_gpu is None:
            # Auto-detect: try CuPy first, fallback to NumPy
            if cp is not None:
                self.xp = cp
                self.use_gpu = True
            else:
                self.xp = np
                self.use_gpu = False
        elif use_gpu:
            # Force GPU
            if cp is not None:
                self.xp = cp
                self.use_gpu = True
            else:
                raise ImportError("CuPy not available but GPU usage was requested")
        else:
            # Force CPU
            self.xp = np
            self.use_gpu = False
        
        # Set FFT frequencies using the appropriate backend
        if self.N is not None and dt is not None:
            self.fft_freqs = self.xp.fft.fftfreq(self.N, dt)
            self.fft_freqs_rfft = self.xp.fft.rfftfreq(self.N, dt)

        # Pre-compute per-channel TDI PSDs at rfft frequencies
        if tdi_gen == 1:
            channels = [A1TDISens, E1TDISens, T1TDISens]
        else:
            channels = [A2TDISens, E2TDISens]  # AE only for 2nd gen
        self.n_chan = len(channels)
        self._channels = channels
        freqs_np = np.fft.rfftfreq(self.N, dt)[1:]


        self.PSD = self.xp.stack([
            self.xp.asarray(get_sensitivity(freqs_np, sens_fn=ch, return_type="PSD"))
            for ch in channels
        ])  # shape (n_chan, N_freq-1)

    def get_backend_info(self):
        """
        Get information about the current backend.
        """
        backend_name = "CuPy (GPU)" if self.use_gpu else "NumPy (CPU)"
        return {
            "backend": backend_name,
            "module": self.xp.__name__,
            "use_gpu": self.use_gpu
        }
    

    def calc_power(self, teuk_modes, ylms, m0mask, m1=None, m2=None, gw_freqs=None):    
        """
        Calculate the power spectrum using the configured backend.

        Parameters:
        teuk_modes (numpy.ndarray): Teukolsky modes.
        ylms (numpy.ndarray): Spherical harmonics.
        m0mask (numpy.ndarray): Boolean mask where m!= 0.
        m1, m2: Mass parameters (optional, for noise weighing).
        gw_freqs: Gravitational wave dimensionless frequencies (optional, for noise weighing).

        Returns:
        numpy.ndarray: Power summed over all trajectory points.
        """


        # Use self.xp for all operations
        full_modes = self.xp.concatenate([teuk_modes, self.xp.conj(teuk_modes[:, m0mask])], axis=1)
        h_lmn = full_modes * ylms[self.xp.newaxis, :]
        power = self.xp.abs(h_lmn)**2

        if m1 is not None and m2 is not None and gw_freqs is not None:
            M = m1 + m2
            M_sec = M * self.MTSUN_SI  # in seconds

            # Convert freq array to 2d [traj_pts, modes]
            gw_freqs_bd = [self.xp.asarray(freq_array) for freq_array in gw_freqs]
            freqs_array = self.xp.stack(gw_freqs_bd, axis=1)

            # Convert dimensionless freq to Hz and take absolute value
            freqs_hz = self.xp.abs(freqs_array) / (2 * self.xp.pi * M_sec)

            freqs_shape = freqs_hz.shape
            # freqs_hz_cpu = freqs_hz.get() if hasattr(freqs_hz, 'get') else freqs_hz

            # Get PSD over traj
            # NOTE:IDK WHAT TO USE?
            Sn = get_sensitivity(freqs_hz.flatten(), sens_fn=A2TDISens, return_type="PSD").reshape(freqs_shape)

            # Apply noise weighing
            power /= Sn
        
        total_power = self.xp.sum(power, axis=0)
        return total_power
      

    # def char_strain(self, hf):
    #     """
    #     Compute the characteristic strain.

    #     Parameters:
    #     hf (numpy.ndarray): Frequency domain waveform.

    #     Returns:
    #     numpy.ndarray: Characteristic strain.
    #     """
        
    #     # Compute the characteristic strain
    #     return 2*fft_freqs[freq_mask]*np.sqrt(np.abs(hf[0,freq_mask])**2+np.abs(hf[1,freq_mask])**2)

    def dist_factor(self, dist, m1, m2):
        """
        Compute the distance factor for gravitational wave signals.

        Parameters:
        dist (float or array): Distance to the source in Gpc.
        mu (float or array): Mass parameter.

        Returns:
        numpy.ndarray or cupy.ndarray: Distance factor (backend-consistent).
        """
        # Calculate reduced mass
        mu = (m1 * m2) / (m1 + m2)
        # Compute and return using the configured backend
        return self.xp.asarray((dist * self.Gpc) / (mu * self.MRSUN_SI))

    def freq_wave(self, wave):
        """
        Compute the frequency domain representation of a waveform.
        NOISE-BASED
        shape (3, N), real AET channels
        """ 
        wave_c = [self.xp.fft.rfft(wave[i]) * self.dt for i in range(wave.shape[0])]
        return self.xp.stack(wave_c)
        # wave_c = self.xp.vstack((wave.real, wave.imag))
        # return self.xp.fft.rfft(wave_c, axis=1) * self.dt

    def generate_colored_noise(self, seed=0):
        # Return time domain noise 
        df = 1 / (self.N * self.dt)
        N_freq = self.PSD.shape[1]

        if self.xp is np:
            np.random.seed(seed)
        else:
            self.xp.random.seed(seed)

        noise_f = self.xp.zeros((self.n_chan, N_freq), dtype=self.xp.complex128)
        for i in range(self.n_chan):
            variance = self.PSD[i] / (2 * df)
            noise_f[i] = (self.xp.random.normal(0, self.xp.sqrt(variance / 2), N_freq) +
                          1j * self.xp.random.normal(0, self.xp.sqrt(variance / 2), N_freq))

        # zero pad
        # NOTE: important bc psd is generated without
        dc = self.xp.zeros((self.n_chan, 1), dtype=self.xp.complex128)
        noise_f_full = self.xp.concatenate([dc, noise_f], axis=1)

        return self.xp.stack([self.xp.fft.irfft(noise_f_full[i] / self.dt, n=self.N) for i in range(self.n_chan)])

    def inner(self, h1f, h2f, return_complex=False):
        df = 1 / (self.N * self.dt)
        total = self.xp.zeros(1, dtype=self.xp.complex128)[0]
        for i in range(self.n_chan):
            total += self.xp.conj(h1f[i, 1:]) @ (h2f[i, 1:] / self.PSD[i])
        inner_prod = 4 * df * total
        if return_complex:
            return inner_prod
        else:
            return self.xp.real(inner_prod)

    def SNR(self, hf):
        """
        Compute the signal-to-noise ratio (SNR) for a gravitational wave signal.

        Parameters:
        hf (numpy.ndarray): Frequency domain waveform.

        Returns:
        float: Signal-to-noise ratio.
        """
        
        # Compute the SNR
        return self.xp.sqrt(self.inner(hf,hf))

    def overlap(self, h1f, h2f):
        """
        Compute the overlap reduction function between two gravitational waveforms.

        Parameters:
        h1f, h2f (numpy.ndarray): Frequency domain waveforms.

        Returns:
        float: Overlap reduction function.
        """
        
        # Compute the overlap reduction function
        return self.inner(h1f,h2f)/(self.SNR(h1f)*self.SNR(h2f))

    """
      For the f function
    """

    def Xstat(self, x, h):
        """
        Compute the standard detection statistic for gravitational wave data.

        Returns |<x|h>| / sqrt(<h|h>)
        """

        xf = self.freq_wave(x)
        hf = self.freq_wave(h)

        # Get COMPLEX inner product for phase maximization
        calc_inner_complex = self.inner(xf, hf, return_complex=True)
        calc_SNR = self.xp.sqrt(self.inner(hf, hf))

        # NOTE: Phase-marginalized (maximizes over Phi_phi0, Phi_r0, etc)
        return self.xp.abs(calc_inner_complex) / calc_SNR

        # NOTE: Phase-dependent version
        # calc_inner_real = self.inner(xf, hf, return_complex=False)
        # return calc_inner_real / calc_SNR

    def Xstat_timemax(self, x, h):
        """
        Compute time-and-phase-maximized detection statistic with PSD weighting.

        Returns max_τ,φ |<x|h(τ,φ)>|_weighted / sqrt(<h|h>)

        Should peak at SNR when x=h.
        Uses FFT-based cross-correlation to efficiently maximize over all time shifts.

        Parameters:
        -----------
        x : array
            Time-domain data signal
        h : array
            Time-domain template waveform

        Returns:
        --------
        float
            Maximum correlation over time and phase shifts normalized by template SNR
        """
        # Time-maximized correlation
        calc_inner_complex = self.inner_timemax(x, h)
        calc_SNR = self.xp.sqrt(self.inner_timemax(h, h))

        # Normalized detection statistic
        return calc_inner_complex / calc_SNR


    def Xmstat(self, x, hm_arr, rho_modes):
        """
        Calculate X_m statistic for each mode
        """
        X_modes = self.xp.empty(len(hm_arr), dtype=self.xp.complex128)

        # Get frequency domain of data once
        xf = self.freq_wave(x)

        for idx, hm in enumerate(hm_arr):
            # Get frequency domain of mode template
            hmf = self.freq_wave(hm)

            # Calculate inner product <x|hm>
            inner_product = self.inner(xf, hmf)

            # X_m = <x|hm> / rho_m
            X_modes[idx] = inner_product / rho_modes[idx]

        return X_modes

    def Xmstat_phasemax(self, x, hm_arr, rho_modes):
        """
        Calculate X_m statistic with phase maximization for each mode.

        Mirrors Xmstat_timemax but uses |<x|hm>| instead of max_τ |<x|hm(τ)>|.
        Each mode is independently phase-maximized.
        """
        xf = self.freq_wave(x)

        X_modes = self.xp.empty(len(hm_arr), dtype=self.xp.float64)
        for idx, hm in enumerate(hm_arr):
            hmf = self.freq_wave(hm)
            inner_product = self.inner(xf, hmf, return_complex=True)
            X_modes[idx] = self.xp.abs(inner_product) / rho_modes[idx]

        return X_modes

    def Xmstat_timemax(self, x, hm_arr, rho_modes):
        """
        Calculate X_m statistic with time maximization for each mode

        Parameters:
        -----------
        x : array
            Time-domain data signal
        hm_arr : list of arrays
            List of time-domain mode waveforms
        rho_modes : array
            SNR values for each mode

        Returns:
        --------
        array
            X_m values with time maximization
        """
        X_modes = self.xp.empty(len(hm_arr), dtype=self.xp.float64)

        for idx, hm in enumerate(hm_arr):
            # Time-maximized correlation (now takes time-domain inputs)
            inner_product = self.inner_timemax(x, hm)

            # X_m = <x|hm>_max / rho_m
            X_modes[idx] = inner_product / rho_modes[idx]

        return X_modes

    def Xmstat_timeonly(self, x, hm_arr, rho_modes):
        """
        Calculate X_m statistic with time-only maximization for each mode.

        Mirrors Xmstat_timemax but uses inner_timeonly (max_τ Re(...)) instead of
        inner_timemax (max_τ |...|). Phase is NOT maximized per mode.
        """
        X_modes = self.xp.empty(len(hm_arr), dtype=self.xp.float64)

        for idx, hm in enumerate(hm_arr):
            X_modes[idx] = self.inner_timeonly(x, hm) / rho_modes[idx]

        return X_modes

    def rhostat(self, h):
        # optimal SNR
        # assuming the h is still in time-domain

        hf = self.freq_wave(h)
        calc_inner = self.inner(hf, hf)
        return self.xp.sqrt(calc_inner)

    def rhostat_timemax(self, h):
        """
        Compute time-maximized SNR.

        Returns sqrt(max_τ,φ <h|h(τ,φ)>)

        This is useful when you want SNR maximized over time shifts.

        Parameters:
        -----------
        h : array
            Time-domain waveform

        Returns:
        --------
        float
            Time-maximized SNR
        """
        # Time-maximized correlation with itself (now takes time-domain input)
        calc_inner = self.inner_timemax(h, h)
        return self.xp.sqrt(calc_inner)

    def rhostat_modes(self, hm_arr):
        rho_modes = self.xp.empty(len(hm_arr), dtype=self.xp.float64)

        for idx, hm in enumerate(hm_arr):
            rho_modes[idx] = self.rhostat_timemax(hm)

        return self.xp.array(rho_modes)

    def inner_timemax(self, h1, h2):
        """
        Find maximum noise-weighted correlation between two waveforms.

        Uses FFT-based correlation with LISA noise weighting to find the maximum
        correlation over all time shifts.

        Parameters:
        -----------
        h1, h2 : array
            Time-domain waveforms (complex arrays)

        Returns:
        --------
        float
            Maximum correlation value over all time shifts
        """
        Y = self.xp.zeros(self.N, dtype=self.xp.complex128)
        for i in range(self.n_chan):
            H1 = self.xp.fft.fft(h1[i]) * self.dt
            H2 = self.xp.fft.fft(h2[i]) * self.dt
            Y[1:self.N//2+1] += H1[1:self.N//2+1] * self.xp.conj(H2[1:self.N//2+1]) / (0.5 * self.PSD[i])
        S = 2 * self.xp.fft.ifft(Y) / self.dt
        return self.xp.max(self.xp.abs(S))

    def wave_fft(self, wave):
        """Compute full per-channel FFTs (for use with inner_timemax_f)."""
        return [self.xp.fft.fft(wave[i]) * self.dt for i in range(self.n_chan)]

    def cross_corr_f(self, H1_list, H2_list):
        """Full cross-correlation time series S[τ] with pre-computed per-channel FFTs."""
        Y = self.xp.zeros(self.N, dtype=self.xp.complex128)
        for i in range(self.n_chan):
            Y[1:self.N//2+1] += H1_list[i][1:self.N//2+1] * self.xp.conj(H2_list[i][1:self.N//2+1]) / (0.5 * self.PSD[i])
        return 2 * self.xp.fft.ifft(Y) / self.dt

    def inner_timemax_f(self, H1_list, H2_list):
        """inner_timemax with pre-computed per-channel FFTs."""
        return self.xp.max(self.xp.abs(self.cross_corr_f(H1_list, H2_list)))

    def inner_timeonly(self, h1, h2):
        """
        max_τ Re(<h1|h2(τ)>) — time-shift maximized, phase NOT maximized.

        Same as inner_timemax but uses Re() instead of abs() before taking the max.
        """
        Y = self.xp.zeros(self.N, dtype=self.xp.complex128)
        for i in range(self.n_chan):
            H1 = self.xp.fft.fft(h1[i]) * self.dt
            H2 = self.xp.fft.fft(h2[i]) * self.dt
            Y[1:self.N//2+1] += H1[1:self.N//2+1] * self.xp.conj(H2[1:self.N//2+1]) / (0.5 * self.PSD[i])
        S = 2 * self.xp.fft.ifft(Y) / self.dt
        return self.xp.max(self.xp.real(S))  # Re() not abs()

    def Xstat_timeonly(self, x, h):
        """
        Time-maximized (no phasemax) detection statistic.

        Returns max_τ Re(<x|h(τ)>) / sqrt(<h|h>)
        """
        calc_inner = self.inner_timeonly(x, h)
        calc_SNR = self.xp.sqrt(self.inner(self.freq_wave(h), self.freq_wave(h)))
        return calc_inner / calc_SNR

    def calc_beta(self, rho_dom_M, rho_tot):
        """
        Calculate beta parameter for F-statistic.
        
        Parameters:
        rho_dom_M: SNR of dominant mode
        rho_tot: Total SNR
        
        Returns:
        float: Beta parameter
        """
        alpha = rho_dom_M / rho_tot
        beta_num = 2 * self.xp.log(alpha * rho_tot)
        beta_denom = (1 - alpha**2) * rho_tot**2
        return beta_num / beta_denom
    
    def chi_sq(self, X_theta, rho_theta):
        """
        Calculate chi square statistic
        """
        diff = X_theta - rho_theta
        return self.xp.linalg.norm(diff)**2

    def F_stat_timemarg_f(self, signal_fft, h_temp_fft, mode_ffts, rho_tot, rho_m, beta):
        """
        Time-marginalized F-statistic: mean_τ F(τ).

        Instead of picking the single best τ (time-max), averages F(τ) over all
        time lags. More robust to noise because:
          max_τ |noise cross-corr| ~ σ√(2 log N) ≈ 5.3σ  for N=1.5M
          mean_τ |noise cross-corr| ~ σ√(π/2)    ≈ 1.25σ
        At the true parameters the coherent peak still dominates the sum.

        All FFT inputs must be pre-computed with wave_fft (full FFT, length N),
        consistent with cross_corr_f.

        Parameters
        ----------
        signal_fft  : list of (N,) complex arrays — wave_fft of data
        h_temp_fft  : list of (N,) complex arrays — wave_fft of full template
        mode_ffts   : list of lists of (N,) arrays — wave_fft per mode group
        rho_tot     : float  — total template SNR sqrt(<h|h>)
        rho_m       : (n_modes,) array — per-mode SNR
        beta        : float  — F-stat beta parameter

        Returns
        -------
        float : mean_τ F(τ)
        """
        n_modes = len(mode_ffts)

        # Full-template cross-correlation for all τ at once, shape (N,)
        S_full = self.cross_corr_f(signal_fft, h_temp_fft)
        X_scalar_all = self.xp.abs(S_full) / rho_tot          # (N,)

        # Per-mode cross-correlations: X_modes_all[m, τ], shape (n_modes, N)
        X_modes_all = self.xp.empty((n_modes, self.N), dtype=self.xp.float64)
        for idx, hf in enumerate(mode_ffts):
            S_mode = self.cross_corr_f(signal_fft, hf)        # (N,)
            X_modes_all[idx] = self.xp.abs(S_mode) / rho_m[idx]

        # chi_sq(τ) = Σ_m (X_modes_all[m,τ] - rho_m[m])², shape (N,)
        chi_sq_all = self.xp.sum(
            (X_modes_all - rho_m[:, None]) ** 2, axis=0
        )

        # F(τ) = X_scalar(τ) × exp(-½β × chi_sq(τ)), then average over τ
        F_all = X_scalar_all * self.xp.exp(-0.5 * beta * chi_sq_all)  # (N,)
        return float(self.xp.mean(F_all))

    def _whiten(self, x):
        """
        Whiten a time-domain series x (shape (n_chan, N)) at full frequency
        resolution: w(f) = sqrt(4 df / PSD), DC = 0, so that the Euclidean
        inner product <a|b> = (N/2) * sum_t a_w b_w reproduces the noise-
        weighted inner product. Shared by the semi-coherent statistics.
        """
        xp = self.xp
        N = x.shape[-1]
        df = 1.0 / (N * self.dt)
        W = xp.zeros((self.n_chan, N // 2 + 1))
        W[:, 1:] = xp.sqrt(4.0 * df / self.PSD)
        return xp.stack([xp.fft.irfft(xp.fft.rfft(x[c]) * self.dt * W[c], n=N)
                         for c in range(self.n_chan)])

    def _semicoherent_inner(self, x_w, h_w, N_seg, phase_max=False,
                            tau_star=None, return_tau=False):
        """
        Per-segment, time-shift-maximized semi-coherent inner products from
        already-whitened series x_w, h_w (shape (n_chan, N)):

            <x|h>_N = Σᵢ max_τ Re(<xᵢ|hᵢ(τ)>)      (phase_max=False)
            <x|h>_N = Σᵢ max_τ |<xᵢ|hᵢ(τ)>|        (phase_max=True)
            <h|h>_N = Σᵢ <hᵢ|hᵢ>          (autocorr max is at τ=0)

        With phase_max=False each segment is maximized over a time shift
        only (Re of the real cross-correlation). With phase_max=True each
        segment is additionally maximized over a constant phase, i.e. the
        magnitude of the analytic (positive-frequency) cross-correlation —
        the per-segment analogue of inner_timemax vs inner_timeonly.

        Common-τ control (used by Sf's common_tau path so per-mode statistics
        are NOT maximized independently):
          tau_star : optional length-N_seg sequence of per-segment lag indices.
                     If given, each segment is evaluated AT that fixed lag
                     instead of being maximized over τ.
          return_tau : if True, also return the length-N_seg array of the
                       per-segment argmax lags actually used.

        Returns (<x|h>_N, <h|h>_N), or (<x|h>_N, <h|h>_N, tau_arr) when
        return_tau=True.
        """
        xp = self.xp
        N = x_w.shape[-1]
        N_per = N // N_seg

        xh_sum = 0.0
        hh_sum = 0.0
        taus = []
        for i in range(N_seg):
            sl = slice(i * N_per, (i + 1) * N_per)
            if phase_max:
                # Analytic (positive-frequency) circular cross-correlation:
                # corr[n] = (N/2) * <x_i|h_i(tau=n dt)>  (complex), so that
                # |corr[n]| is maximized over a constant phase as well as τ.
                Y = xp.zeros(N_per, dtype=xp.complex128)
                for c in range(self.n_chan):
                    Xf = xp.fft.fft(x_w[c, sl])
                    Hf = xp.fft.fft(h_w[c, sl])
                    Y[1:N_per // 2 + 1] += (xp.conj(Xf[1:N_per // 2 + 1])
                                            * Hf[1:N_per // 2 + 1])
                corr = (N / 2.0) * 2.0 * xp.fft.ifft(Y)
                metric = xp.abs(corr)   # max_τ |·|, phase maxed
            else:
                # Euclidean circular cross-correlation within the segment:
                # corr[n] = (N/2) * sum_{ch,t} x_w[t] h_w[t+n] = Re(<x_i|h_i(tau=n dt)>)
                Y = xp.zeros(N_per // 2 + 1, dtype=xp.complex128)
                for c in range(self.n_chan):
                    Y += xp.conj(xp.fft.rfft(x_w[c, sl])) * xp.fft.rfft(h_w[c, sl])
                corr = (N / 2.0) * xp.fft.irfft(Y, n=N_per)
                metric = corr   # max_τ Re, no phase max

            if tau_star is not None:
                # Evaluate at the supplied per-segment lag (no maximization).
                n = int(tau_star[i])
                xh_sum += float(metric[n])
            elif return_tau:
                n = int(xp.argmax(metric))
                taus.append(n)
                xh_sum += float(metric[n])
            else:
                xh_sum += float(xp.max(metric))

            # <h_i|h_i>: time-shift max of the autocorrelation is at tau=0
            hh_sum += float((N / 2.0) * xp.sum(h_w[:, sl] ** 2))
        if return_tau:
            return xh_sum, hh_sum, taus
        return xh_sum, hh_sum

    def SNR_semicoherent(self, x, h, N_seg, phase_max=False):
        """
        Semi-coherent statistic S_N (arXiv:2205.08702, eqs. 34-35):

            <x|h>_N = Σᵢ max_τ Re(<xᵢ|hᵢ(τ)>)
            S_N     = <x|h>_N / sqrt(<h|h>_N)

        Per-segment maximization over an overall time shift only
        (no phase maximization, hence Re not abs). Set phase_max=True to
        additionally maximize each segment over a constant phase
        (max_τ |·| of the analytic cross-correlation).

        Implemented on whitened time series with the Euclidean inner
        product (cf. the paper: "for white noise (or whitened time
        series with the Euclidean inner product), each <x|h>_N reduces
        exactly to <x|h> in the absence of maximization").
        Whitening once at full frequency resolution avoids the spectral
        leakage / PSD mis-weighting that occurs when unwindowed segment
        FFTs of steeply colored noise are weighted by a coarse segment
        PSD — that mis-weighting makes the statistic fluctuate wildly.

        x, h : time-domain, shape (n_chan, N)
        """
        x_w = self._whiten(x)
        h_w = self._whiten(h)
        xh_sum, hh_sum = self._semicoherent_inner(x_w, h_w, N_seg, phase_max)
        return xh_sum / hh_sum ** 0.5

    def SNR_semicoherent_nomax(self, x, h, N_seg):
        """
        Same construction as SNR_semicoherent but WITHOUT the per-segment
        time-shift maximization: each segment contributes the unshifted
        Re(<x_i|h_i>).

        Note: since the segments partition the whitened series,
            sum_i Re(<x_i|h_i>) = Re(<x|h>)
        so this is mathematically identical to the coherent Re(<x|h>)/rho
        (up to the trailing samples dropped when N % N_seg != 0).
        Cross-check of the segmentation/whitening machinery.

        x, h : time-domain, shape (n_chan, N)
        """
        xp = self.xp
        N = x.shape[-1]
        N_per = N // N_seg
        df = 1.0 / (N * self.dt)

        W = xp.zeros((self.n_chan, N // 2 + 1))
        W[:, 1:] = xp.sqrt(4.0 * df / self.PSD)
        x_w = xp.stack([xp.fft.irfft(xp.fft.rfft(x[c]) * self.dt * W[c], n=N)
                        for c in range(self.n_chan)])
        h_w = xp.stack([xp.fft.irfft(xp.fft.rfft(h[c]) * self.dt * W[c], n=N)
                        for c in range(self.n_chan)])

        xh_sum = 0.0
        hh_sum = 0.0
        for i in range(N_seg):
            sl = slice(i * N_per, (i + 1) * N_per)
            # unshifted Euclidean inner product on the segment (lag 0 only)
            xh_sum += float((N / 2.0) * xp.sum(x_w[:, sl] * h_w[:, sl]))
            hh_sum += float((N / 2.0) * xp.sum(h_w[:, sl] ** 2))
        return xh_sum / hh_sum ** 0.5

    def Sf(self, x, h, hm_arr, N_seg, beta=None, rho_modes=None,
           phase_max=False, common_tau=False):
        """
        Semi-coherent f-statistic.

        Mirrors the time/phase-maximized f-statistic
            f = X · exp(-½ β Σ_m (X_m - ρ_m)²)
        but replaces every coherent, time-maximized statistic with its
        semi-coherent counterpart S_N (see SNR_semicoherent), so each mode
        is maximized separately over a per-segment time shift and the
        prefactor X is itself the semi-coherent statistic of the full
        template:

            S    = <x|h>_N   / sqrt(<h|h>_N)          (full template)
            S_m  = <x|h_m>_N / sqrt(<h_m|h_m>_N)      (per mode)
            ρ_m  = sqrt(<h_m|h_m>_N)                  (semi-coherent mode SNR)

            Sf = S · exp(-½ β Σ_m (S_m - ρ_m)²)

        At the true parameters (x = h, no noise) every S_m → ρ_m so the
        chi-square term vanishes and Sf → S = ρ_tot, exactly as for f.
        Whitening is done once on the data and shared across the full
        template and all modes.

        Parameters
        ----------
        x        : (n_chan, N) time-domain data
        h        : (n_chan, N) time-domain full template
        hm_arr   : list of (n_chan, N) time-domain mode templates
        N_seg    : number of semi-coherent segments
        beta     : f-stat beta parameter. If None, computed internally as
                   calc_beta(max_m ρ_m, ρ_tot) from the semi-coherent SNRs
                   ρ_m = sqrt(<h_m|h_m>_N) and ρ_tot = sqrt(<h|h>_N).
        rho_modes: optional (n_modes,) precomputed semi-coherent mode SNRs
                   sqrt(<h_m|h_m>_N); computed from hm_arr if None.
        phase_max: if True, each segment is maximized over a constant phase
                   as well as a time shift (max_τ |·|); if False, over a
                   time shift only (max_τ Re). Applies to S and every S_m.
        common_tau: if False (default) each mode is maximized independently
                   over its own per-segment lag, so noise biases every S_m
                   high and the chi-square term never collapses. If True,
                   the per-segment lags are fixed once by maximizing the
                   FULL template, and every mode is evaluated at those same
                   lags (the semi-coherent analogue of the coherent f-stat's
                   single tau_star) — this restores S_m → ρ_m at the truth
                   and lets the chi-square term (and hence the peak) reappear.

        Returns
        -------
        float : Sf
        """
        xp = self.xp

        # Whiten data once; reuse for the full template and every mode.
        x_w = self._whiten(x)

        # Prefactor: semi-coherent statistic of the full template.
        # With common_tau, also capture the per-segment lags it selects so
        # the modes can be evaluated at the same lags (no independent max).
        h_w = self._whiten(h)
        tau_star = None
        if common_tau:
            xh, hh, tau_star = self._semicoherent_inner(
                x_w, h_w, N_seg, phase_max, return_tau=True)
        else:
            xh, hh = self._semicoherent_inner(x_w, h_w, N_seg, phase_max)
        S = xh / hh ** 0.5

        # Per-mode semi-coherent statistics S_m and self-SNRs ρ_m.
        n_modes = len(hm_arr)
        S_modes = xp.empty(n_modes, dtype=xp.float64)
        rho_sc = xp.empty(n_modes, dtype=xp.float64)
        for idx, hm in enumerate(hm_arr):
            hm_w = self._whiten(hm)
            xh_m, hh_m = self._semicoherent_inner(
                x_w, hm_w, N_seg, phase_max, tau_star=tau_star)
            S_modes[idx] = xh_m / hh_m ** 0.5
            rho_sc[idx] = hh_m ** 0.5

        # beta from the semi-coherent SNRs (dominant mode and full template),
        # the semi-coherent analogue of calc_beta(rho_dom_M, rho_tot).
        if beta is None:
            rho_tot = hh ** 0.5
            rho_dom = rho_sc.max()
            # calc_beta assumes alpha = rho_dom/rho_tot < 1. When the modes
            # destructively interfere the full-template norm can drop below a
            # single mode's, giving alpha >= 1, a negative beta and an
            # exp(+...) blow-up. Fall back to beta=0 (no chi_sq term) there.
            if rho_dom >= rho_tot:
                beta = 0.0
            else:
                beta = self.calc_beta(rho_dom, rho_tot)

        if rho_modes is not None:
            rho_sc = xp.asarray(rho_modes)

        chi_sq = xp.sum((S_modes - rho_sc) ** 2)
        return float(S * xp.exp(-0.5 * beta * chi_sq))

    def chi2_semi(self, x, h, hm_arr, N_seg, beta=None, rho_modes=None,
                  phase_max=False, common_tau=False):
        """
        Semi-coherent chi-square and its exponential suppression factor.

        Exposes the pieces that build the semi-coherent f-statistic Sf
        (see Sf), so the suppression factor can be inspected on its own
        rather than folded into S:

            S_m  = <x|h_m>_N / sqrt(<h_m|h_m>_N)      (per mode)
            ρ_m  = sqrt(<h_m|h_m>_N)                  (semi-coherent mode SNR)

            chi_sq      = Σ_m (S_m - ρ_m)²
            suppression = exp(-½ β chi_sq)

        so that  Sf = S · suppression.  At the true parameters (x = h, no
        noise) every S_m → ρ_m, chi_sq → 0 and suppression → 1.

        Parameters are identical to Sf; beta defaults to the semi-coherent
        calc_beta(max_m ρ_m, ρ_tot) just as in Sf.

        Returns
        -------
        dict with keys:
            'chi_sq'      : float — Σ_m (S_m - ρ_m)²
            'suppression' : float — exp(-½ β chi_sq)
            'beta'        : float — beta actually used
            'S'           : float — full-template semi-coherent statistic
        """
        xp = self.xp

        # Whiten data once; reuse for the full template and every mode.
        x_w = self._whiten(x)

        # Full-template semi-coherent statistic S and its SNR rho_tot.
        # See Sf for the common_tau semantics (fixed per-segment lags).
        h_w = self._whiten(h)
        tau_star = None
        if common_tau:
            xh, hh, tau_star = self._semicoherent_inner(
                x_w, h_w, N_seg, phase_max, return_tau=True)
        else:
            xh, hh = self._semicoherent_inner(x_w, h_w, N_seg, phase_max)
        S = xh / hh ** 0.5
        rho_tot = hh ** 0.5

        # Per-mode semi-coherent statistics S_m and self-SNRs ρ_m.
        n_modes = len(hm_arr)
        S_modes = xp.empty(n_modes, dtype=xp.float64)
        rho_sc = xp.empty(n_modes, dtype=xp.float64)
        for idx, hm in enumerate(hm_arr):
            hm_w = self._whiten(hm)
            xh_m, hh_m = self._semicoherent_inner(
                x_w, hm_w, N_seg, phase_max, tau_star=tau_star)
            S_modes[idx] = xh_m / hh_m ** 0.5
            rho_sc[idx] = hh_m ** 0.5

        if beta is None:
            rho_dom = rho_sc.max()
            # Guard against alpha = rho_dom/rho_tot >= 1 (destructive mode
            # interference) which makes calc_beta negative and Sf blow up;
            # fall back to beta=0 there. See Sf.
            if rho_dom >= rho_tot:
                beta = 0.0
            else:
                beta = self.calc_beta(rho_dom, rho_tot)

        if rho_modes is not None:
            rho_sc = xp.asarray(rho_modes)

        chi_sq = float(xp.sum((S_modes - rho_sc) ** 2))
        suppression = float(xp.exp(-0.5 * beta * chi_sq))
        return {
            'chi_sq': chi_sq,
            'suppression': suppression,
            'beta': float(beta),
            'S': float(S),
        }