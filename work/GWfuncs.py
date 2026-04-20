import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from lisatools.sensitivity import get_sensitivity, CornishLISASens
# from SNR_tutorial_utils import LISA_Noise

class GravWaveAnalysis:
    """
    A module for GW data analysis that I've compiled.
    """

    # Physical constants
    Gpc = 3.0856775814913674e+25 # Gigaparsec in meters
    MRSUN_SI = 1476.6250615036158 # Mass-radius in SI units
    MTSUN_SI = 4.925491025543576e-06 # Mass-time conversion factor in seconds
    YRSID_SI = 31558149.763545603 # Number of seconds in 1 astronomical year

    def __init__(self, T=None, dt=None, use_gpu=None):
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
        self.N = int(T_sec / self.dt) + 1
        
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
            Sn = get_sensitivity(freqs_hz.flatten(), sens_fn=CornishLISASens, return_type="PSD").reshape(freqs_shape)

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
        Zero-pads to data length before FFT.

        Parameters:
        wave (numpy.ndarray): Time domain waveform.

        Returns:
        numpy.ndarray: Frequency domain waveform.
        """ 
        
        wave_c = self.xp.vstack((wave.real, wave.imag))
        return self.xp.fft.rfft(wave_c, axis=1) * self.dt

    def inner(self, h1f, h2f, return_complex=False):
        """
        Compute the inner product of two gravitational waveforms.

        Parameters:
        h1f, h2f (numpy.ndarray or cupy.ndarray): Frequency domain waveforms.
        return_complex (bool): If True, return complex inner product.
                               If False, return real part only (default, backward compatible)

        Returns:
        float or complex: Inner product of the two waveforms.
        """

        df = 1/(self.N*self.dt)  # Frequency resolution

        # Get sensitivity (using rfft frequencies for this method since h1f/h2f come from freq_wave which uses rfft)
        Sn = get_sensitivity(self.fft_freqs_rfft[1:], sens_fn=CornishLISASens, return_type="PSD")
        # Sn = LISA_Noise(self.fft_freqs_rfft[1:])

        # Compute the inner product using backend operations
        plus = self.xp.conj(h1f[0,1:]) @ (h2f[0,1:] / Sn)
        cross = self.xp.conj(h1f[1,1:]) @ (h2f[1,1:] / Sn)

        inner_prod = 4*df*(plus+cross)

        if return_complex:
            return inner_prod
        else:
            # OLD behavior: return real part only
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
        # FFT with dt scaling
        H1 = self.xp.fft.fft(h1) * self.dt
        H2 = self.xp.fft.fft(h2) * self.dt

        # Get PSD at full FFT frequencies (skip DC), using absolute value for negative frequencies
        Sn = self.xp.asarray(get_sensitivity(self.xp.abs(self.fft_freqs[1:]), sens_fn=CornishLISASens, return_type="PSD"))

        # Initialize Y with zeros
        Y = self.xp.zeros_like(H1)

        # Noise-weighted correlation (skip DC)
        Y[1:] = H1[1:] * self.xp.conj(H2[1:]) / (0.5 * Sn)

        # IFFT to time domain with proper normalization
        S = self.xp.fft.ifft(Y) / self.dt

        # Return maximum correlation
        return self.xp.max(self.xp.abs(S))

    def inner_timeonly(self, h1, h2):
        """
        max_τ Re(<h1|h2(τ)>) — time-shift maximized, phase NOT maximized.

        Same as inner_timemax but uses Re() instead of abs() before taking the max.
        """
        H1 = self.xp.fft.fft(h1) * self.dt
        H2 = self.xp.fft.fft(h2) * self.dt

        Sn = self.xp.asarray(get_sensitivity(self.xp.abs(self.fft_freqs[1:]), sens_fn=CornishLISASens, return_type="PSD"))

        Y = self.xp.zeros_like(H1)
        Y[1:] = H1[1:] * self.xp.conj(H2[1:]) / (0.5 * Sn)

        S = self.xp.fft.ifft(Y) / self.dt

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
