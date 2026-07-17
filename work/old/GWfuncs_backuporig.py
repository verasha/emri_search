from lisatools.sensitivity import get_sensitivity, LISASens

class GravWaveAnalysis:
    """
    A module for GW data analysis that I've compiled.
    """

    # Physical constants
    Gpc = 3.0856775814913674e+25 # Gigaparsec in meters
    MRSUN_SI = 1476.6250615036158 # Mass-radius in SI units

    def __init__(self, N=None, dt=None, use_gpu=None):
        """
        Initialize the class with optional parameters.

        Parameters:
        N (int): Number of data points.
        dt (float): Time step for the data.
        use_gpu (bool): Force GPU usage. If None, auto-detect.
        """
        self.N = N
        self.dt = dt
        
        # Auto-detect or set backend
        if use_gpu is None:
            # Auto-detect: try CuPy first, fallback to NumPy
            try:
                import cupy as cp
                self.xp = cp
                self.use_gpu = True
            except ImportError:
                import numpy as np
                self.xp = np
                self.use_gpu = False
        elif use_gpu:
            # Force GPU
            try:
                import cupy as cp
                self.xp = cp
                self.use_gpu = True
            except ImportError:
                raise ImportError("CuPy not available but GPU usage was requested")
        else:
            # Force CPU
            import numpy as np
            self.xp = np
            self.use_gpu = False
        
        # Set FFT frequencies using the appropriate backend
        if N is not None and dt is not None:
            self.fft_freqs = self.xp.fft.rfftfreq(N, dt)

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
    

    def calc_power(self, teuk_modes, ylms, m0mask):
        """
        Calculate the power spectrum using the configured backend (GPU/CPU).

        Parameters:
        teuk_modes (numpy.ndarray): Teukolsky modes.
        ylms (numpy.ndarray): Spherical harmonics.
        m0mask (numpy.ndarray): Boolean mask where m!= 0.

        Returns:
        numpy.ndarray: Power summed over all trajectory points.
        """


        # Use self.xp for all operations
        full_modes = self.xp.concatenate([teuk_modes, self.xp.conj(teuk_modes[:, m0mask])], axis=1)
        h_lmn = full_modes * ylms[self.xp.newaxis, :]
        power = self.xp.abs(h_lmn)**2

        return self.xp.sum(power, axis=0)  # Sum each mode over all trajectory points
      

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

    def dist_factor(self, dist, mu):
        """
        Compute the distance factor for gravitational wave signals.

        Parameters:
        dist (float or array): Distance to the source in Gpc.
        mu (float or array): Mass parameter.

        Returns:
        numpy.ndarray or cupy.ndarray: Distance factor (backend-consistent).
        """
        
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

    def inner(self, h1f, h2f):
        """
        Compute the inner product of two gravitational waveforms.

        Parameters:
        h1f, h2f (numpy.ndarray or cupy.ndarray): Frequency domain waveforms.

        Returns:
        float: Inner product of the two waveforms.
        """

        df = 1/(self.N*self.dt)  # Frequency resolution
        
        # Get sensitivity 
        Sn = get_sensitivity(self.fft_freqs[1:], sens_fn=LISASens, return_type="PSD")
        
        # Compute the inner product using backend operations
        plus = self.xp.conj(h1f[0,1:]) @ (h2f[0,1:] / Sn)
        cross = self.xp.conj(h1f[1,1:]) @ (h2f[1,1:] / Sn)

        return 4*df*self.xp.real(plus+cross)

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
        """
        
        xf = self.freq_wave(x) 
        hf = self.freq_wave(h)
        
        calc_inner = self.inner(xf, hf)
        calc_SNR = self.xp.sqrt(self.inner(hf, hf))  

        return calc_inner / calc_SNR

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

    def rhostat(self, h):
        # optimal SNR 
        # assuming the h is still in time-domain
        
        hf = self.freq_wave(h) 
        calc_inner = self.inner(hf, hf)
        return self.xp.sqrt(calc_inner) 
    
    def rhostat_modes(self, hm_arr): 
        rho_modes = self.xp.empty(len(hm_arr), dtype=self.xp.float64)

        for idx, hm in enumerate(hm_arr):
            rho_modes[idx] = self.rhostat(hm)
        
        return self.xp.array(rho_modes)
    
    def chi_sq(self, X_theta, rho_theta):
        """
        Calculate chi square statistic
        """
        diff = X_theta - rho_theta
        return self.xp.linalg.norm(diff)**2