import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from lisatools.sensitivity import get_sensitivity, LISASens

class GravWaveAnalysis:
    """
    A module for GW data analysis that I've compiled.
    """

    # Physical constants
    Gpc = 3.0856775814913674e+25 # Gigaparsec in meters
    MRSUN_SI = 1476.6250615036158 # Mass-radius in SI units
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
        self.T = T  # Keep T in years
        # Calculate N (number of points) from T and dt, converting T to seconds
        T_sec = T * self.YRSID_SI
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
            self.fft_freqs = self.xp.fft.rfftfreq(self.N, dt)

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


class LogLike:
    """
    Log-likelihood class for gravitational wave parameter estimation.
    
    Usage:
    loglike_obj = LogLike(theta_data, waveform_gen, gwf, traj, amp, interpolate_mode_sum, ylm_gen, dt, T)
    signal = loglike_obj.signal  # Access data signal
    log_likelihood = loglike_obj(theta_template)  # Evaluate likelihood for template parameters
    """
    
    def __init__(self, theta_data, waveform_gen, gwf=None, dt=None, T=None, M_mode=10):
        """
        Initialize LogLike object with data parameters.
        
        Parameters:
        theta_data: Data parameters (m1, m2, a, p0, e0, xI0, theta, phi, dist)
        waveform_gen: Waveform generator object
        gwf: GravWaveAnalysis object (optional - created if not provided)
        dt, T: Time step and total time (optional if gwf provided)
        M_mode: Number of modes to use (default 10)
        """
        self.theta_data = theta_data
        self.waveform_gen = waveform_gen
        self.M_mode = M_mode
        
        # Create or use provided GravWaveAnalysis object
        if gwf is None:
            if dt is None or T is None:
                raise ValueError("Must provide either GravWaveAnalysis object or both dt and T parameters")
            self.gwf = GravWaveAnalysis(T=T, dt=dt)
            self.dt = dt
            self.T = T
        else:
            self.gwf = gwf
            self.dt = gwf.dt
            self.T = gwf.T
        
        # Get components from waveform generator (assuming it has these attributes)
        self.traj = getattr(waveform_gen, 'inspiral_generator', None)
        self.amp = getattr(waveform_gen, 'amplitude_generator', None)
        self.interpolate_mode_sum = getattr(waveform_gen, 'create_waveform', None)
        self.ylm_gen = getattr(waveform_gen, 'ylm_gen', None)
        
        # Generate data signal with data parameters
        m1, m2, a, p0, e0, xI0, theta, phi, dist = theta_data
        self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=self.dt, T=self.T)
    
    def _generate_mode_waveforms(self, total_power, teuk_modes, ylms, t_gpu, factor):
        """
        Generate waveforms for top M modes based on power.
        
        Parameters:
        total_power: Power values for all modes
        teuk_modes: Teukolsky modes
        ylms: Spherical harmonics
        t_gpu: Time array on GPU
        factor: Distance factor
        
        Returns:
        list: Waveforms for selected modes
        """
        # Get mode labels
        mode_labels = [f"({l},{m},{n})" for l,m,n in zip(self.amp.l_arr, self.amp.m_arr, self.amp.n_arr)]

        # Get top M indices
        top_indices_gpu = self.gwf.xp.argsort(total_power)[-self.M_mode:][::-1]
        if hasattr(top_indices_gpu, 'get'):  # CuPy array
            top_indices = top_indices_gpu.get().tolist()
        else:  # NumPy array
            top_indices = top_indices_gpu.tolist()

        # Pick modes based on top M power contributions
        mp_modes = [mode_labels[idx] for idx in top_indices]
        top_indices = [mode_labels.index(mode) for mode in mp_modes]

        # Generate hm_arr for top modes
        waveform_per_mode = []
        for idx in top_indices:
            l = self.amp.l_arr[idx]
            m = self.amp.m_arr[idx]
            n = self.amp.n_arr[idx]

            if m >= 0:
                teuk_modes_single = teuk_modes[:, [idx]]
                ylms_single = ylms[[idx]]
                m_arr = self.amp.m_arr[[idx]]
            else:
                pos_m_mask = (self.amp.l_arr == l) & (self.amp.m_arr == -m) & (self.amp.n_arr == n)
                pos_m_idx = self.gwf.xp.where(pos_m_mask)[0][0]
                teuk_modes_single = (-1)**l * self.gwf.xp.conj(teuk_modes[:, [pos_m_idx]])
                ylms_single = ylms[[idx]]
                m_arr = self.gwf.xp.abs(self.amp.m_arr[[idx]]) 

            waveform = self.interpolate_mode_sum(
                t_gpu, teuk_modes_single, ylms_single,
                self.traj.integrator_spline_t, self.traj.integrator_spline_phase_coeff[:, [0, 2]],
                self.amp.l_arr[[idx]], m_arr, self.amp.n_arr[[idx]], 
                dt=self.dt, T=self.T
            )
            waveform_per_mode.append(waveform / factor)
            
        return waveform_per_mode
    
    def __call__(self, theta_template):
        """
        Evaluate log-likelihood for template parameters.
        
        Parameters:
        theta_template: Template parameters (m1, m2, a, p0, e0, xI0, theta, phi, dist)
        
        Returns:
        float: Log-likelihood value
        """
        
        # Convert theta to parameters
        m1, m2, a, p0, e0, xI0, theta, phi, dist = theta_template
        
        # Calculate the factor for normalization
        factor = self.gwf.dist_factor(dist, m2)

        # Generate trajectory
        (t, p, e, x, Phi_phi, Phi_theta, Phi_r) = self.traj(m1, m2, a, p0, e0, xI0, T=self.T, dt=self.dt)
        t_gpu = self.gwf.xp.asarray(t)

        # Get amplitudes along trajectory
        teuk_modes = self.amp(a, p, e, x)

        # Get Ylms
        ylms = self.ylm_gen(self.amp.unique_l, self.amp.unique_m, theta, phi).copy()[self.amp.inverse_lm]

        # Calculate power for all modes
        m0mask = self.amp.m_arr_no_mask != 0
        total_power = self.gwf.calc_power(teuk_modes, ylms, m0mask)

        # Generate template waveform through interpmodesum
        # # need to prepare arrays for sum with all modes due to +/- m setup
        # ls = self.amp.l_arr[: teuk_modes.shape[1]]
        # ms = self.amp.m_arr[: teuk_modes.shape[1]]
        # ks = self.amp.k_arr[: teuk_modes.shape[1]]
        # ns = self.amp.n_arr[: teuk_modes.shape[1]]

        # keep_modes = self.gwf.xp.arange(teuk_modes.shape[1])
        # temp2 = keep_modes * (keep_modes < self.amp.num_m0) + (keep_modes + self.amp.num_m_1_up) * (
        #     keep_modes >= self.amp.num_m0
        # ) # amp.num_m0 gives number of modes with m == 0, amp.num_m_1_up gives number of modes with m > 0

        # ylmkeep = np.concatenate([keep_modes, temp2])
        # ylms_in = ylms[ylmkeep]
        # h_temp = self.interpolate_mode_sum(
        #     t_gpu,
        #     teuk_modes,
        #     ylms_in,
        #     self.traj.integrator_spline_t,
        #     self.traj.integrator_spline_phase_coeff,
        #     ls,
        #     ms,
        #     ks,
        #     ns,
        #     dt=self.dt,
        #     T=self.T
        # )


        # Generate waveforms for top modes
        waveform_per_mode = self._generate_mode_waveforms(total_power, teuk_modes, ylms, t_gpu, factor)

        # Calculate rho_m
        rho_m = self.gwf.rhostat_modes(waveform_per_mode)
        
        # Calculate Xm 
        X_modes = self.gwf.Xmstat(self.signal, waveform_per_mode, rho_m)

        # Calculate X_scalar
        Xdotrho = self.gwf.xp.sum(X_modes * rho_m)
        rho_tot = self.gwf.xp.sqrt(self.gwf.xp.sum(rho_m**2))
        # rho_tot can be calculated from h_temp for accuracy
        # but we're using the approximation via modes here 
        # TODO: ? should we use h_temp to calculate rho_tot?
        X_scalar = Xdotrho / rho_tot

        # Check actual Xstat value
        # Need to generate h_temp though
        # X_check = self.gwf.Xstat(self.signal, h_temp)

        # Calculate optimal SNR of most dominant mode by power
        rho_dom_M = self.gwf.rhostat(waveform_per_mode[0])

        # Calculate beta parameter
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        # Calculate chi sq
        chi_sq = self.gwf.chi_sq(X_modes, rho_m)

        # Calculate f statistic 
        f_exp = -0.5 * beta * chi_sq 

        # Overflow protection for large exponentials 
        # TODO: Check better way to handle this, ex: mpmath?    
        if f_exp > 700:  
            f_exp = 700 # Cap 
        elif f_exp < -700:
            return -self.gwf.xp.inf
        
        f_stat = X_scalar * self.gwf.xp.exp(f_exp)

        # Convert to float, handling both GPU and CPU backends
        f_stat_real = self.gwf.xp.real(f_stat)
        if hasattr(f_stat_real, 'get'):  # CuPy array
            logl_res = float(f_stat_real.get())
        else:  # NumPy array
            logl_res = float(f_stat_real)
        
        # Check for NaNs 
        if self.gwf.xp.isnan(logl_res):
            return -self.gwf.xp.inf

        return logl_res


class ModeSelector:
    """
    Mode selection algorithm class for gravitational wave analysis.
    
    Implements the power-based mode selection algorithm from the Jupyter notebook.
    
    Usage:
    mode_selector = ModeSelector(teuk_modes, amp, gw_frequencies, gw_phases, delta_T, factor, gwf)
    selected_modes, selected_indices, selected_labels, counts, final_ips = mode_selector.select_modes_power(
        power_values, mode_labels, original_mode_labels, M_sel=10, inner_threshold=0.01
    )
    """
    
    def __init__(self, teuk_modes, amp, gw_frequencies, gw_phases, delta_T, factor, gwf):
        """
        Initialize ModeSelector with trajectory and analysis objects.
        
        Parameters:
        teuk_modes: Teukolsky modes from amplitude calculation
        amp: Amplitude interpolation object
        gw_frequencies: Gravitational wave frequencies per mode
        gw_phases: Gravitational wave phases per mode  
        delta_T: Time step
        gwf: GravWaveAnalysis object
        dist: Distance to source in Gpc
        mu: Mass parameter
        """
        self.teuk_modes = teuk_modes
        self.amp = amp
        self.gw_frequencies = gw_frequencies
        self.gw_phases = gw_phases
        self.delta_T = delta_T
        self.gwf = gwf
        self.factor = factor
    
    def calc_inner_power(self, mode_i, mode_j):
        """
        Calculate inner product between modes using power method.
        
        Parameters:
        mode_i, mode_j: Lists of mode indices
        
        Returns:
        float: Inner product value
        """
        
        total_inner = 0.0
        for idx_i in mode_i:
            for idx_j in mode_j:
                # Obtain the lmn-s
                l_i = self.amp.l_arr[idx_i]
                m_i = self.amp.m_arr[idx_i]
                n_i = self.amp.n_arr[idx_i]
            
                l_j = self.amp.l_arr[idx_j]
                m_j = self.amp.m_arr[idx_j]
                n_j = self.amp.n_arr[idx_j]
            
                # Get Teukolsky modes
                # Check if negative m 
                if m_i >= 0:
                    A_i = self.teuk_modes[:, idx_i]
            
                elif m_i < 0:
                    pos_m_mask_i = (self.amp.l_arr == l_i) & (self.amp.m_arr == -m_i) & (self.amp.n_arr == n_i)
                    pos_m_idx_i = self.gwf.xp.where(pos_m_mask_i)[0][0]
                    A_i_pos = self.teuk_modes[:, pos_m_idx_i]
                    A_i = (-1)**l_i * self.gwf.xp.conj(A_i_pos)
            
                if m_j >= 0:
                    A_j = self.teuk_modes[:, idx_j]
                    
                elif m_j < 0:
                    pos_m_mask_j = (self.amp.l_arr == l_j) & (self.amp.m_arr == -m_j) & (self.amp.n_arr == n_j)
                    pos_m_idx_j = self.gwf.xp.where(pos_m_mask_j)[0][0]
                    A_j_pos = self.teuk_modes[:, pos_m_idx_j]
                    A_j = (-1)**l_j * self.gwf.xp.conj(A_j_pos)
            
                # Get sensitivity for each mode 
                Sn_i = get_sensitivity(self.gw_frequencies[idx_i], sens_fn=LISASens, return_type="PSD")
                Sn_j = get_sensitivity(self.gw_frequencies[idx_j], sens_fn=LISASens, return_type="PSD")
            
                # Get noise-weighted amplitudes
                bar_A_i = A_i.get() / np.sqrt(Sn_i)
                bar_A_j = A_j.get() / np.sqrt(Sn_j)
            
                # Get phase mask
                phase_mask = np.abs(self.gw_phases[idx_i] - self.gw_phases[idx_j]) < 1.0 
            
                # Calculate product
                prod = np.conj(bar_A_i[phase_mask]) * bar_A_j[phase_mask]
            
                # Calculate full inner product
                innerprod = np.sum(np.real(prod)) * self.delta_T * 1/(self.factor**2)

                total_inner += innerprod

        return total_inner
    
    def SNR_approx(self, mode_idx):
        """
        Calculate approximate SNR for given mode indices.
        
        Parameters:
        mode_idx: List of mode indices
        
        Returns:
        float: SNR value
        """
        return np.sqrt(self.calc_inner_power(mode_idx, mode_idx))
    
    def overlap_approx(self, mode_i, mode_j):
        """
        Calculate approximate overlap between two modes.
        
        Parameters:
        mode_i, mode_j: Lists of mode indices
        
        Returns:
        float: Overlap value
        """
        SNR_i = self.SNR_approx(mode_i)
        SNR_j = self.SNR_approx(mode_j)

        cross_inner = self.calc_inner_power(mode_i, mode_j)
        overlap = cross_inner / (SNR_i * SNR_j)
        return overlap
    
    def select_modes_power(self, power_values, mode_labels, original_mode_labels, M_sel=10, inner_threshold=0.01):
        """
        Power-based mode selection algorithm.
        
        Parameters:
        power_values: Power values for modes (sorted)
        mode_labels: Mode labels (sorted by power)
        original_mode_labels: Original unsorted mode labels
        M_sel: Target number of selected modes
        inner_threshold: Inner product threshold for mode combination
        
        Returns:
        tuple: (selected_modes, selected_indices, selected_labels, combination_counts, final_inner_products)
        """
        
        ##########################################
        ### Step 0: Initialize with strongest mode 
        ##########################################

        # Initialize combination counts
        combination_counts = {
            'rejected_modes': 0,
            'remaining_to_hM': 0, 
            'hM_with_selected': 0,
            'total': 0
        }

        # Track which modes were combined with each selected mode
        combined_modes_tracker = {}

        # Create mapping from sorted mode labels to original indices
        mode_label_to_original_idx = {label: idx for idx, label in enumerate(original_mode_labels)}
        
        # Pick the strongest mode h_0
        h0_label = mode_labels[0]
        h0_original_idx = mode_label_to_original_idx[h0_label]
        
        # Initialize selected set S with h_0
        selected_modes = [[h0_original_idx]] # Each element is a list of original mode indices
        selected_indices = [0]  # Index in the sorted list
        selected_labels = [h0_label]

        # Initialize tracking for the first selected mode
        combined_modes_tracker[0] = {'original': h0_label, 'combined_with': []}

        # Keep track of all processed modes (using sorted indices)
        processed_indices = [0]  
        
        print(f"Step 0: Selected strongest mode h_0: {h0_label} with power value {power_values[0]:.4e}")
        print("Using power inner product calculation.")
        h0_inner = self.calc_inner_power([h0_original_idx], [h0_original_idx])
        print(f"Inner product: {h0_inner:.6f}")
        
        ###################################################
        # Step 1, 2, ... N: Iterate through remaining modes
        ###################################################  

        # Iterate till N_max to fulfill cond. <h_i|h_i> > 1 
        for i in range(1, len(power_values)):   
            # Check if we have reached the target number of modes
            if len(selected_modes) >= M_sel:
                print(f"\nReached target of {M_sel} selected modes.")
                break
                
            # Get the next candidate mode h_j'
            hj_prime_label = mode_labels[i]
            hj_prime_original_idx = mode_label_to_original_idx[hj_prime_label]
            processed_indices.append(i)
            
            print(f"\n--- Iteration {i} ---")
            print(f"Currently have {len(selected_modes)} selected modes, target is {M_sel}")
            print(f"Candidate mode h_j': {hj_prime_label} with power value {power_values[i]:.4e}")
            
            # Check max inner product w/ selected modes 
            max_inner = 0
            # Initialize index of mode w/ max inner product
            max_inner_idx = -1  
            
            for k, selected_mode in enumerate(selected_modes):
                # Calculate |<h_selected|h_j'>|
                print(f"  Calculating inner product with selected mode", selected_mode, "and hj_prime_idx:", hj_prime_original_idx)
                # Use calc_power_inner to get the inner product
                cross_term_inner = self.calc_inner_power(selected_mode, [hj_prime_original_idx])
                self_selmode_inner = self.calc_inner_power(selected_mode, selected_mode)
                self_hjprime_inner = self.calc_inner_power([hj_prime_original_idx], [hj_prime_original_idx])
                calc_inner = abs(cross_term_inner/ np.sqrt(self_selmode_inner * self_hjprime_inner))
                print(calc_inner)
                print(f"  calc_inner with selected mode {k} ({selected_labels[k]}): {calc_inner:.6f}")

                # Update max inner product and index if this is the largest so far
                if calc_inner > max_inner:
                    max_inner = calc_inner
                    max_inner_idx = k
            
            # Check if max inner prod is below threshold
            if max_inner < inner_threshold:
                # Accept the mode
                selected_modes.append([hj_prime_original_idx])  # Store as a list of original indices
                selected_indices.append(i)  # Store sorted index
                selected_labels.append(hj_prime_label)
                
                # Initialize tracking for this new selected mode
                combined_modes_tracker[len(selected_modes)-1] = {
                    'original': hj_prime_label, 
                    'combined_with': []
                }
                
                print(f"  ACCEPTED: Max inner product {max_inner:.6f} < {inner_threshold}")
                print(f"  Added mode: {hj_prime_label} (Total selected: {len(selected_modes)})")
            else:
                # Reject and add to most correlated mode
                print(f"  REJECTED: Max inner product {max_inner:.6f} >= {inner_threshold}")
                print(f"  Most correlated with: {selected_labels[max_inner_idx]}")
                
                # Track which mode was combined
                combined_modes_tracker[max_inner_idx]['combined_with'].append(hj_prime_label)
                
                # Add h_j' to the most correlated mode h_k
                selected_modes[max_inner_idx].append(hj_prime_original_idx)
                combination_counts['rejected_modes'] += 1
                print(f"  Combined with mode: {selected_labels[max_inner_idx]}")
        
        ##########################################
        # Step N+1: Handle remaining modes as h_M
        ##########################################

        print(f"\n" + "="*50)
        print("STEP 2: Processing remaining modes as h_M")
        print("="*50)
        
        # Find all remaining modes not processed yet (using sorted indices)
        all_sorted_indices = set(range(len(power_values)))
        remaining_sorted_indices = list(all_sorted_indices - set(processed_indices))
        
        # Only process if there are remaining modes
        if remaining_sorted_indices:
            print(f"There are {len(remaining_sorted_indices)} remaining modes to combine into h_M")
            
            # Convert to original indices and get labels
            remaining_original_indices = [mode_label_to_original_idx[mode_labels[idx]] for idx in remaining_sorted_indices]
            remaining_mode_labels = [mode_labels[idx] for idx in remaining_sorted_indices]
            print(f"Remaining modes: {remaining_mode_labels}")
            
            # Count combinations for h_M creation
            combination_counts['remaining_to_hM'] += len(remaining_original_indices) - 1
            
            # Check condition 1: <h_M|h_M> > 1 : True/False
            h_M_inner_product = self.calc_inner_power(remaining_original_indices, remaining_original_indices)
            cond_one = h_M_inner_product > 1
            print(f"Condition 1: <h_M|h_M> = {h_M_inner_product:.6f} > 1? {cond_one}")
            
            # Check condition 2: <h_i|h_M> << 1 for all selected modes
            # Check inner products with selected modes and get the maximum inner product
            inners_with_selected = []
            max_inner_with_selected = 0
            max_inner_with_selected_idx = -1
            
            for k, selected_mode in enumerate(selected_modes):
                calc_inner = self.calc_inner_power(selected_mode, remaining_original_indices)
                inners_with_selected.append(calc_inner)
                print(f"  <h_{k}|h_M> = {calc_inner:.6f}")
                
                # Update max inner product and index if this is the largest so far
                if calc_inner > max_inner_with_selected:
                    max_inner_with_selected = calc_inner
                    max_inner_with_selected_idx = k
            
            # Check if max inner product with selected modes is below threshold
            cond_two = max_inner_with_selected < inner_threshold
            print(f"Condition 2: max inner = {max_inner_with_selected:.6f} < {inner_threshold}? {cond_two}")
            
            # Decision logic
            print(f"\nDecision for h_M:")
            if cond_one and cond_two: # If both conditions are satisfied
                # Accept h_M as an extra mode
                selected_modes.append(remaining_original_indices)
                selected_indices.append(-1)  # Special index for h_M
                selected_labels.append("h_M (remaining modes)")
                
                # Initialize tracking for h_M
                combined_modes_tracker[len(selected_modes)-1] = {
                    'original': 'h_M', 
                    'combined_with': remaining_mode_labels
                }
                
                print(f"  ACCEPTED h_M: Both conditions satisfied")
                print(f"  Added h_M as extra mode (Total selected: {len(selected_modes)})")
                
            elif not cond_one and cond_two: # If condition 1 is violated but condition 2 is satisfied
                # Throw away h_M (becomes error term)
                print(f"  DISCARDED h_M: Condition 1 violated, Condition 2 satisfied")
                print(f"  h_M becomes error/epsilon term")
                 
            elif cond_one and not cond_two: # If condition 1 is satisfied but condition 2 is violated
                # Add h_M to most correlated selected mode
                print(f"  COMBINED h_M: Condition 1 satisfied, Condition 2 violated")
                print(f"  Most correlated with: {selected_labels[max_inner_with_selected_idx]}")
                
                # Track h_M combination
                combined_modes_tracker[max_inner_with_selected_idx]['combined_with'].extend(remaining_mode_labels)
                
                selected_modes[max_inner_with_selected_idx].extend(remaining_original_indices)
                combination_counts['hM_with_selected'] += 1 
                print(f"  Combined h_M with: {selected_labels[max_inner_with_selected_idx]}")
                
            else:  # If both conditions are violated
                # Throw away h_M
                print(f"  DISCARDED h_M: Both conditions violated")
                print(f"  h_M is thrown away")
        else:
            print("No remaining modes to process as h_M")
        
        if len(selected_modes) >= M_sel:
            print(f"\nReached target of {M_sel} selected modes.")
        else:
            print(f"\nProcessed all modes. Selected {len(selected_modes)} modes.")

        # Calculate total combinations
        combination_counts['total'] = (combination_counts['rejected_modes'] + 
                                      combination_counts['remaining_to_hM'] + 
                                      combination_counts['hM_with_selected'])
        
        ##########################################################
        # Calculate final inner products of selected modes
        ##########################################################
        print(f"\n" + "="*50)
        print("FINAL INNER PRODUCTS OF SELECTED MODES:")
        print("="*50)
        
        final_inner_products = []
        total_final_inner = 0
        
        for i, mode in enumerate(selected_modes):
            final_ip = self.calc_inner_power(mode, mode)
            final_inner_products.append(final_ip)
            total_final_inner += final_ip
            
            # Get combination info for this mode
            mode_info = combined_modes_tracker[i]
            combined_info = ""
            if mode_info['combined_with']:
                combined_info = f" + {len(mode_info['combined_with'])} modes: {mode_info['combined_with']}"
            
            print(f"  <h_{i}|h_{i}> = {final_ip:.6f} | {mode_info['original']}{combined_info}")
        
        print(f"\nTotal final inner product: {total_final_inner:.6f}")
        print(f"Square root of total: {np.sqrt(total_final_inner):.6f}")
        
        ##########################################################
        # Detailed combination breakdown
        ##########################################################
        print(f"\n" + "="*50)
        print("COMBINATION BREAKDOWN:")
        print("="*50)
        
        for i, mode_info in combined_modes_tracker.items():
            if mode_info['combined_with']:
                print(f"Selected mode {i} ({mode_info['original']}):")
                print(f"  Combined with {len(mode_info['combined_with'])} modes: {mode_info['combined_with']}")
            else:
                print(f"Selected mode {i} ({mode_info['original']}): No combinations")
        
        print(f"\n" + "="*50)
        print("COMBINATION SUMMARY:")
        print("="*50)
        print(f"Rejected modes combined: {combination_counts['rejected_modes']}")
        print(f"Remaining modes combined into h_M: {combination_counts['remaining_to_hM']}")
        print(f"h_M combined with selected: {combination_counts['hM_with_selected']}")
        print(f"TOTAL COMBINATIONS: {combination_counts['total']}")
        
        return selected_modes, selected_indices, selected_labels, combination_counts, final_inner_products