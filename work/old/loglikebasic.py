import numpy as np
from modeselector import ModeSelector
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI

class LogLike:
    """
    Log-likelihood class for gravitational wave parameter estimation.
    TODO: save modes, signal as we move through the parameter space
    NOTE: at the start, we need to compute the mode basis (mode selection at the start) -> save mode indices/labels
    shall we need to generate modes when we move far away from the initial point?

    """

    
    def __init__(self, params, waveform_gen, gwf, M_init = 100, M_mode=5, N_traj = 5000, mode_threshold=0.01, verbose=False, waveform_gen_sep=None, noise_weighted=False):
        """
        Initialize the LogLike class.
        
        Parameters and some notes:
        - params: List of DATA/SIGNAL parameters 
                  [m1, m2, a, p0, e0, xI0, theta, phi, dist]
        - waveform_gen: Waveform generator object (separate_modes=TRUEEE!!!)
        - gwf: GravWaveAnalysis object 
        - dt: Time step (uses gwf if None)
        - T: Total time (uses gwf if None)
        - M_init: Initial number of modes to consider for selection for modeselector class (default = 100)
        - M_mode: Number of modes to generate at each point (default = 5)
        - N_traj: Number of trajectory points for mode selection (default = 5000)
        - mode_threshold: Threshold for mode selection (default = 0.01)
        - verbose: Whether to print debug information during mode selection (default = False)
        TODO: Save mode info as we move?
        """
        self.params = params
        self.waveform_gen = waveform_gen
        self.waveform_gen_sep = waveform_gen_sep if waveform_gen_sep is not None else waveform_gen
        self.M_init = M_init
        self.M_mode = M_mode
        self.N_traj = N_traj
        self.mode_threshold = mode_threshold
        self.verbose = verbose
        
        # Initialize based on GravWaveAnalysis object
        self.gwf = gwf
        self.dt = gwf.dt
        self.T = gwf.T
        
        # Get components from the waveform generator 
        self.traj = getattr(waveform_gen.waveform_generator, 'inspiral_generator', None)
        self.amp = getattr(waveform_gen.waveform_generator, 'amplitude_generator', None)
        self.interpolate_mode_sum = getattr(waveform_gen.waveform_generator, 'create_waveform', None)
        self.ylm_gen = getattr(waveform_gen.waveform_generator, 'ylm_gen', None)
    
        # Calc delta_T
        self.delta_T = self.T * YRSID_SI / self.N_traj
        if self.verbose:
            print(f"Delta_T for mode selection: {self.delta_T} seconds")

        # Generate data signal with data parameters
        # OLD: m1, m2, a, p0, e0, xI0, theta, phi, dist = params
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params
        # OLD: self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=self.dt, T=self.T)

        # Generate w GenerateEMRIWaveform
        self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, dist, 
                                   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, 
                                   T=self.T, dt=self.dt)

        # # Save dominant mode history 
        # self.history = []
        
        # Do mode selection once at initialization with data parameters
        if self.verbose:
            print("Generating modes at initialization...")
        
        mode_selector = ModeSelector(self.params, self.traj, self.amp, self.ylm_gen, self.delta_T, self.gwf, verbose=self.verbose)
        
        # Perform mode selection for data parameters
        self.selected_modes, self.selected_labels = mode_selector.select_modes(
            M_init=self.M_init,
            M_sel=self.M_mode,
            threshold=self.mode_threshold,
            noise_weighted=noise_weighted
        )
        
        # Flatten the nested structure for waveform generation
        self.flattened_modes = []
        for group in self.selected_labels:
            self.flattened_modes.extend(group)
        
        # Debug mode selection
        if self.verbose:
            print(f"Selected modes: {self.selected_labels}")
            print(f"Number of selected modes: {len(self.selected_labels)}")
            print(f"Flattened modes: {self.flattened_modes}")
            print(f"Selected modes structure: {self.selected_modes}")

    def _generate_selected_waveforms(self, params, selected_labels):
        """Generate waveforms for selected mode groups."""
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        # NOTE: changing to simple separate_modes feature
        # TODO: not robust for negative modes!! FUCK
        # waveforms_per_group = []
        waveforms = self.waveform_gen_sep(
            m1, m2, a, p0, e0, xI0, dist, 
            qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0, 
            dt=self.dt,
            T=self.T,
            mode_selection=selected_labels, 
            include_minus_mkn=False,
        )

        # Sort waveforms by index_map: hundreds ascending, then remainder descending
        sorted_indices = sorted(range(len(selected_labels)), 
                            key=lambda i: (self.amp.index_map[selected_labels[i]] % 1000 // 100, 
                                            -(self.amp.index_map[selected_labels[i]] % 100)))
        waveforms = waveforms[:, sorted_indices]        
        return waveforms

        # waveforms_per_group = []
        # for group in selected_labels:
        #     h_group = self.waveform_gen(
        #         m1, m2, a, p0, e0, xI0, dist, 
        #         qS, phiS, qK, phiK,
        #         Phi_phi0, Phi_theta0, Phi_r0, 
        #         dt=self.dt,
        #         T=self.T,
        #         mode_selection=group, 
        #         include_minus_mkn=False,
        #     )

        #     waveforms_per_group.append(h_group)

        # return waveforms_per_group
    
    
    def __call__(self, theta_template):
        """
        Evaluate log-likelihood for template parameters.
        
        Parameters:
        theta_template: Template parameters (m1, m2, a, p0, e0, xI0, theta, phi, dist)
        
        Returns:
        float: Log-likelihood value
        """
        # TODO: change all dependancy to detector frame

        # Use the pre-selected modes from initialization
        if self.verbose:
            print(f"Evaluating log-likelihood at parameters: {theta_template}")
            print(f"Using selected modes: {self.selected_labels}")
        waveform_combined = self._generate_selected_waveforms(theta_template, self.flattened_modes)

        # Debug waveform generation
        # if self.verbose:
        #     print(f"waveform_combined type: {type(waveform_combined)}")
        #     print(f"waveform_combined shape: {waveform_combined.shape if hasattr(waveform_combined, 'shape') else 'No shape'}")
            
        # Split the combined waveform into individual mode waveforms
        waveform_per_mode = []
        for i in range(waveform_combined.shape[1]):  # 5 modes
            waveform_per_mode.append(waveform_combined[:, i])
        
        # print(f"Split into {len(waveform_per_mode)} individual waveforms")
        # for i, wf in enumerate(waveform_per_mode):
        #     print(f"  Mode {i} waveform shape: {wf.shape}")

        # Calculate rho_m
        rho_m = self.gwf.rhostat_modes(waveform_per_mode)
        
        # Find actual dominant mode (always calculate, regardless of verbose)
        max_rho_idx = rho_m.argmax()
        
        # Debug inner product calculations
        if self.verbose:
            print(f"Waveform amplitudes:")
            for i, wf in enumerate(waveform_per_mode):
                wf_max = self.gwf.xp.max(self.gwf.xp.abs(wf))
                # Calculate inner product manually for debugging
                hf = self.gwf.freq_wave(wf)
                inner_prod = self.gwf.inner(hf, hf)
                print(f"  Mode {i}: max(|h|) = {wf_max}, <hf|hf> = {inner_prod}, rho = {rho_m[i]}")
                
            # Debug rho values
            print(f"Individual rho values: {rho_m}")
            print(f"Max rho: {max(rho_m)}, Min rho: {min(rho_m)}")
            print(f"Dominant mode rho (first): {rho_m[0]}")
            
            # Find actual dominant mode
            max_rho_idx = rho_m.argmax()
            print(f"Actually dominant mode index: {max_rho_idx}")
            print(f"Actually dominant mode rho: {rho_m[max_rho_idx]}")
            print(f"Mode 0 rho: {rho_m[0]}")
        
        # Calculate Xm 
        X_modes = self.gwf.Xmstat(self.signal, waveform_per_mode, rho_m)

        # Calculate X_scalar
        Xdotrho = self.gwf.xp.sum(X_modes * rho_m)
        if self.verbose:
            print(f"Xdotrho: {Xdotrho}")
        rho_tot = self.gwf.xp.sqrt(self.gwf.xp.sum(rho_m**2))
        # rho_tot can be calculated from h_temp for accuracy
        # but we're using the approximation via modes here 
        # TODO: ? should we use h_temp to calculate rho_tot?
        X_scalar = Xdotrho / rho_tot

        # Check actual Xstat value
        # Need to generate h_temp though
        # X_check = self.gwf.Xstat(self.signal, h_temp)

        # Calculate optimal SNR of actually most dominant mode (not just mode 0)
        rho_dom_M = rho_m[max_rho_idx]
        # print(f"Using actually dominant mode {max_rho_idx} for rho_dom_M: {rho_dom_M}")
        # rho_dom_M = self.gwf.rhostat(waveform_per_mode[0])
        if self.verbose:
            print(f"Using actually dominant mode {max_rho_idx} for rho_dom_M: {rho_dom_M}")

        # Calculate beta parameter
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        if self.verbose:
            print('beta', beta)
            print(f"rho_dom_M: {rho_dom_M}, rho_tot: {rho_tot}, beta: {beta}")
            print(f"X_scalar: {X_scalar}")

        # Calculate chi sq
        chi_sq = self.gwf.chi_sq(X_modes, rho_m)
        if self.verbose:
            print(f"chi_sq: {chi_sq}")
        # Calculate f statistic 
        f_exp = -0.5 * beta * chi_sq 
        if self.verbose:
            print(f"f_exp: {f_exp}")

        # Overflow protection for large exponentials 
        # # TODO: Check better way to handle this, ex: mpmath?    
        # if f_exp > 700:  
        #     f_exp = 700 # Cap 
        # elif f_exp < -700:
        #     return -700.0
        
        f_stat = X_scalar * self.gwf.xp.exp(f_exp)

        # Convert to float, handling both GPU and CPU backends
        f_stat_real = self.gwf.xp.real(f_stat)
        if hasattr(f_stat_real, 'get'):  # CuPy array
            logl_res = float(f_stat_real.get())
        else:  # NumPy array
            logl_res = float(f_stat_real)
        
        # # Check for NaNs 
        # if self.gwf.xp.isnan(logl_res):
        #     return -700.0

        # # Save history
        # self.history.append({
        #     'params': theta_template,
        #     'dom_mode': selected_labels[0] #or [0][0]? TODO check
        # })
        if self.verbose:
            print(f"Log-likelihood: {logl_res}")

        return logl_res