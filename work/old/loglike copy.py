from modeselector import ModeSelector
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI

class LogLike:
    """
    Log-likelihood class for gravitational wave parameter estimation.
    TODO: save modes, signal as we move through the parameter space
    NOTE: at the start, we need to compute the mode basis (mode selection at the start) -> save mode indices/labels
    shall we need to generate modes when we move far away from the initial point?

    """

    
    def __init__(self, params, waveform_gen, gwf, dt=None, T=None, M_init = 100, M_mode=5, N_traj = 5000, threshold=0.01, verbose=False):
        """
        Initialize the LogLike class.
        
        Parameters and some notes:
        - params: List of DATA/SIGNAL parameters 
                  [m1, m2, a, p0, e0, xI0, theta, phi, dist]
        - waveform_gen: Waveform generator object
        - gwf: GravWaveAnalysis object 
        - dt: Time step (uses gwf if None)
        - T: Total time (uses gwf if None)
        - M_init: Initial number of modes to consider for selection for modeselector class (default = 100)
        - M_mode: Number of modes to generate at each point (default = 5)
        - N_traj: Number of trajectory points for mode selection (default = 5000)
        - threshold: Threshold for mode selection (default = 0.01)
        - verbose: Whether to print debug information during mode selection (default = False)
        TODO: Save mode info as we move?
        """
        self.params = params
        self.waveform_gen = waveform_gen
        self.M_init = M_init
        self.M_mode = M_mode
        self.N_traj = N_traj
        self.threshold = threshold
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

        # Generate data signal with data parameters
        # TODO: change every dependancy to detector frame
        # OLD: m1, m2, a, p0, e0, xI0, theta, phi, dist = params
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params
        # OLD: self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, theta, phi, dist=dist, dt=self.dt, T=self.T)

        # Generate w GenerateEMRIWaveform
        self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, dist, 
                                   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, 
                                   T=self.T, dt=self.dt)

        # Save dominant mode history 
        self.history = []
        
        # Cache for selected modes to avoid repeated computation
        # TODO: actually check if this works?
        self._mode_cache = {}

    def _generate_selected_waveforms(self, params, selected_labels):
        """Generate waveforms for selected mode groups."""
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        waveforms_per_group = []


        for group in selected_labels:
            h_group = self.waveform_gen(
                m1, m2, a, p0, e0, xI0, dist, 
                qS, phiS, qK,phiK,
                Phi_phi0, Phi_theta0, Phi_r0, 
                dt=self.dt,
                T=self.T,
                mode_selection=group, 
                include_minus_mkn=False,
            )

            waveforms_per_group.append(h_group)

        return waveforms_per_group
    
    def __call__(self, theta_template):
        """
        Evaluate log-likelihood for template parameters.
        
        Parameters:
        theta_template: Template parameters (m1, m2, a, p0, e0, xI0, theta, phi, dist)
        
        Returns:
        float: Log-likelihood value
        """
        # TODO: change all dependancy to detector frame

        # if self.verbose:
        #     print(f"DEBUG: theta_template shape: {theta_template.shape}")
        #     print(f"DEBUG: theta_template: {theta_template}")
        mode_selector = ModeSelector(theta_template, self.traj, self.amp, self.ylm_gen, self.delta_T, self.gwf, verbose=self.verbose)
            
        # Perform mode selection for theta_template
        selected_modes, selected_labels = mode_selector.select_modes(
            M_init=self.M_init,
            M_sel=self.M_mode,
            threshold=self.threshold
        )

        # Generate waveforms for selected modes using theta_template
        waveform_per_mode = self._generate_selected_waveforms(theta_template, selected_labels)

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

        # Save history
        self.history.append({
            'params': theta_template,
            'dom_mode': selected_labels[0] #or [0][0]? TODO check
        })

        return logl_res