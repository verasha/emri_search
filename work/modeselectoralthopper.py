import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from lisatools.sensitivity import get_sensitivity, CornishLISASens
from few.utils.geodesic import get_fundamental_frequencies


class ModeSelector:
    """ Class to select modes based on a given threshold. """

    # Physical constants
    MTSUN_SI = 4.925491025543576e-06 # Mass-time conversion factor in seconds

    def __init__(self, params, traj, amp, ylm_gen, delta_T, gwf, verbose=False):
        """ Initialize the ModeSelector with the provided parameters. 
        Parameters and some notes:
        - params: List of parameters 
                  [NOTE same order as loglike] 
                  [m1, m2, a, p0, e0, xI0, theta, phi, dist]
        - traj: The trajectory module with delta_T (NOT the finer dt)
                NOTE: easier way would be to access it from waveform_gen
                      but in this case we specify diff args for traj module
        
        - amp: Amplitude module 
        - delta_T: Time step used to generate the Teukolsky modes. 
                    NOT the same as dt used for interpolation
        - gwf: GravWaveAnalysis object 
        - verbose: Whether to print debug information during mode selection
        - sensitivity_fn: Function that takes frequency array and returns PSD for noise weighting

        TODO: calc gw_freqs and gw_phases inside this class? 
        maybe pass parameter set as a vector instead to simplify things...
        """
        self.params = params
        self.traj = traj
        self.amp = amp
        self.ylm_gen = ylm_gen
        self.delta_T = delta_T
        self.gwf = gwf
        self.verbose = verbose

        # Caching
        self._traj_data = None
        self._teuk_modes = None
        self._gw_freqs = None
        self._gw_phases = None

    def _get_viewing_angles(self, qS, phiS, qK, phiK):
        """Transform from the detector frame to the source  frame"""
        cqS = np.cos(qS)
        sqS = np.sin(qS)
        cphiS = np.cos(phiS)
        sphiS = np.sin(phiS)
        cqK = np.cos(qK)
        sqK = np.sin(qK)
        cphiK = np.cos(phiK)
        sphiK = np.sin(phiK)
        # sky location vector
        R = np.array([sqS * cphiS, sqS * sphiS, cqS])
        # spin vector
        S = np.array([sqK * cphiK, sqK * sphiK, cqK])
        # get viewing angles
        phi = -np.pi / 2.0  # by definition of source frame
        theta = np.arccos(-np.dot(R, S))  # normalized vector

        return (theta, phi)

    def _calc_traj(self):
        """ Calculate trajectory """
        if self._traj_data is None:
            # NOTE: OLD
            # m1, m2, a, p0, e0, xI0, _, _, _ = self.params
            # self._traj_data = self.traj(m1, m2, a, p0, e0, xI0, T=self.gwf.T, dt=self.delta_T, upsample=True)
            m1, m2, a, p0, e0, xI0, _, _, _, _, _, Phi_phi0, Phi_theta0, Phi_r0 = self.params
            self._traj_data = self.traj(m1, m2, a, p0, e0, xI0, 
                                       T=self.gwf.T, dt=self.delta_T, upsample=True, 
                                       Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0)    
        return self._traj_data


    @property
    def teuk_modes(self):
        """ Calculate Teukolsky modes """
        if self._teuk_modes is None:
            _, p, e, x, _, _, _ = self._calc_traj()
            # OLD: _, _, a, _, _, _, _, _, _ = self.params
            _, _, a, _, _, _, _, _,  _, _, _, _, _, _ = self.params
            self._teuk_modes = self.amp(a, p, e, x)
        return self._teuk_modes
    
    @property
    def ylms(self):
        """ Calculate spherical harmonics """
        if not hasattr(self, '_ylms'):
            # OLD: _, _, _, _, _, _, theta, phi, _ = self.params
            _, _, _, _, _, _, _, qS, phiS, qK, phiK, _, _, _ = self.params
            theta, phi = self._get_viewing_angles(qS, phiS, qK, phiK)
            self._ylms = self.ylm_gen(self.amp.unique_l, self.amp.unique_m, theta, phi).copy()[self.amp.inverse_lm]
        return self._ylms

    @property
    def gw_freqs(self):
        """ Calculate GW frequencies """
        if self._gw_freqs is None:
            _, p, e, x, _, _, _ = self._calc_traj()
            _, _, a, _, _, _, _, _, _, _, _, _, _, _ = self.params

            # Get fundamental frequencies
            # TODO make it GPU compatible (i always get a problem???)
            OmegaPhi, _, OmegaR = get_fundamental_frequencies(a, p, e, x)

            gw_frequencies_per_mode = []
            for idx in range(len(self.amp.l_arr)):
                m = self.amp.m_arr[idx]
                n = self.amp.n_arr[idx]

                m = m.get() if isinstance(m, cp.ndarray) else m
                n = n.get() if isinstance(n, cp.ndarray) else n

                # Calculate GW frequencies
                f_gw = m * OmegaPhi + n * OmegaR
                gw_frequencies_per_mode.append(f_gw)

            self._gw_freqs = gw_frequencies_per_mode
        return self._gw_freqs

    @property
    def gw_phases(self):
        """ Calculate GW phases """
        if self._gw_phases is None:
            _, _, _, _, Phi_phi, _, Phi_r = self._calc_traj()

            gw_phases_per_mode = []
            for idx in range(len(self.amp.l_arr)):
                m = self.amp.m_arr[idx]
                n = self.amp.n_arr[idx]

                m = m.get() if isinstance(m, cp.ndarray) else m
                n = n.get() if isinstance(n, cp.ndarray) else n

                # Calculate GW phases
                phase_gw = m * Phi_phi + n * Phi_r  # k * Phi_theta = 0 for equatorial
                gw_phases_per_mode.append(phase_gw)

            self._gw_phases = gw_phases_per_mode
        return self._gw_phases
    
    @property
    def factor(self):
        if not hasattr(self, '_factor'):
            # Calculate the distance factor
            m1, m2, _, _, _, _, dist, _, _, _, _, _, _, _ = self.params
            self._factor = self.gwf.dist_factor(dist, m1, m2)
        return self._factor
         


    def inner_approx(self, mode_i, mode_j): 
        """ Calculate the approximate inner product of two modes.
        Based on Eqn. 18-20 in arxiv 2109.14254 [non-local parameter degeneracy paper]

        Parameters:
        mode_i, mode_j: Lists of mode indices (so can be multiple, 
                        this is so that we can calculate
                        the inner product of combinations of modes)"""
        
        # Initilize the inner product
        total_inner = 0.0

        # Loop over all the modes
        for idx_i in mode_i:
            for idx_j in mode_j:
                # Obtain the lmns 
                l_i = self.amp.l_arr[idx_i]
                m_i = self.amp.m_arr[idx_i]
                n_i = self.amp.n_arr[idx_i] 
            
                l_j = self.amp.l_arr[idx_j]
                m_j = self.amp.m_arr[idx_j]
                n_j = self.amp.n_arr[idx_j]

                # Get Teukolsky modes
                # Check if negative m -> use conjugate of positive m mode
                if m_i >= 0:
                    A_i = self.teuk_modes[:, idx_i]
            
                elif m_i < 0:
                    pos_m_mask_i = (self.amp.l_arr == l_i) \
                                    & (self.amp.m_arr == -m_i) \
                                    & (self.amp.n_arr == -n_i)
                    pos_m_idx_i = np.where(pos_m_mask_i)[0][0]
                    A_i_pos = self.teuk_modes[:, pos_m_idx_i]
                    A_i = (-1)**l_i * self.gwf.xp.conj(A_i_pos)
            
                if m_j >= 0:
                    A_j = self.teuk_modes[:, idx_j]
                    
                elif m_j < 0:
                    pos_m_mask_j = (self.amp.l_arr == l_j) \
                                    & (self.amp.m_arr == -m_j) \
                                    & (self.amp.n_arr == -n_j)
                    pos_m_idx_j = np.where(pos_m_mask_j)[0][0]
                    A_j_pos = self.teuk_modes[:, pos_m_idx_j]
                    A_j = (-1)**l_j * self.gwf.xp.conj(A_j_pos)

                # Total mass
                m1, m2 = self.params[0], self.params[1]
                M_tot = m1 + m2
                M_sec = M_tot * self.MTSUN_SI  # in seconds

                # Convert dimensionless freq to Hz
                freq_hz_i = np.abs(self.gw_freqs[idx_i]) / (2 * np.pi * M_sec)
                freq_hz_j = np.abs(self.gw_freqs[idx_j]) / (2 * np.pi * M_sec)

                # NOTE: do i need to make all freqs positive here???
                # Get sensitivity for each mode 
                Sn_i = get_sensitivity(freq_hz_i, 
                                       sens_fn=CornishLISASens, 
                                       return_type="PSD"
                                       )
                
                Sn_j = get_sensitivity(freq_hz_j, 
                                        sens_fn=CornishLISASens, 
                                        return_type="PSD"
                                        )

                # Get noise-weighted amplitudes
                # TODO: make this more compatible GPU-wise?
                bar_A_i = A_i.get() / np.sqrt(Sn_i)
                bar_A_j = A_j.get() / np.sqrt(Sn_j)

                # Define mask where the phase difference is small
                phase_mask = np.abs(self.gw_phases[idx_i] - self.gw_phases[idx_j]) < 1.0 

                # Calculate the product of the two waveforms w/ the phase mask
                prod = np.conj(bar_A_i[phase_mask]) * bar_A_j[phase_mask]
            
                # Calculate full inner product
                innerprod = np.sum(np.real(prod)) * self.delta_T * 1/(self.factor**2)

                # Add to the total inner product
                total_inner += innerprod

        return total_inner
            
    def SNR_approx(self, mode_idx):
        """ Calculate approximate SNR. """
        return np.sqrt(self.inner_approx(mode_idx, mode_idx))
    
    def overlap_approx(self, mode_i, mode_j):
        """ Calculate approximate overlap between two modes. """
        return self.inner_approx(mode_i, mode_j) / (self.SNR_approx(mode_i) * self.SNR_approx(mode_j))
    
    def select_modes(self, 
                     M_sel=5, 
                     n_vals=None,
                        ell=2
                    ):
        """ Select modes indexed by n.
        For each n, sum over all m for fixed l.
        M_sel = number of modes to select in the end (default: 5)
        n_vals = n values to consider
        ell = angular mode l to consider
        """

        if n_vals is None:
            n_vals = np.arange(-1, 3) # n from -1 to 2

        selected_modes = []
        selected_labels = []

        for n in n_vals:
            # find all modes with given l and n
            mask = (self.amp.l_arr == ell) & (self.amp.n_arr == n)
            # print(type(mask), type(self.amp.l_arr), type(self.amp.n_arr))
            mode_indices = np.where(mask)[0]

            # convert to cpu if needed
            mode_indices = mode_indices.get() if isinstance(mode_indices, cp.ndarray) else mode_indices
            if len(mode_indices) > 0:
                # get labels
                mode_labels = []
                for idx in mode_indices:
                    l_int = self.amp.l_arr[idx].item() if hasattr(self.amp.l_arr[idx], 'item') else int(self.amp.l_arr[idx])  # FIX
                    m_int = self.amp.m_arr[idx].item() if hasattr(self.amp.m_arr[idx], 'item') else int(self.amp.m_arr[idx])  # FIX
                    n_int = int(n)  
                    mode_labels.append((l_int, m_int, n_int))
                selected_modes.append(mode_indices.tolist())
                selected_labels.append(mode_labels)
        # sort by approximate SNR and select top M_sel
        if M_sel is not None and M_sel < len(selected_modes):
            snr_list = []
            for mode_idx in selected_modes:  
                snr = self.SNR_approx(mode_idx)
                snr_list.append(snr)
            
            snr_array = np.array(snr_list)
            top_indices = np.argsort(snr_array)[-M_sel:][::-1]

            selected_modes = [selected_modes[i] for i in top_indices]
            selected_labels = [selected_labels[i] for i in top_indices]


        if self.verbose:
            print("Selected modes (l,m,n):", selected_labels)

        return selected_modes, selected_labels