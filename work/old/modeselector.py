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
                    pos_m_idx_i = self.gwf.xp.where(pos_m_mask_i)[0][0]
                    A_i_pos = self.teuk_modes[:, pos_m_idx_i]
                    A_i = (-1)**l_i * self.gwf.xp.conj(A_i_pos)
            
                if m_j >= 0:
                    A_j = self.teuk_modes[:, idx_j]
                    
                elif m_j < 0:
                    pos_m_mask_j = (self.amp.l_arr == l_j) \
                                    & (self.amp.m_arr == -m_j) \
                                    & (self.amp.n_arr == -n_j)
                    pos_m_idx_j = self.gwf.xp.where(pos_m_mask_j)[0][0]
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
                     M_init=100, 
                     M_sel=5, 
                     threshold=0.01,
                     noise_weighted=False
                    ):
        """ Select modes based on a given threshold.
        M_init = number of modes to select initially (for the power sorting)
        M_sel = number of modes to select in the end (default: 5)
        threshold = for accept/reject cond. of inner products between modes
        noise_weighted = whether to use noise-weighted power for initial sorting
        """

        ###### Step 0: Initialization and Setup ######

        # mode_labels =  [(l,m,n) for l,m,n in zip(self.amp.l_arr, self.amp.m_arr, self.amp.n_arr)]
        mode_labels = []
        for l, m, n in zip(self.amp.l_arr, self.amp.m_arr, self.amp.n_arr):
            l_int = l.item() if hasattr(l, 'item') else int(l)
            m_int = m.item() if hasattr(m, 'item') else int(m)
            n_int = n.item() if hasattr(n, 'item') else int(n)
            mode_labels.append((l_int, m_int, n_int))

        # TODO: add mode info here during search 

        # Calculate power and sort 
        m0mask = self.amp.m_arr_no_mask != 0
        if noise_weighted:
            total_power = self.gwf.calc_power(self.teuk_modes, self.ylms, m0mask, 
                                             m1=self.params[0], m2=self.params[1], 
                                             gw_freqs=self.gw_freqs)
        else:
            total_power = self.gwf.calc_power(self.teuk_modes, self.ylms, m0mask)

        # Top M_init indices in descending order (based on power)
        top_indices = self.gwf.xp.argsort(total_power)[-M_init:][::-1] 
        top_indices = top_indices.get() if isinstance(top_indices, cp.ndarray) else top_indices

        # Get sorted mode labels and power values
        # TODO: noise-weighted power mode selection?
        mp_modes = [mode_labels[idx] for idx in top_indices]
        mp_power = total_power[top_indices]

        ### Initialize selected set S with h0

        # Using the original index of the mode for the teuk_modes, amp..
        selected_modes = [[top_indices[0]]] 

        # Below is using h0 index in the SORTED list
        selected_labels = [[mp_modes[0]]]

        # Keep track of all processed modes (using ORIGINAL indices)
        processed_ori_indices = [top_indices[0]]  
        if self.verbose:
            print(f"Initial mode selected: {mp_modes[0]} with power {mp_power[0]}")

        ###### Step 1, 2, ... N
        # Iterate through remaining modes on the sorted list ######
        
        # Iterate till M_init to to fulfill cond. <h_i|h_i> > 1 
        # Do note the idx i runs through the sorted indices
        for i in range(1, M_init):
            # Break if M_sel is reached 
            if len(selected_modes) >= M_sel:
                break

            if self.verbose:
                print(f"Considering mode {i} / {M_init} : {mp_modes[i]} with power {mp_power[i]}")
        
            # Get next candidate mode h_j'
            hj_prime_idx = top_indices[i]
            hj_prime_label = mp_modes[i]
            
            # Keep track of processed indices
            processed_ori_indices.append(hj_prime_idx)

            max_overlap = 0 
            max_overlap_idx = -1 

            for k, selected_mode in enumerate(selected_modes):
                # Calculate with each selected mode |<h_sel|h_j'>/(<h_sel|h_sel>*<h_j'|h_j'>)^(1/2)|
                # Basically abs of overlap 
                calc_overlap = abs(self.overlap_approx(selected_mode, [hj_prime_idx]))
                if self.verbose:
                    print(f" - Overlap with selected mode {k}: {calc_overlap}")

                
                # Check if this is the maximum overlap found so far
                if calc_overlap > max_overlap:
                    max_overlap = calc_overlap
                    max_overlap_idx = k

            # Check if the maximum overlap is below the threshold
            if max_overlap < threshold:
                # Fulfill cond: Accept the mode
                selected_modes.append([hj_prime_idx])
                selected_labels.append([hj_prime_label])
            
            else:
                # Doesn't fulfill cond: Reject the mode and add to the most correlated mode 
                selected_modes[max_overlap_idx].append(hj_prime_idx)
                selected_labels[max_overlap_idx].append(hj_prime_label)
                

        ###### Step N+1: Handle remaining modes as h_M ######
        
        # Get indices of remaining modes with SORTED indices
        all_original_indices = set(top_indices[:M_init])  # All original indices we considered
        remaining_ori_indices = list(all_original_indices - set(processed_ori_indices))

        # Continue only if there are remaining modes
        if remaining_ori_indices:

            # Check condition 1 : <h_M|h_M> > 1
            hM_inner = self.inner_approx(remaining_ori_indices, remaining_ori_indices)
            cond_one = hM_inner > 1

            # Check condition 2 : <h_sel|h_M> << threshold
            inners_w_sel = []
            max_inner_with_sel = 0
            max_inner_with_sel_idx = -1

            for k, selected_mode in enumerate(selected_modes):
                selM_inner = self.inner_approx(selected_mode, remaining_ori_indices)
                inners_w_sel.append(selM_inner)

                if selM_inner > max_inner_with_sel:
                    max_inner_with_sel = selM_inner
                    max_inner_with_sel_idx = k

            cond_two = max_inner_with_sel < threshold

            # Check if both conditions are fulfilled
            if cond_one and cond_two:
                # Fulfill cond: Accept the mode as h_M
                selected_modes.append(remaining_ori_indices)
                hM_labels = [(self.amp.l_arr[idx].item(), self.amp.m_arr[idx].item(), self.amp.n_arr[idx].item()) for idx in remaining_ori_indices]
                selected_labels.append(hM_labels)
                
            # Cond one violated (not ortho), but cond two fulfilled
            elif not cond_one and cond_two:
                # Reject h_M (becomes error term)
                pass

            # Cond one fulfilled, but cond two violated
            elif cond_one and not cond_two:
                # Add h_M to the most correlated mode
                selected_modes[max_inner_with_sel_idx].extend(remaining_ori_indices)
                remaining_labels = [(self.amp.l_arr[idx].item(), self.amp.m_arr[idx].item(), self.amp.n_arr[idx].item()) for idx in remaining_ori_indices]
                selected_labels[max_inner_with_sel_idx].extend(remaining_labels)
            
            # Both conditions violated
            else:
                # Reject h_M (becomes error term)
                pass
        
        if self.verbose:
            print(f"Final selected labels: {selected_labels}")
            print(f"Number of mode groups returned: {len(selected_labels)}")
            
        return selected_modes, selected_labels